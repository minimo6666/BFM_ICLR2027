#!/usr/bin/env python3
"""Official TransformerBD + BinaryDiffusion training on packed BAE-C64 bits."""

import copy
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from hparams import get_sampler_hparams
from models.binaryae import Generator
from utils.log_utils import config_log, load_model, log, log_stats, save_model, start_training_log
from utils.lr_sched import adjust_lr, lr_scheduler
from utils.sampler_utils import get_sampler, retrieve_autoencoder_components_state_dicts
from utils.train_utils import EMA, NativeScalerWithGradNormCount


class PackedBinaryLatentDataset(Dataset):
    def __init__(self, path):
        self.path = str(Path(path).expanduser().resolve())
        self.data = np.load(self.path, mmap_mode="r")
        if self.data.shape[1:] != (2048,) or self.data.dtype != np.uint8:
            raise ValueError(f"Expected uint8 [N,2048], got {self.data.shape} {self.data.dtype}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        packed = np.asarray(self.data[index])
        bits = np.unpackbits(packed, count=256 * 64, bitorder="little")
        return torch.from_numpy(bits.reshape(256, 64).copy()).float()


@torch.no_grad()
def preview(H, sampler, generator, embedding, step):
    was_training = sampler.training
    sampler.eval()
    with torch.cuda.amp.autocast(enabled=H.amp):
        bits = sampler.sample(sample_steps=64, temp=1.0, b=64, return_all=False).float()
        quantized = (bits @ embedding).permute(0, 2, 1).reshape(64, H.emb_dim, 16, 16)
        images = []
        for chunk in torch.split(quantized, 8):
            images.append(generator(chunk).float().cpu().clamp(0, 1))
    images = torch.cat(images)
    output = Path(H.log_dir) / "previews" / f"samples_step_{step:06d}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, output, nrow=8, padding=0)
    sampler.train(was_training)
    print(f"Saved official-BLD 8x8 EMA preview at step {step}: {output}", flush=True)


def main(H):
    if H.sampler != "bld":
        raise ValueError("This entry supports only the official sampler='bld'")
    if not H.p_flip or float(H.focal) != 0.0 or float(H.aux) != 0.0:
        raise ValueError("Required official objective: p_flip=True, focal=0, aux=0")
    if not H.latent_cache:
        raise ValueError("--latent_cache is required")

    device = torch.device("cuda")
    ae_state = retrieve_autoencoder_components_state_dicts(
        H, ["quantize", "generator"], remove_component_from_key=True
    )
    embedding = ae_state.pop("embed.weight").to(device)
    generator = Generator(H)
    generator.load_state_dict(ae_state, strict=False)
    generator = generator.to(device).eval().requires_grad_(False)
    del ae_state

    sampler = get_sampler(H, embedding).to(device).train()
    ema_model = copy.deepcopy(sampler).eval().requires_grad_(False) if H.ema else None
    ema = EMA(H.ema_beta) if H.ema else None
    optimizer = torch.optim.AdamW(
        sampler.parameters(), lr=H.lr, weight_decay=H.weight_decay,
        betas=(0.9, 0.95), eps=H.optim_eps,
    )
    scaler = NativeScalerWithGradNormCount(H.amp, H.init_scale)
    completed = int(H.load_step)
    if completed:
        sampler = load_model(sampler, H.sampler, completed, H.load_dir, device=device).to(device)
        ema_model = load_model(ema_model, f"{H.sampler}_ema", completed, H.load_dir, device=device).to(device)
        if H.load_optim:
            optimizer = load_model(optimizer, f"{H.sampler}_optim", completed, H.load_dir, device=device)
            scaler_path = Path(H.load_dir) / "saved_models" / f"{H.sampler}_scaler_{completed}.th"
            scaler.load_state_dict(torch.load(scaler_path, map_location="cpu"))

    dataset = PackedBinaryLatentDataset(H.latent_cache)
    loader = DataLoader(
        dataset, batch_size=H.batch_size, shuffle=True, drop_last=True,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )
    iterator = iter(loader)
    schedule = lr_scheduler(
        base_value=H.lr, final_value=1e-6, iters=H.train_steps + 1,
        warmup_steps=H.warmup_iters, start_warmup_value=1e-6, lr_type="constant",
    )
    log(f"Official BLD params: {sum(p.numel() for p in sampler.parameters()) / 1e6:.3f}M")
    log(f"Packed cache: {dataset.path}, shape={dataset.data.shape}, dtype={dataset.data.dtype}")
    log(f"Objective: p_flip={H.p_flip}, focal={H.focal}, aux={H.aux}, loss_final={H.loss_final}")
    log(f"Batch={H.batch_size}; optimizer steps={H.train_steps}; EMA beta={H.ema_beta}")

    optimizer.zero_grad(set_to_none=True)
    while completed < H.train_steps:
        try:
            bits = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            bits = next(iterator)
        bits = bits.to(device, non_blocking=True)
        adjust_lr(optimizer, schedule, completed)
        started = time.time()
        with torch.cuda.amp.autocast(enabled=H.amp):
            stats = sampler(bits)
            loss = stats["loss"]
        grad_norm = scaler(
            loss, optimizer, clip_grad=H.grad_norm,
            parameters=sampler.parameters(), update_grad=True,
        )
        optimizer.zero_grad(set_to_none=True)
        completed += 1
        if H.ema and completed % H.steps_per_update_ema == 0:
            ema.update_model_average(ema_model, sampler)

        if completed % H.steps_per_log == 0:
            logged = dict(stats)
            logged.update(lr=optimizer.param_groups[0]["lr"], step_time=time.time() - started, grad_norm=grad_norm)
            if "scale" in scaler.state_dict():
                logged["loss scale"] = scaler.state_dict()["scale"]
            log_stats(completed, logged)
        if completed % H.steps_per_save_output == 0:
            preview(H, ema_model if H.ema else sampler, generator, embedding, completed)
        if completed % H.steps_per_checkpoint == 0:
            save_model(sampler, H.sampler, completed, H.log_dir)
            save_model(optimizer, f"{H.sampler}_optim", completed, H.log_dir)
            save_model(scaler, f"{H.sampler}_scaler", completed, H.log_dir)
            if H.ema:
                save_model(ema_model, f"{H.sampler}_ema", completed, H.log_dir)
    (Path(H.log_dir) / "TRAINING_COMPLETE").touch()


if __name__ == "__main__":
    H = get_sampler_hparams()
    config_log(H.log_dir)
    log("---------------------------------")
    log("Official BLD + packed BAE-C64 LSUN Churches")
    start_training_log(H)
    main(H)
