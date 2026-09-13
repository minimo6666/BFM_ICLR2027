#!/usr/bin/env python3
"""Fixed-t=1 direct-X0 oracle trained with plain BCE only."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision.transforms import CenterCrop, Compose, RandomCrop, RandomHorizontalFlip, Resize, ToTensor


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/data/0/mohao/data/lsun/scenes"))
    parser.add_argument("--ae-load-dir", type=Path, default=PROJECT_ROOT / "logs/BAE_C64")
    parser.add_argument("--ae-load-step", type=int, default=8_100_000)
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--val-images", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--update-freq", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260910)
    return parser.parse_args()


def distributed_setup():
    if "RANK" not in os.environ:
        raise RuntimeError("Launch with torch.distributed.run; this oracle uses baseline DDP.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, dist.get_rank(), dist.get_world_size()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_hparams(args):
    from hparams.defaults.binarygan_default import HparamsBinaryAE
    from hparams.defaults.sampler_defaults import HparamsBianryLatent

    h = HparamsBinaryAE("churches")
    h.vqgan_batch_size = h.batch_size
    h.update(HparamsBianryLatent("churches"))
    h.update(
        sampler="flow_lsun",
        codebook_size=64,
        img_size=256,
        total_steps=64,
        sample_steps=64,
        batch_size=args.batch_size,
        latent_shape=[1, 16, 16],
        loss_final="mean",
        p_flip=False,
        norm_first=True,
        amp=True,
        aux=0.0,
        focal=0,
        use_softmax=False,
        guidance=False,
        cross=False,
        ae_load_dir=str(args.ae_load_dir),
        ae_load_step=args.ae_load_step,
        lr=args.learning_rate,
        warmup_iters=0,
        weight_decay=0.0,
        update_freq=args.update_freq,
    )
    return h


def freeze(module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def load_autoencoder(args, h, device):
    from models.binaryae import BinaryAutoEncoder
    from utils.sampler_utils import retrieve_autoencoder_components_state_dicts

    state = retrieve_autoencoder_components_state_dicts(
        h, ["encoder", "quantize", "generator"], remove_component_from_key=False
    )
    model = BinaryAutoEncoder(h)
    model.load_state_dict(state, strict=True)
    del state
    model = model.to(device)
    freeze(model)
    return model


def make_train_loader(args, rank, world_size):
    transform = Compose(
        [Resize(int(256 * 1.05)), RandomCrop(256), RandomHorizontalFlip(p=0.5), ToTensor()]
    )
    dataset = torchvision.datasets.LSUN(
        str(args.data_root), classes=["church_outdoor_train"], transform=transform
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=True
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed + 200 + rank)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        generator=generator,
    )
    return loader, sampler


def make_val_loader(args):
    transform = Compose([Resize(int(256 * 1.05)), CenterCrop(256), ToTensor()])
    dataset = torchvision.datasets.LSUN(
        str(args.data_root), classes=["church_outdoor_train"], transform=transform
    )
    if args.val_images > len(dataset):
        raise ValueError(f"val-images={args.val_images}, available={len(dataset)}")
    return DataLoader(
        Subset(dataset, range(args.val_images)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def to_flat(code):
    batch, channels, _, _ = code.shape
    return code.view(batch, channels, -1).permute(0, 2, 1).contiguous()


@torch.inference_mode()
def build_validation_latents(autoencoder, loader, device, seed):
    torch.cuda.manual_seed_all(seed)
    batches = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        batches.append(to_flat(autoencoder(images, code_only=True)).cpu())
    return torch.cat(batches, dim=0)


def sample_x1(x0, generator=None):
    # Physical t=1 of the current 64-step linear BFM path.
    tau = 63.0 / 64.0
    probability = (1.0 + (2.0 * x0 - 1.0) * tau) / 2.0
    return torch.bernoulli(probability, generator=generator)


@torch.inference_mode()
def validate(model, val_x0, args, device, step):
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1_000_000)
    totals = {
        "loss_sum": 0.0, "error_bits": 0, "corrupted_bits": 0,
        "corrected_bits": 0, "clean_bits": 0, "introduced_bits": 0, "all_bits": 0,
        "changed_bits": 0,
    }
    for start in range(0, val_x0.shape[0], args.batch_size):
        x0 = val_x0[start : start + args.batch_size].to(device)
        x1 = sample_x1(x0, generator=generator)
        network_time = torch.zeros(x0.shape[0], device=device, dtype=torch.long)
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(x1, time_steps=network_time)
        logits = logits.float()
        pred_x0 = torch.sigmoid(logits) >= 0.5
        truth = x0.bool()
        corrupted = x1.bool() != truth
        clean = ~corrupted
        corrected = corrupted & (pred_x0 == truth)
        introduced = clean & (pred_x0 != truth)
        changed = pred_x0 != x1.bool()
        errors = pred_x0 != truth
        totals["loss_sum"] += float(
            F.binary_cross_entropy_with_logits(logits, x0.float(), reduction="sum").item()
        )
        totals["error_bits"] += int(errors.sum().item())
        totals["corrupted_bits"] += int(corrupted.sum().item())
        totals["corrected_bits"] += int(corrected.sum().item())
        totals["clean_bits"] += int(clean.sum().item())
        totals["introduced_bits"] += int(introduced.sum().item())
        totals["changed_bits"] += int(changed.sum().item())
        totals["all_bits"] += x0.numel()
    model.train()
    return {
        "input_ber": totals["corrupted_bits"] / totals["all_bits"],
        "pred_x0_ber": totals["error_bits"] / totals["all_bits"],
        "correction_recall": totals["corrected_bits"] / totals["corrupted_bits"],
        "introduced_error_rate": totals["introduced_bits"] / totals["clean_bits"],
        "correction_precision": totals["corrected_bits"] / max(totals["changed_bits"], 1),
        "changed_rate": totals["changed_bits"] / totals["all_bits"],
        "loss": totals["loss_sum"] / totals["all_bits"],
    }


def write_rows(path, rows):
    fields = [
        "step", "train_loss", "input_ber", "pred_x0_ber",
        "correction_recall", "introduced_error_rate",
        "correction_precision", "changed_rate", "loss", "step_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path, rows):
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes[0, 0].plot(steps, [100 * row["pred_x0_ber"] for row in rows], "o-", label="pred X0 BER")
    axes[0, 0].plot(steps, [100 * row["input_ber"] for row in rows], "k--", label="identity / input BER")
    axes[0, 0].axhline(0.78125, color="gray", linestyle=":", label="theory 0.78125%")
    axes[0, 0].set_ylabel("Percent (%)")
    axes[0, 0].set_title("BER")
    axes[0, 0].legend()
    axes[0, 1].plot(steps, [100 * row["correction_recall"] for row in rows], "o-", color="#2ca02c")
    axes[0, 1].set_ylabel("Percent (%)")
    axes[0, 1].set_title("Correction recall on corrupted bits")
    axes[1, 0].plot(steps, [100 * row["introduced_error_rate"] for row in rows], "o-", color="#d62728")
    axes[1, 0].set_ylabel("Percent (%)")
    axes[1, 0].set_title("Introduced error on clean bits")
    axes[1, 1].plot(steps, [row["loss"] for row in rows], "o-", color="#9467bd")
    axes[1, 1].set_ylabel("Plain BCE")
    axes[1, 1].set_title("Validation loss")
    for axis in axes.flat:
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.25)
    fig.suptitle("t=1 Direct-X0 Plain-BCE Oracle")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_and_record(model, val_x0, args, device, step, rows, extra=None):
    dist.barrier()
    if dist.get_rank() == 0:
        metrics = validate(model.module, val_x0, args, device, step)
        row = {"step": step, **(extra or {}), **metrics}
        rows.append(row)
        write_rows(args.output_dir / "metrics.csv", rows)
        write_plot(args.output_dir / "t1_direct_x0_bce_oracle.png", rows)
        print(json.dumps(row), flush=True)
    dist.barrier()


def main():
    from models.transformer import TransformerBD
    from utils.train_utils import NativeScalerWithGradNormCount

    args = parse_args()
    local_rank, rank, world_size = distributed_setup()
    device = torch.device("cuda", local_rank)
    if world_size != 2:
        raise RuntimeError(f"Expected baseline 2-rank DDP, got world_size={world_size}")
    set_seed(args.seed)
    h = build_hparams(args)
    autoencoder = load_autoencoder(args, h, device)

    # Every rank constructs exactly the same from-scratch predictor.
    model = TransformerBD(h).to(device).train()
    model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        model.module.parameters(), lr=args.learning_rate, weight_decay=0.0,
        betas=(0.9, 0.95), eps=1e-8,
    )
    scaler = NativeScalerWithGradNormCount(True, 0)
    train_loader, train_sampler = make_train_loader(args, rank, world_size)

    val_x0 = None
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        val_x0 = build_validation_latents(
            autoencoder, make_val_loader(args), device, args.seed + 300
        )
        print(f"learning_rate = {args.learning_rate}", flush=True)
        print("warmup_steps = 0", flush=True)
        print("scheduler = constant", flush=True)
        print("physical_t = 1", flush=True)
        print("network_time = 0", flush=True)
        print("prediction_target = direct_X0", flush=True)
        print("loss = binary_cross_entropy_with_logits(logits, X0_GT.float())", flush=True)
        print("p_flip = False", flush=True)
        print("evaluation_split = church_outdoor_train", flush=True)
        print(f"evaluation_images = {args.val_images}", flush=True)
        print("evaluation_corruption_seed = seed + 1000000 + step", flush=True)
        print(
            f"world_size = {world_size}; per_rank_batch = {args.batch_size}; "
            f"update_freq = {args.update_freq}; global_batch = "
            f"{world_size * args.batch_size * args.update_freq}",
            flush=True,
        )

    # Training corruption stays fresh and is independent across ranks.
    set_seed(args.seed + 10_000 + rank)
    rows = []
    evaluate_and_record(model, val_x0, args, device, 0, rows)
    optimizer.zero_grad(set_to_none=True)
    train_sampler.set_epoch(0)
    data_iterator = iter(train_loader)
    epoch = 0

    for step in range(1, args.train_steps + 1):
        step_start = time.time()
        train_losses = []
        for micro_step in range(args.update_freq):
            try:
                images, _ = next(data_iterator)
            except StopIteration:
                epoch += 1
                train_sampler.set_epoch(epoch)
                data_iterator = iter(train_loader)
                images, _ = next(data_iterator)
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                x0 = to_flat(autoencoder(images, code_only=True)).detach()
                x1 = sample_x1(x0)
            network_time = torch.zeros(x0.shape[0], device=device, dtype=torch.long)
            sync_context = model.no_sync() if micro_step < args.update_freq - 1 else nullcontext()
            with sync_context:
                with torch.cuda.amp.autocast(enabled=True):
                    logits = model(x1, time_steps=network_time)
                    loss = F.binary_cross_entropy_with_logits(logits, x0.float())
                    scaled_loss = loss / args.update_freq
                scaler(
                    scaled_loss, optimizer, clip_grad=0.0,
                    parameters=model.module.parameters(), create_graph=False,
                    update_grad=(micro_step == args.update_freq - 1),
                )
            train_losses.append(float(loss.detach().item()))
        optimizer.zero_grad(set_to_none=True)

        if step % args.eval_every == 0:
            if rank == 0:
                print(
                    json.dumps({
                        "completed_step": step,
                        "train_loss": float(np.mean(train_losses)),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "step_seconds": time.time() - step_start,
                    }),
                    flush=True,
                )
            evaluate_and_record(model, val_x0, args, device, step, rows)

    if rank == 0:
        print("COMPLETE t1_direct_x0_bce_oracle", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
