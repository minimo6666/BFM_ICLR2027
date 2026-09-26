#!/usr/bin/env python3
"""Train official SEDD-small on BAE binary latents without changing its loss."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image

from sedd_binary_common import (
    PROJECT_ROOT,
    ExponentialMovingAverage,
    build_strict_64_nfe_sampler,
    create_sedd_components,
    decode_bits,
    encode_images,
    latent_to_tokens,
    load_autoencoder,
    losses,
    tokens_to_latent,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="/mnt/data/0/mohao/data/lsun/scenes")
    parser.add_argument(
        "--latent-cache",
        default="",
        help=(
            "Optional uint8 .npy array shaped [N,2048], produced with "
            "numpy.packbits(..., bitorder='little'). When supplied, training "
            "reads the cached BAE bits directly and does not run the encoder."
        ),
    )
    parser.add_argument("--ae-load-dir", required=True)
    parser.add_argument("--ae-load-step", type=int, default=8_100_000)
    parser.add_argument("--train-steps", type=int, required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--preview-every", type=int, default=2000)
    parser.add_argument("--preview-num-images", type=int, default=64)
    parser.add_argument("--preview-batch-size", type=int, default=2)
    parser.add_argument("--preview-decode-batch-size", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-micro-steps", type=int, default=0)
    return parser.parse_args()


class PackedLatentDataset(Dataset):
    """Memory-mapped LSUN BAE latents stored as little-endian packed bits."""

    PACKED_WIDTH = 16384 // 8

    def __init__(self, path: str):
        self.path = str(Path(path).expanduser().resolve())
        self.array = np.load(self.path, mmap_mode="r")
        if self.array.ndim != 2 or self.array.shape[1] != self.PACKED_WIDTH:
            raise ValueError(
                f"Expected packed latent cache [N,{self.PACKED_WIDTH}], "
                f"got {self.array.shape}"
            )
        if self.array.dtype != np.uint8:
            raise ValueError(f"Expected uint8 packed latent cache, got {self.array.dtype}")

    def __len__(self):
        return self.array.shape[0]

    def __getitem__(self, index):
        packed = np.asarray(self.array[index])
        bits = np.unpackbits(
            packed, count=16384, bitorder="little"
        ).reshape(16384)
        return torch.from_numpy(bits)


def save_checkpoint(path: Path, state: dict, model_core, noise_core, cfg, args):
    checkpoint = {
        "optimizer": state["optimizer"].state_dict(),
        "model": model_core.state_dict(),
        "noise": noise_core.state_dict(),
        "ema": state["ema"].state_dict(),
        "step": int(state["step"]),
        "protocol": {
            "method": "official SEDD binary specialization",
            "tokens": 2,
            "latent_shape": [64, 16, 16],
            "sequence_length": 16384,
            "transformer_sequence_length": 1024,
            "graph": "Uniform(2)",
            "noise": "GeometricNoise(1e-4,20)",
            "loss": "official Score Entropy",
            "model": "official SEDD-small DDiT",
            "micro_batch_per_gpu": args.batch_size,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "gradient_accumulation": args.accum,
            "effective_global_batch": args.batch_size
            * args.accum
            * (dist.get_world_size() if dist.is_initialized() else 1),
            "sampling": "official analytic predictor, 63 predictor + 1 denoise",
            "sampling_nfe": 64,
        },
        "config": json.loads(json.dumps(dict(cfg), default=str)),
    }
    torch.save(checkpoint, path)


@torch.no_grad()
def save_preview(
        model_core, noise_core, graph, ema, autoencoder, preview_dir, step,
        device, num_images, sample_batch_size, decode_batch_size):
    ema.store(model_core.parameters())
    ema.copy_to(model_core.parameters())
    model_core.eval()
    try:
        decoded = []
        torch.cuda.empty_cache()
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(2020)
            torch.cuda.manual_seed_all(2020)
            for start in range(0, num_images, sample_batch_size):
                batch_size = min(sample_batch_size, num_images - start)
                sampler = build_strict_64_nfe_sampler(
                    graph, noise_core, batch_size, device
                )
                tokens = sampler(model_core)
                bits = tokens_to_latent(tokens)
                for offset in range(0, batch_size, decode_batch_size):
                    images = decode_bits(
                        autoencoder, bits[offset:offset + decode_batch_size]
                    )
                    decoded.append(images.float().cpu().clamp(0.0, 1.0))
                del tokens, bits
        images = torch.cat(decoded, dim=0)
        save_image(
            images,
            preview_dir / f"samples_step_{step:06d}.png",
            nrow=8,
            padding=0,
        )
    finally:
        ema.restore(model_core.parameters())
        model_core.train()
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    if args.preview_num_images <= 0:
        raise ValueError("preview-num-images must be positive")
    if args.preview_batch_size <= 0 or args.preview_decode_batch_size <= 0:
        raise ValueError("preview batch sizes must be positive")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    preview_dir = output_dir / "previews"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "launch_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    # Cached latents remove the BAE encoder from the hot training path. Keep it
    # available only when previews were explicitly requested.
    autoencoder = None
    if not args.latent_cache or args.preview_every > 0:
        autoencoder = load_autoencoder(args.ae_load_dir, args.ae_load_step, device)
    cfg, graph, noise_core, model_core = create_sedd_components(device)
    cfg.training.accum = args.accum
    model = (
        DDP(model_core, device_ids=[local_rank], static_graph=True, find_unused_parameters=True)
        if distributed
        else model_core
    )
    noise = (
        DDP(noise_core, device_ids=[local_rank], static_graph=True)
        if distributed
        else noise_core
    )
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.training.ema)
    optimizer = losses.get_optimizer(cfg, chain(model.parameters(), noise.parameters()))
    scaler = torch.cuda.amp.GradScaler()
    state = {
        "optimizer": optimizer,
        "scaler": scaler,
        "model": model,
        "noise": noise,
        "ema": ema,
        "step": 0,
    }
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model_core.load_state_dict(checkpoint["model"], strict=True)
        noise_core.load_state_dict(checkpoint.get("noise", {}), strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema.load_state_dict(checkpoint["ema"])
        state["step"] = int(checkpoint["step"])

    if args.latent_cache:
        dataset = PackedLatentDataset(args.latent_cache)
        cache_sampler = (
            DistributedSampler(dataset, shuffle=True, seed=args.seed)
            if distributed
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=cache_sampler is None,
            sampler=cache_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
        )
        if rank == 0:
            print(
                f"Training from packed latent cache: {dataset.path} "
                f"({len(dataset)} examples)",
                flush=True,
            )
    else:
        os.chdir(PROJECT_ROOT)
        from utils.reliable_data_utils import get_data_loaders

        loader, _ = get_data_loaders(
            "churches",
            256,
            args.batch_size,
            custom_dataset_path=args.data_root,
            num_workers=args.num_workers,
            distributed=distributed,
            random=True,
            get_val_dataloader=False,
            args=args,
        )
    sampler = loader.sampler
    epoch = 0
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    iterator = iter(loader)
    optimize_fn = losses.optimization_manager(cfg)
    train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, args.accum)
    began = time.time()
    micro_steps = 0

    if rank == 0:
        parameters = sum(parameter.numel() for parameter in model_core.parameters())
        print(f"Official SEDD parameters: {parameters/1e6:.3f}M", flush=True)
        print(
            f"Effective global batch: {args.batch_size * args.accum * world_size}",
            flush=True,
        )
        print("Loss: unmodified official Score Entropy; Uniform(2); coordinates=16384; DDiT tokens=1024", flush=True)
        print("Sampling protocol: 63 analytic steps + denoise = exactly 64 NFE", flush=True)

    while state["step"] < args.train_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
        if args.latent_cache:
            tokens = batch.to(device=device, dtype=torch.long, non_blocking=True)
        else:
            images, _ = batch
            images = images.to(device, non_blocking=True)
            tokens = latent_to_tokens(encode_images(autoencoder, images))
            del images
        previous_step = int(state["step"])
        loss = train_step_fn(state, tokens)
        micro_steps += 1
        completed = int(state["step"])

        if args.smoke_micro_steps and micro_steps >= args.smoke_micro_steps:
            if rank == 0:
                print(f"Smoke complete after {micro_steps} micro-steps", flush=True)
            break
        if completed == previous_step:
            continue

        reduced_loss = loss.detach().clone()
        if distributed:
            dist.all_reduce(reduced_loss)
            reduced_loss /= world_size
        if rank == 0 and completed % args.log_every == 0:
            record = {
                "step": completed,
                "score_entropy_loss": float(reduced_loss),
                "elapsed_seconds": time.time() - began,
            }
            print(json.dumps(record), flush=True)
            with (output_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")

        preview_due = args.preview_every > 0 and completed % args.preview_every == 0
        if preview_due and rank == 0:
            save_preview(
                model_core,
                noise_core,
                graph,
                ema,
                autoencoder,
                preview_dir,
                completed,
                device,
                args.preview_num_images,
                args.preview_batch_size,
                args.preview_decode_batch_size,
            )
            print(
                f"Saved {args.preview_num_images}-image 8x8 strict-64-NFE "
                f"preview at step {completed}",
                flush=True,
            )
        if preview_due and distributed:
            dist.barrier()

        checkpoint_due = completed % args.checkpoint_every == 0 or completed == args.train_steps
        if checkpoint_due and rank == 0:
            checkpoint_path = checkpoint_dir / f"binary_sedd_ema_step_{completed:06d}.pt"
            save_checkpoint(checkpoint_path, state, model_core, noise_core, cfg, args)
            save_checkpoint(checkpoint_dir / "latest.pt", state, model_core, noise_core, cfg, args)
            print(f"Saved {checkpoint_path}", flush=True)
        if checkpoint_due and distributed:
            dist.barrier()

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
