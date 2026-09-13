#!/usr/bin/env python3
"""Stage 1: masked clean-pattern pretraining on cached deterministic X0."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[4]
ORACLE_SCRIPTS = (
    PROJECT_ROOT
    / "experiments/ICLR27/diagnostics/t1_direct_x0_bce_oracle/scripts"
)
for path in (PROJECT_ROOT, ORACLE_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_direct_x0 as base
import train_cached_x0 as cache_utils
from experiments.ICLR27.diagnostics.masked_bit_pretrain_core import (
    MaskedBitConfig,
    MaskedBitPatternPretrainer,
    T1_OBSERVATION_RELIABILITY,
    export_transformerbd_state,
    sample_bit_mask,
)
from utils.train_utils import NativeScalerWithGradNormCount


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-images", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--update-freq", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--ae-load-dir", type=Path, default=PROJECT_ROOT / "logs/BAE_C64"
    )
    parser.add_argument("--ae-load-step", type=int, default=8_100_000)
    return parser.parse_args()


@torch.inference_mode()
def validate(model, eval_x0, args, device):
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 2_000_000)
    totals = {
        "bce_sum": 0.0,
        "error_bits": 0,
        "correct_confidence_sum": 0.0,
        "strong_context_bits": 0,
        "masked_bits": 0,
        "all_bits": 0,
    }
    for start in range(0, eval_x0.shape[0], args.batch_size):
        x0 = eval_x0[start : start + args.batch_size].to(device)
        bit_mask = sample_bit_mask(x0, args.mask_ratio, generator=generator)
        with torch.cuda.amp.autocast(enabled=True):
            out = model(x0, bit_mask=bit_mask)
        logits = out["logits"].float()
        target = x0.bool()
        pred = logits >= 0.0
        probabilities = torch.sigmoid(logits)
        correct_probability = torch.where(target, probabilities, 1.0 - probabilities)
        per_bit_bce = F.binary_cross_entropy_with_logits(
            logits, x0.float(), reduction="none"
        )
        totals["bce_sum"] += float(per_bit_bce[bit_mask].sum().item())
        totals["error_bits"] += int(((pred != target) & bit_mask).sum().item())
        totals["correct_confidence_sum"] += float(
            correct_probability[bit_mask].sum().item()
        )
        totals["strong_context_bits"] += int(
            (correct_probability[bit_mask] > T1_OBSERVATION_RELIABILITY)
            .sum()
            .item()
        )
        totals["masked_bits"] += int(bit_mask.sum().item())
        totals["all_bits"] += x0.numel()
    model.train()
    masked_bits = totals["masked_bits"]
    masked_ber = totals["error_bits"] / masked_bits
    return {
        "masked_bce": totals["bce_sum"] / masked_bits,
        "masked_ber": masked_ber,
        "masked_accuracy": 1.0 - masked_ber,
        "masked_gt_confidence": totals["correct_confidence_sum"] / masked_bits,
        "strong_context_fraction": totals["strong_context_bits"] / masked_bits,
        "mask_fraction": masked_bits / totals["all_bits"],
    }


def write_rows(path, rows):
    fields = [
        "step",
        "train_loss",
        "masked_bce",
        "masked_ber",
        "masked_accuracy",
        "masked_gt_confidence",
        "strong_context_fraction",
        "mask_fraction",
        "step_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path, rows):
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes[0, 0].plot(
        steps, [100 * row["masked_ber"] for row in rows], "o-", label="masked BER"
    )
    axes[0, 0].set_ylabel("Percent (%)")
    axes[0, 0].set_title("Masked-position BER")
    axes[0, 1].plot(
        steps,
        [row["masked_gt_confidence"] for row in rows],
        "o-",
        color="#2ca02c",
    )
    axes[0, 1].axhline(
        T1_OBSERVATION_RELIABILITY,
        color="gray",
        linestyle=":",
        label="t=1 observed-bit reliability",
    )
    axes[0, 1].set_title("Mean P(correct clean bit)")
    axes[0, 1].legend()
    axes[1, 0].plot(
        steps,
        [100 * row["strong_context_fraction"] for row in rows],
        "o-",
        color="#d62728",
    )
    axes[1, 0].set_ylabel("Percent (%)")
    axes[1, 0].set_title("Context confidence > 0.9921875")
    axes[1, 1].plot(
        steps, [row["masked_bce"] for row in rows], "o-", color="#9467bd"
    )
    axes[1, 1].set_title("Masked-only plain BCE")
    for axis in axes.flat:
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.25)
    fig.suptitle("Stage 1: 10% Masked Clean-Pattern Pretraining")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_and_record(model, eval_x0, args, device, step, rows, extra):
    dist.barrier()
    if dist.get_rank() == 0:
        metrics = validate(model.module, eval_x0, args, device)
        row = {"step": step, **extra, **metrics}
        rows.append(row)
        write_rows(args.output_dir / "metrics.csv", rows)
        write_plot(args.output_dir / "masked_pretrain.png", rows)
        print(json.dumps(row), flush=True)
    dist.barrier()


def main():
    args = parse_args()
    local_rank, rank, world_size = base.distributed_setup()
    device = torch.device("cuda", local_rank)
    if world_size != 2:
        raise RuntimeError(f"expected baseline 2-rank DDP, got {world_size}")

    metadata = cache_utils.verify_cache(args.cache.resolve())
    base.set_seed(args.seed)
    h = base.build_hparams(args)
    h.p_flip = False
    h.deterministic = True
    cfg = MaskedBitConfig(mask_ratio=args.mask_ratio, network_time=0)
    model = MaskedBitPatternPretrainer(h, cfg).to(device).train()
    model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        model.module.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scaler = NativeScalerWithGradNormCount(True, 0)
    dataset = cache_utils.PackedX0Dataset(args.cache)
    loader, sampler = cache_utils.make_loader(dataset, args, rank, world_size)

    eval_x0 = None
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        eval_x0 = cache_utils.load_eval_x0(dataset, args.eval_images, device)
        sanity_x0 = eval_x0[: args.batch_size].to(device)
        sanity_generator = torch.Generator(device=device)
        sanity_generator.manual_seed(args.seed + 2_000_000)
        sanity_mask = sample_bit_mask(
            sanity_x0, args.mask_ratio, generator=sanity_generator
        )
        sanity_input = sanity_x0.float().clone()
        sanity_input.masked_fill_(sanity_mask, 0.5)
        if torch.unique(sanity_x0).cpu().tolist() != [0.0, 1.0]:
            raise RuntimeError("cached X0 is not binary")
        if not torch.all(sanity_input[sanity_mask] == 0.5):
            raise RuntimeError("masked input does not use neutral value 0.5")
        print(f"sanity_X0_shape = {list(sanity_x0.shape)}", flush=True)
        print(f"sanity_X0_values = {torch.unique(sanity_x0).cpu().tolist()}", flush=True)
        print(f"sanity_mask_fraction = {sanity_mask.float().mean().item()}", flush=True)
        print("masked_input_value = 0.5; embedding_contribution = 0", flush=True)
        print(f"cache = {args.cache.resolve()}", flush=True)
        print(f"cache_samples = {len(dataset)}", flush=True)
        print(f"cache_format = {metadata['format']}", flush=True)
        print("training_input = cached deterministic clean X0; no RGB; no BinaryAE", flush=True)
        print("stage = 1 masked clean-pattern pretraining", flush=True)
        print(f"mask_ratio = {args.mask_ratio}; training_mask = fresh", flush=True)
        print("network_time = 0; prediction_target = direct clean X0", flush=True)
        print("loss = plain BCE only on masked positions", flush=True)
        print("p_flip = False; BFM_corruption = False", flush=True)
        print("focal = 0; class_balance = False; sensitivity = False", flush=True)
        print(f"learning_rate = {args.learning_rate}; warmup_steps = 0", flush=True)
        print("scheduler = constant", flush=True)
        print("evaluation_split = first cached training latents", flush=True)
        print("evaluation_mask_seed = seed + 2000000 (fixed)", flush=True)
        print(
            f"world_size = {world_size}; per_rank_batch = {args.batch_size}; "
            f"update_freq = {args.update_freq}; global_batch = "
            f"{world_size * args.batch_size * args.update_freq}",
            flush=True,
        )

    base.set_seed(args.seed + 20_000 + rank)
    rows = []
    evaluate_and_record(
        model,
        eval_x0,
        args,
        device,
        0,
        rows,
        {"train_loss": float("nan"), "step_seconds": float("nan")},
    )
    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    sampler.set_epoch(epoch)
    iterator = iter(loader)

    for step in range(1, args.train_steps + 1):
        step_start = time.time()
        update_losses = []
        for micro in range(args.update_freq):
            try:
                packed = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                packed = next(iterator)
            x0 = cache_utils.unpack_x0(packed.to(device, non_blocking=True))
            sync_context = model.no_sync() if micro < args.update_freq - 1 else nullcontext()
            with sync_context:
                with torch.cuda.amp.autocast(enabled=True):
                    out = model(x0)
                    loss = out["loss"]
                    scaled_loss = loss / args.update_freq
                scaler(
                    scaled_loss,
                    optimizer,
                    clip_grad=0.0,
                    parameters=model.module.parameters(),
                    create_graph=False,
                    update_grad=(micro == args.update_freq - 1),
                )
            update_losses.append(float(loss.detach().item()))
        optimizer.zero_grad(set_to_none=True)

        if step % args.eval_every == 0:
            evaluate_and_record(
                model,
                eval_x0,
                args,
                device,
                step,
                rows,
                {
                    "train_loss": float(np.mean(update_losses)),
                    "step_seconds": time.time() - step_start,
                },
            )

    dist.barrier()
    if rank == 0:
        checkpoint = args.output_dir / "transformerbd_mask_pretrained.pt"
        export_transformerbd_state(
            model,
            str(checkpoint),
            extra={
                "step": args.train_steps,
                "seed": args.seed,
                "cache": str(args.cache.resolve()),
                "eval_images": args.eval_images,
                "learning_rate": args.learning_rate,
            },
        )
        (args.output_dir / "complete.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "stage": "masked_bit_pattern_pretrain",
                    "step": args.train_steps,
                    "checkpoint": str(checkpoint.resolve()),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"checkpoint = {checkpoint.resolve()}", flush=True)
        print("COMPLETE masked_bit_pattern_pretrain", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
