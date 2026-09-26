#!/usr/bin/env python3
"""Single-GPU official BiGR-L training on the packed LSUN Churches C64 cache."""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_c64 import PackedC64Dataset
from official_bigr_c64 import build_official_bigr_l


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


@torch.no_grad()
def update_ema(ema, model, decay):
    for ema_param, param in zip(ema.parameters(), model.parameters()):
        ema_param.mul_(decay).add_(param, alpha=1.0 - decay)


def infinite_batches(loader):
    while True:
        yield from loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-cache", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--micro-batch", type=int, default=48)
    parser.add_argument("--effective-batch", type=int, default=96)
    parser.add_argument("--train-steps", type=int, default=100_000)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--preview-every", type=int, default=2_000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.effective_batch % args.micro_batch:
        raise ValueError("effective batch must be divisible by micro batch")
    accumulation = args.effective_batch // args.micro_batch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    checkpoints = args.output_dir / "checkpoints"
    queue = args.output_dir / "preview_queue"
    logs = args.output_dir / "logs"
    for directory in (checkpoints, queue, logs):
        directory.mkdir(parents=True, exist_ok=True)

    dataset = PackedC64Dataset(args.latent_cache)
    loader = DataLoader(
        dataset, batch_size=args.micro_batch, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    batches = infinite_batches(loader)
    model = build_official_bigr_l().cuda().train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=2e-2
    )
    start_step = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(payload["model"])
        ema.load_state_dict(payload["ema"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        print(f"RESUMED step={start_step} from={args.resume}", flush=True)

    config = vars(args).copy()
    config.update(grad_accum=accumulation, model="official BiGR-L", precision="bf16")
    (args.output_dir / "train_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n"
    )
    log_path = logs / "train.jsonl"
    started = time.time()
    for step in range(start_step + 1, args.train_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "bce_loss": 0.0, "acc": 0.0}
        for _ in range(accumulation):
            bits, labels = next(batches)
            bits = bits.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, stats = model(inp=bits, cond_idx=labels, targets=bits)
                loss = stats["loss"] / accumulation
            loss.backward()
            for key in totals:
                totals[key] += float(stats[key].detach()) / accumulation
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient at optimizer step {step}")
        optimizer.step()
        update_ema(ema, model, 0.9999)

        if step == 1 or step % args.log_every == 0:
            record = dict(
                step=step, **totals, grad_norm=float(grad_norm),
                elapsed_sec=time.time() - started,
            )
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print("TRAIN", json.dumps(record), flush=True)

        checkpoint_payload = None
        if step % args.checkpoint_every == 0 or step == args.train_steps:
            checkpoint_payload = dict(
                step=step, model=model.state_dict(), ema=ema.state_dict(),
                optimizer=optimizer.state_dict(), config=config,
            )
            path = checkpoints / f"checkpoint_step_{step:06d}.pt"
            atomic_save(checkpoint_payload, path)
            print(f"CHECKPOINT {path}", flush=True)

        if step % args.preview_every == 0:
            preview = queue / f"ema_step_{step:06d}.pt"
            atomic_save(dict(step=step, ema=ema.state_dict(), config=config), preview)
            print(f"PREVIEW_QUEUED {preview}", flush=True)

    (args.output_dir / "TRAINING_COMPLETE").touch()
    print(f"TRAINING_COMPLETE steps={args.train_steps}", flush=True)


if __name__ == "__main__":
    main()
