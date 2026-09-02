#!/usr/bin/env python3
"""Resumable 10k sampling for a trained V9 boundary head."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[4]
for path in (REPO_ROOT, SCRIPT.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from v9_runtime import (  # noqa: E402
    build_hparams,
    ensure_device,
    load_autoencoder,
    read_completed,
    reset_sampling_seed,
    seed_everything,
    sequence_to_binary_code,
    validate_existing_images,
    write_json,
)
from models.binarylatent_flow_boundary_repair_v9 import (  # noqa: E402
    BinaryDiffusionFlowBoundaryRepairV9,
)
from models.transformer_boundary_repair_v9 import (  # noqa: E402
    TransformerBDBoundaryRepairV9,
)
from train_boundary_repair_v9 import load_v8_into_v9, torch_load  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8-checkpoint", type=Path, required=True)
    parser.add_argument("--boundary-checkpoint", type=Path, required=True)
    parser.add_argument("--ae-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nfe", type=int, choices=(64, 32), required=True)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--head-off",
        action="store_true",
        help="Disable the trained terminal head for an exact V8 alpha=1 identity check.",
    )
    return parser.parse_args()


def load_head(model, checkpoint: Path):
    payload = torch_load(checkpoint.expanduser().resolve())
    if payload.get("format") != "bfm_v9_boundary_head_v1":
        raise ValueError(f"Not a V9 boundary checkpoint: {checkpoint}")
    model._denoise_fn.terminal_carry.load_state_dict(
        payload["ema_head_state"], strict=True
    )
    return int(payload["step"])


def main():
    args = parse_args()
    if args.num_samples < 1 or args.batch_size < 1:
        raise ValueError("sample and batch counts must be positive")
    seed_everything(args.seed)
    device = ensure_device(args.device)
    hparams = build_hparams(batch_size=args.batch_size)
    denoiser = TransformerBDBoundaryRepairV9(hparams)
    model = BinaryDiffusionFlowBoundaryRepairV9(
        hparams, denoiser, hparams.codebook_size
    )
    load_v8_into_v9(model, args.v8_checkpoint.expanduser().resolve())
    trained_step = load_head(model, args.boundary_checkpoint)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    autoencoder = load_autoencoder(hparams, args.ae_checkpoint, device)

    image_dir = args.output.expanduser().resolve()
    progress_path = image_dir.parent / "progress.json"
    protocol_path = image_dir.parent / "protocol.json"
    image_dir.mkdir(parents=True, exist_ok=True)
    validate_existing_images(image_dir, args.num_samples)
    protocol = {
        "v8_checkpoint": str(args.v8_checkpoint.expanduser().resolve()),
        "boundary_checkpoint": str(args.boundary_checkpoint.expanduser().resolve()),
        "boundary_head": "ema",
        "trained_step": trained_step,
        "nfe": args.nfe,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "temperature": args.temperature,
        "hard_final": True,
        "adapter_scale": 1.0,
        "boundary_repair": not args.head_off,
        "seed_rule": "seed + 1000003*nfe + 7919*batch_index",
    }
    if protocol_path.is_file():
        with protocol_path.open("r", encoding="utf-8") as handle:
            if json.load(handle) != protocol:
                raise RuntimeError("Existing output uses another protocol")
    write_json(protocol_path, protocol)

    completed = read_completed(progress_path)
    print(f"V9 NFE={args.nfe}: resuming at {completed}/{args.num_samples}")
    with torch.inference_mode():
        while completed < args.num_samples:
            current_batch = min(args.batch_size, args.num_samples - completed)
            batch_index = completed // args.batch_size
            paired_seed = args.seed + 1000003 * args.nfe + 7919 * batch_index
            reset_sampling_seed(paired_seed, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=args.amp
            ):
                latent = model.sample(
                    sample_steps=args.nfe,
                    temp=args.temperature,
                    b=current_batch,
                    adapter_scale=1.0,
                    boundary_repair=not args.head_off,
                )
                code = sequence_to_binary_code(latent)
                decoded, _, _ = autoencoder(None, code=code)
            for offset, image in enumerate(decoded.float().clamp(0, 1).cpu()):
                torchvision.utils.save_image(
                    image, image_dir / f"{completed + offset:06d}.png", padding=0
                )
            completed += current_batch
            write_json(
                progress_path,
                {
                    "completed": completed,
                    "num_samples": args.num_samples,
                    "nfe": args.nfe,
                    "last_paired_seed": paired_seed,
                },
            )
            if completed % 512 == 0 or completed == args.num_samples:
                print(f"V9 NFE={args.nfe}: {completed}/{args.num_samples}", flush=True)

    count = len(list(image_dir.glob("*.png")))
    if count != args.num_samples:
        raise RuntimeError(f"Expected {args.num_samples} PNGs, found {count}")
    print(f"V9 NFE={args.nfe} generation complete: {image_dir}")


if __name__ == "__main__":
    main()
