#!/usr/bin/env python3
"""Encode every LSUN Churches train image once to deterministic packed clean X0 latents."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
POSTHOC = REPO_ROOT / "experiments/ICLR27/new_theory_sensity_aware_loss/BFM_V8_posthoc_analysis/experiments/ICLR27/new_theory_sensity_aware_loss/bfm_v8_posthoc_analysis"
for path in (REPO_ROOT, POSTHOC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis_common import (  # noqa: E402
    binary_code_to_sequence,
    build_hparams,
    ensure_device,
    load_autoencoder,
    make_analysis_loader,
    seed_everything,
    write_json,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=126227)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = ensure_device(args.device)
    H = build_hparams(batch_size=args.batch_size, ae_deterministic=True)
    ae = load_autoencoder(H, args.ae_checkpoint, device)
    loader = make_analysis_loader(
        H=H,
        data_root=args.data_root,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=args.seed,
    )

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_path = out / "latents_packed.npy"
    progress_path = out / "progress.json"
    packed_width = (16 * 16 * 64) // 8
    if data_path.exists():
        packed = np.load(data_path, mmap_mode="r+")
        done = int(json.loads(progress_path.read_text())["completed"]) if progress_path.is_file() else 0
    else:
        packed = np.lib.format.open_memmap(
            data_path, mode="w+", dtype=np.uint8, shape=(args.num_samples, packed_width)
        )
        done = 0
    if packed.shape != (args.num_samples, packed_width) or packed.dtype != np.uint8:
        raise RuntimeError("real latent memmap contract mismatch")

    encoded = 0
    with torch.inference_mode():
        for batch in loader:
            image = batch[0] if isinstance(batch, (tuple, list)) else batch
            batch_end = encoded + int(image.shape[0])
            if batch_end <= done:
                encoded = batch_end
                continue
            if encoded != done:
                raise RuntimeError("resume position is not aligned to dataloader batches")
            remaining = args.num_samples - encoded
            if remaining <= 0:
                break
            image = image[:remaining].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
                code = ae(image, code_only=True)
            sequence = binary_code_to_sequence(code)
            if sequence.shape[1:] != (256, 64):
                raise RuntimeError(f"unexpected real latent shape {tuple(sequence.shape)}")
            if not bool(torch.logical_or(sequence == 0, sequence == 1).all()):
                raise RuntimeError("BAE did not return hard binary codes")
            bits = sequence.to(torch.uint8).reshape(sequence.shape[0], -1).cpu().numpy()
            packed[encoded : encoded + len(bits)] = np.packbits(bits, axis=1, bitorder="little")
            packed.flush()
            encoded += len(bits)
            done = encoded
            write_json(progress_path, {"completed": done, "num_samples": args.num_samples})
            print(f"real: {done}/{args.num_samples}", flush=True)
            if done >= args.num_samples:
                break
    if done != args.num_samples:
        raise RuntimeError(f"dataset ended at {done}, expected {args.num_samples}")
    protocol = {
        "format": "t1_oracle_deterministic_clean_x0_packed_v1",
        "num_samples": args.num_samples,
        "source": "all images of deterministic-order LSUN Churches train split",
        "data_root": str(args.data_root.resolve()),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "amp": bool(args.amp),
        "ae_deterministic": True,
        "ae_checkpoint": str(args.ae_checkpoint.resolve()),
        "ae_checkpoint_sha256": file_sha256(args.ae_checkpoint.resolve()),
        "latent_shape": [256, 64],
        "packing": "numpy.packbits(flattened [256,64], bitorder=little)",
        "decoder_used": False,
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "complete.json", {"complete": True, **protocol})


if __name__ == "__main__":
    main()
