#!/usr/bin/env python3
"""Generate decoded images with official SEDD analytic sampling at 64 NFE."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sedd_binary_common import (
    ExponentialMovingAverage,
    build_strict_64_nfe_sampler,
    create_sedd_components,
    decode_bits,
    load_autoencoder,
    tokens_to_latent,
)


def save_batch(images: torch.Tensor, output_dir: Path, start: int):
    arrays = images.clamp(0, 1).mul(255).add(0.5).to(torch.uint8)
    arrays = arrays.permute(0, 2, 3, 1).cpu().numpy()
    for offset, array in enumerate(arrays):
        Image.fromarray(array, mode="RGB").save(
            output_dir / f"{start + offset:06d}.png", compress_level=1
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ae-load-dir", required=True)
    parser.add_argument("--ae-load-step", type=int, default=8_100_000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--decode-batch-size", type=int, default=2)
    parser.add_argument("--nfe", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.nfe != 64:
        raise ValueError("This comparison is locked to exactly 64 NFE")

    device = torch.device("cuda", 0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg, graph, noise, model = create_sedd_components(device)
    # Completed runs may be compacted to EMA-only checkpoints. The EMA shadow
    # parameters fully replace the randomly initialized trainable model below;
    # retaining the non-EMA model copy is unnecessary for sampling.
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    noise.load_state_dict(checkpoint.get("noise", {}), strict=False)
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.training.ema)
    ema.load_state_dict(checkpoint["ema"])
    ema.copy_to(model.parameters())
    model.eval()
    autoencoder = load_autoencoder(args.ae_load_dir, args.ae_load_step, device)
    autoencoder.encoder = None

    protocol = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "method": "official binary SEDD",
        "tokens": 2,
        "latent_shape": [64, 16, 16],
        "serialization": "HWC spatial-major",
        "graph": "Uniform(2)",
        "noise": "GeometricNoise(1e-4,20)",
        "predictor": "official AnalyticPredictor",
        "predictor_steps": 63,
        "final_official_denoiser_calls": 1,
        "total_nfe": 64,
        "start_index": args.start_index,
        "num_samples": args.num_samples,
        "seed": args.seed,
    }
    (output_dir / f"protocol_{args.start_index:06d}.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )

    end = args.start_index + args.num_samples
    done = 0
    began = time.time()
    for start in range(args.start_index, end, args.batch_size):
        batch = min(args.batch_size, end - start)
        image_paths = [output_dir / f"{index:06d}.png" for index in range(start, start + batch)]
        if args.resume and all(path.exists() for path in image_paths):
            done += batch
            continue
        torch.manual_seed(args.seed + start)
        torch.cuda.manual_seed_all(args.seed + start)
        sampling_fn = build_strict_64_nfe_sampler(graph, noise, batch, device)
        tokens = sampling_fn(model)
        bits = tokens_to_latent(tokens)
        for offset in range(0, batch, args.decode_batch_size):
            images = decode_bits(autoencoder, bits[offset : offset + args.decode_batch_size])
            save_batch(images, output_dir, start + offset)
        done += batch
        if done % max(args.batch_size * 10, 1) == 0:
            print(f"sampled {done}/{args.num_samples} in {time.time()-began:.1f}s", flush=True)


if __name__ == "__main__":
    main()
