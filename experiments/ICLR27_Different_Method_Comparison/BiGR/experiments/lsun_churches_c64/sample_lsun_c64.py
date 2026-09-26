#!/usr/bin/env python3
"""Generate resumable packed C64 samples from an EMA BiGR checkpoint."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from official_bigr_c64 import bind_cfg_off_sampling, build_official_bigr_l, generate_cfg_off


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--outer-iterations", type=int, default=20)
    parser.add_argument("--inner-steps", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gumbel-temp", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    payload = torch.load(args.checkpoint, map_location="cpu")
    state = payload["ema"] if "ema" in payload else payload
    model = build_official_bigr_l()
    model.load_state_dict(state, strict=True)
    model = bind_cfg_off_sampling(model.cuda().eval())
    total_seconds = 0.0
    for start in range(0, args.num_samples, args.batch_size):
        count = min(args.batch_size, args.num_samples - start)
        path = args.output_dir / f"packed_{start:06d}_{start + count:06d}.npy"
        if path.exists():
            existing = np.load(path, mmap_mode="r")
            if existing.shape == (count, 2048) and existing.dtype == np.uint8:
                continue
        before = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            bits = generate_cfg_off(
                model, count, args.outer_iterations, args.inner_steps,
                args.temperature, args.gumbel_temp,
            )
        torch.cuda.synchronize()
        total_seconds += time.time() - before
        packed = np.packbits(
            bits.to(torch.uint8).cpu().numpy().reshape(count, -1),
            axis=1, bitorder="little",
        )
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, packed)
        temporary.replace(path)
        print(f"SAMPLED {start + count}/{args.num_samples} file={path.name}", flush=True)
    measured = sum(np.load(path, mmap_mode="r").shape[0] for path in args.output_dir.glob("packed_*.npy"))
    sec_per_image = total_seconds / measured if total_seconds and measured else float("nan")
    (args.output_dir / "sampling_metrics.txt").write_text(
        f"global_transformer_calls_per_image={args.outer_iterations}\n"
        # Official BinaryDiffusion.sample evaluates its two CFG branches even
        # when the experiment wrapper gives them identical context at scale 1.
        f"inner_denoiser_calls_per_image={2 * args.outer_iterations * args.inner_steps}\n"
        f"wall_clock_sec_per_image_this_run={sec_per_image}\n"
    )
    print(f"SAMPLING_COMPLETE count={measured} sec_per_image={sec_per_image}", flush=True)


if __name__ == "__main__":
    main()
