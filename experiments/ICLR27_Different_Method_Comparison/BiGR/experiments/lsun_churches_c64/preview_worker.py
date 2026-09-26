#!/usr/bin/env python3
"""GPU-isolated watcher that converts queued EMA snapshots into 8x8 previews."""

import argparse
import os
import subprocess
import time
from pathlib import Path


def run(command, env):
    print("RUN", " ".join(map(str, command)), flush=True)
    subprocess.run([str(item) for item in command], check=True, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="5")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    queue = args.experiment_dir / "preview_queue"
    previews = args.experiment_dir / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    bigr_python = "/home/mohao/WorkSpace/Envs/anaconda3/envs/BiGR/bin/python"
    bld_python = "/home/mohao/WorkSpace/Envs/anaconda3/envs/BLD/bin/python"

    while True:
        did_work = False
        for checkpoint in sorted(queue.glob("ema_step_*.pt")):
            step = int(checkpoint.stem.rsplit("_", 1)[-1])
            final_grid = previews / f"samples_step_{step:06d}.png"
            if final_grid.exists():
                checkpoint.unlink(missing_ok=True)
                continue
            work = previews / f".step_{step:06d}"
            packed = work / "packed"
            decoded = work / "decoded"
            run([
                bigr_python, here / "sample_lsun_c64.py", "--checkpoint", checkpoint,
                "--output-dir", packed, "--num-samples", 64, "--batch-size", 8,
            ], env)
            # Merge eight resumable chunks for the one-shot grid decoder.
            merge_script = here / "merge_packed.py"
            merged = work / "preview.npy"
            run([bigr_python, merge_script, "--packed-dir", packed, "--output", merged], env)
            run([
                bld_python, here / "decode_bae_c64.py", "--packed", merged,
                "--output-dir", decoded, "--grid-name", final_grid.name,
            ], env)
            (decoded / final_grid.name).replace(final_grid)
            checkpoint.unlink()
            (previews / f"samples_step_{step:06d}.done").touch()
            did_work = True
        if (args.experiment_dir / "TRAINING_COMPLETE").exists() and not list(queue.glob("ema_step_*.pt")):
            return
        if not did_work:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
