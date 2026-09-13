#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/b/mohao/Projects/BinaryLatentDiffusion
EXP="$ROOT/experiments/ICLR27/diagnostics/t1_direct_x0_bce_oracle"
PYTHON=/home/minimo/.conda/envs/BLD/bin/python
GPU_IDS="${GPU_IDS:-0,1}"
MASTER_PORT="${MASTER_PORT:-12710}"

mkdir -p "$EXP"
cd "$ROOT"

test -f "$EXP/cache/complete.json"
echo "$$" > "$EXP/launcher.pid"
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -m torch.distributed.run \
  --nproc_per_node=2 --master_port="$MASTER_PORT" \
  "$EXP/scripts/train_cached_x0.py" \
  --output-dir "$EXP" \
  --cache "$EXP/cache/latents_packed.npy" \
  --train-steps 10000 --eval-every 500 --eval-images 300 \
  --batch-size 48 --update-freq 2 --num-workers 4 \
  --learning-rate 2e-4 --seed 20260910 \
  > "$EXP/train.log" 2>&1

test -f "$EXP/metrics.csv"
test -f "$EXP/t1_direct_x0_bce_oracle.png"
