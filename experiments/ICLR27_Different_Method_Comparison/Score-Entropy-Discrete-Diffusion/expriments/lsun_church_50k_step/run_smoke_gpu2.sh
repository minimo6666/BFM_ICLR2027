#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEDD_ROOT="$(cd "$HERE/../.." && pwd)"
PROJECT_ROOT="$(cd "$SEDD_ROOT/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/minimo/miniconda3/envs/sedd/bin/python}"
SMOKE_DIR="$HERE/runs/smoke"

CUDA_VISIBLE_DEVICES=2 "$PYTHON_BIN" "$HERE/train_binary_sedd.py" \
  --output-dir "$SMOKE_DIR" --data-root /mnt/data/0/mohao/data/lsun/scenes \
  --ae-load-dir "$PROJECT_ROOT/logs/BAE_C64" --train-steps 1 \
  --batch-size 1 --accum 2 --preview-every 1 --checkpoint-every 1 --log-every 1

CUDA_VISIBLE_DEVICES=2 "$PYTHON_BIN" "$HERE/sample_binary_sedd.py" \
  --checkpoint "$SMOKE_DIR/checkpoints/binary_sedd_ema_step_000001.pt" \
  --output-dir "$SMOKE_DIR/images" --ae-load-dir "$PROJECT_ROOT/logs/BAE_C64" \
  --num-samples 1 --batch-size 1 --decode-batch-size 1 --nfe 64 --seed 2020

test "$(find "$SMOKE_DIR/images" -maxdepth 1 -name '*.png' | wc -l)" -eq 1

