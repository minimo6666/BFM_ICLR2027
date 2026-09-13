#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/b/mohao/Projects/BinaryLatentDiffusion
EXP="$ROOT/experiments/ICLR27/diagnostics/full64_mask_pretrain_direct_x0_bce"

cd "$ROOT"
bash "$EXP/run_group_to_500k.sh" mask_pretrained 0 1 29630 \
  > "$EXP/runs/mask_pretrained/pipeline_to_500k.log" 2>&1 &
MASK_PID=$!
bash "$EXP/run_group_to_500k.sh" random_init 2 3 29631 \
  > "$EXP/runs/random_init/pipeline_to_500k.log" 2>&1 &
RANDOM_PID=$!

echo "mask_pretrained pid=$MASK_PID GPUs=0,1"
echo "random_init pid=$RANDOM_PID GPUs=2,3"

STATUS=0
wait "$MASK_PID" || STATUS=$?
wait "$RANDOM_PID" || STATUS=$?
exit "$STATUS"
