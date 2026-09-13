#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/b/mohao/Projects/BinaryLatentDiffusion
EXP="$ROOT/experiments/ICLR27/diagnostics/t1_masked_pattern_oracle"
STAGE1="$EXP/stage1_masked_pretrain"
STAGE2="$EXP/stage2_t1_finetune"
CACHE="$ROOT/experiments/ICLR27/diagnostics/t1_direct_x0_bce_oracle/cache/latents_packed.npy"
STAGE1_SCRIPT="$EXP/scripts/train_masked_pretrain.py"
STAGE2_SCRIPT="$ROOT/experiments/ICLR27/diagnostics/t1_direct_x0_bce_oracle/scripts/train_cached_x0.py"
PYTHON=/home/minimo/.conda/envs/BLD/bin/python
GPU_IDS="${GPU_IDS:-0,1}"
STAGE1_PORT="${STAGE1_PORT:-12720}"
STAGE2_PORT="${STAGE2_PORT:-12721}"

mkdir -p "$STAGE1" "$STAGE2"
test -f "$CACHE"
test -f "${CACHE%/*}/complete.json"
echo "$$" > "$EXP/launcher.pid"

cd "$ROOT"

CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -m torch.distributed.run \
  --nproc_per_node=2 --master_port="$STAGE1_PORT" \
  "$STAGE1_SCRIPT" \
  --output-dir "$STAGE1" \
  --cache "$CACHE" \
  --train-steps 10000 --eval-every 500 --eval-images 300 \
  --batch-size 48 --update-freq 2 --num-workers 4 \
  --learning-rate 2e-4 --mask-ratio 0.10 --seed 20260910 \
  > "$STAGE1/train.log" 2>&1

CHECKPOINT="$STAGE1/transformerbd_mask_pretrained.pt"
test -f "$STAGE1/complete.json"
test -f "$STAGE1/metrics.csv"
test -f "$STAGE1/masked_pretrain.png"
test -f "$CHECKPOINT"

CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -m torch.distributed.run \
  --nproc_per_node=2 --master_port="$STAGE2_PORT" \
  "$STAGE2_SCRIPT" \
  --output-dir "$STAGE2" \
  --cache "$CACHE" \
  --init-transformerbd-checkpoint "$CHECKPOINT" \
  --train-steps 10000 --eval-every 500 --eval-images 300 \
  --batch-size 48 --update-freq 2 --num-workers 4 \
  --learning-rate 2e-4 --seed 20260910 \
  > "$STAGE2/train.log" 2>&1

test -f "$STAGE2/metrics.csv"
test -f "$STAGE2/t1_direct_x0_bce_oracle.png"

echo "COMPLETE two_stage_masked_pattern_t1_oracle" > "$EXP/complete.txt"
