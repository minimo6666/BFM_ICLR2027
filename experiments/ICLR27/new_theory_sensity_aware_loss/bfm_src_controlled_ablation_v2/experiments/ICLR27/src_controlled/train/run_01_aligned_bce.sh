#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/common.sh"

# EXPERIMENT 0B / LOSS BASELINE:
# Only conceptual change vs original: t -> t-1 in train and sample.
# This becomes the baseline for ALL subsequent loss ablations.
export EXPERIMENT_VARIANT=aligned_bce

python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port="${MASTER_PORT}" \
  "${HERE}/train_controlled_src.py" \
  --log_dir "${LOG_ROOT}/01_aligned_bce" \
  "${COMMON_ARGS[@]}"
