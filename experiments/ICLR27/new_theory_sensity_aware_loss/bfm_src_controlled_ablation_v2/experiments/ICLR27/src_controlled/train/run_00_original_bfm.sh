#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/common.sh"

# EXPERIMENT 0A: historical/current BFM convention.
# Only use this to isolate the t-vs-(t-1) issue.
export EXPERIMENT_VARIANT=original

python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port="${MASTER_PORT}" \
  "${HERE}/train_controlled_src.py" \
  --log_dir "${LOG_ROOT}/00_original_t" \
  "${COMMON_ARGS[@]}"
