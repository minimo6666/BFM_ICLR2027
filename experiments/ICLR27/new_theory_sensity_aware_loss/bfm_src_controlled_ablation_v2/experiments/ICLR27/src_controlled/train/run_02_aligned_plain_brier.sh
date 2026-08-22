#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/common.sh"

# LOSS CONTROL:
# SAME model, SAME t-1 convention, SAME sampler as aligned_bce.
# Only adds an UNWEIGHTED Brier auxiliary.
# This tells us whether any gain is just "extra MSE/Brier regularization".
export EXPERIMENT_VARIANT=aligned_brier
export BFM_SRC_LAMBDA="${BFM_SRC_LAMBDA:-1.0}"

python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port="${MASTER_PORT}" \
  "${HERE}/train_controlled_src.py" \
  --log_dir "${LOG_ROOT}/02_aligned_plain_brier_lam${BFM_SRC_LAMBDA}" \
  "${COMMON_ARGS[@]}"
