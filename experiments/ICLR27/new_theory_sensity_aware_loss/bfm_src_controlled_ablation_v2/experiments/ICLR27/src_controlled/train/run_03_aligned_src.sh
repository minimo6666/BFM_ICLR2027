#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/common.sh"

# PROPOSED LOSS:
# SAME model, SAME t-1 convention, SAME sampler as aligned_bce/brier.
# ONLY difference from plain_brier is the relative S(t-1,t)^2 weighting.
export EXPERIMENT_VARIANT=aligned_src
export BFM_SRC_LAMBDA="${BFM_SRC_LAMBDA:-1.0}"

python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port="${MASTER_PORT}" \
  "${HERE}/train_controlled_src.py" \
  --log_dir "${LOG_ROOT}/03_aligned_src_lam${BFM_SRC_LAMBDA}" \
  "${COMMON_ARGS[@]}"
