#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/common_v4.sh"

# Optional reproducibility baseline. If you already have the aligned-BFM BCE checkpoint,
# do not waste compute retraining this unless you need an exactly matched rerun.
export EXPERIMENT_VARIANT=aligned_bce

python -m torch.distributed.launch \
  --use_env \
  --nproc_per_node=2 \
  --master_port="${MASTER_PORT}" \
  "${HERE}/train_multi_nfe_src_v4.py" \
  --log_dir "${LOG_ROOT}/01_aligned_bce" \
  "${COMMON_ARGS[@]}"
