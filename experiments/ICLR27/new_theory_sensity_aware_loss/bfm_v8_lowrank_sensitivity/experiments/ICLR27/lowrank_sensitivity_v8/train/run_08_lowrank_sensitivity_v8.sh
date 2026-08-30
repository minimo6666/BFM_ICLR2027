#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/common_v8.sh"

export EXPERIMENT_VARIANT=lowrank_sensitivity_v8
export BFM_LR_RANK="${BFM_LR_RANK:-32}"
export BFM_LR_LAMBDA="${BFM_LR_LAMBDA:-1.0}"
export BFM_LR_NFES="${BFM_LR_NFES:-64,32,16,8}"
export BFM_LR_NFE_WEIGHTS="${BFM_LR_NFE_WEIGHTS:-}"

python -m torch.distributed.launch \
  --use_env \
  --nproc_per_node="${NPROC_PER_NODE:-2}" \
  --master_port="${MASTER_PORT}" \
  "${HERE}/train_lowrank_sensitivity_v8.py" \
  --log_dir "${LOG_ROOT}/08_lowrank_sensitivity_v8_r${BFM_LR_RANK}_lam${BFM_LR_LAMBDA}" \
  "${COMMON_ARGS[@]}"
