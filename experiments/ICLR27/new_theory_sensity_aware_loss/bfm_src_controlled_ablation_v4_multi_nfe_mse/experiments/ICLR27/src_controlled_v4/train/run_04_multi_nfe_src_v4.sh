#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/common_v4.sh"

# V4 proposed method:
#   L = L_base + lambda * W_multi(t) * (m_theta(X_t,t)-X0)^2
# W_multi is derived from the ACTUAL 64/32/16/8 np.linspace sampler grids.
export EXPERIMENT_VARIANT=multi_nfe_src_v4
export BFM_V4_LAMBDA="${BFM_V4_LAMBDA:-0.25}"
export BFM_V4_NFES="${BFM_V4_NFES:-64,32,16,8}"
# Empty means equal deployment weights. Example override: 0.4,0.2,0.2,0.2
export BFM_V4_NFE_WEIGHTS="${BFM_V4_NFE_WEIGHTS:-}"

python -m torch.distributed.launch \
  --use_env \
  --nproc_per_node=2 \
  --master_port="${MASTER_PORT}" \
  "${HERE}/train_multi_nfe_src_v4.py" \
  --log_dir "${LOG_ROOT}/04_multi_nfe_src_v4_lam${BFM_V4_LAMBDA}" \
  "${COMMON_ARGS[@]}"
