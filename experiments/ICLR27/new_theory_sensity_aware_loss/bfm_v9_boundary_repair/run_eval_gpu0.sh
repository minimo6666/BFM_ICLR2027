#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"
cd "${REPO_ROOT}"

: "${V8_CKPT:?Set V8_CKPT to the exact V8 EMA 50k .th file}"
: "${AE_CKPT:?Set AE_CKPT to binaryae_ema_8100000.th}"
: "${DATA_ROOT:?Set DATA_ROOT to the LSUN Churches root}"

BOUNDARY_CKPT="${BOUNDARY_CKPT:-${HERE}/runs/01_boundary_repair_64_32/checkpoints/boundary_head_latest.pt}"
OUTPUT="${OUTPUT:-${HERE}/runs/02_boundary_validation}"

export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"
export BFM_LR_NFES="${BFM_LR_NFES:-64,32,16,8}"
export BFM_LR_RANK="${BFM_LR_RANK:-32}"
export BFM_LR_LAMBDA="${BFM_LR_LAMBDA:-1.0}"
export BFM_V9_BOUNDARY_NFES="64,32"
export BFM_V9_BOUNDARY_RANK="${BFM_V9_BOUNDARY_RANK:-32}"
export BFM_V9_BRANCHES="${BFM_V9_BRANCHES:-4}"
export BFM_V9_TERMINAL_CHUNK="${BFM_V9_TERMINAL_CHUNK:-16}"

python "${HERE}/evaluate_boundary_repair_v9.py" \
  --v8-checkpoint "${V8_CKPT}" \
  --boundary-checkpoint "${BOUNDARY_CKPT}" \
  --ae-checkpoint "${AE_CKPT}" \
  --data-root "${DATA_ROOT}" \
  --device cuda:0 \
  --nfes 64,32 \
  --branches "${EVAL_BRANCHES:-32}" \
  --max-images "${MAX_IMAGES:-128}" \
  --batch-size "${EVAL_BATCH_SIZE:-4}" \
  --workers "${EVAL_WORKERS:-4}" \
  --amp \
  --output "${OUTPUT}"
