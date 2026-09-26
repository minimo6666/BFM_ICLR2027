#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIGR_ROOT="$(cd -- "${HERE}/../.." && pwd)"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${HERE}}"
LATENT_CACHE="${LATENT_CACHE:-/home/mohao/WorkSpace/Datasets/lsun/bitcache/latents_packed.npy}"
BIGR_PYTHON="/home/mohao/WorkSpace/Envs/anaconda3/envs/BiGR/bin/python"
TRAIN_GPU="${TRAIN_GPU:-3}"
PREVIEW_GPU="${PREVIEW_GPU:-5}"

mkdir -p "${EXPERIMENT_DIR}/logs"
cd "${BIGR_ROOT}"

CUDA_VISIBLE_DEVICES="${PREVIEW_GPU}" "${BIGR_PYTHON}" -u \
  "${HERE}/preview_worker.py" --experiment-dir "${EXPERIMENT_DIR}" --gpu "${PREVIEW_GPU}" \
  >"${EXPERIMENT_DIR}/logs/preview_gpu${PREVIEW_GPU}.log" 2>&1 &
preview_pid=$!
echo "PREVIEW_WORKER_PID=${preview_pid}"

resume_args=()
latest="$(find "${EXPERIMENT_DIR}/checkpoints" -maxdepth 1 -name 'checkpoint_step_*.pt' -type f 2>/dev/null | sort | tail -n 1 || true)"
if [[ -n "${latest}" ]]; then
  resume_args=(--resume "${latest}")
fi

set +e
CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" "${BIGR_PYTHON}" -u \
  "${HERE}/train_lsun_c64.py" \
  --latent-cache "${LATENT_CACHE}" --output-dir "${EXPERIMENT_DIR}" \
  --micro-batch 48 --effective-batch 96 --train-steps 100000 \
  --checkpoint-every 10000 --preview-every 2000 \
  "${resume_args[@]}" \
  >"${EXPERIMENT_DIR}/logs/train_gpu${TRAIN_GPU}.log" 2>&1
train_status=$?
set -e
if [[ "${train_status}" -ne 0 ]]; then
  kill "${preview_pid}" 2>/dev/null || true
  exit "${train_status}"
fi
wait "${preview_pid}"
