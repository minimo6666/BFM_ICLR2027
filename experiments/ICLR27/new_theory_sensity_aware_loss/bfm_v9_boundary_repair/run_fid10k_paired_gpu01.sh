#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"
POSTHOC_ROOT="${REPO_ROOT}/experiments/ICLR27/new_theory_sensity_aware_loss/BFM_V8_posthoc_analysis"
POSTHOC_RUNS="${POSTHOC_ROOT}/runs/02_scale_sweep"
FID_SCRIPT="${REPO_ROOT}/experiments/ICLR27/comparasion_with_bld_100w_training_steps_64_sampling_steps/churches/common/eval_50k_comparison.py"
FID_CACHE="${REPO_ROOT}/experiments/ICLR27/comparasion_with_bld_100w_training_steps_64_sampling_steps/churches/common/fid_cache"
PYTHON_BIN="${PYTHON_BIN:-/home/minimo/.conda/envs/BLD/bin/python}"

: "${V8_CHECKPOINT:?Set V8_CHECKPOINT to flow_lowrank_sensitivity_v8_ema_50000.th}"
AE_CHECKPOINT="${AE_CHECKPOINT:-${REPO_ROOT}/logs/BAE_C64/saved_models/binaryae_ema_8100000.th}"
DATA_ROOT="${DATA_ROOT:-/mnt/data/0/mohao/data/lsun/scenes}"
BOUNDARY_CHECKPOINT="${BOUNDARY_CHECKPOINT:-${HERE}/runs/01_boundary_repair_64_32/checkpoints/boundary_head_latest.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HERE}/runs/04_fid10k_paired}"
GPU_IDS="${GPU_IDS:-0,1}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEED="${SEED:-20260821}"

IFS=',' read -r GPU_64 GPU_32 <<< "${GPU_IDS}"
if [[ -z "${GPU_64:-}" || -z "${GPU_32:-}" ]]; then
  echo "GPU_IDS must contain exactly two physical GPU IDs, e.g. 0,1" >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_ROOT}/logs" "${FID_CACHE}"

run_condition() {
  local gpu="$1"
  local nfe="$2"
  local mode="$3"
  local condition="${OUTPUT_ROOT}/nfe_$(printf '%03d' "${nfe}")/v9_head_${mode}"
  local image_dir="${condition}/images"
  local result="${OUTPUT_ROOT}/fid/nfe_$(printf '%03d' "${nfe}")_v9_head_${mode}.json"
  local head_args=()
  if [[ "${mode}" == "off" ]]; then
    head_args+=(--head-off)
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${HERE}/generate_v9_10k.py" \
    --v8-checkpoint "${V8_CHECKPOINT}" \
    --boundary-checkpoint "${BOUNDARY_CHECKPOINT}" \
    --ae-checkpoint "${AE_CHECKPOINT}" \
    --output "${image_dir}" \
    --nfe "${nfe}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --temperature 1.0 \
    --device cuda:0 \
    "${head_args[@]}"

  local count
  count="$(find "${image_dir}" -maxdepth 1 -type f -name '*.png' | wc -l)"
  if [[ "${count}" -ne "${NUM_SAMPLES}" ]]; then
    echo "Expected ${NUM_SAMPLES} PNGs for NFE=${nfe} head=${mode}; found ${count}" >&2
    exit 1
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${FID_SCRIPT}" fid \
    --output-dir "${image_dir}" \
    --checkpoint "${V8_CHECKPOINT}" \
    --sampler-implementation expectation_consistent_boundary_repair \
    --bfm-model-variant tminus1 \
    --total-steps 64 \
    --sample-steps "${nfe}" \
    --num-samples "${NUM_SAMPLES}" \
    --num-real 50000 \
    --temperature 1.0 \
    --lsun-root "${DATA_ROOT}" \
    --cache-root "${FID_CACHE}" \
    --fake-cache-name "v9-boundary-${mode}-nfe${nfe}-n${NUM_SAMPLES}-seed${SEED}" \
    --result-json "${result}" \
    --fid-batch-size 64
}

mkdir -p "${OUTPUT_ROOT}/fid"
(
  run_condition "${GPU_64}" 64 off
  run_condition "${GPU_64}" 64 on
) >"${OUTPUT_ROOT}/logs/nfe64.log" 2>&1 &
PID_64=$!
(
  run_condition "${GPU_32}" 32 off
  run_condition "${GPU_32}" 32 on
) >"${OUTPUT_ROOT}/logs/nfe32.log" 2>&1 &
PID_32=$!

status=0
wait "${PID_64}" || status=$?
wait "${PID_32}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  echo "At least one V9 FID branch failed; see ${OUTPUT_ROOT}/logs" >&2
  exit "${status}"
fi

"${PYTHON_BIN}" "${HERE}/summarize_fid10k.py" --v9-root "${OUTPUT_ROOT}"
