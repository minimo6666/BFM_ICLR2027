#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"
cd "${REPO_ROOT}"

GPU_IDS="${GPU_IDS:-0,1}"
MASTER_PORT="${MASTER_PORT:-12409}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
UPDATE_FREQ="${UPDATE_FREQ:-1}"
EXPERIMENT_SEED="${EXPERIMENT_SEED:-20260901}"
AE_LOAD_DIR="${AE_LOAD_DIR:-logs/BAE_C64}"
DATA_ROOT="${DATA_ROOT:-/mnt/data/0/mohao/data/lsun/scenes}"
LOG_DIR="${LOG_DIR:-${HERE}/runs/01_boundary_repair_64_32}"

: "${V8_CHECKPOINT:?Set V8_CHECKPOINT to the exact V8 EMA 50k .th file}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export EXPERIMENT_SEED
export V8_CHECKPOINT

# The released V8 checkpoint was trained with all four deployment grids.  V9's
# first causal repair is deliberately restricted to the two failing grids.
export BFM_LR_NFES="${BFM_LR_NFES:-64,32,16,8}"
export BFM_LR_NFE_WEIGHTS="${BFM_LR_NFE_WEIGHTS:-}"
export BFM_LR_RANK="${BFM_LR_RANK:-32}"
export BFM_LR_LAMBDA="${BFM_LR_LAMBDA:-1.0}"
export BFM_V9_BOUNDARY_NFES="${BFM_V9_BOUNDARY_NFES:-64,32}"
export BFM_V9_BOUNDARY_RANK="${BFM_V9_BOUNDARY_RANK:-32}"
export BFM_V9_BRANCHES="${BFM_V9_BRANCHES:-4}"
export BFM_V9_HARD_TAU="${BFM_V9_HARD_TAU:-0.5}"
export BFM_V9_LAMBDA_BCE="${BFM_V9_LAMBDA_BCE:-1.0}"
export BFM_V9_LAMBDA_PROB="${BFM_V9_LAMBDA_PROB:-1.0}"
export BFM_V9_LAMBDA_HARD="${BFM_V9_LAMBDA_HARD:-1.0}"
export BFM_V9_LAMBDA_ANCHOR="${BFM_V9_LAMBDA_ANCHOR:-0.0}"
export BFM_V9_TERMINAL_CHUNK="${BFM_V9_TERMINAL_CHUNK:-16}"
export BFM_V9_LR="${BFM_V9_LR:-0.001}"
export BFM_V9_EMA_DECAY="${BFM_V9_EMA_DECAY:-0.995}"
export BFM_V9_WORKERS="${BFM_V9_WORKERS:-4}"

python -m torch.distributed.launch \
  --use_env \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${HERE}/train_boundary_repair_v9.py" \
  --dataset churches \
  --sampler bld \
  --codebook_size 64 \
  --img_size 256 \
  --path_to_data "${DATA_ROOT}" \
  --total_steps 64 \
  --sample_steps 64 \
  --beta_type linear \
  --amp \
  --train_steps "${TRAIN_STEPS}" \
  --ae_load_dir "${AE_LOAD_DIR}" \
  --ae_load_step 8100000 \
  --batch_size "${BATCH_SIZE}" \
  --update_freq "${UPDATE_FREQ}" \
  --latent_shape 1 16 16 \
  --loss_final mean \
  --p_flip \
  --norm_first \
  --aux 0 \
  --steps_per_log "${STEPS_PER_LOG:-20}" \
  --steps_per_checkpoint "${STEPS_PER_CHECKPOINT:-500}" \
  --log_dir "${LOG_DIR}"
