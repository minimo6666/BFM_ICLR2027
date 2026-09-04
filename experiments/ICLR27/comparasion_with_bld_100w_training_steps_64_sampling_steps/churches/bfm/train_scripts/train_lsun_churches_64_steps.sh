#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHURCH_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$CHURCH_DIR/../../../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export EXPERIMENT_METHOD=bfm
export BFM_MODEL_VARIANT="${BFM_MODEL_VARIANT:-tminus1}"
export BFM_AUX_INTERVAL_MODE="${BFM_AUX_INTERVAL_MODE:-mixed}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

PYTHON_BIN="${PYTHON_BIN:-/home/minimo/.conda/envs/BLD/bin/python}"
MASTER_PORT="${MASTER_PORT:-12340}"
TARGET_STEPS="${TARGET_STEPS:-200000}"
LOAD_STEP="${LOAD_STEP:-0}"
LOG_DIR="$CHURCH_DIR/bfm/models"
AE_LOAD_DIR="${AE_LOAD_DIR:-$PROJECT_ROOT/logs/BAE_C64}"
LSUN_ROOT="${LSUN_ROOT:-/mnt/data/0/mohao/data/lsun/scenes}"

resume_args=()
if (( LOAD_STEP > 0 )); then
    resume_args=(
        --load_step "$LOAD_STEP"
        --load_dir "$LOG_DIR"
        --load_optim
    )
fi

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node 2 \
    --master_port "$MASTER_PORT" \
    train_sampler_online_comparison.py \
    --sampler bld \
    --dataset churches \
    --ema \
    --steps_per_checkpoint 100000 \
    --codebook_size 64 \
    --img_size 256 \
    --steps_per_display_output 200000 \
    --steps_per_save_output 5000 \
    --steps_per_log 100 \
    --total_steps 64 \
    --sample_steps 64 \
    --beta_type linear \
    --amp \
    --train_steps "$TARGET_STEPS" \
    --ae_load_dir "$AE_LOAD_DIR" \
    --ae_load_step 8100000 \
    --batch_size 48 \
    --update_freq 2 \
    --latent_shape 1 16 16 \
    --path_to_data "$LSUN_ROOT" \
    --log_dir "$LOG_DIR" \
    --loss_final mean \
    --aux 0.0 \
    --p_flip \
    --norm_first \
    "${resume_args[@]}"
