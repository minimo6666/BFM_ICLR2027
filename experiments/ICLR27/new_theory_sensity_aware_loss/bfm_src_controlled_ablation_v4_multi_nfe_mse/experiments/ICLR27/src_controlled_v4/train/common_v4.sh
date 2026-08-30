#!/usr/bin/env bash
# Common configuration for the V4 controlled run.
# Keep these identical to the aligned-BFM baseline unless deliberately testing a new variable.
set -euo pipefail

GPU_IDS="${GPU_IDS:-5,6}"
MASTER_PORT="${MASTER_PORT:-12368}"
TRAIN_STEPS="${TRAIN_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-48}"
UPDATE_FREQ="${UPDATE_FREQ:-2}"
EXPERIMENT_SEED="${EXPERIMENT_SEED:-20260821}"

AE_LOAD_DIR="${AE_LOAD_DIR:-logs/BAE_C64}"
LOG_ROOT="${LOG_ROOT:-experiments/ICLR27/src_controlled_v4/runs}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export EXPERIMENT_SEED

COMMON_ARGS=(
  --dataset churches
  --sampler bld
  --codebook_size 64
  --img_size 256
  --path_to_data /mnt/data/0/mohao/data/lsun/scenes
  --ema
  --total_steps 64
  --sample_steps 64
  --beta_type linear
  --amp
  --train_steps "${TRAIN_STEPS}"
  --ae_load_dir "${AE_LOAD_DIR}"
  --ae_load_step 8100000
  --batch_size "${BATCH_SIZE}"
  --update_freq "${UPDATE_FREQ}"
  --latent_shape 1 16 16
  --loss_final mean
  --p_flip
  --norm_first
  --aux 0
)
