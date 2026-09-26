#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {bedrooms|ffhq} GPU_ID" >&2
  exit 2
fi

DATASET="$1"
GPU_ID="$2"
ROOT="/home/mohao/WorkSpace/Projects/BFM_ICLR2027"
HERE="${ROOT}/experiments/ICLR27_Different_Method_Comparison/BLD"
PYTHON="/home/mohao/WorkSpace/Envs/anaconda3/envs/BLD/bin/python"
TRAIN_DIR="${HERE}/${DATASET}/train_100k_official"
AE_DIR="${ROOT}/logs/BAE_C64"

case "${DATASET}" in
  bedrooms) LATENT_CACHE="/home/mohao/WorkSpace/Datasets/bedrooms/cache/latents_packed.npy" ;;
  ffhq) LATENT_CACHE="/home/mohao/WorkSpace/Datasets/ffhq/cache/latents_packed.npy" ;;
  *) echo "Unsupported dataset: ${DATASET}" >&2; exit 2 ;;
esac

resume_step=0
for ((step=200000; step>=100000; step-=10000)); do
  prefix="${TRAIN_DIR}/saved_models/bld"
  if [[ -f "${prefix}_${step}.th" && -f "${prefix}_ema_${step}.th" && \
        -f "${prefix}_optim_${step}.th" && -f "${prefix}_scaler_${step}.th" ]]; then
    resume_step="${step}"
    break
  fi
done
(( resume_step >= 100000 )) || { echo "No complete >=100k checkpoint set in ${TRAIN_DIR}" >&2; exit 1; }

if (( resume_step >= 200000 )); then
  touch "${TRAIN_DIR}/TRAINING_200K_COMPLETE"
  exit 0
fi

mkdir -p "${TRAIN_DIR}/logs"
cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" -u \
  "${HERE}/train_bld_lsun_bitcache.py" \
  --sampler bld --dataset "${DATASET}" --ema \
  --steps_per_checkpoint 10000 --codebook_size 64 --img_size 256 \
  --steps_per_display_output 5000 --steps_per_save_output 5000 \
  --steps_per_log 100 --total_steps 64 --sample_steps 64 \
  --beta_type linear --amp --train_steps 200000 \
  --ae_load_dir "${AE_DIR}" --ae_load_step 8100000 \
  --batch_size 96 --latent_shape 1 16 16 \
  --log_dir "${TRAIN_DIR}" --latent_cache "${LATENT_CACHE}" \
  --loss_final mean --p_flip --norm_first \
  --load_dir "${TRAIN_DIR}" --load_step "${resume_step}" --load_optim \
  >>"${TRAIN_DIR}/logs/train_to200k_gpu${GPU_ID}.log" 2>&1
touch "${TRAIN_DIR}/TRAINING_200K_COMPLETE"
