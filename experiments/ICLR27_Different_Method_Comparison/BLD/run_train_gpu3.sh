#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mohao/WorkSpace/Projects/BFM_ICLR2027"
HERE="${ROOT}/experiments/ICLR27_Different_Method_Comparison/BLD"
TRAIN_DIR="${HERE}/train_2m_official"
PYTHON="/home/mohao/WorkSpace/Envs/anaconda3/envs/BLD/bin/python"
LATENT_CACHE="/home/mohao/WorkSpace/Datasets/lsun/bitcache/latents_packed.npy"
mkdir -p "${TRAIN_DIR}/logs"

# Resume only checkpoints produced by this new official-objective run. The old
# train_100k directory is intentionally ignored because it used focal=-1.
resume=()
for ((step=2000000; step>=10000; step-=10000)); do
  prefix="${TRAIN_DIR}/saved_models/bld"
  if [[ -f "${prefix}_${step}.th" && -f "${prefix}_ema_${step}.th" && \
        -f "${prefix}_optim_${step}.th" && -f "${prefix}_scaler_${step}.th" ]]; then
    resume=(--load_dir "${TRAIN_DIR}" --load_step "${step}" --load_optim)
    break
  fi
done

cd "${ROOT}"
CUDA_VISIBLE_DEVICES=3 exec "${PYTHON}" -u "${HERE}/train_bld_lsun_bitcache.py" \
  --sampler bld --dataset churches --ema \
  --steps_per_checkpoint 10000 \
  --codebook_size 64 --img_size 256 \
  --steps_per_display_output 5000 --steps_per_save_output 5000 \
  --steps_per_log 100 \
  --total_steps 64 --sample_steps 64 --beta_type linear --amp \
  --train_steps 2000000 \
  --ae_load_dir "${ROOT}/logs/BAE_C64" --ae_load_step 8100000 \
  --batch_size 96 --latent_shape 1 16 16 \
  --log_dir "${TRAIN_DIR}" --latent_cache "${LATENT_CACHE}" \
  --loss_final mean --p_flip --norm_first \
  "${resume[@]}" >>"${TRAIN_DIR}/logs/train_gpu3.log" 2>&1
