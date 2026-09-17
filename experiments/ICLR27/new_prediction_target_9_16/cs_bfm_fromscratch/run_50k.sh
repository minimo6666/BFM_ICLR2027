#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/b/mohao/Projects/BinaryLatentDiffusion
EXP="$ROOT/experiments/ICLR27/new_prediction_target_9_16/cs_bfm_fromscratch"
RUN="$EXP/runs/cs_bfm_fromscratch"
CACHE="$ROOT/experiments/ICLR27/diagnostics/t1_direct_x0_bce_oracle/cache/latents_packed.npy"
TRAIN="$EXP/train_cs_bfm.py"
PYTHON=/home/minimo/.conda/envs/BLD/bin/python
GPU0=${GPU0:-5}
GPU1=${GPU1:-6}
MASTER_PORT=${MASTER_PORT:-29616}
SEED=20260910
BATCH=48
UPDATE_FREQ=2

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export EXPERIMENT_VARIANT=cached_cs_bfm
export EXPERIMENT_SEED="$SEED"
export BFM_CACHED_X0_PATH="$CACHE"
export BFM_FIXED_EVAL_EVERY=500
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
unset BFM_MASK_PRETRAIN_CHECKPOINT || true
unset BFM_ENDPOINT_CHECKPOINT || true
unset BFM_BARRIER_RETENTION_SCALE || true

mkdir -p "$RUN/logs"
test -f "$CACHE"
test -f "${CACHE%/*}/complete.json"
cd "$ROOT"


for TARGET in 10000 20000 30000 40000 50000; do
  TAG="$(printf '%06d' "$TARGET")"
  CKPT="$RUN/saved_models/flow_cached_cs_bfm_ema_${TARGET}.th"
  if [[ ! -f "$CKPT" ]]; then
    PREVIOUS=$((TARGET - 10000))
    RESUME=()
    if (( PREVIOUS > 0 )); then
      RESUME=(--load_step "$PREVIOUS" --load_dir "$RUN" --load_optim)
    fi
    echo "[$(date -Is)] train CS-BFM $PREVIOUS -> $TARGET on GPUs $GPU0,$GPU1"
    CUDA_VISIBLE_DEVICES="$GPU0,$GPU1" "$PYTHON" -m torch.distributed.run       --nproc_per_node=2 --master_port="$MASTER_PORT" "$TRAIN"       --dataset churches --sampler bld --codebook_size 64 --img_size 256       --path_to_data /mnt/data/0/mohao/data/lsun/scenes --ema       --total_steps 64 --sample_steps 64 --beta_type linear --amp       --train_steps "$TARGET" --warmup_iters 10000       --ae_load_dir "$ROOT/logs/BAE_C64" --ae_load_step 8100000       --batch_size "$BATCH" --update_freq "$UPDATE_FREQ"       --latent_shape 1 16 16 --loss_final mean --focal -1 --norm_first --aux 0       --steps_per_save_output 2000 --steps_per_checkpoint 10000 --steps_per_log 100       --log_dir "$RUN" "${RESUME[@]}"       > "$RUN/logs/train_to_${TAG}.log" 2>&1
  fi

  echo "[$(date -Is)] training step $TARGET complete"
done

echo "TRAINING_COMPLETE $(date -Is)" > "$RUN/training_complete.txt"
