#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 GROUP_NAME GPU0 GPU1 MASTER_PORT" >&2
  exit 2
fi

GROUP_NAME="$1"
GPU0="$2"
GPU1="$3"
MASTER_PORT="$4"

ROOT=/mnt/data/b/mohao/Projects/BinaryLatentDiffusion
EXP="$ROOT/experiments/ICLR27/diagnostics/full64_mask_pretrain_direct_x0_bce"
RUN_DIR="$EXP/runs/$GROUP_NAME"
CACHE="$ROOT/experiments/ICLR27/diagnostics/t1_direct_x0_bce_oracle/cache/latents_packed.npy"
TRAIN_ENTRY="$ROOT/experiments/ICLR27/src_controlled_v4/train/train_bitdance_joint.py"
SAMPLE_ENTRY="$ROOT/experiments/ICLR27/new_theory_sensity_aware_loss/bfm_src_controlled_ablation_v4_multi_nfe_mse/eval/sample_50k_nfe.py"
FID_ENTRY="$ROOT/metrics/fid_compute_algorithm_1.py"
FID_CACHE="$ROOT/experiments/ICLR27/comparasion_with_bld_100w_training_steps_64_sampling_steps/churches/common/fid_cache"
PYTHON=/home/minimo/.conda/envs/BLD/bin/python
SEED=20260910
BATCH=48
UPDATE_FREQ=2
NUM_FAKE=50000

case "$GROUP_NAME" in
  mask_pretrained|random_init) ;;
  *) echo "unknown group: $GROUP_NAME" >&2; exit 2 ;;
esac

# Both runs resume their full 50K state. Stage-1 initialization must not be
# applied again before loading the resumed predictor.
unset BFM_MASK_PRETRAIN_CHECKPOINT || true
export EXPERIMENT_VARIANT=cached_direct_x0_bce
export EXPERIMENT_SEED="$SEED"
export BFM_CACHED_X0_PATH="$CACHE"
export BFM_FIXED_EVAL_EVERY=500
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/fid" "$FID_CACHE"
test -f "$CACHE"
cd "$ROOT"

require_resume_state() {
  local step="$1"
  local prefix="$RUN_DIR/saved_models/flow_cached_direct_x0_bce"
  test -f "${prefix}_${step}.th"
  test -f "${prefix}_ema_${step}.th"
  test -f "${prefix}_optim_${step}.th"
  test -f "${prefix}_scaler_${step}.th"
}

for TARGET in 100000 150000 200000 250000 300000 350000 400000 450000 500000; do
  PREVIOUS=$((TARGET - 50000))
  TAG="$(printf '%06d' "$TARGET")"
  CKPT="$RUN_DIR/saved_models/flow_cached_direct_x0_bce_ema_${TARGET}.th"

  if [[ ! -f "$CKPT" ]]; then
    require_resume_state "$PREVIOUS"
    echo "[$(date -Is)] $GROUP_NAME train $PREVIOUS -> $TARGET on GPUs $GPU0,$GPU1"
    CUDA_VISIBLE_DEVICES="$GPU0,$GPU1" "$PYTHON" -m torch.distributed.run \
      --nproc_per_node=2 --master_port="$MASTER_PORT" "$TRAIN_ENTRY" \
      --dataset churches --sampler bld --codebook_size 64 --img_size 256 \
      --path_to_data /mnt/data/0/mohao/data/lsun/scenes --ema \
      --total_steps 64 --sample_steps 64 --beta_type linear --amp \
      --train_steps "$TARGET" --warmup_iters 10000 \
      --ae_load_dir "$ROOT/logs/BAE_C64" --ae_load_step 8100000 \
      --batch_size "$BATCH" --update_freq "$UPDATE_FREQ" \
      --latent_shape 1 16 16 --loss_final mean --focal -1 --norm_first --aux 0 \
      --steps_per_save_output 2000 --steps_per_checkpoint 50000 --steps_per_log 100 \
      --log_dir "$RUN_DIR" --load_step "$PREVIOUS" --load_dir "$RUN_DIR" --load_optim \
      > "$RUN_DIR/logs/train_to_${TAG}.log" 2>&1
  fi

  STEP_DIR="$RUN_DIR/fid/step_${TAG}_fid50k"
  IMAGE_DIR="$STEP_DIR/direct_x0_plain_bce_tminus1/nfe_64"
  RESULT="$STEP_DIR/fid.json"
  mkdir -p "$IMAGE_DIR"
  COUNT="$(find "$IMAGE_DIR" -maxdepth 1 -type f -name '*.png' | wc -l)"
  if [[ "$COUNT" -ne "$NUM_FAKE" ]]; then
    MID=$((NUM_FAKE / 2))
    echo "[$(date -Is)] $GROUP_NAME sample FID50K at step $TARGET"
    CUDA_VISIBLE_DEVICES="$GPU0" "$PYTHON" -u "$SAMPLE_ENTRY" \
      --variant direct_x0_plain_bce_tminus1 --checkpoint "$CKPT" \
      --output-root "$STEP_DIR" --nfes 64 --num-samples "$NUM_FAKE" \
      --start-index 0 --end-index "$MID" --allow-incomplete \
      --batch-size 64 --decode-batch-size 5 --temperature 1.0 --seed "$SEED" \
      --ae-load-dir "$ROOT/logs/BAE_C64" --ae-load-step 8100000 \
      > "$STEP_DIR/sample_gpu${GPU0}.log" 2>&1 &
    PID0=$!
    CUDA_VISIBLE_DEVICES="$GPU1" "$PYTHON" -u "$SAMPLE_ENTRY" \
      --variant direct_x0_plain_bce_tminus1 --checkpoint "$CKPT" \
      --output-root "$STEP_DIR" --nfes 64 --num-samples "$NUM_FAKE" \
      --start-index "$MID" --end-index "$NUM_FAKE" --allow-incomplete \
      --batch-size 64 --decode-batch-size 5 --temperature 1.0 --seed "$SEED" \
      --ae-load-dir "$ROOT/logs/BAE_C64" --ae-load-step 8100000 \
      > "$STEP_DIR/sample_gpu${GPU1}.log" 2>&1 &
    PID1=$!
    wait "$PID0"
    wait "$PID1"
  fi

  COUNT="$(find "$IMAGE_DIR" -maxdepth 1 -type f -name '*.png' | wc -l)"
  [[ "$COUNT" -eq "$NUM_FAKE" ]]
  if [[ ! -f "$RESULT" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU0" "$PYTHON" -u "$FID_ENTRY" \
      "$IMAGE_DIR" --lsun-root /mnt/data/0/mohao/data/lsun/scenes \
      --num-real 50000 --expected-num-fake 50000 --fid-batch-size 64 \
      --cache-root "$FID_CACHE" \
      --fake-cache-name "full64-${GROUP_NAME}-step${TARGET}-n50000-seed${SEED}" \
      --result-json "$RESULT" > "$STEP_DIR/fid.log" 2>&1
  fi
  "$PYTHON" "$EXP/plot_fid_progress.py"
  echo "[$(date -Is)] $GROUP_NAME step $TARGET complete: FID50K"
done

echo "COMPLETE through 500000" > "$RUN_DIR/complete_500k.txt"
