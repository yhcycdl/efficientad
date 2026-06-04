#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
GPUS_CSV="${GPUS:-0}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS_CSV"

DATA_ROOT="${DATA_ROOT:-datasets/MVTecAD}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
LOG_ROOT="${LOG_ROOT:-$OUTPUTS_ROOT/full_experiment_logs}"
PRECACHE_MODELS="${PRECACHE_MODELS:-1}"

CATEGORIES_CSV="${CATEGORIES:-bottle,hazelnut,metal_nut}"
IFS=',' read -r -a CATEGORIES_LIST <<< "$CATEGORIES_CSV"

PATCHCORE_PRESET="${PATCHCORE_PRESET:-env}"
EFFICIENTAD_PRESET="${EFFICIENTAD_PRESET:-final100}"

PATCHCORE_TRAIN_BATCH_SIZE="${PATCHCORE_TRAIN_BATCH_SIZE:-16}"
PATCHCORE_EVAL_BATCH_SIZE="${PATCHCORE_EVAL_BATCH_SIZE:-16}"
EFFICIENTAD_TRAIN_BATCH_SIZE="${EFFICIENTAD_TRAIN_BATCH_SIZE:-4}"
EFFICIENTAD_EVAL_BATCH_SIZE="${EFFICIENTAD_EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"

mkdir -p "$RESULTS_ROOT" "$OUTPUTS_ROOT" "$LOG_ROOT"

echo "[full] env check"
"$PYTHON" scripts/check_env.py | tee "$LOG_ROOT/check_env.log"

if [[ "$PRECACHE_MODELS" == "1" ]]; then
  echo "[full] pre-cache PatchCore backbone before parallel jobs"
  bash scripts/precache_models.sh --models patchcore | tee "$LOG_ROOT/precache_models.log"
fi

run_job() {
  local gpu="$1"
  shift
  echo "[full] GPU=$gpu $*"
  CUDA_VISIBLE_DEVICES="$gpu" "$@"
}

wait_for_jobs() {
  local pids=("$@")
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

echo "[full] train PatchCore categories=${CATEGORIES_CSV} preset=${PATCHCORE_PRESET}"
pids=()
idx=0
for category in "${CATEGORIES_LIST[@]}"; do
  gpu="${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}"
  log="$LOG_ROOT/train_patchcore_${category}.log"
  run_job "$gpu" "$PYTHON" train.py \
    --model patchcore \
    --category "$category" \
    --preset "$PATCHCORE_PRESET" \
    --data-root "$DATA_ROOT" \
    --results-root "$RESULTS_ROOT" \
    --train-batch-size "$PATCHCORE_TRAIN_BATCH_SIZE" \
    --eval-batch-size "$PATCHCORE_EVAL_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" >"$log" 2>&1 &
  pids+=("$!")
  idx=$((idx + 1))
done
wait_for_jobs "${pids[@]}"

echo "[full] train EfficientAD categories=${CATEGORIES_CSV} preset=${EFFICIENTAD_PRESET}"
pids=()
idx=0
for category in "${CATEGORIES_LIST[@]}"; do
  gpu="${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}"
  log="$LOG_ROOT/train_efficientad_${category}.log"
  run_job "$gpu" "$PYTHON" train.py \
    --model efficientad \
    --category "$category" \
    --preset "$EFFICIENTAD_PRESET" \
    --data-root "$DATA_ROOT" \
    --results-root "$RESULTS_ROOT" \
    --train-batch-size "$EFFICIENTAD_TRAIN_BATCH_SIZE" \
    --eval-batch-size "$EFFICIENTAD_EVAL_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" >"$log" 2>&1 &
  pids+=("$!")
  idx=$((idx + 1))
done
wait_for_jobs "${pids[@]}"

echo "[full] evaluate EfficientAD categories=${CATEGORIES_CSV}"
pids=()
idx=0
for category in "${CATEGORIES_LIST[@]}"; do
  gpu="${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}"
  log="$LOG_ROOT/eval_efficientad_${category}.log"
  run_job "$gpu" "$PYTHON" eval.py \
    --model efficientad \
    --category "$category" \
    --data-root "$DATA_ROOT" \
    --results-root "$RESULTS_ROOT" \
    --output-dir "$OUTPUTS_ROOT/eval" \
    --threshold-strategies fixed otsu percentile \
    --save-visuals \
    --visual-limit 8 >"$log" 2>&1 &
  pids+=("$!")
  idx=$((idx + 1))
done
wait_for_jobs "${pids[@]}"

echo "[full] done"
echo "[full] logs: $LOG_ROOT"
echo "[full] metrics: $OUTPUTS_ROOT/eval/metrics_all.csv plus per-category metrics.csv"
