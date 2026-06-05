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
PREPARE_MVTEC="${PREPARE_MVTEC:-1}"
OFFLINE="${OFFLINE:-0}"

CATEGORIES_CSV="${CATEGORIES:-bottle,hazelnut,metal_nut}"
IFS=',' read -r -a CATEGORIES_LIST <<< "$CATEGORIES_CSV"

PATCHCORE_PRESET="${PATCHCORE_PRESET:-env}"
EFFICIENTAD_PRESET="${EFFICIENTAD_PRESET:-final100}"
EFFICIENTAD_MAX_STEPS="${EFFICIENTAD_MAX_STEPS:-}"

PATCHCORE_TRAIN_BATCH_SIZE="${PATCHCORE_TRAIN_BATCH_SIZE:-16}"
PATCHCORE_EVAL_BATCH_SIZE="${PATCHCORE_EVAL_BATCH_SIZE:-16}"
EFFICIENTAD_TRAIN_BATCH_SIZE="${EFFICIENTAD_TRAIN_BATCH_SIZE:-1}"
EFFICIENTAD_EVAL_BATCH_SIZE="${EFFICIENTAD_EVAL_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RUN_PATCHCORE="${RUN_PATCHCORE:-1}"
RUN_EFFICIENTAD="${RUN_EFFICIENTAD:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

mkdir -p "$RESULTS_ROOT" "$OUTPUTS_ROOT" "$LOG_ROOT"

echo "[full] env check"
"$PYTHON" scripts/check_env.py | tee "$LOG_ROOT/check_env.log"

if [[ "$OFFLINE" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  echo "[full] offline mode enabled; skipping model pre-cache downloads"
elif [[ "$PRECACHE_MODELS" == "1" ]]; then
  precache_model_args=()
  if [[ "$RUN_PATCHCORE" == "1" ]]; then
    precache_model_args+=("patchcore")
  fi
  if [[ "$RUN_EFFICIENTAD" == "1" ]]; then
    precache_model_args+=("efficientad")
  fi
  if [[ "${#precache_model_args[@]}" -gt 0 ]]; then
    echo "[full] pre-cache model assets before parallel jobs: ${precache_model_args[*]}"
    bash scripts/precache_models.sh --models "${precache_model_args[@]}" | tee "$LOG_ROOT/precache_models.log"
  fi
fi

if [[ "$PREPARE_MVTEC" == "1" ]]; then
  echo "[full] prepare MVTec AD categories before parallel jobs"
  CATEGORIES="$CATEGORIES_CSV" DATA_ROOT="$DATA_ROOT" bash scripts/download_mvtec_categories.sh | tee "$LOG_ROOT/download_mvtec.log"
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

show_recent_logs() {
  local pattern="$1"
  echo "[full] recent logs for $pattern"
  for log in $LOG_ROOT/$pattern; do
    [[ -f "$log" ]] || continue
    echo "========== $log =========="
    tail -n 80 "$log"
  done
}

if [[ "$RUN_PATCHCORE" == "1" ]]; then
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
  if ! wait_for_jobs "${pids[@]}"; then
    echo "[full] PatchCore stage failed"
    show_recent_logs "train_patchcore_*.log"
    exit 1
  fi
else
  echo "[full] skip PatchCore stage"
fi

if [[ "$RUN_EFFICIENTAD" == "1" ]]; then
  echo "[full] train EfficientAD categories=${CATEGORIES_CSV} preset=${EFFICIENTAD_PRESET}"
  pids=()
  idx=0
  efficientad_step_args=()
  if [[ -n "$EFFICIENTAD_MAX_STEPS" ]]; then
    efficientad_step_args=(--max-steps "$EFFICIENTAD_MAX_STEPS")
    echo "[full] EfficientAD will use max_steps=$EFFICIENTAD_MAX_STEPS"
  fi
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
      --num-workers "$NUM_WORKERS" \
      "${efficientad_step_args[@]}" >"$log" 2>&1 &
    pids+=("$!")
    idx=$((idx + 1))
  done
  if ! wait_for_jobs "${pids[@]}"; then
    echo "[full] EfficientAD stage failed"
    show_recent_logs "train_efficientad_*.log"
    exit 1
  fi
else
  echo "[full] skip EfficientAD stage"
fi

if [[ "$RUN_EVAL" == "1" ]]; then
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
  if ! wait_for_jobs "${pids[@]}"; then
    echo "[full] evaluation stage failed"
    show_recent_logs "eval_efficientad_*.log"
    exit 1
  fi
  "$PYTHON" scripts/merge_eval_metrics.py \
    --output-dir "$OUTPUTS_ROOT/eval" \
    --model efficientad \
    --categories "$CATEGORIES_CSV"
else
  echo "[full] skip evaluation stage"
fi

echo "[full] done"
echo "[full] logs: $LOG_ROOT"
echo "[full] metrics: $OUTPUTS_ROOT/eval/metrics_all.csv plus per-category metrics.csv"
