#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
GPUS_CSV="${GPUS:-0,1,2}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS_CSV"

DATA_ROOT="${DATA_ROOT:-datasets/MVTecAD}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
LOG_ROOT="${LOG_ROOT:-$OUTPUTS_ROOT/extra_baseline_logs}"
CATEGORIES_CSV="${CATEGORIES:-bottle,hazelnut,metal_nut}"
IFS=',' read -r -a CATEGORIES_LIST <<< "$CATEGORIES_CSV"
PRECACHE_MODELS="${PRECACHE_MODELS:-1}"

BASELINES_CSV="${BASELINES:-padim,stfpm}"
IFS=',' read -r -a BASELINE_LIST <<< "$BASELINES_CSV"

STFPM_PRESET="${STFPM_PRESET:-initial20}"
PADIM_PRESET="${PADIM_PRESET:-env}"
NUM_WORKERS="${NUM_WORKERS:-8}"

mkdir -p "$LOG_ROOT"

if [[ "${USE_HF_MIRROR:-0}" == "1" && -z "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT="https://hf-mirror.com"
fi
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

run_job() {
  local gpu="$1"
  shift
  echo "[baseline] GPU=$gpu $*"
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
  echo "[baseline] recent logs for $pattern"
  for log in $LOG_ROOT/$pattern; do
    [[ -f "$log" ]] || continue
    echo "========== $log =========="
    tail -n 80 "$log"
  done
}

for model in "${BASELINE_LIST[@]}"; do
  model="${model// /}"
  if [[ "$model" == "padim" ]]; then
    preset="$PADIM_PRESET"
    train_bs="${PADIM_TRAIN_BATCH_SIZE:-16}"
    eval_bs="${PADIM_EVAL_BATCH_SIZE:-16}"
  elif [[ "$model" == "stfpm" ]]; then
    preset="$STFPM_PRESET"
    train_bs="${STFPM_TRAIN_BATCH_SIZE:-8}"
    eval_bs="${STFPM_EVAL_BATCH_SIZE:-8}"
  else
    echo "[baseline] unsupported model: $model"
    exit 1
  fi

  if [[ "$PRECACHE_MODELS" == "1" ]]; then
    echo "[baseline] pre-cache $model assets before parallel jobs"
    bash scripts/precache_models.sh --models "$model" | tee "$LOG_ROOT/precache_${model}.log"
  fi

  echo "[baseline] train $model categories=${CATEGORIES_CSV} preset=${preset}"
  pids=()
  idx=0
  for category in "${CATEGORIES_LIST[@]}"; do
    gpu="${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}"
    log="$LOG_ROOT/train_${model}_${category}.log"
    run_job "$gpu" "$PYTHON" train.py \
      --model "$model" \
      --category "$category" \
      --preset "$preset" \
      --data-root "$DATA_ROOT" \
      --results-root "$RESULTS_ROOT" \
      --train-batch-size "$train_bs" \
      --eval-batch-size "$eval_bs" \
      --num-workers "$NUM_WORKERS" >"$log" 2>&1 &
    pids+=("$!")
    idx=$((idx + 1))
  done
  if ! wait_for_jobs "${pids[@]}"; then
    echo "[baseline] train stage failed for $model"
    show_recent_logs "train_${model}_*.log"
    exit 1
  fi

  echo "[baseline] evaluate $model"
  pids=()
  idx=0
  for category in "${CATEGORIES_LIST[@]}"; do
    gpu="${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}"
    log="$LOG_ROOT/eval_${model}_${category}.log"
    run_job "$gpu" "$PYTHON" eval.py \
      --model "$model" \
      --category "$category" \
      --data-root "$DATA_ROOT" \
      --results-root "$RESULTS_ROOT" \
      --output-dir "$OUTPUTS_ROOT/eval_${model}" \
      --threshold-strategies fixed otsu percentile \
      --save-visuals \
      --visual-limit 8 >"$log" 2>&1 &
    pids+=("$!")
    idx=$((idx + 1))
  done
  if ! wait_for_jobs "${pids[@]}"; then
    echo "[baseline] eval stage failed for $model"
    show_recent_logs "eval_${model}_*.log"
    exit 1
  fi
  "$PYTHON" scripts/merge_eval_metrics.py --output-dir "$OUTPUTS_ROOT/eval_${model}" --model "$model" --categories "$CATEGORIES_CSV"
done

echo "[baseline] done"
