#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
GPUS_CSV="${GPUS:-1,2,3}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS_CSV"

DATA_ROOT="${DATA_ROOT:-datasets/MVTecAD}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
LOG_ROOT="${LOG_ROOT:-$OUTPUTS_ROOT/efficientad_tuning_logs}"
CATEGORIES_CSV="${CATEGORIES:-bottle,hazelnut,metal_nut}"
IFS=',' read -r -a CATEGORIES_LIST <<< "$CATEGORIES_CSV"
TUNING_MODELS_CSV="${TUNING_MODELS:-efficientad_m}"
IFS=',' read -r -a TUNING_MODELS_LIST <<< "$TUNING_MODELS_CSV"

EFFICIENTAD_TUNE_PRESET="${EFFICIENTAD_TUNE_PRESET:-final100}"
EFFICIENTAD_TRAIN_BATCH_SIZE="${EFFICIENTAD_TRAIN_BATCH_SIZE:-1}"
EFFICIENTAD_EVAL_BATCH_SIZE="${EFFICIENTAD_EVAL_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECACHE_MODELS="${PRECACHE_MODELS:-1}"

mkdir -p "$LOG_ROOT"

if [[ "${USE_HF_MIRROR:-0}" == "1" && -z "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT="https://hf-mirror.com"
fi
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

run_job() {
  local gpu="$1"
  shift
  echo "[tune] GPU=$gpu $*"
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
  echo "[tune] recent logs for $pattern"
  for log in $LOG_ROOT/$pattern; do
    [[ -f "$log" ]] || continue
    echo "========== $log =========="
    tail -n 80 "$log"
  done
}

echo "[tune] train/evaluate tuned EfficientAD models: $TUNING_MODELS_CSV preset=$EFFICIENTAD_TUNE_PRESET"
BASELINES="$TUNING_MODELS_CSV" \
  GPUS="$GPUS_CSV" \
  DATA_ROOT="$DATA_ROOT" \
  RESULTS_ROOT="$RESULTS_ROOT" \
  OUTPUTS_ROOT="$OUTPUTS_ROOT" \
  CATEGORIES="$CATEGORIES_CSV" \
  PRECACHE_MODELS="$PRECACHE_MODELS" \
  EFFICIENTAD_TUNE_PRESET="$EFFICIENTAD_TUNE_PRESET" \
  EFFICIENTAD_TRAIN_BATCH_SIZE="$EFFICIENTAD_TRAIN_BATCH_SIZE" \
  EFFICIENTAD_EVAL_BATCH_SIZE="$EFFICIENTAD_EVAL_BATCH_SIZE" \
  NUM_WORKERS="$NUM_WORKERS" \
  bash scripts/run_extra_baselines.sh

for model in "${TUNING_MODELS_LIST[@]}"; do
  model="${model// /}"
  model="$(printf '%s' "$model" | tr '[:upper:]' '[:lower:]')"
  model="${model//-/_}"
  output_dir="$OUTPUTS_ROOT/eval_${model}_fusion_morph"
  echo "[tune] evaluate enhanced postprocess for $model -> $output_dir"
  pids=()
  idx=0
  for category in "${CATEGORIES_LIST[@]}"; do
    gpu="${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}"
    log="$LOG_ROOT/eval_${model}_fusion_morph_${category}.log"
    run_job "$gpu" "$PYTHON" eval.py \
      --model "$model" \
      --category "$category" \
      --data-root "$DATA_ROOT" \
      --results-root "$RESULTS_ROOT" \
      --output-dir "$output_dir" \
      --fusion-scales 224 256 288 \
      --threshold-strategies fixed otsu percentile best_f1 \
      --threshold-val-ratio 0.2 \
      --mask-postprocess morph_cc \
      --min-area 64 \
      --close-size 5 \
      --save-visuals \
      --visual-limit 8 >"$log" 2>&1 &
    pids+=("$!")
    idx=$((idx + 1))
  done
  if ! wait_for_jobs "${pids[@]}"; then
    echo "[tune] enhanced eval failed for $model"
    show_recent_logs "eval_${model}_fusion_morph_*.log"
    exit 1
  fi
  "$PYTHON" scripts/merge_eval_metrics.py --output-dir "$output_dir" --model "$model" --categories "$CATEGORIES_CSV"
done

"$PYTHON" scripts/summarize_experiments.py
echo "[tune] done"
