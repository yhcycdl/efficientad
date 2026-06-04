#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-datasets/MVTecAD}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs/day1_smoke}"

mkdir -p "$RESULTS_ROOT" "$OUTPUTS_ROOT"

echo "[day1] Environment check"
python scripts/check_env.py

echo "[day1] PatchCore bottle, 1 epoch via project wrapper"
python train.py \
  --model patchcore \
  --category bottle \
  --preset env \
  --data-root "$DATA_ROOT" \
  --results-root "$RESULTS_ROOT" \
  --train-batch-size "${PATCHCORE_TRAIN_BATCH_SIZE:-8}" \
  --eval-batch-size "${PATCHCORE_EVAL_BATCH_SIZE:-8}"

echo "[day1] EfficientAD bottle, 5 epochs via project wrapper"
python train.py \
  --model efficientad \
  --category bottle \
  --preset smoke \
  --data-root "$DATA_ROOT" \
  --results-root "$RESULTS_ROOT" \
  --train-batch-size "${EFFICIENTAD_TRAIN_BATCH_SIZE:-1}" \
  --eval-batch-size "${EFFICIENTAD_EVAL_BATCH_SIZE:-1}"

CKPT="$(find "$RESULTS_ROOT/efficientad/bottle" -name '*.ckpt' | sort | tail -n 1)"
IMAGE="$(find "$DATA_ROOT/bottle/test" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) | sort | head -n 1)"

if [[ -z "$CKPT" ]]; then
  echo "[day1] No EfficientAD checkpoint found, skipping inference."
  exit 1
fi

if [[ -z "$IMAGE" ]]; then
  echo "[day1] No bottle test image found, skipping inference."
  exit 1
fi

echo "[day1] Generate heatmap from $IMAGE"
python infer.py \
  --model efficientad \
  --ckpt "$CKPT" \
  --input "$IMAGE" \
  --output-dir "$OUTPUTS_ROOT/efficientad_bottle" \
  --threshold-strategy otsu \
  --smooth-sigma 4

echo "[day1] Done. Check $OUTPUTS_ROOT/efficientad_bottle for heatmap/overlay/mask."
