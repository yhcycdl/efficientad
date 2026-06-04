#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-datasets/MVTecAD}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs/day1_cli_smoke}"

mkdir -p "$RESULTS_ROOT" "$OUTPUTS_ROOT"

echo "[day1-cli] Environment check"
python scripts/check_env.py

echo "[day1-cli] PatchCore bottle, 1 epoch via Anomalib CLI"
anomalib train \
  --model Patchcore \
  --data anomalib.data.MVTecAD \
  --data.root "$DATA_ROOT" \
  --data.category bottle \
  --data.train_batch_size "${PATCHCORE_TRAIN_BATCH_SIZE:-8}" \
  --data.eval_batch_size "${PATCHCORE_EVAL_BATCH_SIZE:-8}" \
  --trainer.max_epochs 1 \
  --trainer.default_root_dir "$RESULTS_ROOT/cli_patchcore_bottle_1epoch"

echo "[day1-cli] EfficientAD bottle, 5 epochs via Anomalib CLI"
anomalib train \
  --model EfficientAd \
  --data anomalib.data.MVTecAD \
  --data.root "$DATA_ROOT" \
  --data.category bottle \
  --data.train_batch_size "${EFFICIENTAD_TRAIN_BATCH_SIZE:-1}" \
  --data.eval_batch_size "${EFFICIENTAD_EVAL_BATCH_SIZE:-1}" \
  --trainer.max_epochs 5 \
  --trainer.default_root_dir "$RESULTS_ROOT/cli_efficientad_bottle_5epochs"

CKPT="$(find "$RESULTS_ROOT/cli_efficientad_bottle_5epochs" -name '*.ckpt' | sort | tail -n 1)"
IMAGE="$(find "$DATA_ROOT/bottle/test" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) | sort | head -n 1)"

if [[ -z "$CKPT" ]]; then
  echo "[day1-cli] No EfficientAD checkpoint found, skipping inference."
  exit 1
fi

if [[ -z "$IMAGE" ]]; then
  echo "[day1-cli] No bottle test image found, skipping inference."
  exit 1
fi

echo "[day1-cli] Generate heatmap from $IMAGE"
python infer.py \
  --model efficientad \
  --ckpt "$CKPT" \
  --input "$IMAGE" \
  --output-dir "$OUTPUTS_ROOT/efficientad_bottle" \
  --threshold-strategy otsu \
  --smooth-sigma 4

echo "[day1-cli] Done. Check $OUTPUTS_ROOT/efficientad_bottle for heatmap/overlay/mask."
