#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-datasets/MVTecAD}"

echo "[offline-check] data root: $DATA_ROOT"
missing=0

for category in bottle hazelnut metal_nut; do
  for split in train/good test; do
    path="$DATA_ROOT/$category/$split"
    if [[ -d "$path" ]]; then
      count="$(find "$path" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) | wc -l)"
      echo "[offline-check] OK $path images=$count"
    else
      echo "[offline-check] MISSING $path"
      missing=1
    fi
  done
done

echo "[offline-check] huggingface cache:"
if [[ -n "${HF_HOME:-}" ]]; then
  echo "  HF_HOME=$HF_HOME"
  find "$HF_HOME" -maxdepth 4 -type d -name 'models--timm--wide_resnet50_2*' 2>/dev/null | head
else
  echo "  HF_HOME not set; default is ~/.cache/huggingface"
  find "$HOME/.cache/huggingface" -maxdepth 4 -type d -name 'models--timm--wide_resnet50_2*' 2>/dev/null | head
fi

if [[ "$missing" == "1" ]]; then
  echo "[offline-check] Dataset is incomplete. Upload MVTec AD before training."
  exit 1
fi

echo "[offline-check] done"
