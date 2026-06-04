#!/usr/bin/env bash
set -euo pipefail

CUDA_EXTRA="${1:-cu121}"
ENV_NAME="${ENV_NAME:-efficientad-ad}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[setup] conda not found. Install Miniconda/Miniforge first."
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup] conda env '$ENV_NAME' already exists"
else
  echo "[setup] creating conda env '$ENV_NAME'"
  conda create -n "$ENV_NAME" python=3.11 -y
fi

conda activate "$ENV_NAME"
python -m pip install -U pip wheel setuptools
python -m pip install "anomalib[$CUDA_EXTRA]" gradio scipy matplotlib
python scripts/check_env.py

echo "[setup] done. Activate with: conda activate $ENV_NAME"
