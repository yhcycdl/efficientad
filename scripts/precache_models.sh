#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

if [[ "${USE_HF_MIRROR:-0}" == "1" && -z "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT="https://hf-mirror.com"
fi

"$PYTHON" scripts/precache_models.py "$@"
