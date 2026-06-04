#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-datasets/MVTecAD}"
CATEGORIES_CSV="${CATEGORIES:-bottle,hazelnut,metal_nut}"
TMP_ROOT="${TMP_ROOT:-.cache/mvtec_downloads}"
SOURCE="${SOURCE:-hf}"
HF_REPO="${HF_REPO:-micguida1/mvtech_anomaly_detection}"
HF_ARCHIVE="${HF_ARCHIVE:-mvtec_anomaly_detection.tar.xz}"

mkdir -p "$DATA_ROOT" "$TMP_ROOT"

declare -A URLS
URLS[bottle]="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937370-1629958698/bottle.tar.xz"
URLS[hazelnut]="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937545-1629959162/hazelnut.tar.xz"
URLS[metal_nut]="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937637-1629959294/metal_nut.tar.xz"

download_file() {
  local url="$1"
  local dest="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --retry-delay 3 -o "$dest" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$dest" "$url"
  else
    echo "[mvtec] neither curl nor wget is available"
    exit 1
  fi
}

IFS=',' read -r -a CATEGORIES_LIST <<< "$CATEGORIES_CSV"

extract_from_archive() {
  local archive="$1"
  local listing="$TMP_ROOT/archive_listing.txt"
  tar -tf "$archive" > "$listing"

  for category in "${CATEGORIES_LIST[@]}"; do
    category="${category// /}"
    if [[ -d "$DATA_ROOT/$category/train" && -d "$DATA_ROOT/$category/test" ]]; then
      echo "[mvtec] $category already exists under $DATA_ROOT, skipping"
      continue
    fi

    echo "[mvtec] extracting $category from $archive"
    if grep -q "^${category}/" "$listing"; then
      tar -xJf "$archive" -C "$DATA_ROOT" "$category"
    else
      prefix="$(grep -m 1 "/${category}/train/" "$listing" | sed "s#/${category}/train/.*##")"
      if [[ -z "$prefix" ]]; then
        echo "[mvtec] could not find category '$category' in archive"
        echo "[mvtec] archive top entries:"
        head -20 "$listing"
        exit 1
      fi
      tar -xJf "$archive" -C "$TMP_ROOT" "${prefix}/${category}"
      mv "$TMP_ROOT/${prefix}/${category}" "$DATA_ROOT/"
    fi

    if [[ ! -d "$DATA_ROOT/$category/train" || ! -d "$DATA_ROOT/$category/test" ]]; then
      echo "[mvtec] extraction did not create expected folders for $category"
      echo "[mvtec] expected: $DATA_ROOT/$category/train and $DATA_ROOT/$category/test"
      exit 1
    fi
  done
}

download_from_hf() {
  local local_dir="$TMP_ROOT/huggingface"
  local archive="$local_dir/$HF_ARCHIVE"
  mkdir -p "$local_dir"

  if [[ "${USE_HF_MIRROR:-0}" == "1" && -z "${HF_ENDPOINT:-}" ]]; then
    export HF_ENDPOINT="https://hf-mirror.com"
  fi

  echo "[mvtec] HF_ENDPOINT=${HF_ENDPOINT:-<default>}"
  echo "[mvtec] downloading $HF_REPO/$HF_ARCHIVE"

  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$HF_REPO" "$HF_ARCHIVE" --repo-type dataset --local-dir "$local_dir"
  else
    python - <<PY
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="$HF_REPO",
    filename="$HF_ARCHIVE",
    repo_type="dataset",
    local_dir="$local_dir",
)
PY
  fi

  if [[ ! -f "$archive" ]]; then
    echo "[mvtec] archive not found after HuggingFace download: $archive"
    exit 1
  fi

  extract_from_archive "$archive"
}

if [[ "$SOURCE" == "hf" ]]; then
  download_from_hf
  echo "[mvtec] ready: $DATA_ROOT"
  exit 0
fi

for category in "${CATEGORIES_LIST[@]}"; do
  category="${category// /}"
  if [[ -z "${URLS[$category]:-}" ]]; then
    echo "[mvtec] no URL configured for category: $category"
    exit 1
  fi

  if [[ -d "$DATA_ROOT/$category/train" && -d "$DATA_ROOT/$category/test" ]]; then
    echo "[mvtec] $category already exists under $DATA_ROOT, skipping"
    continue
  fi

  archive="$TMP_ROOT/${category}.tar.xz"
  echo "[mvtec] downloading $category -> $archive"
  download_file "${URLS[$category]}" "$archive"

  echo "[mvtec] extracting $archive -> $DATA_ROOT"
  tar -xJf "$archive" -C "$DATA_ROOT"

  if [[ ! -d "$DATA_ROOT/$category/train" || ! -d "$DATA_ROOT/$category/test" ]]; then
    echo "[mvtec] extraction did not create expected folders for $category"
    echo "[mvtec] expected: $DATA_ROOT/$category/train and $DATA_ROOT/$category/test"
    exit 1
  fi
done

echo "[mvtec] ready: $DATA_ROOT"
