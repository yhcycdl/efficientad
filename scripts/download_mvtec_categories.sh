#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-datasets/MVTecAD}"
CATEGORIES_CSV="${CATEGORIES:-bottle,hazelnut,metal_nut}"
TMP_ROOT="${TMP_ROOT:-.cache/mvtec_downloads}"

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
