#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

file_id="1DaZoibKFyPS681e728Tzs613qscMgv4u"
archive_sha="4659858da8ff53f2c09769527ac486d182eef6e35c12ebeacfcc5d7ff6fdc103"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/relcompat3d_models.XXXXXX")"
trap 'find "$tmp" -depth -delete' EXIT
archive="$tmp/relcompat3d_models_3dssg_v1.tar.zst"

echo "Downloading RelCompat3D fitted models..."
curl --fail --location --retry 3 \
  "https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t" \
  --output "$archive"

printf '%s  %s\n' "$archive_sha" "$archive" | sha256sum --check --status || {
  echo "Model archive checksum mismatch." >&2
  exit 1
}

tar --zstd --extract --file "$archive" --directory "$root"
sha256sum --check configs/model_files.sha256
echo "RelCompat3D models restored and verified."
