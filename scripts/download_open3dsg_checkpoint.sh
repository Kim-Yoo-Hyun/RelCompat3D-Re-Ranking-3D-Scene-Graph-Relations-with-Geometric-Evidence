#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
file_id="1PJNduscoRAB6cQcggBOo-ErzkiBs_QDG"
expected_size="419735447"
expected_sha="c1302882da43a7b985c10dd4f50177d5161ff6619090cd0eb4d5ff0411d64511"
destination="${1:-$root/local_dataset/Open3DSG_staged/training_repro/mlops/opensg/mlflow/363094050435167554/25da9c4c00214f3b880cedbb2a124177/checkpoints/epoch=13-step=13104.ckpt}"
partial="${destination}.part"

verify_checkpoint() {
  local checkpoint="$1"
  [[ -f "$checkpoint" ]] || return 1
  [[ "$(stat -c '%s' "$checkpoint")" == "$expected_size" ]] || return 1
  printf '%s  %s\n' "$expected_sha" "$checkpoint" |
    sha256sum --check --status
}

if verify_checkpoint "$destination"; then
  echo "Open3DSG checkpoint already restored and verified: $destination"
  exit 0
fi

mkdir -p "$(dirname "$destination")"
echo "Downloading the selected Open3DSG checkpoint..."
curl --fail --location --retry 3 --continue-at - \
  "https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t" \
  --output "$partial"

if ! verify_checkpoint "$partial"; then
  echo "Open3DSG checkpoint size or checksum mismatch: $partial" >&2
  exit 1
fi

mv "$partial" "$destination"
echo "Open3DSG checkpoint restored and verified: $destination"
