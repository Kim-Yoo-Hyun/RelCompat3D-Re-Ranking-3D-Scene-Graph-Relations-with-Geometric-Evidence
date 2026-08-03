#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

file_id="1DaZoibKFyPS681e728Tzs613qscMgv4u"
archive_sha="4659858da8ff53f2c09769527ac486d182eef6e35c12ebeacfcc5d7ff6fdc103"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/relcompat3d_models.XXXXXX")"
trap 'find "$tmp" -depth -delete' EXIT
archive="$tmp/relcompat3d_models_3dssg_v1.tar.zst"
extracted="$tmp/extracted"

echo "Downloading RelCompat3D fitted models..."
curl --fail --location --retry 3 \
  "https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t" \
  --output "$archive"

printf '%s  %s\n' "$archive_sha" "$archive" | sha256sum --check --status || {
  echo "Model archive checksum mismatch." >&2
  exit 1
}

mkdir -p "$extracted"
tar --zstd --extract --file "$archive" --directory "$extracted"

# Map the experiment paths stored in the model archive to the current
# repository layout without modifying model contents.
while IFS='|' read -r source destination; do
  mkdir -p "$(dirname "$root/$destination")"
  cp "$extracted/$source" "$root/$destination"
done <<'MODELS'
experiments/RelCompat3D_geom_reliability/no_family_indicator_v1/fit/structured_models.json|experiments/RelCompat3D_geom_reliability/main_experiment/fit/structured_models.json
experiments/RelCompat3D_geom_reliability/no_family_indicator_v1/fit/strict_models.json|experiments/RelCompat3D_geom_reliability/main_experiment/fit/strict_models.json
experiments/RelCompat3D_geom_reliability/no_family_indicator_v1/evaluation/nonlinear/models.json|experiments/RelCompat3D_geom_reliability/main_experiment/evaluation/nonlinear/models.json
experiments/RelCompat3D_geom_reliability/factor_isolation_protocol/fitted_v1/models.json|experiments/RelCompat3D_geom_reliability/factor_controls/fitted/models.json
experiments/RelCompat3D_geom_reliability/no_family_indicator_v1/evaluation/counterfactual_sensitivity/models.json|experiments/RelCompat3D_geom_reliability/main_experiment/evaluation/counterfactual_sensitivity/models.json
experiments/RelCompat3D_geom_reliability/no_family_indicator_v1/evaluation/held_out_primitive/models.json|experiments/RelCompat3D_geom_reliability/main_experiment/evaluation/held_out_primitive/models.json
experiments/RelCompat3D_geom_reliability/train_only_reestablishment_v1/calibration/fitted/models.json|experiments/RelCompat3D_geom_reliability/training_protocol/calibration/fitted/models.json
experiments/RelCompat3D_geom_reliability/component_diagnostics_v1/evaluation/models.json|experiments/RelCompat3D_geom_reliability/component_analysis/evaluation/models.json
MODELS

sha256sum --check configs/model_files.sha256
echo "RelCompat3D models restored and verified."
