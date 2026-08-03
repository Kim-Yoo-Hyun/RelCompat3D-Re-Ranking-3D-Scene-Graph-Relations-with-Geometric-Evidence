#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bundle="${1:-}"
mode="${2:-core}"

if [[ -z "$bundle" ]]; then
  echo "usage: $0 /path/to/RelCompat3D_AAAI27_release_20260730 [core|all]" >&2
  exit 2
fi

bundle="$(cd "$bundle" && pwd)"
cd "$root"
if [[ ! -f "$bundle/MANIFEST.sha256" ]]; then
  echo "Missing recovery-archive manifest: $bundle/MANIFEST.sha256" >&2
  exit 1
fi
if [[ "$mode" != "core" && "$mode" != "all" ]]; then
  echo "mode must be core or all" >&2
  exit 2
fi

echo "Verifying maintainer recovery archive..."
(cd "$bundle" && sha256sum --check MANIFEST.sha256)

tmp="$(mktemp -d "${TMPDIR:-/tmp}/relcompat3d_private_restore.XXXXXX")"
trap 'find "$tmp" -depth -delete' EXIT

install_file() {
  local source="$1"
  local destination="$2"
  mkdir -p "$(dirname "$root/$destination")"
  install -m 0644 "$source" "$root/$destination"
}

echo "Restoring fitted RelCompat3D models..."
models="$tmp/models"
mkdir -p "$models"
tar --zstd --extract \
  --file "$bundle/checkpoints/relcompat3d/relcompat3d_models_3dssg_v1.tar.zst" \
  --directory "$models"

while IFS='|' read -r source destination; do
  install_file "$models/$source" "$destination"
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

echo "Mapping legacy paper rows to paper_reproduction/.../table_rows..."
rows="$tmp/rows"
mkdir -p "$rows"
tar --zstd --extract \
  --file "$bundle/artifacts/relcompat3d_paper_table_rows_3dssg_v1.tar.zst" \
  --directory "$rows" \
  row_reproduction_v1/artifacts/derived_rows
for file in ground_truth.csv.gz open3dsg_candidates.csv.gz sgfn_candidates.csv.gz vlsat_candidates.csv.gz schema.json manifest.json; do
  install_file \
    "$rows/row_reproduction_v1/artifacts/derived_rows/$file" \
    "experiments/RelCompat3D_geom_reliability/paper_reproduction/artifacts/table_rows/$file"
done

if [[ "$mode" == "all" ]]; then
  echo "Restoring training/development and point/mesh inputs..."
  training="$tmp/training"
  surface="$tmp/surface"
  mkdir -p "$training" "$surface"
  tar --zstd --extract \
    --file "$bundle/artifacts/relcompat3d_training_inputs_3dssg_train1061_dev117_v1.tar.zst" \
    --directory "$training"
  tar --zstd --extract \
    --file "$bundle/artifacts/relcompat3d_point_mesh_audit_measurements_v1.tar.zst" \
    --directory "$surface"

  install_file "$training/training_inputs/train_table.jsonl" \
    "experiments/RelCompat3D_geom_reliability/training_protocol/calibration/export/table.jsonl"
  install_file "$training/training_inputs/train_table.jsonl" \
    "local_dataset/RelCompat3D/calibration/table.jsonl"
  install_file "$training/training_inputs/dev_ground_truth.jsonl" \
    "experiments/RelCompat3D_geom_reliability/training_protocol/internal_dev_ground_truth.jsonl"
  install_file "$training/training_inputs/dev_geometry_verification.jsonl" \
    "experiments/RelCompat3D_geom_reliability/training_protocol/internal_dev/geometry/verification.jsonl"

  install_file "$surface/point_mesh_audit/training_measurements.jsonl" \
    "local_dataset/RelCompat3D/surface_audit/training_measurements.jsonl"
  install_file "$surface/point_mesh_audit/evaluation_measurements.jsonl" \
    "local_dataset/RelCompat3D/surface_audit/evaluation_measurements.jsonl"
  install_file "$surface/point_mesh_audit/mlp_additional_evaluation_measurements.jsonl" \
    "local_dataset/RelCompat3D/surface_audit/additional_evaluation_measurements.jsonl"
fi

echo "Validating restored public paths..."
"$root/scripts/validate.sh" --require-models
echo "Maintainer recovery archive restored in '$mode' mode."
echo "Run scripts/reproduce_tables.sh to regenerate Tables 1--3 and Figure 3."
