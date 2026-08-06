#!/usr/bin/env bash
set -euo pipefail

stage="${1:-}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

compose=(docker compose -f configs/relcompat3d/compose.yaml)

run_service() {
  env UID="$(id -u)" GID="$(id -g)" "${compose[@]}" run --rm "$1"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing required file: $1" >&2
    exit 2
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "missing required directory: $1" >&2
    exit 2
  fi
}

require_empty_output() {
  if [[ -d "$1" ]] && [[ -n "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "refusing to overwrite nonempty output: $1" >&2
    exit 2
  fi
}

case "$stage" in
  prepare)
    for source in vlsat sgfn open3dsg; do
      require_file "local_dataset/RelCompat3D/source_outputs/$source/raw.jsonl"
    done
    require_file local_dataset/RelCompat3D/3DSSG_subset/relationships_validation.json
    require_file local_dataset/RelCompat3D/3DSSG_subset/relationships.txt
    run_service relcompat3d_adapt_vlsat
    run_service relcompat3d_adapt_sgfn
    run_service relcompat3d_adapt_open3dsg
    run_service relcompat3d_build_ground_truth
    run_service relcompat3d_verify_vlsat
    run_service relcompat3d_verify_sgfn
    run_service relcompat3d_verify_open3dsg
    ;;
  evaluate)
    scripts/validate.sh --require-models
    for source in vlsat sgfn open3dsg; do
      require_file "local_dataset/RelCompat3D/prepared/$source/verification.jsonl"
    done
    require_file local_dataset/RelCompat3D/prepared/ground_truth.jsonl
    output=experiments/RelCompat3D_geom_reliability/main_experiment/regenerated/public_evaluation
    require_empty_output "$output"
    run_service relcompat3d_evaluate_public
    ;;
  train)
    require_file local_dataset/RelCompat3D/3DSSG_subset/relationships_train.json
    require_file local_dataset/RelCompat3D/3DSSG_subset/relationships.txt
    require_dir local_dataset/3RScan/scans
    require_empty_output experiments/RelCompat3D_geom_reliability/training_protocol/calibration/regenerated
    require_empty_output experiments/RelCompat3D_geom_reliability/main_experiment/regenerated/base
    require_empty_output experiments/RelCompat3D_geom_reliability/main_experiment/regenerated/fit
    require_empty_output experiments/RelCompat3D_geom_reliability/main_experiment/regenerated/nonlinear
    run_service relcompat3d_build_training_rows
    run_service relcompat3d_fit_base
    run_service relcompat3d_fit
    run_service relcompat3d_fit_mlp
    ;;
  evaluate-trained)
    for source in vlsat sgfn open3dsg; do
      require_file "local_dataset/RelCompat3D/prepared/$source/verification.jsonl"
    done
    require_file local_dataset/RelCompat3D/prepared/ground_truth.jsonl
    require_file experiments/RelCompat3D_geom_reliability/main_experiment/regenerated/fit/structured_models.json
    require_file experiments/RelCompat3D_geom_reliability/main_experiment/regenerated/nonlinear/models.json
    require_empty_output experiments/RelCompat3D_geom_reliability/main_experiment/regenerated/trained_evaluation
    run_service relcompat3d_evaluate_trained
    ;;
  audit-trained)
    require_dir local_dataset/3RScan/scans
    require_file experiments/RelCompat3D_geom_reliability/training_protocol/calibration/regenerated/table.jsonl
    require_file experiments/RelCompat3D_geom_reliability/main_experiment/regenerated/fit/strict_models.json
    require_file experiments/RelCompat3D_geom_reliability/main_experiment/regenerated/fit/structured_models.json
    require_empty_output experiments/RelCompat3D_geom_reliability/main_experiment/regenerated/public_surface_audit
    run_service relcompat3d_surface_audit_trained
    ;;
  tables-trained)
    require_file experiments/RelCompat3D_geom_reliability/main_experiment/regenerated/fit/structured_models.json
    require_file experiments/RelCompat3D_geom_reliability/main_experiment/regenerated/nonlinear/models.json
    require_file experiments/RelCompat3D_geom_reliability/main_experiment/regenerated/public_surface_audit/thresholds.json
    require_file experiments/RelCompat3D_geom_reliability/main_experiment/regenerated/public_surface_audit/evaluation_measurements.jsonl
    key=local_dataset/RelCompat3D/secrets/table_rows_hmac_key.txt
    if [[ ! -f "$key" ]]; then
      run_service relcompat3d_create_local_key
    fi
    require_empty_output experiments/RelCompat3D_geom_reliability/paper_reproduction/regenerated/public_rows
    require_empty_output experiments/RelCompat3D_geom_reliability/paper_reproduction/regenerated/public_tables
    run_service relcompat3d_export_trained_rows
    run_service relcompat3d_reproduce_trained_rows
    ;;
  tables)
    run_service relcompat3d_export_rows
    scripts/reproduce_tables.sh
    ;;
  full)
    "$0" prepare
    "$0" train
    "$0" evaluate-trained
    "$0" audit-trained
    "$0" tables-trained
    ;;
  *)
    echo "usage: $0 {prepare|train|evaluate|evaluate-trained|audit-trained|tables|tables-trained|full}" >&2
    exit 2
    ;;
esac
