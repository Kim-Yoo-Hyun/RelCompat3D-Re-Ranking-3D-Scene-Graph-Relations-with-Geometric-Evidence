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
  train)
    require_file local_dataset/RelCompat3D/3DSSG_subset/relationships_train.json
    require_file local_dataset/RelCompat3D/3DSSG_subset/relationships.txt
    require_dir local_dataset/3RScan/scans
    require_empty_output experiments/relcompat3d/training/calibration/regenerated
    require_empty_output experiments/relcompat3d/main/regenerated/base
    require_empty_output experiments/relcompat3d/main/regenerated/fit
    require_empty_output experiments/relcompat3d/main/regenerated/mlp
    run_service relcompat3d_build_training_rows
    run_service relcompat3d_fit_base
    run_service relcompat3d_fit
    run_service relcompat3d_fit_mlp
    ;;
  evaluate)
    for source in vlsat sgfn open3dsg; do
      require_file "local_dataset/RelCompat3D/prepared/$source/verification.jsonl"
    done
    require_file local_dataset/RelCompat3D/prepared/ground_truth.jsonl
    require_file experiments/relcompat3d/main/regenerated/fit/linear_models.json
    require_file experiments/relcompat3d/main/regenerated/mlp/models.json
    require_empty_output experiments/relcompat3d/main/regenerated/evaluation
    run_service relcompat3d_evaluate
    ;;
  audit)
    require_dir local_dataset/3RScan/scans
    require_file experiments/relcompat3d/training/calibration/regenerated/table.jsonl
    require_file experiments/relcompat3d/main/regenerated/fit/base_models.json
    require_file experiments/relcompat3d/main/regenerated/fit/linear_models.json
    require_empty_output experiments/relcompat3d/main/regenerated/point_mesh_analysis
    run_service relcompat3d_point_mesh_analysis
    ;;
  tables)
    require_file experiments/relcompat3d/main/regenerated/fit/linear_models.json
    require_file experiments/relcompat3d/main/regenerated/mlp/models.json
    require_file experiments/relcompat3d/main/regenerated/point_mesh_analysis/thresholds.json
    require_file experiments/relcompat3d/main/regenerated/point_mesh_analysis/evaluation_measurements.jsonl
    key=local_dataset/RelCompat3D/secrets/table_rows_hmac_key.txt
    if [[ ! -f "$key" ]]; then
      run_service relcompat3d_create_local_key
    fi
    require_empty_output experiments/relcompat3d/paper_results/regenerated/rows
    require_empty_output experiments/relcompat3d/paper_results/regenerated/tables
    run_service relcompat3d_export_rows
    run_service relcompat3d_generate_tables
    ;;
  full)
    "$0" prepare
    "$0" train
    "$0" evaluate
    "$0" audit
    "$0" tables
    ;;
  *)
    echo "usage: $0 {prepare|train|evaluate|audit|tables|full}" >&2
    exit 2
    ;;
esac
