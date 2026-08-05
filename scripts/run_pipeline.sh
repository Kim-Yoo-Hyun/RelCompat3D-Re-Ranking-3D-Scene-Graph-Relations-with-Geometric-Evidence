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
      require_file "local_dataset/RelCompat3D/canonical/$source/verification.jsonl"
    done
    require_file local_dataset/RelCompat3D/canonical/ground_truth.jsonl
    output=experiments/RelCompat3D_geom_reliability/main_experiment/regenerated/public_evaluation
    if [[ -d "$output" ]] && [[ -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "refusing to overwrite nonempty output: $output" >&2
      exit 2
    fi
    run_service relcompat3d_evaluate_public
    ;;
  tables)
    run_service relcompat3d_export_rows
    scripts/reproduce_tables.sh
    ;;
  *)
    echo "usage: $0 {prepare|evaluate|tables}" >&2
    exit 2
    ;;
esac
