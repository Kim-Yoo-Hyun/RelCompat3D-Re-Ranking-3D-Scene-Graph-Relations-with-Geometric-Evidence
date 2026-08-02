#!/usr/bin/env bash
set -euo pipefail

phase="${1:-}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

compose=(docker compose -f configs/relcompat3d/compose.yaml)

run_service() {
  env UID="$(id -u)" GID="$(id -g)" "${compose[@]}" run --rm "$1"
}

run_if_missing() {
  local service="$1"
  local output="$2"
  local manifest="$output/manifest.json"
  if [[ -f "$manifest" ]] && [[ "$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "$manifest")" == "completed" ]]; then
    echo "skip completed: $service"
    return
  fi
  if [[ -d "$output" ]] && [[ -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "refusing nonempty incomplete output: $output" >&2
    return 1
  fi
  run_service "$service"
}

case "$phase" in
  initial)
    run_if_missing relcompat3d_evaluate_linear experiments/RelCompat3D_geom_reliability/main_experiment/evaluation/structured_main
    run_if_missing relcompat3d_support_routing experiments/RelCompat3D_geom_reliability/main_experiment/evaluation/support_routing
    run_if_missing relcompat3d_open3dsg_route experiments/RelCompat3D_geom_reliability/main_experiment/evaluation/open3dsg_route
    run_if_missing relcompat3d_evaluate_mlp experiments/RelCompat3D_geom_reliability/main_experiment/evaluation/nonlinear
    ;;
  downstream)
    if [[ ! -f experiments/RelCompat3D_geom_reliability/main_experiment/protocols/downstream_freeze.json ]]; then
      run_service relcompat3d_freeze_downstream
    fi
    run_if_missing relcompat3d_compare_rankings experiments/RelCompat3D_geom_reliability/main_experiment/evaluation/routed_comparators
    run_if_missing relcompat3d_linear_controls experiments/RelCompat3D_geom_reliability/main_experiment/evaluation/routed_ablation
    run_if_missing relcompat3d_bootstrap experiments/RelCompat3D_geom_reliability/main_experiment/evaluation/scan_cluster
    run_if_missing relcompat3d_linear_bootstrap experiments/RelCompat3D_geom_reliability/main_experiment/evaluation/structured_scan_cluster
    run_if_missing relcompat3d_surface_audit experiments/RelCompat3D_geom_reliability/main_experiment/evaluation/surface_audit
    run_if_missing relcompat3d_feature_removal experiments/RelCompat3D_geom_reliability/main_experiment/evaluation/held_out_primitive
    run_if_missing relcompat3d_counterfactual_controls experiments/RelCompat3D_geom_reliability/main_experiment/evaluation/counterfactual_sensitivity
    ;;
  *)
    echo "usage: $0 {initial|downstream}" >&2
    exit 2
    ;;
esac
