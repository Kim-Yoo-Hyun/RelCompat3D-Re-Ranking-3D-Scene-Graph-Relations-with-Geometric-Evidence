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
    run_if_missing no_family_indicator_structured_main experiments/RelCompat3D_geom_reliability/no_family_indicator_v1/evaluation/structured_main
    run_if_missing no_family_indicator_support_routing experiments/RelCompat3D_geom_reliability/no_family_indicator_v1/evaluation/support_routing
    run_if_missing no_family_indicator_open3dsg_route experiments/RelCompat3D_geom_reliability/no_family_indicator_v1/evaluation/open3dsg_route
    run_if_missing no_family_indicator_nonlinear experiments/RelCompat3D_geom_reliability/no_family_indicator_v1/evaluation/nonlinear
    ;;
  downstream)
    if [[ ! -f experiments/RelCompat3D_geom_reliability/no_family_indicator_v1/protocols/downstream_freeze.json ]]; then
      run_service no_family_indicator_freeze_downstream
    fi
    run_if_missing no_family_indicator_routed_comparators experiments/RelCompat3D_geom_reliability/no_family_indicator_v1/evaluation/routed_comparators
    run_if_missing no_family_indicator_routed_ablation experiments/RelCompat3D_geom_reliability/no_family_indicator_v1/evaluation/routed_ablation
    run_if_missing no_family_indicator_scan_cluster experiments/RelCompat3D_geom_reliability/no_family_indicator_v1/evaluation/scan_cluster
    run_if_missing no_family_indicator_structured_scan_cluster experiments/RelCompat3D_geom_reliability/no_family_indicator_v1/evaluation/structured_scan_cluster
    run_if_missing no_family_indicator_surface_audit experiments/RelCompat3D_geom_reliability/no_family_indicator_v1/evaluation/surface_audit
    run_if_missing no_family_indicator_held_out_primitive experiments/RelCompat3D_geom_reliability/no_family_indicator_v1/evaluation/held_out_primitive
    run_if_missing no_family_indicator_counterfactual_sensitivity experiments/RelCompat3D_geom_reliability/no_family_indicator_v1/evaluation/counterfactual_sensitivity
    ;;
  *)
    echo "usage: $0 {initial|downstream}" >&2
    exit 2
    ;;
esac
