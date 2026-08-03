#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose=(docker compose -f "$root/configs/open3dsg/compose.yaml")
runtime="$root/local_dataset/Open3DSG"
feature_record="$runtime/feature_directory.txt"

mkdir -p \
  "$runtime/data/3RScan/3DSSG_subset" \
  "$runtime/output/features" \
  "$runtime/output/checkpoints" \
  "$runtime/mlops/opensg/mlflow" \
  "$root/local_dataset/model_cache/huggingface" \
  "$root/local_dataset/model_cache/torch" \
  "$root/local_dataset/model_cache/home" \
  "$root/local_dataset/model_cache/xdg"

run_service() {
  env UID="$(id -u)" GID="$(id -g)" "${compose[@]}" run --rm "$@"
}

prepare() {
  run_service open3dsg_environment
  run_service open3dsg_stage_splits
}

preprocess() {
  run_service open3dsg_views_train
  run_service open3dsg_views_development
  run_service open3dsg_preprocess
  run_service open3dsg_filter_splits
}

features() {
  run_service open3dsg_features
  local latest
  latest="$(find "$runtime/output/features" -mindepth 1 -maxdepth 1 \
    -type d -name 'clip_features_*' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
  if [[ -z "$latest" ]]; then
    echo "Open3DSG feature extraction did not create a feature directory." >&2
    exit 1
  fi
  printf '%s\n' "$latest" > "$feature_record"
  echo "Recorded feature directory: $latest"
}

train() {
  local feature_dir="${OPEN3DSG_FEATURE_DIR:-}"
  if [[ -z "$feature_dir" && -f "$feature_record" ]]; then
    feature_dir="$(<"$feature_record")"
  fi
  if [[ -z "$feature_dir" || ! -d "$feature_dir" ]]; then
    echo "Run the features stage or set OPEN3DSG_FEATURE_DIR." >&2
    exit 1
  fi
  local container_feature_dir="/workspace${feature_dir#"$root"}"
  env UID="$(id -u)" GID="$(id -g)" OPEN3DSG_FEATURE_DIR="$container_feature_dir" \
    "${compose[@]}" run --rm open3dsg_train
}

select_checkpoint() {
  run_service open3dsg_select_checkpoint
}

case "${1:-}" in
  prepare) prepare ;;
  preprocess) preprocess ;;
  features) features ;;
  train) train ;;
  select) select_checkpoint ;;
  all)
    prepare
    preprocess
    features
    train
    select_checkpoint
    ;;
  *)
    echo "Usage: $0 {prepare|preprocess|features|train|select|all}" >&2
    exit 2
    ;;
esac
