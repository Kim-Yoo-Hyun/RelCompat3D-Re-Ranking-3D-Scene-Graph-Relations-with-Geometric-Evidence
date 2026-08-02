#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

rows="experiments/RelCompat3D_geom_reliability/paper_reproduction/artifacts/table_rows"
required=(
  ground_truth.csv.gz
  open3dsg_candidates.csv.gz
  sgfn_candidates.csv.gz
  vlsat_candidates.csv.gz
  schema.json
)

for file in "${required[@]}"; do
  if [[ ! -f "$rows/$file" ]]; then
    echo "Missing $rows/$file" >&2
    echo "See docs/data.md for the licensed-input export route." >&2
    exit 1
  fi
done

env UID="$(id -u)" GID="$(id -g)" docker compose \
  -f configs/relcompat3d/compose.yaml \
  run --rm relcompat3d_reproduce_rows

python scripts/validate_repository.py --regenerated
