#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

docker compose -f configs/relcompat3d/compose.yaml config --quiet
docker compose -f configs/open3dsg/compose.yaml config --quiet

cache="$(mktemp -d "${TMPDIR:-/tmp}/relcompat3d_pycache.XXXXXX")"
trap 'find "$cache" -depth -delete' EXIT
PYTHONPYCACHEPREFIX="$cache" python -m compileall -q src/relcompat3d

python scripts/validate_repository.py "$@"
