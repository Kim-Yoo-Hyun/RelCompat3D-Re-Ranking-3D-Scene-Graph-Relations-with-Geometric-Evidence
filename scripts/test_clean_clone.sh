#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_repo="${1:-$root}"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/relcompat3d_clean_clone.XXXXXX")"
trap 'find "$tmp" -depth -delete' EXIT

echo "Creating isolated checkout..."
git clone --quiet --local "$source_repo" "$tmp/repo"
cd "$tmp/repo"

echo "Creating synthetic official-output contract..."
python tests/create_synthetic_workspace.py --root "$tmp/repo" >/dev/null

echo "Restoring RelCompat3D models..."
scripts/download_models.sh

echo "Running source adaptation and geometry verification..."
scripts/run_pipeline.sh prepare

echo "Running public evaluation..."
scripts/run_pipeline.sh evaluate

python - <<'PY'
import json
from pathlib import Path

root = Path.cwd()
manifest = (
    root
    / "experiments/RelCompat3D_geom_reliability/main_experiment/regenerated"
    / "public_evaluation/manifest.json"
)
payload = json.loads(manifest.read_text(encoding="utf-8"))
if payload.get("status") != "completed":
    raise SystemExit(f"clean-clone evaluation failed: {payload.get('status')}")
if not all(payload.get("validations", {}).values()):
    raise SystemExit("clean-clone evaluation invariants failed")
print(json.dumps({"status": "passed", "manifest": str(manifest)}, sort_keys=True))
PY
