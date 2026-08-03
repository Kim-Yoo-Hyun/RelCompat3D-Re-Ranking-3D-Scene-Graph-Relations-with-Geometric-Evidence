#!/usr/bin/env python3
"""Validate the public repository structure and frozen numerical evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/RelCompat3D_geom_reliability"
REFERENCE = EXPERIMENT / "paper_reproduction/evaluation"
REGENERATED = EXPERIMENT / "paper_reproduction/regenerated"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_json() -> int:
    count = 0
    for pattern in ("experiments/**/*.json", "results/**/*.json"):
        for path in sorted(ROOT.glob(pattern)):
            with path.open(encoding="utf-8") as handle:
                json.load(handle)
            count += 1
    return count


def validate_models(required: bool) -> tuple[int, int]:
    checksum_file = ROOT / "configs/model_files.sha256"
    present = 0
    missing = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative
        if not path.is_file():
            missing += 1
            continue
        if sha256(path) != expected:
            raise SystemExit(f"model checksum mismatch: {relative}")
        present += 1
    if required and missing:
        raise SystemExit(
            f"{missing} model files are missing; run scripts/download_models.sh"
        )
    return present, missing


def validate_cells(directory: Path) -> tuple[int, float]:
    path = directory / "canonical_validation.csv"
    if not path.is_file():
        raise SystemExit(f"missing canonical validation: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 291:
        raise SystemExit(f"expected 291 canonical cells, found {len(rows)}")
    if any(row["passed"] != "True" for row in rows):
        raise SystemExit(f"one or more canonical cells failed in {path}")
    maximum = max(float(row["abs_error"]) for row in rows)
    if maximum > 1e-12:
        raise SystemExit(f"maximum error {maximum} exceeds tolerance 1e-12")
    return len(rows), maximum


def validate_result_index() -> tuple[int, int]:
    index_path = ROOT / "results/relcompat3d_geom_reliability/manifest.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    artifact_count = 0
    for relative in index["canonical_artifacts"].values():
        if not (ROOT / relative).exists():
            raise SystemExit(f"missing indexed artifact: {relative}")
        artifact_count += 1

    manifest_paths = {
        "score_robustness_manifest_sha256": "score_robustness",
        "routing_controls_manifest_sha256": "routing_controls",
        "measurement_audit_manifest_sha256": "measurement_audit",
        "component_analysis_manifest_sha256": "component_analysis",
        "seed_robustness_manifest_sha256": "seed_robustness",
        "paper_reproduction_manifest_sha256": "paper_reproduction",
        "candidate_oracle_manifest_sha256": "candidate_oracle",
    }
    verified = 0
    for key, directory in manifest_paths.items():
        path = EXPERIMENT / directory / "evaluation/manifest.json"
        expected = index["posthoc_integrity"][key]
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"indexed manifest mismatch: {path.relative_to(ROOT)}")
        verified += 1
    return artifact_count, verified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-models", action="store_true")
    parser.add_argument("--regenerated", action="store_true")
    args = parser.parse_args()

    required = [
        ROOT / "configs/relcompat3d/Dockerfile",
        ROOT / "configs/relcompat3d/compose.yaml",
        ROOT / "configs/open3dsg/Dockerfile",
        ROOT / "configs/open3dsg/compose.yaml",
        ROOT / "configs/open3dsg/protocol.json",
        ROOT / "configs/open3dsg/development_scans.txt",
        ROOT / "docs/open3dsg-training.md",
        ROOT / "docs/source-adapters.md",
        ROOT / "scripts/train_open3dsg.sh",
        ROOT / "src/relcompat3d/adapt_source_predictions.py",
        ROOT / "src/relcompat3d/configure_open3dsg.py",
        ROOT / "src/relcompat3d/prepare_open3dsg_splits.py",
        ROOT / "src/relcompat3d/select_open3dsg_checkpoint.py",
        EXPERIMENT / "active_method.json",
        ROOT / "results/relcompat3d_geom_reliability/manifest.json",
        REFERENCE / "table1.csv",
        REFERENCE / "table2.csv",
        REFERENCE / "table3.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files: " + ", ".join(missing))

    json_count = validate_json()
    model_count, missing_models = validate_models(args.require_models)
    cell_count, maximum = validate_cells(REGENERATED if args.regenerated else REFERENCE)
    artifact_count, manifest_count = validate_result_index()

    print(
        json.dumps(
            {
                "status": "passed",
                "json_files": json_count,
                "verified_models": model_count,
                "missing_optional_models": missing_models,
                "canonical_cells": cell_count,
                "maximum_absolute_error": maximum,
                "indexed_artifacts": artifact_count,
                "verified_evidence_manifests": manifest_count,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
