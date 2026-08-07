#!/usr/bin/env python3
"""Validate repository structure and reported numerical results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/relcompat3d"
REFERENCE = EXPERIMENT / "paper_results/evaluation"
REGENERATED = EXPERIMENT / "paper_results/regenerated"


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


def validate_cells(directory: Path) -> tuple[int, float]:
    path = directory / "result_check.csv"
    if not path.is_file():
        raise SystemExit(f"missing reported-value validation: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 291:
        raise SystemExit(f"expected 291 reported values, found {len(rows)}")
    if any(row["passed"] != "True" for row in rows):
        raise SystemExit(f"one or more reported values failed in {path}")
    maximum = max(float(row["abs_error"]) for row in rows)
    if maximum > 1e-12:
        raise SystemExit(f"maximum error {maximum} exceeds tolerance 1e-12")
    return len(rows), maximum


def validate_result_index() -> tuple[int, int]:
    index_path = ROOT / "results/relcompat3d/manifest.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    artifact_count = 0
    for relative in index["reported_results"].values():
        if not (ROOT / relative).exists():
            raise SystemExit(f"missing indexed artifact: {relative}")
        artifact_count += 1

    manifest_paths = {
        "score_robustness_manifest_sha256": "score_robustness",
        "routing_controls_manifest_sha256": "routing_controls",
        "measurement_analysis_manifest_sha256": "measurement_analysis",
        "component_analysis_manifest_sha256": "component_analysis",
        "seed_robustness_manifest_sha256": "seed_robustness",
        "paper_results_manifest_sha256": "paper_results",
        "candidate_oracle_manifest_sha256": "candidate_oracle",
    }
    verified = 0
    for key, directory in manifest_paths.items():
        path = EXPERIMENT / directory / "evaluation/manifest.json"
        expected = index["analysis_hashes"][key]
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"indexed manifest mismatch: {path.relative_to(ROOT)}")
        verified += 1
    return artifact_count, verified


def validate_compose_output_boundaries() -> int:
    compose_path = ROOT / "configs/relcompat3d/compose.yaml"
    lines = compose_path.read_text(encoding="utf-8").splitlines()
    outputs: list[str] = []
    for index, line in enumerate(lines[:-1]):
        if line.strip() not in {"- --out", "- --output-dir"}:
            continue
        value = lines[index + 1].strip()
        if not value.startswith("- "):
            raise SystemExit(f"invalid Compose --out entry at line {index + 2}")
        outputs.append(value[2:])

    allowed_local = "/workspace/local_dataset/"
    unsafe = [
        output
        for output in outputs
        if allowed_local not in output
        and "/regenerated" not in output
    ]
    if unsafe:
        raise SystemExit(
            "Compose outputs must use local data, local table rows, or regenerated paths: "
            + ", ".join(unsafe)
        )
    return len(outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerated", action="store_true")
    args = parser.parse_args()

    required = [
        ROOT / "configs/relcompat3d/Dockerfile",
        ROOT / "configs/relcompat3d/compose.yaml",
        ROOT / "configs/open3dsg/Dockerfile",
        ROOT / "configs/open3dsg/compose.yaml",
        ROOT / "configs/open3dsg/protocol.json",
        ROOT / "configs/open3dsg/development_scans.txt",
        ROOT / "configs/open3dsg/README.md",
        ROOT / "src/relcompat3d/README.md",
        EXPERIMENT / "README.md",
        ROOT / "results/README.md",
        ROOT / "third_party_licenses.md",
        ROOT / "scripts/train_open3dsg.sh",
        ROOT / "src/relcompat3d/adapt_source_predictions.py",
        ROOT / "src/relcompat3d/build_ground_truth.py",
        ROOT / "src/relcompat3d/build_verification_rows.py",
        ROOT / "src/relcompat3d/create_local_key.py",
        ROOT / "src/relcompat3d/evaluate_predictions.py",
        ROOT / "src/relcompat3d/configure_open3dsg.py",
        ROOT / "src/relcompat3d/prepare_open3dsg_splits.py",
        ROOT / "src/relcompat3d/select_open3dsg_checkpoint.py",
        EXPERIMENT / "method_config.json",
        EXPERIMENT / "main/protocols/point_mesh_analysis.json",
        EXPERIMENT / "paper_results/reproduction_protocol.json",
        ROOT / "results/relcompat3d/manifest.json",
        REFERENCE / "table1.csv",
        REFERENCE / "table2.csv",
        REFERENCE / "table3.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files: " + ", ".join(missing))

    json_count = validate_json()
    cell_count, maximum = validate_cells(REGENERATED if args.regenerated else REFERENCE)
    artifact_count, manifest_count = validate_result_index()
    compose_output_count = validate_compose_output_boundaries()

    print(
        json.dumps(
            {
                "status": "passed",
                "json_files": json_count,
                "reported_values": cell_count,
                "maximum_absolute_error": maximum,
                "indexed_artifacts": artifact_count,
                "verified_analysis_manifests": manifest_count,
                "validated_compose_outputs": compose_output_count,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
