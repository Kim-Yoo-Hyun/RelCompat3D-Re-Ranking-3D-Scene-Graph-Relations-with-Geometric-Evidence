#!/usr/bin/env python3
"""Build a hash-verified construct-dependence evidence package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def relpath(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty_csv:{path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def point_mesh_claim(path: Path) -> dict[str, Any]:
    rows = [
        row for row in read_csv(path)
        if row["audit"] == "consensus"
        and row["method"] == "relcompat3d"
    ]
    deltas = [float(row["delta_violation"]) for row in rows]
    upper = [float(row["delta_violation_ci_high"]) for row in rows]
    return {
        "cells": len(rows),
        "negative_delta_cells": sum(value < 0.0 for value in deltas),
        "tied_delta_cells": sum(value == 0.0 for value in deltas),
        "positive_delta_cells": sum(value > 0.0 for value in deltas),
        "paired_ci_strictly_below_zero_cells": sum(value < 0.0 for value in upper),
        "maximum_delta": max(deltas),
        "minimum_delta": min(deltas),
    }


def uncertainty_claim(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    comparisons: list[dict[str, Any]] = []
    for source in ("vlsat", "open3dsg", "sgfn"):
        for estimator, method in (
            ("linear", "linear__identity"),
            ("mlp", "mlp__identity"),
        ):
            for k in (5, 10, 20, 50, 100):
                report = payload["sources"][source]["results"]
                source_row = report["source"][str(k)]
                method_row = report[method][str(k)]
                comparisons.append(
                    {
                        "source": source,
                        "estimator": estimator,
                        "k": k,
                        "primary_delta": (
                            method_row["violation_all"]["point"]
                            - source_row["violation_all"]["point"]
                        ),
                        "decidable_delta": (
                            method_row["violation_decidable"]["point"]
                            - source_row["violation_decidable"]["point"]
                        ),
                        "pessimistic_delta": (
                            (
                                method_row["violation_all"]["point"]
                                + method_row["uncertainty_rate"]["point"]
                            )
                            - (
                                source_row["violation_all"]["point"]
                                + source_row["uncertainty_rate"]["point"]
                            )
                        ),
                    }
                )
    return {
        "cells": len(comparisons),
        "primary_nonincreasing_cells": sum(
            row["primary_delta"] <= 0.0 for row in comparisons
        ),
        "decidable_nonincreasing_cells": sum(
            row["decidable_delta"] <= 0.0 for row in comparisons
        ),
        "pessimistic_nonincreasing_cells": sum(
            row["pessimistic_delta"] <= 0.0 for row in comparisons
        ),
        "comparisons": comparisons,
    }


def feature_removal_claim(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    conditions = sorted({row["method"] for row in rows})
    expected = {
        "source",
        "main_route",
        "exact_scalar_held_out",
        "primitive_family_held_out",
        "alternative_evidence_only",
    }
    return {
        "rows": len(rows),
        "conditions": conditions,
        "expected_conditions_present": expected <= set(conditions),
        "sources": sorted({row["source"] for row in rows}),
        "ks": sorted({int(row["k"]) for row in rows}),
    }


def component_claim(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    return {
        "rows": len(rows),
        "conditions": sorted({row["condition"] for row in rows}),
        "sources": sorted({row["source"] for row in rows}),
        "ks": sorted({int(row["k"]) for row in rows}),
    }


def counterfactual_claim(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    return {
        "rows": len(rows),
        "conditions": sorted({row["condition"] for row in rows}),
        "sources": sorted({row["source"] for row in rows}),
        "ks": sorted({int(row["k"]) for row in rows}),
    }


def markdown(summary: dict[str, Any]) -> str:
    claims = summary["claims"]
    linear = claims["point_mesh_linear"]
    mlp = claims["point_mesh_mlp"]
    uncertainty = claims["uncertainty_policy"]
    lines = [
        "# Construct-Dependence Evidence Package",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This package does not create independent physical-validity ground truth. "
        "It records exactly which information is shared across training-target "
        "construction, the primary OBB verifier, and the point/mesh audit, then "
        "hash-verifies the compact analyses used to probe that dependence.",
        "",
        "## Verified evidence",
        "",
        f"- Linear point/mesh agreement: {linear['negative_delta_cells']}/"
        f"{linear['cells']} cells have a negative Violation change.",
        f"- MLP point/mesh agreement: {mlp['negative_delta_cells']}/"
        f"{mlp['cells']} cells have a negative Violation change.",
        f"- Uncertainty-policy check: primary/decidable/pessimistic Violation is "
        f"non-increasing in {uncertainty['primary_nonincreasing_cells']}/"
        f"{uncertainty['cells']}, {uncertainty['decidable_nonincreasing_cells']}/"
        f"{uncertainty['cells']}, and "
        f"{uncertainty['pessimistic_nonincreasing_cells']}/"
        f"{uncertainty['cells']} cells, respectively.",
        "- Feature-removal, counterfactual-sensitivity, and direct component-"
        "removal artifacts are present and hash-locked.",
        "",
        "The dependency matrix and complete evidence index are stored in the "
        "adjacent CSV files.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    protocol_path = resolve(root, args.protocol)
    out = resolve(root, args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"nonempty_output:{out}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "ready":
        raise ValueError("protocol_version_mismatch")
    paths = {
        name: resolve(root, value)
        for name, value in protocol["inputs"].items()
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing_inputs:{missing}")
    for name, expected in protocol["expected_sha256"].items():
        actual = sha256_file(paths[name])
        if actual != expected:
            raise ValueError(
                f"hash_mismatch:{name}:expected={expected}:actual={actual}"
            )

    manifests = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
        if name.endswith("_manifest")
    }
    claims = {
        "feature_removal": feature_removal_claim(
            paths["feature_removal_metrics"]
        ),
        "point_mesh_linear": point_mesh_claim(
            paths["point_mesh_linear_metrics"]
        ),
        "point_mesh_mlp": point_mesh_claim(
            paths["point_mesh_mlp_metrics"]
        ),
        "uncertainty_policy": uncertainty_claim(
            paths["score_robustness_summary"]
        ),
        "component_removals": component_claim(
            paths["component_removal_metrics"]
        ),
        "counterfactual_sensitivity": counterfactual_claim(
            paths["counterfactual_metrics"]
        ),
    }
    dependency_rows = protocol["dependency_matrix"]
    evidence_rows = protocol["evidence_index"]
    validations = {
        "all_input_hashes_match": True,
        "all_manifests_completed": all(
            manifest.get("status") == "completed"
            for manifest in manifests.values()
        ),
        "feature_removal_conditions_present": claims["feature_removal"][
            "expected_conditions_present"
        ],
        "linear_point_mesh_all_cells_nonpositive": (
            claims["point_mesh_linear"]["positive_delta_cells"] == 0
        ),
        "mlp_point_mesh_all_cells_nonpositive": (
            claims["point_mesh_mlp"]["positive_delta_cells"] == 0
        ),
        "all_uncertainty_policies_nonincreasing": all(
            claims["uncertainty_policy"][name]
            == claims["uncertainty_policy"]["cells"]
            for name in (
                "primary_nonincreasing_cells",
                "decidable_nonincreasing_cells",
                "pessimistic_nonincreasing_cells",
            )
        ),
        "dependency_matrix_complete": {
            row["information"] for row in dependency_rows
        } == {
            "evaluation_candidate_rows",
            "source_relation_score",
            "primary_verifier_status_labels",
            "obb_derived_measurements",
            "point_mesh_measurements",
            "evaluation_scene_identities",
            "relation_ontology",
        },
    }
    status = "completed" if all(validations.values()) else "failed_validation"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "dependency_matrix.csv", dependency_rows)
    write_csv(out / "evidence_index.csv", evidence_rows)
    write_json(out / "claims.json", claims)
    summary = {
        "schema_version": "relcompat3d_measurement_analysis_package_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "claims": claims,
        "validations": validations,
        "evaluation_scope": protocol["evaluation_scope"],
    }
    write_json(out / "summary.json", summary)
    (out / "summary.md").write_text(markdown(summary), encoding="utf-8")
    outputs = (
        "dependency_matrix.csv",
        "evidence_index.csv",
        "claims.json",
        "summary.json",
        "summary.md",
    )
    write_json(
        out / "manifest.json",
        {
            "schema_version": "relcompat3d_measurement_analysis_manifest_v1",
            "status": status,
            "protocol": {
                "path": relpath(root, protocol_path),
                "sha256": sha256_file(protocol_path),
            },
            "inputs": {
                name: {
                    "path": relpath(root, path),
                    "sha256": sha256_file(path),
                }
                for name, path in paths.items()
            },
            "outputs": {
                name: {
                    "path": relpath(root, out / name),
                    "sha256": sha256_file(out / name),
                }
                for name in outputs
            },
            "validations": validations,
            "docker_command": (
                "env UID=$(id -u) GID=$(id -g) docker compose "
                "-f configs/relcompat3d/compose.yaml run --rm "
                "relcompat3d_measurement_analysis"
            ),
        },
    )
    print(json.dumps({"status": status, "validations": validations}))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
