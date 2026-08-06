#!/usr/bin/env python3
"""Evaluate restored RelCompat3D models on locally prepared rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evaluate_comparators as comparators
import evaluate_main as linear
import evaluate_train_only as strict
import fit_mlp as nonlinear


SOURCES = ("vlsat", "open3dsg", "sgfn")
METHOD_LABELS = {
    "source_score": "Source",
    "routed_product": "RelCompat3D-Linear",
    "routed_rank_average": "RankAvg",
    "routed_rrf": "RRF",
    "routed_matched_mlp": "RelCompat3D-MLP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--structured-models", type=Path, required=True)
    parser.add_argument("--nonlinear-models", type=Path, required=True)
    parser.add_argument("--vlsat", type=Path, required=True)
    parser.add_argument("--open3dsg", type=Path, required=True)
    parser.add_argument("--sgfn", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    inputs = {
        "annotations": args.annotations,
        "ground_truth": args.ground_truth,
        "structured_models": args.structured_models,
        "nonlinear_models": args.nonlinear_models,
        "vlsat": args.vlsat,
        "open3dsg": args.open3dsg,
        "sgfn": args.sgfn,
    }
    missing = [name for name, path in inputs.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"nonempty output: {args.out}")

    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    contexts = sorted({f"{row['scan']}_{row['split']}" for row in annotations["scans"]})
    gt, _ = strict.load_gt(args.ground_truth)
    structured = json.loads(args.structured_models.read_text(encoding="utf-8"))
    nonlinear_models = json.loads(args.nonlinear_models.read_text(encoding="utf-8"))
    scorer = linear.make_structured_scorer(structured)
    bce_model = nonlinear_models["shared_nonlinear_bce"]
    matched_model = nonlinear_models["shared_nonlinear_structured"]

    source_paths = {source: getattr(args, source) for source in SOURCES}
    sources: dict[str, Any] = {}
    route_checks: dict[str, Any] = {}
    for index, (source, path) in enumerate(source_paths.items()):
        sources[source], route_checks[source] = comparators.evaluate_source(
            path,
            args.ground_truth,
            contexts,
            scorer,
            bce_model,
            matched_model,
            args.bootstrap_resamples,
            args.bootstrap_seed + index,
        )

    validations = {
        "contexts_match_annotations": all(
            payload["counts"]["evaluation_contexts"] == len(contexts)
            for payload in sources.values()
        ),
        "ground_truth_denominator_matches": all(
            payload["counts"]["gt_denominator"] == sum(len(rows) for rows in gt.values())
            for payload in sources.values()
        ),
        "family_composition_preserved": all(
            value["family_composition_exact"] for value in route_checks.values()
        ),
        "support_contact_selection_preserved": all(
            value["support_selection_exact"] for value in route_checks.values()
        ),
        "models_exclude_source_inputs": (
            structured.get("source_score_used") is False
            and structured.get("source_identity_used") is False
            and not matched_model["feature_contract"]["source_score_input"]
            and not matched_model["feature_contract"]["source_identity_input"]
        ),
    }
    status = "completed" if all(validations.values()) else "failed_validation"
    summary = {
        "schema_version": "relcompat3d_public_evaluation_v1",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contexts": len(contexts),
        "ground_truth_denominator": sum(len(rows) for rows in gt.values()),
        "bootstrap": {
            "unit": "scan",
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
        },
        "methods": METHOD_LABELS,
        "sources": sources,
        "route_checks": route_checks,
        "validations": validations,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.json"
    write_json(summary_path, summary)

    rows: list[dict[str, Any]] = []
    for source, payload in sources.items():
        for method, label in METHOD_LABELS.items():
            for k in linear.KS:
                cell = payload["results"][method][str(k)]
                rows.append(
                    {
                        "predictor": source,
                        "method": label,
                        "k": k,
                        "recall": cell["recall"]["point"],
                        "recall_ci_low": cell["recall"]["scan_cluster_ci95"][0],
                        "recall_ci_high": cell["recall"]["scan_cluster_ci95"][1],
                        "violation": cell["violation_all"]["point"],
                        "violation_ci_low": cell["violation_all"]["scan_cluster_ci95"][0],
                        "violation_ci_high": cell["violation_all"]["scan_cluster_ci95"][1],
                    }
                )
    metrics_path = args.out / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema_version": "relcompat3d_public_evaluation_manifest_v1",
        "status": status,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "outputs": {
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
            "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
        },
        "validations": validations,
    }
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps({"status": status, "validations": validations}, sort_keys=True))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
