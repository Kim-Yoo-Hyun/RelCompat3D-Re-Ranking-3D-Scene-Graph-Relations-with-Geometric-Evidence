#!/usr/bin/env python3
"""Evaluate direct removals of the pairwise loss and transformation averaging."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

import evaluate_all_families as base
import evaluate_support_bootstrap as scan_bootstrap
import evaluate_base_models as model_eval
import relation_consistency as algebra


METHODS = (
    "source",
    "full_linear",
    "no_pairwise_loss",
    "no_transformation_averaging",
)
REMOVALS = METHODS[2:]
METRICS = ("recall", "violation_all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def make_scorer(
    models: dict[str, Any], attempt: str, average: bool
) -> Callable[[str, str, dict[str, float]], float]:
    family_models = models["attempts"][attempt]

    def direct(family: str, predicate: str, raw: dict[str, float]) -> float:
        return algebra.existing_probability(
            family_models[family], family, predicate, raw
        )

    def score(family: str, predicate: str, raw: dict[str, float]) -> float:
        value = direct(family, predicate, raw)
        transformed = algebra.transformed_view(family, predicate, raw)
        if not average or transformed is None:
            return value
        transformed_predicate, transformed_raw = transformed
        return 0.5 * (
            value + direct(family, transformed_predicate, transformed_raw)
        )

    return score


def load_candidates(
    path: Path,
    scorers: dict[str, Callable[[str, str, dict[str, float]], float]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    digest = hashlib.sha256()
    input_rows = 0
    in_scope_rows = 0
    transformation_errors = {
        method: {
            family: {"rows": 0, "sum": 0.0, "max": 0.0}
            for family in ("proximity", "relative_vertical")
        }
        for method in METHODS[1:]
    }
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            input_rows += 1
            row = json.loads(raw_line)
            family = row["predicate"]["predicate_family"]
            if family not in base.FAMILIES:
                continue
            in_scope_rows += 1
            predicate = row["predicate"]["predicate_label"]
            raw = model_eval.raw_numeric(row)
            semantic = model_eval.finite((row.get("semantic") or {}).get("ranking_score"))
            if semantic is None:
                raise ValueError(f"missing_semantic:{row['prediction_id']}")
            compatibility = {
                method: scorer(family, predicate, raw)
                for method, scorer in scorers.items()
            }
            transformed = algebra.transformed_view(family, predicate, raw)
            if transformed is not None:
                transformed_predicate, transformed_raw = transformed
                for method, scorer in scorers.items():
                    error = abs(
                        compatibility[method]
                        - scorer(family, transformed_predicate, transformed_raw)
                    )
                    cell = transformation_errors[method][family]
                    cell["rows"] += 1
                    cell["sum"] += error
                    cell["max"] = max(cell["max"], error)
            grouped[row["subgraph_id"]].append(
                {
                    "id": row["prediction_id"],
                    "scan": row["scan_id"],
                    "key": model_eval.candidate_key(row),
                    "family": family,
                    "semantic": float(semantic),
                    "status": row.get("verification_status")
                    or (row.get("verification") or {}).get("verification_status"),
                    "compatibility": compatibility,
                    "scores": {"source": float(semantic)},
                }
            )
    diagnostics: dict[str, Any] = {}
    for method, by_family in transformation_errors.items():
        diagnostics[method] = {}
        for family, cell in by_family.items():
            diagnostics[method][family] = {
                "rows": cell["rows"],
                "mean_abs_error": cell["sum"] / cell["rows"] if cell["rows"] else None,
                "max_abs_error": cell["max"] if cell["rows"] else None,
            }
    return grouped, {
        "input_rows": input_rows,
        "in_scope_rows": in_scope_rows,
        "input_sha256": digest.hexdigest(),
        "transformation_error": diagnostics,
    }


def add_family_routes(
    grouped: dict[str, list[dict[str, Any]]]
) -> dict[str, bool]:
    checks = {"family_sequence_exact": True, "support_subsequence_exact": True}
    for candidates in grouped.values():
        source_order = sorted(
            candidates, key=lambda row: (-row["semantic"], row["key"])
        )
        for method in METHODS[1:]:
            queues: dict[str, list[dict[str, Any]]] = {}
            for family in base.FAMILIES:
                rows = [row for row in candidates if row["family"] == family]
                if family == "support_contact":
                    queues[family] = sorted(
                        rows, key=lambda row: (-row["semantic"], row["key"])
                    )
                else:
                    queues[family] = sorted(
                        rows,
                        key=lambda row: (
                            -row["semantic"] * row["compatibility"][method],
                            row["key"],
                        ),
                    )
            offsets = {family: 0 for family in base.FAMILIES}
            routed: list[dict[str, Any]] = []
            for source_row in source_order:
                family = source_row["family"]
                routed.append(queues[family][offsets[family]])
                offsets[family] += 1
            for rank, row in enumerate(routed, 1):
                row["scores"][method] = float(len(routed) - rank + 1)
            for k in base.KS:
                source_top = source_order[:k]
                routed_top = routed[:k]
                checks["family_sequence_exact"] &= [
                    row["family"] for row in source_top
                ] == [row["family"] for row in routed_top]
                checks["support_subsequence_exact"] &= [
                    row["id"]
                    for row in source_top
                    if row["family"] == "support_contact"
                ] == [
                    row["id"]
                    for row in routed_top
                    if row["family"] == "support_contact"
                ]
    return checks


def weighted_ratio(
    numerator: np.ndarray, denominator: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    boot_num = weights @ numerator
    boot_den = weights @ denominator
    return np.divide(
        boot_num,
        boot_den,
        out=np.full(boot_num.shape, np.nan, dtype=np.float64),
        where=boot_den > 0,
    )


def summarize(values: dict[str, Any], weights: np.ndarray) -> dict[str, Any]:
    report: dict[str, Any] = {method: {} for method in METHODS}
    cache: dict[str, Any] = {method: {} for method in METHODS}
    for method in METHODS:
        for ki, k in enumerate(base.KS):
            report[method][str(k)] = {}
            cache[method][str(k)] = {}
            for metric in METRICS:
                numerator, denominator = base.ratio_arrays(
                    values[method], metric, ki
                )
                point = (
                    float(numerator.sum() / denominator.sum())
                    if denominator.sum()
                    else None
                )
                boot = weighted_ratio(numerator, denominator, weights)
                report[method][str(k)][metric] = {
                    "point": point,
                    "bootstrap_intervals_ci95": base.ci95(boot),
                    "numerator": int(numerator.sum()),
                    "denominator": int(denominator.sum()),
                }
                cache[method][str(k)][metric] = boot
    report["removal_minus_full"] = {}
    for method in REMOVALS:
        report["removal_minus_full"][method] = {}
        for k in base.KS:
            report["removal_minus_full"][method][str(k)] = {}
            for metric in METRICS:
                delta = (
                    cache[method][str(k)][metric]
                    - cache["full_linear"][str(k)][metric]
                )
                report["removal_minus_full"][method][str(k)][metric] = {
                    "point": (
                        report[method][str(k)][metric]["point"]
                        - report["full_linear"][str(k)][metric]["point"]
                    ),
                    "paired_bootstrap_intervals_ci95": base.ci95(delta),
                }
    return report


def evaluate_source(
    path: Path,
    gt_path: Path,
    contexts: list[str],
    scorers: dict[str, Callable[[str, str, dict[str, float]], float]],
    resamples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, bool]]:
    grouped, counts = load_candidates(path, scorers)
    route_checks = add_family_routes(grouped)
    gt, gt_family = model_eval.load_gt(gt_path)
    previous_methods = base.METHODS
    base.METHODS = METHODS
    try:
        overall, _, _ = base.contributions(grouped, gt, gt_family, contexts)
    finally:
        base.METHODS = previous_methods
    weights, cluster_counts = scan_bootstrap.scan_weights(
        grouped, contexts, resamples, seed
    )
    return {
        "counts": {
            **counts,
            **cluster_counts,
            "evaluation_contexts": len(contexts),
            "zero_prediction_contexts": len(set(contexts) - set(grouped)),
            "gt_denominator": sum(len(rows) for rows in gt.values()),
        },
        "results": summarize(overall, weights),
    }, route_checks


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
        name: resolve(root, spec["path"])
        for name, spec in protocol["inputs"].items()
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing_inputs:{missing}")
    for name, path in paths.items():
        expected = protocol["inputs"][name]["sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"hash_mismatch:{name}")

    models = json.loads(paths["linear_models"].read_text(encoding="utf-8"))
    scorers = {
        "full_linear": make_scorer(models, "orbit_pairwise", True),
        "no_pairwise_loss": make_scorer(models, "orbit_augmented", True),
        "no_transformation_averaging": make_scorer(
            models, "orbit_pairwise", False
        ),
    }
    annotations = json.loads(
        paths["official_context_annotations"].read_text(encoding="utf-8")
    )
    contexts = sorted(
        {f"{row['scan']}_{row['split']}" for row in annotations["scans"]}
    )
    source_paths = {
        source: paths[f"{source}_verification"]
        for source in ("vlsat", "open3dsg", "sgfn")
    }
    resamples = int(protocol["evaluation"]["bootstrap_resamples"])
    seed = int(protocol["evaluation"]["bootstrap_seed"])
    sources: dict[str, Any] = {}
    route_checks: dict[str, Any] = {}
    for index, (source, path) in enumerate(source_paths.items()):
        sources[source], route_checks[source] = evaluate_source(
            path,
            paths["ground_truth"],
            contexts,
            scorers,
            resamples,
            seed + index,
        )

    reference = json.loads(paths["main_reference"].read_text(encoding="utf-8"))
    full_matches_reference = True
    for source in source_paths:
        for k in base.KS:
            for metric in METRICS:
                expected = reference["sources"][source]["results"]["relcompat3d_linear"][
                    str(k)
                ][metric]["point"]
                actual = sources[source]["results"]["full_linear"][str(k)][metric][
                    "point"
                ]
                full_matches_reference &= abs(actual - expected) <= 1e-12
    projected_errors_zero = all(
        sources[source]["counts"]["transformation_error"][method][family][
            "max_abs_error"
        ]
        <= 1e-12
        for source in source_paths
        for method in ("full_linear", "no_pairwise_loss")
        for family in ("proximity", "relative_vertical")
    )
    no_average_errors_observed = any(
        sources[source]["counts"]["transformation_error"][
            "no_transformation_averaging"
        ][family]["max_abs_error"]
        > 1e-12
        for source in source_paths
        for family in ("proximity", "relative_vertical")
    )
    validations = {
        "all_input_hashes_match": True,
        "official_context_universe_548": len(contexts) == 548,
        "all_sources_157_scans": all(
            payload["counts"]["scans"] == 157 for payload in sources.values()
        ),
        "all_gt_denominators_3972": all(
            payload["counts"]["gt_denominator"] == 3972
            for payload in sources.values()
        ),
        "open3dsg_missing_predictions_are_empty": sources["open3dsg"]["counts"][
            "zero_prediction_contexts"
        ]
        == 15,
        "linear_results_match_reported_results": full_matches_reference,
        "family_sequence_exact": all(
            cell["family_sequence_exact"] for cell in route_checks.values()
        ),
        "support_subsequence_exact": all(
            cell["support_subsequence_exact"] for cell in route_checks.values()
        ),
        "projected_conditions_have_exact_transformation_consistency": projected_errors_zero,
        "no_averaging_condition_exposes_nonzero_transformation_error": no_average_errors_observed,
    }
    status = "completed" if all(validations.values()) else "failed_validation"
    summary = {
        "schema_version": "relcompat3d_component_removal_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "conditions": {
            "full_linear": "BCE, relation-preserving augmentation, linked pairwise loss, and transformation averaging",
            "no_pairwise_loss": "the same Linear estimator without the linked pairwise term; relation-preserving augmentation and transformation averaging remain",
            "no_transformation_averaging": "the same fitted full Linear estimator scored directly without inference-time transformation averaging",
        },
        "bootstrap_unit": "scan_id cluster",
        "bootstrap_resamples": resamples,
        "sources": sources,
        "route_checks": route_checks,
        "validations": validations,
        "evaluation_scope": protocol["evaluation_scope"],
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary)
    rows: list[dict[str, Any]] = []
    for source, payload in sources.items():
        for method in METHODS:
            for k in base.KS:
                cell = payload["results"][method][str(k)]
                rows.append(
                    {
                        "source": source,
                        "condition": method,
                        "k": k,
                        "recall": cell["recall"]["point"],
                        "violation": cell["violation_all"]["point"],
                    }
                )
    with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Direct Component Removals",
        "",
        f"Status: `{status}`",
        "",
        "All conditions use RelCompat3D-Linear, the same fixed train rows and family-slot route.",
        "",
        "| Source | Condition | R@50 | V@50 | R@100 | V@100 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for source, payload in sources.items():
        for method in METHODS:
            k50 = payload["results"][method]["50"]
            k100 = payload["results"][method]["100"]
            lines.append(
                f"| {source} | {method} | {k50['recall']['point']:.4f} | "
                f"{k50['violation_all']['point']:.4f} | "
                f"{k100['recall']['point']:.4f} | "
                f"{k100['violation_all']['point']:.4f} |"
            )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = ("summary.json", "summary.md", "metrics.csv")
    write_json(
        out / "manifest.json",
        {
            "schema_version": "relcompat3d_component_removal_manifest_v1",
            "status": status,
            "protocol": {
                "path": str(protocol_path.relative_to(root)),
                "sha256": sha256_file(protocol_path),
            },
            "inputs": {
                name: {
                    "path": str(path.relative_to(root)),
                    "sha256": sha256_file(path),
                }
                for name, path in paths.items()
            },
            "outputs": {
                name: sha256_file(out / name) for name in outputs
            },
            "validations": validations,
            "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm relcompat3d_component_removals",
        },
    )
    print(json.dumps({"status": status, "validations": validations}))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
