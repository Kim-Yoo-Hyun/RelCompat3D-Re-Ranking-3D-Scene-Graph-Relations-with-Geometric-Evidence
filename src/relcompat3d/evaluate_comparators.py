#!/usr/bin/env python3
"""Evaluate fusion comparators under the same RelCompat3D family-slot route."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_all_families as base
import fit_mlp as nonlinear
import evaluate_support_bootstrap as scan_bootstrap
import evaluate_base_models as model_eval


METHODS = (
    "source",
    "relcompat3d_linear",
    "rank_average",
    "rrf",
    "relcompat3d_mlp",
)
ROUTE_INPUTS = {
    "relcompat3d_linear": "all_family_product",
    "rank_average": "rank_average_all_families",
    "rrf": "rrf_all_families",
    "relcompat3d_mlp": "shared_mlp_pairwise_product",
}
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


def add_rank_scores(grouped: dict[str, list[dict[str, Any]]]) -> None:
    for candidates in grouped.values():
        denominator = max(len(candidates) - 1, 1)
        source_order = sorted(candidates, key=lambda row: (-row["semantic"], row["key"]))
        geometry_order = sorted(candidates, key=lambda row: (-row["linear"], row["key"]))
        source_rank = {row["id"]: rank for rank, row in enumerate(source_order, 1)}
        geometry_rank = {row["id"]: rank for rank, row in enumerate(geometry_order, 1)}
        for row in candidates:
            rz, rc = source_rank[row["id"]], geometry_rank[row["id"]]
            qz = 1.0 - (rz - 1) / denominator
            qc = 1.0 - (rc - 1) / denominator
            row["scores"]["rank_average_all_families"] = 0.5 * (qz + qc)
            row["scores"]["rrf_all_families"] = 1.0 / (60 + rz) + 1.0 / (60 + rc)


def add_family_slot_routes(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, bool]:
    checks = {"family_composition_exact": True, "support_selection_exact": True}
    for candidates in grouped.values():
        source_order = sorted(candidates, key=lambda row: (-row["semantic"], row["key"]))
        for method, score_name in ROUTE_INPUTS.items():
            queues: dict[str, list[dict[str, Any]]] = {}
            for family in base.FAMILIES:
                rows = [row for row in candidates if row["family"] == family]
                family_score = "source" if family == "support_contact" else score_name
                queues[family] = sorted(rows, key=lambda row: (-row["scores"][family_score], row["key"]))
            offsets = {family: 0 for family in base.FAMILIES}
            ranked: list[dict[str, Any]] = []
            for source_row in source_order:
                family = source_row["family"]
                ranked.append(queues[family][offsets[family]])
                offsets[family] += 1
            n = len(ranked)
            for rank, row in enumerate(ranked, 1):
                row["scores"][method] = float(n - rank + 1)
            for k in base.KS:
                source_top = source_order[:k]
                routed_top = ranked[:k]
                checks["family_composition_exact"] &= (
                    [row["family"] for row in source_top]
                    == [row["family"] for row in routed_top]
                )
                checks["support_selection_exact"] &= (
                    [row["id"] for row in source_top if row["family"] == "support_contact"]
                    == [row["id"] for row in routed_top if row["family"] == "support_contact"]
                )
    return checks


def weighted_ratio(numerator: np.ndarray, denominator: np.ndarray, weights: np.ndarray) -> np.ndarray:
    boot_num, boot_den = weights @ numerator, weights @ denominator
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
            report[method][str(k)], cache[method][str(k)] = {}, {}
            for metric in METRICS:
                numerator, denominator = base.ratio_arrays(values[method], metric, ki)
                point = float(numerator.sum() / denominator.sum()) if denominator.sum() else None
                boot = weighted_ratio(numerator, denominator, weights)
                report[method][str(k)][metric] = {
                    "point": point,
                    "bootstrap_intervals_ci95": base.ci95(boot),
                    "numerator": int(numerator.sum()),
                    "denominator": int(denominator.sum()),
                }
                cache[method][str(k)][metric] = boot
    report["deltas_vs_source_score"] = {}
    for method in ROUTE_INPUTS:
        report["deltas_vs_source_score"][method] = {}
        for k in base.KS:
            report["deltas_vs_source_score"][method][str(k)] = {}
            for metric in METRICS:
                delta = cache[method][str(k)][metric] - cache["source"][str(k)][metric]
                report["deltas_vs_source_score"][method][str(k)][metric] = {
                    "point": report[method][str(k)][metric]["point"]
                    - report["source"][str(k)][metric]["point"],
                    "paired_bootstrap_intervals_ci95": base.ci95(delta),
                }
    return report


def evaluate_source(
    path: Path,
    gt_path: Path,
    contexts: list[str],
    scorer: Any,
    bce_model: dict[str, Any],
    matched_model: dict[str, Any],
    resamples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, bool]]:
    grouped, counts = nonlinear.load_candidates(path, scorer, bce_model, matched_model)
    add_rank_scores(grouped)
    ranking_checks = add_family_slot_routes(grouped)
    gt, gt_family = model_eval.load_gt(gt_path)
    old_methods = base.METHODS
    base.METHODS = METHODS
    try:
        overall, _, _ = base.contributions(grouped, gt, gt_family, contexts)
    finally:
        base.METHODS = old_methods
    weights, cluster_counts = scan_bootstrap.scan_weights(grouped, contexts, resamples, seed)
    return {
        "counts": {
            **counts,
            **cluster_counts,
            "evaluation_contexts": len(contexts),
            "zero_prediction_contexts": len(set(contexts) - set(grouped)),
            "gt_denominator": sum(len(rows) for rows in gt.values()),
        },
        "results": summarize(overall, weights),
    }, ranking_checks


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    protocol_path, out = resolve(root, args.protocol), resolve(root, args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"nonempty_output:{out}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "ready":
        raise ValueError("protocol_version_mismatch")
    paths = {name: resolve(root, value) for name, value in protocol["inputs"].items()}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing_inputs:{missing}")
    for name, expected in protocol["expected_sha256"].items():
        if base.sha256_file(paths[name]) != expected:
            raise ValueError(f"hash_mismatch:{name}")

    linear_models = json.loads(paths["linear_models"].read_text(encoding="utf-8"))
    nonlinear_models = json.loads(paths["nonlinear_models"].read_text(encoding="utf-8"))
    scorer = base.make_linear_scorer(linear_models)
    bce_model = nonlinear_models["shared_mlp_bce"]
    matched_model = nonlinear_models["shared_mlp_pairwise"]
    annotations = json.loads(paths["official_context_annotations"].read_text(encoding="utf-8"))
    contexts = sorted({f"{row['scan']}_{row['split']}" for row in annotations["scans"]})
    source_paths = {
        "vlsat": paths["vlsat_verification"],
        "open3dsg": paths["open3dsg_verification"],
        "sgfn": paths["sgfn_verification"],
    }
    resamples = int(protocol["evaluation"]["bootstrap_resamples"])
    seed = int(protocol["evaluation"]["bootstrap_seed"])
    sources: dict[str, Any] = {}
    ranking_checks: dict[str, Any] = {}
    for index, (source, path) in enumerate(source_paths.items()):
        sources[source], ranking_checks[source] = evaluate_source(
            path,
            paths["ground_truth"],
            contexts,
            scorer,
            bce_model,
            matched_model,
            resamples,
            seed + index,
        )

    routing_reference = json.loads(paths["routing_reference"].read_text(encoding="utf-8"))
    open_reference = json.loads(paths["open3dsg_reference"].read_text(encoding="utf-8"))
    point_match = True
    for source in source_paths:
        for k in base.KS:
            for metric in METRICS:
                if source == "open3dsg":
                    expected = open_reference["routes"]["official_full_548"]["overall"]["family_slot_rerank"][str(k)][metric]["point"]
                else:
                    expected = routing_reference["sources"][source]["overall"]["family_slot_rerank"][str(k)][metric]["point"]
                actual = sources[source]["results"]["relcompat3d_linear"][str(k)][metric]["point"]
                point_match &= abs(actual - expected) <= 1e-12
    validations = {
        "locked_model_hashes_match": True,
        "official_context_universe_548": len(contexts) == 548,
        "all_sources_157_scans_and_548_contexts": all(
            payload["counts"]["scans"] == 157 and payload["counts"]["evaluation_contexts"] == 548
            for payload in sources.values()
        ),
        "all_gt_denominators_3972": all(payload["counts"]["gt_denominator"] == 3972 for payload in sources.values()),
        "open3dsg_missing_predictions_are_empty": sources["open3dsg"]["counts"]["zero_prediction_contexts"] == 15,
        "relcompat3d_linear_matches_reported_results": point_match,
        "family_composition_exact_for_all_comparators": all(cell["family_composition_exact"] for cell in ranking_checks.values()),
        "support_contact_selection_exact_for_all_comparators": all(cell["support_selection_exact"] for cell in ranking_checks.values()),
        "matched_mlp_excludes_source_score_and_identity": (
            not matched_model["feature_spec"]["source_score_input"]
            and not matched_model["feature_spec"]["source_identity_input"]
        ),
    }
    status = "completed" if all(validations.values()) else "failed_validation"
    summary = {
        "schema_version": "relcompat3d_ranking_baselines_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "methods": {
            "relcompat3d_linear": "family-slot routing with the projected compatibility product",
            "rank_average": "the same route with within-context source/compatibility percentile averaging",
            "rrf": "the same route with reciprocal-rank fusion (c=60)",
            "relcompat3d_mlp": "the same ranking rule with the fitted MLP compatibility model",
        },
        "bootstrap_unit": "scan_id cluster",
        "bootstrap_resamples": resamples,
        "sources": sources,
        "ranking_checks": ranking_checks,
        "validations": validations,
        "evaluation_scope": protocol["evaluation_scope"],
    }
    out.mkdir(parents=True, exist_ok=True)
    base.write_json(out / "summary.json", summary)
    rows: list[dict[str, Any]] = []
    for source, payload in sources.items():
        for method in METHODS:
            for k in base.KS:
                cell = payload["results"][method][str(k)]
                rows.append({
                    "source": source,
                    "method": method,
                    "k": k,
                    "recall": cell["recall"]["point"],
                    "recall_ci_low": cell["recall"]["bootstrap_intervals_ci95"][0],
                    "recall_ci_high": cell["recall"]["bootstrap_intervals_ci95"][1],
                    "violation": cell["violation_all"]["point"],
                    "violation_ci_low": cell["violation_all"]["bootstrap_intervals_ci95"][0],
                    "violation_ci_high": cell["violation_all"]["bootstrap_intervals_ci95"][1],
                })
    with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Same-Route Fusion Comparator Evaluation", "", f"Status: `{status}`", "",
        "All methods use the same family-slot composition, support/contact pass-through, official 548-context universe, and scan-cluster resampling.", "",
        "| Source | Method | R@50 | V@50 | R@100 | V@100 |", "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for source, payload in sources.items():
        for method in METHODS:
            k50, k100 = payload["results"][method]["50"], payload["results"][method]["100"]
            lines.append(
                f"| {source} | {method} | {k50['recall']['point']:.4f} | {k50['violation_all']['point']:.4f} | "
                f"{k100['recall']['point']:.4f} | {k100['violation_all']['point']:.4f} |"
            )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = ("summary.json", "summary.md", "metrics.csv")
    base.write_json(out / "manifest.json", {
        "schema_version": "relcompat3d_ranking_baselines_manifest_v1",
        "status": status,
        "protocol": {"path": base.relpath(root, protocol_path), "sha256": base.sha256_file(protocol_path)},
        "inputs": {name: {"path": base.relpath(root, path), "sha256": base.sha256_file(path)} for name, path in paths.items()},
        "outputs": {name: base.sha256_file(out / name) for name in outputs},
        "validations": validations,
        "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm relcompat3d_compare_rankings",
    })
    print(json.dumps({"status": status, "validations": validations}))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
