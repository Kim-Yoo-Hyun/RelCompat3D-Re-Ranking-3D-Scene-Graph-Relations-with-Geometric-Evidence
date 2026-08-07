#!/usr/bin/env python3
"""Scan-cluster bootstrap sensitivity for support/contact applicability routing."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_all_families as base
import evaluate_support_order as routing


METHODS = ("source", "family_slot_rerank")
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


def scan_weights(
    grouped: dict[str, list[dict[str, Any]]],
    contexts: list[str],
    resamples: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    context_to_scan: dict[str, str] = {}
    for context in contexts:
        candidates = grouped.get(context, [])
        scans = {str(row["scan"]) for row in candidates}
        if len(scans) > 1:
            raise ValueError(f"context_maps_to_multiple_scans:{context}")
        context_to_scan[context] = next(iter(scans)) if scans else context.rsplit("_", 1)[0]
    scans = sorted(set(context_to_scan.values()))
    scan_index = {scan: index for index, scan in enumerate(scans)}
    context_scan = np.asarray([scan_index[context_to_scan[c]] for c in contexts])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(scans), size=(resamples, len(scans)))
    counts = np.zeros((resamples, len(scans)), dtype=np.float64)
    for index in range(resamples):
        counts[index] = np.bincount(sampled[index], minlength=len(scans))
    per_scan = Counter(context_to_scan.values())
    return counts[:, context_scan], {
        "scans": len(scans),
        "contexts": len(contexts),
        "min_contexts_per_scan": min(per_scan.values()),
        "max_contexts_per_scan": max(per_scan.values()),
    }


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
    report["family_slot_rerank_minus_source_score"] = {}
    for k in base.KS:
        report["family_slot_rerank_minus_source_score"][str(k)] = {}
        for metric in METRICS:
            left = report["family_slot_rerank"][str(k)][metric]["point"]
            right = report["source"][str(k)][metric]["point"]
            delta = cache["family_slot_rerank"][str(k)][metric] - cache["source"][str(k)][metric]
            report["family_slot_rerank_minus_source_score"][str(k)][metric] = {
                "point": left - right,
                "paired_bootstrap_intervals_ci95": base.ci95(delta),
            }
    return report


def main() -> int:
    args = parse_args()
    root, out = args.repo_root.resolve(), resolve(args.repo_root.resolve(), args.out)
    protocol_path = resolve(root, args.protocol)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"nonempty_output:{out}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "ready":
        raise ValueError("protocol_version_mismatch")
    routing_protocol = json.loads(resolve(root, protocol["routing_protocol"]).read_text(encoding="utf-8"))
    paths = {name: resolve(root, value) for name, value in routing_protocol["inputs"].items()}
    if base.sha256_file(paths["linear_models"]) != protocol["model_sha256"]:
        raise ValueError("linear_model_hash_mismatch")
    models = json.loads(paths["linear_models"].read_text(encoding="utf-8"))
    base_models = json.loads(paths["base_models"].read_text(encoding="utf-8"))
    scorer = base.make_linear_scorer(models)
    gt, gt_family = base.model_eval.load_gt(paths["final_ground_truth"])
    source_paths = {
        "vlsat": paths["vlsat_verification"],
        "open3dsg": resolve(root, protocol["open3dsg_official_verification"]),
        "sgfn": paths["sgfn_verification"],
    }
    routed_summary = json.loads(resolve(root, protocol["routing_summary"]).read_text(encoding="utf-8"))
    open_summary = json.loads(resolve(root, protocol["open3dsg_official_summary"]).read_text(encoding="utf-8"))
    annotations = json.loads(resolve(root, protocol["official_context_annotations"]).read_text(encoding="utf-8"))
    official_contexts = {f"{row['scan']}_{row['split']}" for row in annotations["scans"]}
    sources: dict[str, Any] = {}
    validations = {
        "model_hash_matches": True,
        "all_sources_have_157_scans": True,
        "all_sources_have_548_contexts": True,
        "point_estimates_match_routing_summary": True,
        "k100_recall_ci_positive": True,
        "k100_violation_ci_below_zero": True,
    }
    original_methods = base.METHODS
    base.METHODS = METHODS
    try:
        for index, (source, path) in enumerate(source_paths.items()):
            grouped, _ = base.load_candidates(path, scorer, base_models)
            routing.add_routing_scores(grouped)
            contexts = sorted(official_contexts if source == "open3dsg" else set(grouped) | set(gt))
            overall, _, _ = base.contributions(grouped, gt, gt_family, contexts)
            weights, counts = scan_weights(
                grouped,
                contexts,
                int(protocol["bootstrap_resamples"]),
                int(protocol["bootstrap_seed"]) + index,
            )
            results = summarize(overall, weights)
            sources[source] = {"counts": counts, "results": results}
            validations["all_sources_have_157_scans"] &= counts["scans"] == 157
            validations["all_sources_have_548_contexts"] &= counts["contexts"] == 548
            for method in METHODS:
                for k in base.KS:
                    for metric in METRICS:
                        if source == "open3dsg":
                            expected = open_summary["routes"]["official_full_548"]["overall"][method][str(k)][metric]["point"]
                        else:
                            expected = routed_summary["sources"][source]["overall"][method][str(k)][metric]["point"]
                        actual = results[method][str(k)][metric]["point"]
                        validations["point_estimates_match_routing_summary"] &= abs(actual - expected) <= 1e-12
            k100 = results["family_slot_rerank_minus_source_score"]["100"]
            validations["k100_recall_ci_positive"] &= k100["recall"]["paired_bootstrap_intervals_ci95"][0] > 0.0
            validations["k100_violation_ci_below_zero"] &= k100["violation_all"]["paired_bootstrap_intervals_ci95"][1] < 0.0
    finally:
        base.METHODS = original_methods
    status = "completed" if all(validations.values()) else "failed_validation"
    payload = {
        "schema_version": "relcompat3d_relcompat3d_linear_bootstrap_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "bootstrap_unit": "scan_id cluster",
        "bootstrap_resamples": int(protocol["bootstrap_resamples"]),
        "sources": sources,
        "validations": validations,
        "evaluation_scope": "dependence sensitivity for the development-selected applicability route",
    }
    out.mkdir(parents=True, exist_ok=True)
    base.write_json(out / "summary.json", payload)
    lines = [
        "# Applicability-Routing Scan-Cluster Sensitivity", "", f"Status: `{status}`", "",
        "| Source | delta Recall@100 (95% CI) | delta V@100 (95% CI) |", "| --- | ---: | ---: |",
    ]
    for source in ("vlsat", "open3dsg", "sgfn"):
        cell = sources[source]["results"]["family_slot_rerank_minus_source_score"]["100"]
        dr, dv = cell["recall"], cell["violation_all"]
        lines.append(
            f"| {source} | {dr['point']:+.4f} [{dr['paired_bootstrap_intervals_ci95'][0]:+.4f}, {dr['paired_bootstrap_intervals_ci95'][1]:+.4f}] | "
            f"{dv['point']:+.4f} [{dv['paired_bootstrap_intervals_ci95'][0]:+.4f}, {dv['paired_bootstrap_intervals_ci95'][1]:+.4f}] |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    base.write_json(out / "manifest.json", {
        "schema_version": "relcompat3d_relcompat3d_linear_bootstrap_manifest_v1",
        "status": status,
        "protocol": {"path": base.relpath(root, protocol_path), "sha256": base.sha256_file(protocol_path)},
        "outputs": {name: base.sha256_file(out / name) for name in ("summary.json", "summary.md")},
        "validations": validations,
        "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm relcompat3d_linear_bootstrap",
    })
    print(json.dumps({"status": status, "validations": validations}))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
