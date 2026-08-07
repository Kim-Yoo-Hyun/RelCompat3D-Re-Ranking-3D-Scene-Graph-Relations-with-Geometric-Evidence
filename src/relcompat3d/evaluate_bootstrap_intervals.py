#!/usr/bin/env python3
"""Scan-cluster bootstrap intervals for the reported RelCompat3D rankings."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_all_families as main_eval


METHODS = ("source", "all_family_product")
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


def weighted_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    boot_num = weights @ numerator
    boot_den = weights @ denominator
    return np.divide(
        boot_num,
        boot_den,
        out=np.full(boot_num.shape, np.nan, dtype=np.float64),
        where=boot_den > 0,
    )


def scan_weights(
    grouped: dict[str, list[dict[str, Any]]],
    contexts: list[str],
    n_resamples: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    context_to_scan: dict[str, str] = {}
    for context in contexts:
        candidates = grouped.get(context, [])
        if candidates:
            scans = {str(row["scan"]) for row in candidates}
            if len(scans) != 1:
                raise ValueError(f"context_maps_to_multiple_scans:{context}:{sorted(scans)}")
            context_to_scan[context] = next(iter(scans))
        else:
            context_to_scan[context] = context.rsplit("_", 1)[0]

    scans = sorted(set(context_to_scan.values()))
    scan_index = {scan: index for index, scan in enumerate(scans)}
    context_scan_index = np.asarray(
        [scan_index[context_to_scan[context]] for context in contexts],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    sampled_scans = rng.integers(
        0,
        len(scans),
        size=(n_resamples, len(scans)),
    )
    counts = np.zeros((n_resamples, len(scans)), dtype=np.float64)
    for bootstrap_index in range(n_resamples):
        counts[bootstrap_index] = np.bincount(
            sampled_scans[bootstrap_index], minlength=len(scans)
        )
    weights = counts[:, context_scan_index]
    contexts_per_scan = Counter(context_to_scan.values())
    return weights, {
        "scans": len(scans),
        "contexts": len(contexts),
        "min_contexts_per_scan": min(contexts_per_scan.values()),
        "max_contexts_per_scan": max(contexts_per_scan.values()),
        "mean_contexts_per_scan": len(contexts) / len(scans),
    }


def summarize(
    values: dict[str, dict[str, np.ndarray]],
    weights: np.ndarray,
) -> dict[str, Any]:
    report: dict[str, Any] = {method: {} for method in METHODS}
    cache: dict[str, dict[str, dict[str, np.ndarray]]] = {
        method: {} for method in METHODS
    }
    for method in METHODS:
        for ki, k in enumerate(main_eval.KS):
            report[method][str(k)] = {}
            cache[method][str(k)] = {}
            for metric in METRICS:
                numerator, denominator = main_eval.ratio_arrays(
                    values[method], metric, ki
                )
                point = (
                    float(numerator.sum() / denominator.sum())
                    if denominator.sum()
                    else None
                )
                samples = weighted_ratio(numerator, denominator, weights)
                report[method][str(k)][metric] = {
                    "point": point,
                    "bootstrap_intervals_ci95": main_eval.ci95(samples),
                    "numerator": int(numerator.sum()),
                    "denominator": int(denominator.sum()),
                }
                cache[method][str(k)][metric] = samples

    report["all_family_product_minus_source_score"] = {}
    for k in main_eval.KS:
        report["all_family_product_minus_source_score"][str(k)] = {}
        for metric in METRICS:
            left = report["all_family_product"][str(k)][metric]["point"]
            right = report["source"][str(k)][metric]["point"]
            delta = (
                cache["all_family_product"][str(k)][metric]
                - cache["source"][str(k)][metric]
            )
            report["all_family_product_minus_source_score"][str(k)][metric] = {
                "point": left - right,
                "paired_bootstrap_intervals_ci95": main_eval.ci95(delta),
            }
    return report


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Scan-Cluster Bootstrap Sensitivity",
        "",
        f"Status: `{payload['status']}`",
        "",
        "The reported rankings and point estimates are unchanged. This analysis resamples 157 scans with replacement and keeps the relation contexts of each sampled scan together.",
        "",
        "| Source | dRecall@100 (95% scan-cluster CI) | dVerifier-V@100 (95% scan-cluster CI) |",
        "| --- | ---: | ---: |",
    ]
    for source in ("vlsat", "open3dsg", "sgfn"):
        cell = payload["sources"][source]["results"][
            "all_family_product_minus_source_score"
        ]["100"]
        dr = cell["recall"]
        dv = cell["violation_all"]
        lines.append(
            f"| {main_eval.SOURCE_LABELS[source]} | {dr['point']:+.4f} "
            f"[{dr['paired_bootstrap_intervals_ci95'][0]:+.4f}, {dr['paired_bootstrap_intervals_ci95'][1]:+.4f}] | "
            f"{dv['point']:+.4f} [{dv['paired_bootstrap_intervals_ci95'][0]:+.4f}, "
            f"{dv['paired_bootstrap_intervals_ci95'][1]:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "At K=100, no Recall interval crosses below zero (the VL-SAT lower bound reaches zero), and all verifier-V intervals remain below zero. This is a dependence sensitivity, not a new score-selection result.",
            "",
        ]
    )
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

    main_protocol_path = resolve(root, protocol["main_protocol"])
    main_protocol = json.loads(main_protocol_path.read_text(encoding="utf-8"))
    paths = {
        name: resolve(root, value) for name, value in main_protocol["inputs"].items()
    }
    linear_models = json.loads(paths["linear_models"].read_text(encoding="utf-8"))
    if main_eval.sha256_file(paths["linear_models"]) != protocol["model_sha256"]:
        raise ValueError("linear_model_hash_mismatch")
    base_models = json.loads(paths["base_models"].read_text(encoding="utf-8"))
    scorer = main_eval.make_linear_scorer(linear_models)
    gt, gt_family = main_eval.model_eval.load_gt(paths["ground_truth"])
    source_paths = {
        "vlsat": paths["vlsat_verification"],
        "open3dsg": paths["open3dsg_verification"],
        "sgfn": paths["sgfn_verification"],
    }
    current_summary = json.loads(
        resolve(root, protocol["current_summary"]).read_text(encoding="utf-8")
    )

    sources: dict[str, Any] = {}
    validations: dict[str, bool] = {
        "model_hash_matches": True,
        "all_sources_have_157_scans": True,
        "all_sources_have_548_contexts": True,
        "point_estimates_match_reported_results": True,
        "k100_recall_ci_nonnegative": True,
        "k100_violation_ci_below_zero": True,
    }
    for source_index, (source, path) in enumerate(source_paths.items()):
        grouped, _ = main_eval.load_candidates(path, scorer, base_models)
        contexts = sorted(set(grouped) | set(gt))
        overall, _, _ = main_eval.contributions(grouped, gt, gt_family, contexts)
        weights, counts = scan_weights(
            grouped,
            contexts,
            int(protocol["bootstrap_resamples"]),
            int(protocol["bootstrap_seed"]) + source_index,
        )
        results = summarize(overall, weights)
        sources[source] = {"counts": counts, "results": results}
        validations["all_sources_have_157_scans"] &= counts["scans"] == 157
        validations["all_sources_have_548_contexts"] &= counts["contexts"] == 548
        for method in METHODS:
            for k in main_eval.KS:
                for metric in METRICS:
                    expected = current_summary["sources"][source]["overall"][method][str(k)][metric]["point"]
                    actual = results[method][str(k)][metric]["point"]
                    validations["point_estimates_match_reported_results"] &= abs(actual - expected) <= 1e-12
        k100 = results["all_family_product_minus_source_score"]["100"]
        validations["k100_recall_ci_nonnegative"] &= k100["recall"]["paired_bootstrap_intervals_ci95"][0] >= 0
        validations["k100_violation_ci_below_zero"] &= k100["violation_all"]["paired_bootstrap_intervals_ci95"][1] < 0

    status = "completed" if all(validations.values()) else "failed_validation"
    payload = {
        "schema_version": "relcompat3d_bootstrap_intervals_bootstrap_sensitivity_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "bootstrap_unit": "scan_id cluster",
        "bootstrap_resamples": int(protocol["bootstrap_resamples"]),
        "bootstrap_seed": int(protocol["bootstrap_seed"]),
        "sources": sources,
        "validations": validations,
        "evaluation_scope": "scan-cluster intervals for the reported rankings",
    }
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary_md_path = out / "summary.md"
    main_eval.write_json(summary_path, payload)
    summary_md_path.write_text(markdown(payload), encoding="utf-8")
    manifest = {
        "schema_version": "relcompat3d_bootstrap_intervals_bootstrap_manifest_v1",
        "status": status,
        "protocol": main_eval.relpath(root, protocol_path),
        "outputs": {
            "summary.json": main_eval.sha256_file(summary_path),
            "summary.md": main_eval.sha256_file(summary_md_path),
        },
        "validations": validations,
        "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm relcompat3d_bootstrap",
    }
    main_eval.write_json(out / "manifest.json", manifest)
    print(json.dumps({"status": status, "out": str(out), "validations": validations}, indent=2))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
