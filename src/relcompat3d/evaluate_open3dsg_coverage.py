#!/usr/bin/env python3
"""Compare the unmodified Open3DSG public route with the 548-context recovery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_all_families as base
import evaluate_support_order as routing
import evaluate_base_models as model_eval


METHODS = (
    "source",
    "family_slot_rerank",
    "all_family_product",
    "rank_average_all_families",
    "rrf_all_families",
    "pooled_product",
    "hard_rule_filter",
)
KS = base.KS
FAMILIES = base.FAMILIES


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate(
    path: Path,
    gt_path: Path,
    linear_score: Any,
    base_models: dict[str, Any],
    context_policy: str,
    context_universe: set[str],
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    grouped, load_info = base.load_candidates(path, linear_score, base_models)
    routing.add_routing_scores(grouped)
    gt, gt_family = model_eval.load_gt(gt_path)
    if context_policy == "public_pipeline_eligible":
        eligible = set(grouped)
        gt = {context: rows for context, rows in gt.items() if context in eligible}
        gt_family = {context: rows for context, rows in gt_family.items() if context in eligible}
        contexts = sorted(eligible)
    elif context_policy == "official_validation":
        contexts = sorted(context_universe)
    else:
        raise ValueError(f"unknown_context_policy:{context_policy}")
    samples = np.random.default_rng(seed).integers(0, len(contexts), size=(resamples, len(contexts)))
    original_methods = base.METHODS
    base.METHODS = METHODS
    try:
        overall_values, within_values, global_values = base.contributions(grouped, gt, gt_family, contexts)
        overall, _ = base.summarize(overall_values, samples)
        within, global_slice = {}, {}
        for family in FAMILIES:
            within[family], _ = base.summarize(within_values[family], samples)
            global_slice[family], _ = base.summarize(global_values[family], samples)
    finally:
        base.METHODS = original_methods
    return {
        "context_policy": context_policy,
        "counts": {
            **load_info,
            "candidate_contexts": len(grouped),
            "evaluation_contexts": len(contexts),
            "zero_prediction_contexts": len(set(contexts) - set(grouped)),
            "gt_denominator": sum(len(rows) for rows in gt.values()),
        },
        "overall": overall,
        "within_family": within,
        "global_topk_family_slice": global_slice,
    }


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
    diagnosis = json.loads(paths["missing15_diagnosis"].read_text(encoding="utf-8"))
    annotations = json.loads(paths["official_context_annotations"].read_text(encoding="utf-8"))
    context_universe = {
        f"{record['scan']}_{record['split']}" for record in annotations["scans"]
    }
    linear_models = json.loads(paths["linear_models"].read_text(encoding="utf-8"))
    base_models = json.loads(paths["base_models"].read_text(encoding="utf-8"))
    linear_score = base.make_linear_scorer(linear_models)
    seed, resamples = int(protocol["evaluation"]["bootstrap_seed"]), int(protocol["evaluation"]["bootstrap_resamples"])
    routes = {
        "official_eligible_533": evaluate(paths["official_verification"], paths["ground_truth"], linear_score, base_models, "public_pipeline_eligible", context_universe, seed, resamples),
        "official_full_548": evaluate(paths["official_verification"], paths["ground_truth"], linear_score, base_models, "official_validation", context_universe, seed, resamples),
        "recovered_full_548": evaluate(paths["recovered_verification"], paths["ground_truth"], linear_score, base_models, "official_validation", context_universe, seed, resamples),
    }
    validations = {
        "public_gate_is_four_visible_objects": "< 4" in diagnosis["official_drop_condition"]["condition"],
        "missing_contexts_15": diagnosis["diagnosed_contexts"] == 15,
        "official_context_universe_548": len(context_universe) == 548,
        "official_candidate_contexts_533": routes["official_eligible_533"]["counts"]["candidate_contexts"] == 533,
        "official_eligible_evaluation_contexts_533": routes["official_eligible_533"]["counts"]["evaluation_contexts"] == 533,
        "strict_full_has_15_zero_prediction_contexts": routes["official_full_548"]["counts"]["evaluation_contexts"] == 548 and routes["official_full_548"]["counts"]["zero_prediction_contexts"] == 15,
        "recovered_contexts_548": routes["recovered_full_548"]["counts"]["candidate_contexts"] == routes["recovered_full_548"]["counts"]["evaluation_contexts"] == 548,
        "full_routes_gt_denominator_3972": routes["official_full_548"]["counts"]["gt_denominator"] == routes["recovered_full_548"]["counts"]["gt_denominator"] == 3972,
        "all_routes_methods_and_k": all(set(route["overall"]) - {"deltas_vs_source_score"} == set(METHODS) and set(route["overall"]["family_slot_rerank"]) == {str(k) for k in KS} for route in routes.values()),
    }
    status = "completed" if all(validations.values()) else "failed_validation"
    summary = {
        "schema_version": "relcompat3d_relcompat3d_open3dsg_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "route_interpretation": {
            "official_eligible_533": "faithful public-pipeline evaluation on contexts retained by the unmodified four-visible-object gate",
            "official_full_548": "common-target sensitivity retaining all official GT contexts and assigning no predictions to the 15 public-pipeline drops",
            "recovered_full_548": "documented min-visible-two plus two-scan view-regeneration recovery"
        },
        "routes": routes,
        "validations": validations,
        "evaluation_scope": protocol["evaluation_scope"],
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary)
    rows: list[dict[str, Any]] = []
    for route_name, route in routes.items():
        for method in ("source", "family_slot_rerank", "all_family_product"):
            for k in KS:
                cell = route["overall"][method][str(k)]
                rows.append({"route": route_name, "contexts": route["counts"]["evaluation_contexts"], "gt_denominator": route["counts"]["gt_denominator"], "method": method, "k": k, "recall": cell["recall"]["point"], "violation": cell["violation_all"]["point"]})
    with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# Open3DSG Public-Route Sensitivity", "", f"Status: `{status}`", "", "| Route | Contexts | GT | Method | R@100 | V@100 |", "| --- | ---: | ---: | --- | ---: | ---: |"]
    for route_name, route in routes.items():
        for method in METHODS:
            cell = route["overall"][method]["100"]
            lines.append(f"| {route_name} | {route['counts']['evaluation_contexts']} | {route['counts']['gt_denominator']} | {method} | {cell['recall']['point']:.4f} | {cell['violation_all']['point']:.4f} |")
    lines.extend(["", "The public route drops 15 contexts because fewer than four annotated objects retain view metadata. The 533-context row is the faithful unmodified-pipeline evaluation; the strict-548 row is the conservative common-target sensitivity; the recovered row is reported separately.", ""])
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    outputs = [out / "summary.json", out / "summary.md", out / "metrics.csv"]
    write_json(out / "manifest.json", {"schema_version": "relcompat3d_open3dsg_official_route_manifest_v1", "status": status, "protocol": {"path": relpath(root, protocol_path), "sha256": sha256(protocol_path)}, "inputs": {name: {"path": relpath(root, path), "sha256": sha256(path)} for name, path in paths.items()}, "outputs": {path.name: {"path": relpath(root, path), "sha256": sha256(path)} for path in outputs}, "validations": validations, "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm relcompat3d_open3dsg_evaluation"})
    print(json.dumps({"status": status, "validations": validations}))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
