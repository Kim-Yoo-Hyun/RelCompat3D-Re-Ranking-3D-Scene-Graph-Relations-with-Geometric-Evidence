#!/usr/bin/env python3
"""Evaluate pre-specified support/contact applicability routing."""

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
import evaluate_base_models as model_eval


METHODS = (
    "source",
    "all_family_product",
    "support_passthrough_product",
    "family_slot_rerank",
)
FAMILIES = base.FAMILIES
KS = base.KS


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


def add_routing_scores(grouped: dict[str, list[dict[str, Any]]]) -> None:
    for candidates in grouped.values():
        for item in candidates:
            item["scores"]["support_passthrough_product"] = (
                item["semantic"] if item["family"] == "support_contact"
                else item["scores"]["all_family_product"]
            )
        source_order = sorted(candidates, key=lambda item: (-item["semantic"], item["key"]))
        family_queues: dict[str, list[dict[str, Any]]] = {}
        for family in FAMILIES:
            rows = [item for item in candidates if item["family"] == family]
            score_name = "source" if family == "support_contact" else "all_family_product"
            family_queues[family] = sorted(rows, key=lambda item: (-item["scores"][score_name], item["key"]))
        offsets = {family: 0 for family in FAMILIES}
        routed_order: list[dict[str, Any]] = []
        for source_item in source_order:
            family = source_item["family"]
            routed_order.append(family_queues[family][offsets[family]])
            offsets[family] += 1
        n = len(routed_order)
        for rank, item in enumerate(routed_order, 1):
            item["scores"]["family_slot_rerank"] = float(n - rank + 1)


def evaluate_source(
    path: Path,
    gt_path: Path,
    linear_score: Any,
    base_models: dict[str, Any],
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    grouped, load_info = base.load_candidates(path, linear_score, base_models)
    add_routing_scores(grouped)
    gt, gt_family = model_eval.load_gt(gt_path)
    contexts = sorted(set(grouped) | set(gt))
    samples = np.random.default_rng(seed).integers(0, len(contexts), size=(resamples, len(contexts)))
    original_methods = base.METHODS
    base.METHODS = METHODS
    try:
        overall_values, within_values, global_values = base.contributions(grouped, gt, gt_family, contexts)
        overall, _ = base.summarize(overall_values, samples)
        within: dict[str, Any] = {}
        within_cache: dict[str, Any] = {}
        global_slice: dict[str, Any] = {}
        global_cache: dict[str, Any] = {}
        for family in FAMILIES:
            within[family], within_cache[family] = base.summarize(within_values[family], samples)
            global_slice[family], global_cache[family] = base.summarize(global_values[family], samples)
        base.add_simultaneous_family_ci(within, within_cache)
        base.add_simultaneous_family_ci(global_slice, global_cache)
    finally:
        base.METHODS = original_methods
    return {
        "counts": {
            **load_info,
            "contexts": len(contexts),
            "gt_denominator": sum(len(rows) for rows in gt.values()),
        },
        "overall": overall,
        "within_family": within,
        "global_topk_family_slice": global_slice,
    }


def point(cell: dict[str, Any], metric: str) -> float | None:
    return cell[metric]["point"]


def make_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, payload in results.items():
        for method in METHODS:
            for k in KS:
                cell = payload["overall"][method][str(k)]
                rows.append({
                    "source": source,
                    "method": method,
                    "k": k,
                    "recall": point(cell, "recall"),
                    "violation": point(cell, "violation_all"),
                    "uncertainty": point(cell, "uncertainty_rate"),
                    "selected": cell["counts"]["selected"],
                })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    paths = {name: resolve(root, value) for name, value in protocol["inputs"].items()}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing_inputs:{missing}")
    models = json.loads(paths["linear_models"].read_text(encoding="utf-8"))
    base_models = json.loads(paths["base_models"].read_text(encoding="utf-8"))
    linear_score = base.make_linear_scorer(models)
    resamples = int(protocol["evaluation"]["bootstrap_resamples"])
    seed = int(protocol["evaluation"]["bootstrap_seed"])
    sources = {
        "development": (paths["development_verification"], paths["development_ground_truth"]),
        "vlsat": (paths["vlsat_verification"], paths["final_ground_truth"]),
        "open3dsg": (paths["open3dsg_verification"], paths["final_ground_truth"]),
        "sgfn": (paths["sgfn_verification"], paths["final_ground_truth"]),
    }
    results = {
        source: evaluate_source(path, gt_path, linear_score, base_models, seed + index, resamples)
        for index, (source, (path, gt_path)) in enumerate(sources.items())
    }

    dev = results["development"]
    dev_delta = dev["overall"]["deltas_vs_source_score"]["family_slot_rerank"]["100"]
    support_source = dev["global_topk_family_slice"]["support_contact"]["source"]
    support_routed = dev["global_topk_family_slice"]["support_contact"]["family_slot_rerank"]
    support_exact = all(
        support_source[str(k)][metric]["numerator"] == support_routed[str(k)][metric]["numerator"]
        and support_source[str(k)][metric]["denominator"] == support_routed[str(k)][metric]["denominator"]
        for k in KS for metric in ("recall", "violation_all")
    )
    composition_exact = all(
        dev["global_topk_family_slice"][family]["source"][str(k)]["counts"]["selected"]
        == dev["global_topk_family_slice"][family]["family_slot_rerank"][str(k)]["counts"]["selected"]
        for family in FAMILIES for k in KS
    )
    selected_method = "family_slot_rerank" if (
        support_exact
        and composition_exact
        and dev_delta["recall"]["paired_ci95"][0] >= -0.01
        and dev_delta["violation_all"]["paired_ci95"][1] < 0.0
    ) else None
    final_checks = {
        source: {
            "k100_delta_recall": results[source]["overall"]["deltas_vs_source_score"]["family_slot_rerank"]["100"]["recall"],
            "k100_delta_violation": results[source]["overall"]["deltas_vs_source_score"]["family_slot_rerank"]["100"]["violation_all"],
            "support_contact_global_slice_exact": all(
                results[source]["global_topk_family_slice"]["support_contact"]["source"][str(k)][metric]["numerator"]
                == results[source]["global_topk_family_slice"]["support_contact"]["family_slot_rerank"][str(k)][metric]["numerator"]
                and results[source]["global_topk_family_slice"]["support_contact"]["source"][str(k)][metric]["denominator"]
                == results[source]["global_topk_family_slice"]["support_contact"]["family_slot_rerank"][str(k)][metric]["denominator"]
                for k in KS for metric in ("recall", "violation_all")
            ),
        }
        for source in ("vlsat", "open3dsg", "sgfn")
    }
    validations = {
        "inputs_exist": not missing,
        "all_sources_evaluated": set(results) == set(sources),
        "development_support_selection_exact": support_exact,
        "development_family_composition_exact": composition_exact,
        "development_gate_selected_family_slot_rerank": selected_method == "family_slot_rerank",
        "all_final_support_contact_slices_exact": all(value["support_contact_global_slice_exact"] for value in final_checks.values()),
    }
    status = "completed" if all(validations.values()) else "failed_validation"
    summary = {
        "schema_version": "relcompat3d_relcompat3d_support_order_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "selected_on_development": selected_method,
        "methods": list(METHODS),
        "development_gate": {
            "support_contact_selection_exact": support_exact,
            "family_composition_exact": composition_exact,
            "k100_delta_recall": dev_delta["recall"],
            "k100_delta_violation": dev_delta["violation_all"],
        },
        "final_benchmark": final_checks,
        "sources": results,
        "validations": validations,
        "evaluation_scope": protocol["evaluation_scope"],
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary)
    rows = make_rows(results)
    write_csv(out / "metrics.csv", rows)
    lines = [
        "# Support/Contact Applicability Routing", "", f"Status: `{status}`", "",
        f"Internal-development selection: `{selected_method}`", "",
        "| Source | Method | R@100 | V@100 | dR vs source | dV vs source | support/contact exact |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for source in ("vlsat", "open3dsg", "sgfn"):
        cell = results[source]["overall"]["family_slot_rerank"]["100"]
        dr = final_checks[source]["k100_delta_recall"]["point"]
        dv = final_checks[source]["k100_delta_violation"]["point"]
        lines.append(f"| {source} | family-slot rerank | {point(cell, 'recall'):.4f} | {point(cell, 'violation_all'):.4f} | {dr:+.4f} | {dv:+.4f} | {final_checks[source]['support_contact_global_slice_exact']} |")
    lines.extend(["", "The selected route preserves the source-ranked family composition at every K and leaves support/contact selections unchanged; only proximity and vertical candidates are reordered within their source family slots.", ""])
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    output_paths = [out / "summary.json", out / "summary.md", out / "metrics.csv"]
    write_json(out / "manifest.json", {
        "schema_version": "relcompat3d_relcompat3d_support_order_manifest_v1",
        "status": status,
        "protocol": {"path": relpath(root, protocol_path), "sha256": sha256(protocol_path)},
        "inputs": {name: {"path": relpath(root, path), "sha256": sha256(path)} for name, path in paths.items()},
        "outputs": {path.name: {"path": relpath(root, path), "sha256": sha256(path)} for path in output_paths},
        "validations": validations,
        "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm relcompat3d_support_order",
    })
    print(json.dumps({"status": status, "selected": selected_method, "validations": validations}))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
