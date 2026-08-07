#!/usr/bin/env python3
"""Evaluate RelCompat3D with all-family compatibility comparisons on 3DSSG."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

import relation_consistency as algebra
import evaluate_base_models as model_eval


FAMILIES = ("support_contact", "proximity", "relative_vertical")
KS = (5, 10, 20, 50, 100)
METHODS = (
    "source",
    "all_family_product",
    "rank_average_all_families",
    "rrf_all_families",
    "pooled_product",
    "hard_rule_filter",
    "family_product_continuity",
    "compatibility_only_all_families",
)
RATIO_METRICS = (
    "recall",
    "violation_all",
    "violation_decidable",
    "uncertainty_rate",
    "pessimistic_violation",
    "decidable_coverage",
    "status_coverage",
)
SOURCE_LABELS = {"vlsat": "VL-SAT", "open3dsg": "Open3DSG", "sgfn": "SGFN"}


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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_scans(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def point_status(row: dict[str, Any]) -> str | None:
    return (
        (row.get("verification_variants") or {})
        .get("point_subtype", {})
        .get("verification_status")
    )


def make_linear_scorer(models: dict[str, Any]) -> Callable[[str, str, dict[str, float]], float]:
    orbit_models = models["attempts"]["orbit_pairwise"]

    def direct(_: str, family: str, predicate: str, raw: dict[str, float]) -> float:
        return algebra.existing_probability(orbit_models[family], family, predicate, raw)

    projected = algebra.build_scorer(direct)

    def score(family: str, predicate: str, raw: dict[str, float]) -> float:
        return projected("orbit_pairwise_projected", family, predicate, raw)

    return score


def load_candidates(
    path: Path,
    linear_score: Callable[[str, str, dict[str, float]], float],
    base_models: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    digest = hashlib.sha256()
    input_rows = 0
    in_scope_rows = 0
    transform = {
        family: {"rows": 0, "max_abs_error": 0.0, "sum_abs_error": 0.0}
        for family in ("proximity", "relative_vertical")
    }
    family_models = base_models["family_models"]
    pooled_model = base_models["factor_models"]["M_int"]
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            input_rows += 1
            row = json.loads(raw_line)
            family = row["predicate"]["predicate_family"]
            if family not in FAMILIES:
                continue
            in_scope_rows += 1
            predicate = row["predicate"]["predicate_label"]
            raw = model_eval.raw_numeric(row)
            semantic = model_eval.finite((row.get("semantic") or {}).get("ranking_score"))
            if semantic is None:
                raise ValueError(f"missing_semantic:{row['prediction_id']}")
            linear = linear_score(family, predicate, raw)
            previous = model_eval.probability(family_models[family], family, predicate, raw)
            pooled = model_eval.probability(pooled_model, family, predicate, raw)
            transformed = algebra.transformed_view(family, predicate, raw)
            if transformed is not None:
                transformed_predicate, transformed_raw = transformed
                error = abs(linear - linear_score(family, transformed_predicate, transformed_raw))
                cell = transform[family]
                cell["rows"] += 1
                cell["sum_abs_error"] += error
                cell["max_abs_error"] = max(cell["max_abs_error"], error)
            grouped[row["subgraph_id"]].append(
                {
                    "id": row["prediction_id"],
                    "scan": row["scan_id"],
                    "key": model_eval.candidate_key(row),
                    "family": family,
                    "predicate": predicate,
                    "semantic": float(semantic),
                    "linear": float(linear),
                    "previous": float(previous),
                    "pooled": float(pooled),
                    "status": row.get("verification_status")
                    or (row.get("verification") or {}).get("verification_status"),
                    "point_status": point_status(row),
                    "scores": {},
                }
            )
    for candidates in grouped.values():
        denominator = max(len(candidates) - 1, 1)
        semantic_order = sorted(candidates, key=lambda item: (-item["semantic"], item["key"]))
        structured_order = sorted(candidates, key=lambda item: (-item["linear"], item["key"]))
        semantic_rank = {item["id"]: rank for rank, item in enumerate(semantic_order, 1)}
        structured_rank = {item["id"]: rank for rank, item in enumerate(structured_order, 1)}
        for item in candidates:
            rank_z = semantic_rank[item["id"]]
            rank_c = structured_rank[item["id"]]
            q_z = 1.0 - (rank_z - 1) / denominator
            q_c = 1.0 - (rank_c - 1) / denominator
            item["scores"] = {
                "source": item["semantic"],
                "all_family_product": item["semantic"] * item["linear"],
                "rank_average_all_families": 0.5 * (q_z + q_c),
                "rrf_all_families": 1.0 / (60 + rank_z) + 1.0 / (60 + rank_c),
                "pooled_product": item["semantic"] * item["pooled"],
                "hard_rule_filter": item["semantic"],
                "family_product_continuity": item["semantic"] * item["previous"],
                "compatibility_only_all_families": item["linear"],
            }
    for family, cell in transform.items():
        cell["mean_abs_error"] = cell["sum_abs_error"] / cell["rows"] if cell["rows"] else None
        cell.pop("sum_abs_error")
    return grouped, {
        "input_rows": input_rows,
        "in_scope_rows": in_scope_rows,
        "input_sha256": digest.hexdigest(),
        "transformation": transform,
    }


def empty_arrays(contexts: list[str]) -> dict[str, dict[str, np.ndarray]]:
    names = (
        "recall_num",
        "recall_den",
        "selected",
        "satisfied",
        "uncertain",
        "violated",
        "other",
    )
    return {
        method: {
            name: np.zeros((len(KS), len(contexts)), dtype=np.float64)
            for name in names
        }
        for method in METHODS
    }


def select(candidates: list[dict[str, Any]], method: str, k: int) -> list[dict[str, Any]]:
    pool = candidates
    if method == "hard_rule_filter":
        pool = [row for row in pool if row["point_status"] in {"satisfied", "uncertain"}]
    return sorted(pool, key=lambda item: (-item["scores"][method], item["key"]))[:k]


def add_cell(
    target: dict[str, np.ndarray],
    ki: int,
    ci: int,
    selected: list[dict[str, Any]],
    gt: set[tuple[Any, ...]],
) -> None:
    target["recall_num"][ki, ci] = len({row["key"] for row in selected} & gt)
    target["recall_den"][ki, ci] = len(gt)
    target["selected"][ki, ci] = len(selected)
    for row in selected:
        status = row["status"]
        if status in {"satisfied", "uncertain", "violated"}:
            target[status][ki, ci] += 1
        else:
            target["other"][ki, ci] += 1


def contributions(
    grouped: dict[str, list[dict[str, Any]]],
    gt: dict[str, set[tuple[Any, ...]]],
    gt_family: dict[str, dict[str, set[tuple[Any, ...]]]],
    contexts: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    overall = empty_arrays(contexts)
    within = {family: empty_arrays(contexts) for family in FAMILIES}
    global_slice = {family: empty_arrays(contexts) for family in FAMILIES}
    for ci, context in enumerate(contexts):
        candidates = grouped.get(context, [])
        for method in METHODS:
            for ki, k in enumerate(KS):
                chosen = select(candidates, method, k)
                add_cell(overall[method], ki, ci, chosen, gt.get(context, set()))
                for family in FAMILIES:
                    family_candidates = [row for row in candidates if row["family"] == family]
                    family_chosen = select(family_candidates, method, k)
                    family_gt = gt_family.get(context, {}).get(family, set())
                    add_cell(within[family][method], ki, ci, family_chosen, family_gt)
                    add_cell(
                        global_slice[family][method],
                        ki,
                        ci,
                        [row for row in chosen if row["family"] == family],
                        family_gt,
                    )
    return overall, within, global_slice


def ci95(samples: np.ndarray) -> list[float | None]:
    finite = samples[np.isfinite(samples)]
    if not len(finite):
        return [None, None]
    return [float(value) for value in np.percentile(finite, (2.5, 97.5))]


def ratio_arrays(values: dict[str, np.ndarray], metric: str, ki: int) -> tuple[np.ndarray, np.ndarray]:
    satisfied = values["satisfied"][ki]
    uncertain = values["uncertain"][ki]
    violated = values["violated"][ki]
    status = satisfied + uncertain + violated
    decidable = satisfied + violated
    definitions = {
        "recall": (values["recall_num"][ki], values["recall_den"][ki]),
        "violation_all": (violated, status),
        "violation_decidable": (violated, decidable),
        "uncertainty_rate": (uncertain, status),
        "pessimistic_violation": (violated + uncertain, status),
        "decidable_coverage": (decidable, status),
        "status_coverage": (status, values["selected"][ki]),
    }
    return definitions[metric]


def summarize(
    values: dict[str, Any], samples: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    report: dict[str, Any] = {}
    cache: dict[str, Any] = defaultdict(lambda: defaultdict(dict))
    for method in METHODS:
        report[method] = {}
        for ki, k in enumerate(KS):
            counts = {
                name: int(values[method][name][ki].sum())
                for name in ("selected", "satisfied", "uncertain", "violated", "other")
            }
            report[method][str(k)] = {"counts": counts}
            for metric in RATIO_METRICS:
                numerator, denominator = ratio_arrays(values[method], metric, ki)
                point = float(numerator.sum() / denominator.sum()) if denominator.sum() else None
                boot_num = numerator[samples].sum(axis=1)
                boot_den = denominator[samples].sum(axis=1)
                boot = np.divide(
                    boot_num,
                    boot_den,
                    out=np.full_like(boot_num, np.nan),
                    where=boot_den > 0,
                )
                report[method][str(k)][metric] = {
                    "point": point,
                    "ci95": ci95(boot),
                    "numerator": int(numerator.sum()),
                    "denominator": int(denominator.sum()),
                }
                cache[method][str(k)][metric] = boot
    report["deltas_vs_source_score"] = {}
    for method in METHODS[1:]:
        report["deltas_vs_source_score"][method] = {}
        for k in KS:
            report["deltas_vs_source_score"][method][str(k)] = {}
            for metric in RATIO_METRICS:
                left = report[method][str(k)][metric]["point"]
                right = report["source"][str(k)][metric]["point"]
                delta = (
                    cache[method][str(k)][metric]
                    - cache["source"][str(k)][metric]
                )
                report["deltas_vs_source_score"][method][str(k)][metric] = {
                    "point": left - right if left is not None and right is not None else None,
                    "paired_ci95": ci95(delta),
                }
    return report, cache


def add_simultaneous_family_ci(
    reports: dict[str, Any], caches: dict[str, Any]
) -> None:
    for method in METHODS[1:]:
        for k in KS:
            for metric in ("recall", "violation_all"):
                family_boot: list[np.ndarray] = []
                points: list[float] = []
                active: list[str] = []
                for family in FAMILIES:
                    item = reports[family]["deltas_vs_source_score"][method][str(k)][metric]
                    point = item["point"]
                    boot = (
                        caches[family][method][str(k)][metric]
                        - caches[family]["source"][str(k)][metric]
                    )
                    if point is not None and np.any(np.isfinite(boot)):
                        family_boot.append(boot)
                        points.append(point)
                        active.append(family)
                radius = None
                if family_boot:
                    matrix = np.column_stack(family_boot)
                    centered = np.abs(matrix - np.asarray(points)[None, :])
                    radius = float(np.nanpercentile(np.nanmax(centered, axis=1), 95.0))
                for family in FAMILIES:
                    item = reports[family]["deltas_vs_source_score"][method][str(k)][metric]
                    point = item["point"]
                    item["simultaneous_familywise_ci95"] = (
                        [point - radius, point + radius]
                        if radius is not None and family in active and point is not None
                        else [None, None]
                    )


def make_csv_rows(sources: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    for source, payload in sources.items():
        overall = payload["overall"]
        for method in METHODS:
            for k in KS:
                cell = overall[method][str(k)]
                metric_rows.append(
                    {
                        "source": source,
                        "method": method,
                        "k": k,
                        "recall": cell["recall"]["point"],
                        "recall_ci95_low": cell["recall"]["ci95"][0],
                        "recall_ci95_high": cell["recall"]["ci95"][1],
                        "violation_all": cell["violation_all"]["point"],
                        "violation_ci95_low": cell["violation_all"]["ci95"][0],
                        "violation_ci95_high": cell["violation_all"]["ci95"][1],
                        "selected": cell["counts"]["selected"],
                    }
                )
                uncertainty_rows.append(
                    {
                        "source": source,
                        "method": method,
                        "k": k,
                        **{
                            metric: cell[metric]["point"]
                            for metric in RATIO_METRICS[1:]
                        },
                        **cell["counts"],
                    }
                )
    return metric_rows, uncertainty_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty_csv:{path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Linear Main Evaluation",
        "",
        f"Status: `{summary['status']}`",
        "",
        "The main paper compatibility is relation-algebra-constrained compatibility; `orbit_pairwise_projected` is retained only as its artifact ID.",
        "",
        "## K=100 overall",
        "",
        "| Source | Method | Recall | verifier V | uncertainty | pessimistic V |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for source, payload in summary["sources"].items():
        for method in METHODS:
            cell = payload["overall"][method]["100"]
            lines.append(
                f"| {SOURCE_LABELS[source]} | {method} | {cell['recall']['point']:.4f} | "
                f"{cell['violation_all']['point']:.4f} | {cell['uncertainty_rate']['point']:.4f} | "
                f"{cell['pessimistic_violation']['point']:.4f} |"
            )
    lines.extend(
        [
            "",
            "All scores use the same 548 contexts and 3,972 exact-label denominator. The hard-rule diagnostic retains satisfied and uncertain point-subtype rows, adds no synthetic rows, and therefore may select fewer than K candidates.",
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
    paths = {name: resolve(root, value) for name, value in protocol["inputs"].items()}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing_inputs:{missing}")
    if sha256_file(paths["linear_models"]) != protocol["main_compatibility"]["model_bundle_sha256"]:
        raise ValueError("linear_model_hash_mismatch")
    relation_manifest = json.loads(paths["relation_manifest"].read_text(encoding="utf-8"))
    if relation_manifest.get("status") != "completed" or not all(relation_manifest["validations"].values()):
        raise ValueError("relation_model_not_integrity_validated")
    linear_models = json.loads(paths["linear_models"].read_text(encoding="utf-8"))
    base_models = json.loads(paths["base_models"].read_text(encoding="utf-8"))
    if base_models.get("schema_version") != "relcompat3d_base_models_v1":
        raise ValueError("base_models_not_training_only")
    train = read_scans(paths["train_scans"])
    dev = read_scans(paths["development_scans"])
    final = read_scans(paths["final_validation_scans"])
    if train & dev or train & final or dev & final:
        raise ValueError("data_split_overlap")
    linear_score = make_linear_scorer(linear_models)
    gt, gt_family = model_eval.load_gt(paths["ground_truth"])
    source_paths = {
        "vlsat": paths["vlsat_verification"],
        "open3dsg": paths["open3dsg_verification"],
        "sgfn": paths["sgfn_verification"],
    }
    source_results: dict[str, Any] = {}
    source_checks: dict[str, Any] = {}
    for source_index, (source, path) in enumerate(source_paths.items()):
        grouped, load_info = load_candidates(path, linear_score, base_models)
        contexts = sorted(set(grouped) | set(gt))
        samples = np.random.default_rng(
            int(protocol["evaluation"]["bootstrap_seed"]) + source_index
        ).integers(
            0,
            len(contexts),
            size=(int(protocol["evaluation"]["bootstrap_resamples"]), len(contexts)),
        )
        overall_values, within_values, global_values = contributions(
            grouped, gt, gt_family, contexts
        )
        overall, _ = summarize(overall_values, samples)
        within: dict[str, Any] = {}
        within_cache: dict[str, Any] = {}
        global_slice: dict[str, Any] = {}
        global_cache: dict[str, Any] = {}
        for family in FAMILIES:
            within[family], within_cache[family] = summarize(within_values[family], samples)
            global_slice[family], global_cache[family] = summarize(global_values[family], samples)
        add_simultaneous_family_ci(within, within_cache)
        add_simultaneous_family_ci(global_slice, global_cache)
        source_results[source] = {
            "counts": {
                **load_info,
                "contexts": len(contexts),
                "gt_denominator": sum(len(rows) for rows in gt.values()),
            },
            "overall": overall,
            "within_family": within,
            "global_topk_family_slice": global_slice,
        }
        source_checks[source] = {
            "contexts_548": len(contexts) == 548,
            "gt_denominator_3972": sum(len(rows) for rows in gt.values()) == 3972,
            "proximity_swap_exact": load_info["transformation"]["proximity"]["max_abs_error"] <= 1e-10,
            "vertical_inverse_exact": load_info["transformation"]["relative_vertical"]["max_abs_error"] <= 1e-10,
        }
    expected_rows = {"vlsat": 220848, "open3dsg": 160596, "sgfn": 220848}
    validations = {
        "split_counts_1061_117_157": (len(train), len(dev), len(final)) == (1061, 117, 157),
        "split_sets_pairwise_disjoint": not (train & dev or train & final or dev & final),
        "base_model_schema": base_models.get("schema_version") == "relcompat3d_base_models_v1",
        "linear_model_hash_matches": sha256_file(paths["linear_models"]) == protocol["main_compatibility"]["model_bundle_sha256"],
        "linear_model_excludes_source": linear_models.get("source_score_used") is False and linear_models.get("source_identity_used") is False,
        "all_methods_reported": all(
            set(payload["overall"]) - {"deltas_vs_source_score"} == set(METHODS)
            for payload in source_results.values()
        ),
        "all_k_reported": all(
            set(payload["overall"]["all_family_product"]) == {str(k) for k in KS}
            for payload in source_results.values()
        ),
        "familywise_intervals_reported": all(
            "simultaneous_familywise_ci95"
            in source_results[source][slice_name][family]["deltas_vs_source_score"]
            ["all_family_product"]["100"][metric]
            for source in source_results
            for slice_name in ("within_family", "global_topk_family_slice")
            for family in FAMILIES
            for metric in ("recall", "violation_all")
        ),
        "source_context_and_transform_checks": all(
            all(checks.values()) for checks in source_checks.values()
        ),
        "source_row_counts": all(
            source_results[source]["counts"]["in_scope_rows"] == expected
            for source, expected in expected_rows.items()
        ),
        "hard_filter_has_zero_violations": all(
            source_results[source]["overall"]["hard_rule_filter"][str(k)]["violation_all"]["point"] == 0.0
            for source in source_results
            for k in KS
        ),
    }
    status = "completed" if all(validations.values()) else "failed_validation"
    relation_diagnostics = json.loads(paths["relation_diagnostics"].read_text(encoding="utf-8"))
    summary = {
        "schema_version": "relcompat3d_relcompat3d_evaluate_linear_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "classification": protocol["classification"],
        "main_compatibility": protocol["main_compatibility"],
        "methods": list(METHODS),
        "ks": list(KS),
        "sources": source_results,
        "source_checks": source_checks,
        "train_internal_diagnostics": {
            "source": relpath(root, paths["relation_diagnostics"]),
            "linked_counterfactual": relation_diagnostics["diagnostics"]["linked_counterfactual"]["orbit_pairwise_projected"],
            "transformation": relation_diagnostics["diagnostics"]["transformation"]["orbit_pairwise_projected"],
        },
        "validations": validations,
        "evaluation_scope": protocol["evaluation_scope"],
        "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm relcompat3d_evaluate_linear",
    }
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    write_json(summary_path, summary)
    markdown_path = out / "summary.md"
    markdown_path.write_text(markdown(summary), encoding="utf-8")
    metrics, uncertainty = make_csv_rows(source_results)
    metrics_path = out / "metrics.csv"
    uncertainty_path = out / "uncertainty.csv"
    write_csv(metrics_path, metrics)
    write_csv(uncertainty_path, uncertainty)
    compact_inputs = {
        name: {
            "path": relpath(root, path),
            "size_bytes": path.stat().st_size,
            "sha256": (
                source_results[name.removesuffix("_verification")]["counts"]["input_sha256"]
                if name in {"vlsat_verification", "open3dsg_verification", "sgfn_verification"}
                else sha256_file(path)
            ),
        }
        for name, path in paths.items()
    }
    manifest = {
        "schema_version": "relcompat3d_all_family_comparison_manifest_v1",
        "created_at_utc": summary["created_at_utc"],
        "status": status,
        "protocol": {"path": relpath(root, protocol_path), "sha256": sha256_file(protocol_path)},
        "inputs": compact_inputs,
        "outputs": {
            path.name: {"path": relpath(root, path), "sha256": sha256_file(path)}
            for path in (summary_path, markdown_path, metrics_path, uncertainty_path)
        },
        "validations": validations,
        "docker_command": summary["docker_command"],
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"status": status, "validations": validations, "out": relpath(root, out)}))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
