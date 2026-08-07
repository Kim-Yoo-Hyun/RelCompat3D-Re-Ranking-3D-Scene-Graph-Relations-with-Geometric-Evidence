#!/usr/bin/env python3
"""Evaluate the locked RelCompat3D ranking rule on ReplicaSSG and diagnose transfer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import relation_consistency as algebra
import evaluate_all_families as linear_eval
import evaluate_base_models as model_eval


FAMILIES = ("proximity", "relative_vertical")
KS = (5, 10, 20, 50, 100)
METHODS = (
    "source",
    "relcompat3d_linear",
    "unrestricted_product",
    "rank_average",
    "global_rank_average",
    "global_rrf_c60",
    "routed_compatibility_only",
)
PRIMARY_METHOD = "relcompat3d_linear"


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
        return str(path.resolve())


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


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def candidate_key(row: dict[str, Any]) -> tuple[str, int, int, int, str]:
    edge = row["edge"]
    return (
        str(row["scan_id"]),
        int(row["subset_split_id"]),
        int(edge["subject_id"]),
        int(edge["object_id"]),
        str(row["predicate"]["predicate_label"]),
    )


def gt_key(row: dict[str, Any]) -> tuple[str, int, int, int, str]:
    return (
        str(row["scan_id"]),
        int(row["subset_split_id"]),
        int(row["subject_id"]),
        int(row["object_id"]),
        str(row["predicate_label"]),
    )


def load_gt(
    path: Path, contexts: list[str]
) -> tuple[dict[str, set[tuple[Any, ...]]], dict[str, dict[str, set[tuple[Any, ...]]]]]:
    overall = {context: set() for context in contexts}
    by_family = {
        context: {family: set() for family in FAMILIES} for context in contexts
    }
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            context = str(row["scan_id"])
            family = str(row["predicate_family"])
            if context not in overall or family not in FAMILIES:
                raise ValueError("ground_truth_outside_protocol_scope")
            key = gt_key(row)
            overall[context].add(key)
            by_family[context][family].add(key)
    return overall, by_family


def percentile_ranks(
    candidates: list[dict[str, Any]], value_name: str
) -> dict[str, float]:
    order = sorted(candidates, key=lambda item: (-item[value_name], item["key"]))
    denominator = max(len(order) - 1, 1)
    return {
        item["id"]: 1.0 - (rank - 1) / denominator
        for rank, item in enumerate(order, 1)
    }


def routed_order(
    candidates: list[dict[str, Any]], family_value_name: str
) -> list[dict[str, Any]]:
    source_order = sorted(candidates, key=lambda item: (-item["semantic"], item["key"]))
    queues = {
        family: iter(
            sorted(
                [item for item in candidates if item["family"] == family],
                key=lambda item: (-item[family_value_name], item["key"]),
            )
        )
        for family in FAMILIES
    }
    return [next(queues[item["family"]]) for item in source_order]


def build_rankings(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    source_pct = percentile_ranks(candidates, "semantic")
    compat_pct = percentile_ranks(candidates, "compatibility")
    for item in candidates:
        item["product"] = item["semantic"] * item["compatibility"]
        item["global_rank_average"] = 0.5 * (
            source_pct[item["id"]] + compat_pct[item["id"]]
        )
    source_order = sorted(candidates, key=lambda item: (-item["semantic"], item["key"]))
    source_rank = {item["id"]: rank for rank, item in enumerate(source_order, 1)}
    compat_order = sorted(
        candidates, key=lambda item: (-item["compatibility"], item["key"])
    )
    compat_rank = {item["id"]: rank for rank, item in enumerate(compat_order, 1)}
    for item in candidates:
        item["global_rrf_c60"] = 1.0 / (60 + source_rank[item["id"]]) + 1.0 / (
            60 + compat_rank[item["id"]]
        )

    for family in FAMILIES:
        subset = [item for item in candidates if item["family"] == family]
        family_source_pct = percentile_ranks(subset, "semantic")
        family_compat_pct = percentile_ranks(subset, "compatibility")
        for item in subset:
            item["family_rank_average"] = 0.5 * (
                family_source_pct[item["id"]] + family_compat_pct[item["id"]]
            )

    return {
        "source": source_order,
        "relcompat3d_linear": routed_order(candidates, "product"),
        "unrestricted_product": sorted(
            candidates, key=lambda item: (-item["product"], item["key"])
        ),
        "rank_average": routed_order(candidates, "family_rank_average"),
        "global_rank_average": sorted(
            candidates, key=lambda item: (-item["global_rank_average"], item["key"])
        ),
        "global_rrf_c60": sorted(
            candidates, key=lambda item: (-item["global_rrf_c60"], item["key"])
        ),
        "routed_compatibility_only": routed_order(candidates, "compatibility"),
    }


def load_candidates(
    path: Path,
    contexts: list[str],
    models: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    scorer = linear_eval.make_linear_scorer(models)
    grouped = {context: [] for context in contexts}
    input_rows = 0
    family_rows: Counter[str] = Counter()
    predicate_rows: Counter[str] = Counter()
    transform_errors: dict[str, list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            input_rows += 1
            row = json.loads(line)
            context = str(row["scan_id"])
            family = str(row["predicate"]["predicate_family"])
            if context not in grouped or family not in FAMILIES:
                raise ValueError("candidate_outside_protocol_scope")
            predicate = str(row["predicate"]["predicate_label"])
            raw = model_eval.raw_numeric(row)
            semantic = finite((row.get("semantic") or {}).get("ranking_score"))
            if semantic is None:
                raise ValueError(f"missing_source_score:{row['prediction_id']}")
            compatibility = scorer(family, predicate, raw)
            transformed = algebra.transformed_view(family, predicate, raw)
            if transformed is None:
                raise ValueError(f"missing_declared_transform:{family}:{predicate}")
            transformed_predicate, transformed_raw = transformed
            transform_errors[family].append(
                abs(compatibility - scorer(family, transformed_predicate, transformed_raw))
            )
            status = row.get("verification_status") or (
                row.get("verification") or {}
            ).get("verification_status")
            grouped[context].append(
                {
                    "id": str(row["prediction_id"]),
                    "key": candidate_key(row),
                    "context": context,
                    "family": family,
                    "predicate": predicate,
                    "semantic": float(semantic),
                    "compatibility": float(compatibility),
                    "status": status,
                    "raw": raw,
                }
            )
            family_rows[family] += 1
            predicate_rows[predicate] += 1
    rankings = {context: build_rankings(rows) for context, rows in grouped.items()}
    return grouped, {
        "rankings": rankings,
        "input_rows": input_rows,
        "family_rows": dict(sorted(family_rows.items())),
        "predicate_rows": dict(sorted(predicate_rows.items())),
        "transformation": {
            family: {
                "rows": len(values),
                "max_abs_error": max(values) if values else None,
                "mean_abs_error": float(np.mean(values)) if values else None,
            }
            for family, values in sorted(transform_errors.items())
        },
    }


def ratio_ci(
    numerators: np.ndarray, denominators: np.ndarray, samples: np.ndarray
) -> tuple[float | None, list[float | None], np.ndarray]:
    point = (
        float(numerators.sum() / denominators.sum()) if denominators.sum() else None
    )
    boot_num = numerators[samples].sum(axis=1)
    boot_den = denominators[samples].sum(axis=1)
    boot = np.divide(
        boot_num,
        boot_den,
        out=np.full(boot_num.shape, np.nan, dtype=np.float64),
        where=boot_den > 0,
    )
    finite_boot = boot[np.isfinite(boot)]
    interval = (
        [float(value) for value in np.percentile(finite_boot, (2.5, 97.5))]
        if len(finite_boot)
        else [None, None]
    )
    return point, interval, boot


def evaluate(
    rankings: dict[str, dict[str, list[dict[str, Any]]]],
    gt: dict[str, set[tuple[Any, ...]]],
    gt_family: dict[str, dict[str, set[tuple[Any, ...]]]],
    contexts: list[str],
    samples: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    shape = (len(KS), len(contexts))
    arrays = {
        method: {
            name: np.zeros(shape, dtype=np.float64)
            for name in ("recall_num", "recall_den", "violation_num", "violation_den")
        }
        for method in METHODS
    }
    family_arrays = {
        family: {
            method: {
                name: np.zeros(shape, dtype=np.float64)
                for name in (
                    "recall_num",
                    "recall_den",
                    "violation_num",
                    "violation_den",
                    "selected",
                )
            }
            for method in METHODS
        }
        for family in FAMILIES
    }
    per_scene: dict[str, Any] = {context: {} for context in contexts}
    for ci, context in enumerate(contexts):
        for method in METHODS:
            ranked = rankings[context][method]
            for ki, k in enumerate(KS):
                selected = ranked[:k]
                selected_keys = {item["key"] for item in selected}
                hit_count = len(selected_keys & gt[context])
                status_rows = [
                    item
                    for item in selected
                    if item["status"] in {"satisfied", "uncertain", "violated"}
                ]
                arrays[method]["recall_num"][ki, ci] = hit_count
                arrays[method]["recall_den"][ki, ci] = len(gt[context])
                arrays[method]["violation_num"][ki, ci] = sum(
                    item["status"] == "violated" for item in status_rows
                )
                arrays[method]["violation_den"][ki, ci] = len(status_rows)
                for family in FAMILIES:
                    family_selected = [
                        item for item in selected if item["family"] == family
                    ]
                    family_status = [
                        item
                        for item in family_selected
                        if item["status"] in {"satisfied", "uncertain", "violated"}
                    ]
                    cell = family_arrays[family][method]
                    cell["recall_num"][ki, ci] = len(
                        {item["key"] for item in family_selected}
                        & gt_family[context][family]
                    )
                    cell["recall_den"][ki, ci] = len(gt_family[context][family])
                    cell["violation_num"][ki, ci] = sum(
                        item["status"] == "violated" for item in family_status
                    )
                    cell["violation_den"][ki, ci] = len(family_status)
                    cell["selected"][ki, ci] = len(family_selected)

    report: dict[str, Any] = {}
    cache: dict[str, Any] = {}
    for method in METHODS:
        report[method], cache[method] = {}, {}
        for ki, k in enumerate(KS):
            report[method][str(k)], cache[method][str(k)] = {}, {}
            for metric in ("recall", "violation"):
                point, interval, boot = ratio_ci(
                    arrays[method][f"{metric}_num"][ki],
                    arrays[method][f"{metric}_den"][ki],
                    samples,
                )
                report[method][str(k)][metric] = {
                    "point": point,
                    "scene_bootstrap_ci95": interval,
                    "numerator": int(arrays[method][f"{metric}_num"][ki].sum()),
                    "denominator": int(arrays[method][f"{metric}_den"][ki].sum()),
                }
                cache[method][str(k)][metric] = boot
    report["deltas_vs_source_score"] = {}
    for method in METHODS[1:]:
        report["deltas_vs_source_score"][method] = {}
        for k in KS:
            report["deltas_vs_source_score"][method][str(k)] = {}
            for metric in ("recall", "violation"):
                point = (
                    report[method][str(k)][metric]["point"]
                    - report["source"][str(k)][metric]["point"]
                )
                boot = (
                    cache[method][str(k)][metric]
                    - cache["source"][str(k)][metric]
                )
                valid = boot[np.isfinite(boot)]
                report["deltas_vs_source_score"][method][str(k)][metric] = {
                    "point": point,
                    "paired_scene_ci95": [
                        float(value) for value in np.percentile(valid, (2.5, 97.5))
                    ],
                }

    family_report: dict[str, Any] = {}
    for family in FAMILIES:
        family_report[family] = {}
        for method in METHODS:
            family_report[family][method] = {}
            for ki, k in enumerate(KS):
                family_report[family][method][str(k)] = {}
                for metric in ("recall", "violation"):
                    point, interval, _ = ratio_ci(
                        family_arrays[family][method][f"{metric}_num"][ki],
                        family_arrays[family][method][f"{metric}_den"][ki],
                        samples,
                    )
                    family_report[family][method][str(k)][metric] = {
                        "point": point,
                        "scene_bootstrap_ci95": interval,
                    }
                family_report[family][method][str(k)]["selected"] = int(
                    family_arrays[family][method]["selected"][ki].sum()
                )

    k_index = KS.index(100)
    for ci, context in enumerate(contexts):
        source_r_den = arrays["source"]["recall_den"][k_index, ci]
        source_v_den = arrays["source"]["violation_den"][k_index, ci]
        per_scene[context] = {}
        for method in METHODS:
            recall = (
                arrays[method]["recall_num"][k_index, ci] / source_r_den
                if source_r_den
                else None
            )
            violation = (
                arrays[method]["violation_num"][k_index, ci]
                / arrays[method]["violation_den"][k_index, ci]
                if arrays[method]["violation_den"][k_index, ci]
                else None
            )
            source_recall = (
                arrays["source"]["recall_num"][k_index, ci] / source_r_den
                if source_r_den
                else None
            )
            source_violation = (
                arrays["source"]["violation_num"][k_index, ci] / source_v_den
                if source_v_den
                else None
            )
            per_scene[context][method] = {
                "recall": recall,
                "violation": violation,
                "delta_recall": recall - source_recall if recall is not None else None,
                "delta_violation": (
                    violation - source_violation
                    if violation is not None and source_violation is not None
                    else None
                ),
            }
    return {"overall": report, "family_slices": family_report}, per_scene


def average_rank(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = np.asarray(values)[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = average_rank(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "rows": len(values),
        "mean": float(array.mean()) if len(array) else None,
        "std": float(array.std()) if len(array) else None,
        "p01": float(np.percentile(array, 1)) if len(array) else None,
        "p05": float(np.percentile(array, 5)) if len(array) else None,
        "median": float(np.median(array)) if len(array) else None,
        "p95": float(np.percentile(array, 95)) if len(array) else None,
        "p99": float(np.percentile(array, 99)) if len(array) else None,
        "min": float(array.min()) if len(array) else None,
        "max": float(array.max()) if len(array) else None,
    }


def feature_shift(
    grouped: dict[str, list[dict[str, Any]]], models: dict[str, Any]
) -> dict[str, Any]:
    orbit_models = models["attempts"]["orbit_pairwise"]
    report: dict[str, Any] = {}
    all_z: list[float] = []
    missing_cells = 0
    total_cells = 0
    for family in FAMILIES:
        rows = [item for values in grouped.values() for item in values if item["family"] == family]
        model = orbit_models[family]
        feature_report: dict[str, Any] = {}
        for feature in model["feature_names"]:
            if not feature.startswith("num:"):
                continue
            name = feature.split(":", 1)[1]
            stat = model["numeric_stats"][name]
            z_values: list[float] = []
            raw_values: list[float] = []
            missing = 0
            for item in rows:
                aligned = model_eval.align_predicate(item["raw"], item["predicate"])
                value = finite(aligned.get(name))
                if value is None:
                    missing += 1
                    value = float(stat["mean"])
                else:
                    raw_values.append(value)
                z_values.append((value - stat["mean"]) / (stat["std"] or 1.0))
            z_array = np.asarray(z_values)
            feature_report[name] = {
                "train_mean": stat["mean"],
                "train_std": stat["std"],
                "external_observed_mean": float(np.mean(raw_values)) if raw_values else None,
                "missing_fraction": missing / len(rows) if rows else None,
                "mean_z": float(z_array.mean()) if len(z_array) else None,
                "std_z": float(z_array.std()) if len(z_array) else None,
                "abs_z_gt_3_fraction": float(np.mean(np.abs(z_array) > 3)) if len(z_array) else None,
                "abs_z_gt_5_fraction": float(np.mean(np.abs(z_array) > 5)) if len(z_array) else None,
            }
            all_z.extend(z_values)
            missing_cells += missing
            total_cells += len(rows)
        compatibilities = [item["compatibility"] for item in rows]
        report[family] = {
            "rows": len(rows),
            "features": feature_report,
            "compatibility": {
                **distribution(compatibilities),
                "below_0_05_fraction": float(
                    np.mean(np.asarray(compatibilities) < 0.05)
                ),
                "above_0_95_fraction": float(
                    np.mean(np.asarray(compatibilities) > 0.95)
                ),
            },
        }
    z_array = np.asarray(all_z)
    feature_cells = [
        (family, name, value)
        for family, family_report in report.items()
        for name, value in family_report["features"].items()
    ]
    return {
        "by_family": report,
        "aggregate": {
            "feature_cells": total_cells,
            "missing_fraction": missing_cells / total_cells if total_cells else None,
            "abs_z_gt_3_fraction": float(np.mean(np.abs(z_array) > 3)),
            "abs_z_gt_5_fraction": float(np.mean(np.abs(z_array) > 5)),
            "largest_absolute_mean_z": [
                {
                    "family": family,
                    "feature": name,
                    "mean_z": value["mean_z"],
                }
                for family, name, value in sorted(
                    feature_cells,
                    key=lambda cell: abs(cell[2]["mean_z"]),
                    reverse=True,
                )[:10]
            ],
        },
    }


def source_diagnostics(
    grouped: dict[str, list[dict[str, Any]]], contexts: list[str]
) -> dict[str, Any]:
    rows = [item for context in contexts for item in grouped[context]]
    scores = [item["semantic"] for item in rows]
    boundary_ties: dict[str, Any] = {}
    for k in KS:
        tied_contexts = 0
        excess_rows = 0
        for context in contexts:
            ordered = sorted(grouped[context], key=lambda item: (-item["semantic"], item["key"]))
            if not ordered:
                continue
            cutoff = min(k, len(ordered))
            boundary = ordered[cutoff - 1]["semantic"]
            total_tied = sum(item["semantic"] == boundary for item in ordered)
            selected_tied = sum(item["semantic"] == boundary for item in ordered[:cutoff])
            if total_tied > selected_tied:
                tied_contexts += 1
                excess_rows += total_tied - selected_tied
        boundary_ties[str(k)] = {
            "contexts_with_unresolved_boundary_tie": tied_contexts,
            "excess_tied_rows": excess_rows,
        }
    return {
        "score_distribution": distribution(scores),
        "zero_fraction": sum(score == 0.0 for score in scores) / len(scores),
        "one_fraction": sum(score == 1.0 for score in scores) / len(scores),
        "nonzero_fraction": sum(score > 0.0 for score in scores) / len(scores),
        "distinct_values": len(set(scores)),
        "boundary_ties": boundary_ties,
        "context_candidate_counts": {
            context: len(grouped[context]) for context in contexts
        },
        "contexts_with_candidates_le_100": sum(
            len(grouped[context]) <= 100 for context in contexts
        ),
    }


def rank_diagnostics(
    rankings: dict[str, dict[str, list[dict[str, Any]]]],
    gt: dict[str, set[tuple[Any, ...]]],
    contexts: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in METHODS[1:]:
        displacements: list[int] = []
        topk: dict[str, list[float]] = {str(k): [] for k in KS}
        transitions: dict[str, Counter[str]] = {str(k): Counter() for k in KS}
        composition_exact = True
        for context in contexts:
            source = rankings[context]["source"]
            ranked = rankings[context][method]
            source_rank = {item["id"]: rank for rank, item in enumerate(source, 1)}
            method_rank = {item["id"]: rank for rank, item in enumerate(ranked, 1)}
            displacements.extend(
                abs(source_rank[item_id] - method_rank[item_id])
                for item_id in source_rank
            )
            for k in KS:
                source_top = source[:k]
                method_top = ranked[:k]
                source_ids = {item["id"] for item in source_top}
                method_ids = {item["id"] for item in method_top}
                union = source_ids | method_ids
                topk[str(k)].append(
                    len(source_ids & method_ids) / len(union) if union else 1.0
                )
                source_hits = {item["key"] for item in source_top} & gt[context]
                method_hits = {item["key"] for item in method_top} & gt[context]
                transitions[str(k)]["gt_gained"] += len(method_hits - source_hits)
                transitions[str(k)]["gt_lost"] += len(source_hits - method_hits)
                transitions[str(k)]["rows_entered"] += len(method_ids - source_ids)
                transitions[str(k)]["rows_left"] += len(source_ids - method_ids)
                if Counter(item["family"] for item in source_top) != Counter(
                    item["family"] for item in method_top
                ):
                    composition_exact = False
        array = np.asarray(displacements)
        result[method] = {
            "mean_absolute_rank_displacement": float(array.mean()),
            "p95_absolute_rank_displacement": float(np.percentile(array, 95)),
            "maximum_absolute_rank_displacement": int(array.max()),
            "mean_topk_jaccard": {
                k: float(np.mean(values)) for k, values in topk.items()
            },
            "selection_transitions": {
                k: dict(value) for k, value in transitions.items()
            },
            "family_composition_exact_at_all_k": composition_exact,
        }
    return result


def construct_alignment(
    grouped: dict[str, list[dict[str, Any]]],
    gt: dict[str, set[tuple[Any, ...]]],
    contexts: list[str],
) -> dict[str, Any]:
    rows = [item for context in contexts for item in grouped[context]]
    result: dict[str, Any] = {}
    slices = {
        "all": rows,
        **{
            family: [item for item in rows if item["family"] == family]
            for family in FAMILIES
        },
    }
    for name, subset in slices.items():
        gt_labels = [int(item["key"] in gt[item["context"]]) for item in subset]
        decidable = [
            item for item in subset if item["status"] in {"satisfied", "violated"}
        ]
        verifier_labels = [int(item["status"] == "satisfied") for item in decidable]
        result[name] = {
            "rows": len(subset),
            "exact_gt_positives": sum(gt_labels),
            "decidable_rows": len(decidable),
            "gt_auc": {
                "source": auc(gt_labels, [item["semantic"] for item in subset]),
                "compatibility": auc(
                    gt_labels, [item["compatibility"] for item in subset]
                ),
                "product": auc(gt_labels, [item["product"] for item in subset]),
            },
            "verifier_satisfaction_auc": {
                "source": auc(
                    verifier_labels, [item["semantic"] for item in decidable]
                ),
                "compatibility": auc(
                    verifier_labels, [item["compatibility"] for item in decidable]
                ),
                "product": auc(
                    verifier_labels, [item["product"] for item in decidable]
                ),
            },
        }
    return result


def markdown(report: dict[str, Any]) -> str:
    overall = report["metrics"]["overall"]
    lines = [
        "# ReplicaSSG Final-Method Cross-Dataset Evaluation",
        "",
        f"Status: `{report['status']}`",
        "",
        "This is a benchmark evaluation of the locked final method on a previously observed external target; it is not untouched prospective confirmation.",
        "",
        "| Method | R@10 | V@10 | R@50 | V@50 | R@100 | V@100 | dR@100 | dV@100 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        delta = overall["deltas_vs_source_score"].get(method, {}).get("100", {})
        lines.append(
            "| {} | {:.5f} | {:.5f} | {:.5f} | {:.5f} | {:.5f} | {:.5f} | {:+.5f} | {:+.5f} |".format(
                method,
                overall[method]["10"]["recall"]["point"],
                overall[method]["10"]["violation"]["point"],
                overall[method]["50"]["recall"]["point"],
                overall[method]["50"]["violation"]["point"],
                overall[method]["100"]["recall"]["point"],
                overall[method]["100"]["violation"]["point"],
                delta.get("recall", {}).get("point", 0.0),
                delta.get("violation", {}).get("point", 0.0),
            )
        )
    source = report["decomposition"]["source"]
    shift = report["decomposition"]["geometry_shift"]["aggregate"]
    alignment = report["decomposition"]["construct_alignment"]["all"]
    lines.extend(
        [
            "",
            f"Primary joint gate: `{report['primary_gate']['decision']}`.",
            "",
            "## Negative-transfer decomposition",
            "",
            f"- Source-score zeros: {source['zero_fraction']:.3%}; ones: {source['one_fraction']:.3%}; distinct values: {source['distinct_values']}.",
            f"- External feature cells with |train-standardized z|>3: {shift['abs_z_gt_3_fraction']:.3%}; missing: {shift['missing_fraction']:.3%}.",
            f"- Exact-GT AUC: source={alignment['gt_auc']['source_score']:.4f}, compatibility={alignment['gt_auc']['compatibility']:.4f}, product={alignment['gt_auc']['product']:.4f}.",
            f"- Verifier-satisfaction AUC: source={alignment['verifier_satisfaction_auc']['source_score']:.4f}, compatibility={alignment['verifier_satisfaction_auc']['compatibility']:.4f}, product={alignment['verifier_satisfaction_auc']['product']:.4f}.",
            f"- Scope: {report['counts']['gt_denominator']} exact-label GT relations, three predicates, two families, and {report['counts']['contexts']} scene-level bootstrap units.",
            "",
            "Full feature-shift, rank-displacement, per-scene, family, and selection-transition diagnostics are in `summary.json`.",
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
    paths = {
        name: resolve(root, value["path"])
        for name, value in protocol["inputs"].items()
    }
    models = json.loads(paths["models"].read_text(encoding="utf-8"))
    contexts = list(protocol["dataset"]["test_scenes"])
    gt, gt_family = load_gt(paths["ground_truth"], contexts)
    grouped, candidate_meta = load_candidates(paths["verification"], contexts, models)
    rankings = candidate_meta.pop("rankings")
    rng = np.random.default_rng(protocol["evaluation"]["bootstrap"]["seed"])
    samples = rng.integers(
        0,
        len(contexts),
        size=(protocol["evaluation"]["bootstrap"]["resamples"], len(contexts)),
    )
    metrics, per_scene = evaluate(rankings, gt, gt_family, contexts, samples)
    feature_report = feature_shift(grouped, models)
    source_report = source_diagnostics(grouped, contexts)
    rank_report = rank_diagnostics(rankings, gt, contexts)
    alignment_report = construct_alignment(grouped, gt, contexts)

    gt_denominator = sum(len(values) for values in gt.values())
    candidate_keys = {
        item["key"] for context in contexts for item in grouped[context]
    }
    gt_keys = {key for context in contexts for key in gt[context]}
    coverage_hits = len(candidate_keys & gt_keys)
    primary_delta = metrics["overall"]["deltas_vs_source_score"][PRIMARY_METHOD]["100"]
    recall_interval = primary_delta["recall"]["paired_scene_ci95"]
    violation_interval = primary_delta["violation"]["paired_scene_ci95"]
    recall_pass = recall_interval[0] > -0.01
    violation_pass = violation_interval[1] < 0.0

    input_hashes = {
        name: sha256_file(path) for name, path in paths.items()
    }
    expected_hashes = {
        name: value["sha256"] for name, value in protocol["inputs"].items()
    }
    composition_methods = (
        "relcompat3d_linear",
        "rank_average",
        "routed_compatibility_only",
    )
    all_metrics = [
        cell[metric]["point"]
        for method in METHODS
        for cell in metrics["overall"][method].values()
        for metric in ("recall", "violation")
    ]
    validations = {
        "protocol_ready": protocol.get("status")
        == "ready",
        "classification_discloses_prior_target_observation": protocol.get("classification")
        == "cross_dataset_benchmark_evaluation_on_previously_observed_target_no_target_tuning",
        "all_input_hashes_match": input_hashes == expected_hashes,
        "evaluator_hash_matches": sha256_file(Path(__file__).resolve())
        == protocol["implementation"]["evaluator_sha256"],
        "compose_hash_matches": sha256_file(root / protocol["implementation"]["compose_path"])
        == protocol["implementation"]["compose_sha256"],
        "locked_model_hash_matches": input_hashes["models"]
        == protocol["method"]["model_sha256"],
        "model_excludes_source_score": models.get("source_score_used") is False,
        "model_excludes_predictor_identity": models.get("source_identity_used") is False,
        "contexts_exact": set(grouped) == set(contexts) and len(contexts) == 11,
        "candidate_rows_exact": candidate_meta["input_rows"] == 4293,
        "gt_denominator_exact": gt_denominator == 172,
        "candidate_coverage_computed": 0 <= coverage_hits <= gt_denominator,
        "families_exact": set(candidate_meta["family_rows"]) == set(FAMILIES),
        "predicates_exact": set(candidate_meta["predicate_rows"])
        == {"close by", "higher than", "lower than"},
        "all_methods_reported": set(METHODS)
        == set(metrics["overall"]) - {"deltas_vs_source_score"},
        "primary_route_family_composition_exact": all(
            rank_report[method]["family_composition_exact_at_all_k"]
            for method in composition_methods
        ),
        "relation_algebra_exact": all(
            cell["max_abs_error"] <= 1e-12
            for cell in candidate_meta["transformation"].values()
        ),
        "bootstrap_configuration_matches": protocol["evaluation"]["bootstrap"]
        == {"unit": "scene", "resamples": 1000, "seed": 20260715, "shared_indices": True},
        "no_target_tuning": protocol["method"]["target_specific_fit_rows"] == 0
        and protocol["method"]["target_specific_hyperparameters"] == 0,
        "all_metrics_finite": all(
            value is not None and math.isfinite(value) for value in all_metrics
        ),
    }
    status = (
        "completed_external_dataset_benchmark_with_negative_transfer_decomposition"
        if all(validations.values())
        else "blocked_external_dataset_evaluation"
    )
    report = {
        "schema_version": "relcompat3d_relcompat3d_transfer_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "classification": protocol["classification"],
        "counts": {
            "contexts": len(contexts),
            "candidate_rows": candidate_meta["input_rows"],
            "gt_denominator": gt_denominator,
            "candidate_supported_gt": coverage_hits,
            "candidate_recall_ceiling": coverage_hits / gt_denominator,
            "families": candidate_meta["family_rows"],
            "predicates": candidate_meta["predicate_rows"],
        },
        "metrics": metrics,
        "primary_gate": {
            "decision": "pass" if recall_pass and violation_pass else "fail",
            "recall_guardrail_pass": recall_pass,
            "violation_gate_pass": violation_pass,
            "rule": "paired scene-bootstrap dRecall@100 lower > -0.01 and dViolation@100 upper < 0",
            "delta_recall": primary_delta["recall"],
            "delta_violation": primary_delta["violation"],
        },
        "decomposition": {
            "source": source_report,
            "geometry_shift": feature_report,
            "rank_and_selection": rank_report,
            "construct_alignment": alignment_report,
            "per_scene_k100": per_scene,
            "ontology_and_scope": {
                "mapped_source_predicates": ["near", "above", "under"],
                "paper_predicates": ["close by", "higher than", "lower than"],
                "families_evaluated": list(FAMILIES),
                "main_family_absent": "support_contact",
                "excluded_replica_predicates_include": ["on", "against"],
                "reason": "no exact mapping to the paper's support/contact subtypes",
            },
            "candidate_coverage": {
                "exact_gt": gt_denominator,
                "candidate_supported_gt": coverage_hits,
                "recall_ceiling": coverage_hits / gt_denominator,
            },
        },
        "candidate_metadata": candidate_meta,
        "validations": validations,
        "evaluation_scope": {
            "reported": "cross-dataset behavior of the fixed ranking rule on exact near/above/under mappings",
            "not_evaluated": [
                "dataset-level generalization",
                "support/contact transfer",
                "universal ranking performance",
                "independent geometric-validity annotations",
            ],
        },
        "inputs": {
            name: {"path": relpath(root, path), "sha256": input_hashes[name]}
            for name, path in paths.items()
        },
        "protocol": {
            "path": relpath(root, protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm relcompat3d_transfer_audit",
    }
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "summary.json", report)
    (out / "summary.md").write_text(markdown(report), encoding="utf-8")
    with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "k", "recall", "violation", "delta_recall", "delta_violation"),
        )
        writer.writeheader()
        for method in METHODS:
            for k in KS:
                delta = metrics["overall"]["deltas_vs_source_score"].get(method, {}).get(str(k), {})
                writer.writerow(
                    {
                        "method": method,
                        "k": k,
                        "recall": metrics["overall"][method][str(k)]["recall"]["point"],
                        "violation": metrics["overall"][method][str(k)]["violation"]["point"],
                        "delta_recall": delta.get("recall", {}).get("point", 0.0),
                        "delta_violation": delta.get("violation", {}).get("point", 0.0),
                    }
                )
    outputs = {
        name: {"path": relpath(root, out / name), "sha256": sha256_file(out / name)}
        for name in ("summary.json", "summary.md", "metrics.csv")
    }
    write_json(
        out / "manifest.json",
        {
            "schema_version": "relcompat3d_relcompat3d_transfer_audit_manifest_v1",
            "created_at_utc": report["created_at_utc"],
            "status": status,
            "validations": validations,
            "outputs": outputs,
            "docker_command": report["docker_command"],
        },
    )
    print(
        json.dumps(
            {
                "status": status,
                "counts": report["counts"],
                "primary_gate": report["primary_gate"],
            },
            sort_keys=True,
        )
    )
    return 0 if all(validations.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
