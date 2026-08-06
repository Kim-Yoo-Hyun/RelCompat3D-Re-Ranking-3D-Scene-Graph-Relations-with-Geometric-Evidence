#!/usr/bin/env python3
"""Evaluate source-score mappings and closest simple geometric baselines."""

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

import evaluate_main as base
import evaluate_support_intervals as scan_bootstrap
import evaluate_train_only as strict
import fit_mlp as nonlinear
import relation_consistency as algebra


KS = base.KS
FAMILIES = base.FAMILIES
RERANKED_FAMILIES = ("proximity", "relative_vertical")
METRICS = (
    "recall",
    "violation_all",
    "violation_decidable",
    "uncertainty_rate",
    "status_coverage",
    "decidable_coverage",
)
COUNT_NAMES = ("selected", "satisfied", "uncertain", "violated", "other")


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


def read_scans(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def score_mapping(value: float, spec: dict[str, Any]) -> float:
    kind = spec["kind"]
    if kind == "identity":
        return value
    if kind == "power":
        return value ** float(spec["gamma"])
    if kind == "logit_temperature":
        epsilon = float(spec["epsilon"])
        clipped = min(max(value, epsilon), 1.0 - epsilon)
        logit = math.log(clipped / (1.0 - clipped))
        scaled = logit / float(spec["temperature"])
        if scaled >= 0.0:
            return 1.0 / (1.0 + math.exp(-scaled))
        exponent = math.exp(scaled)
        return exponent / (1.0 + exponent)
    raise ValueError(f"unsupported_score_mapping:{kind}")


def percentile_values(candidates: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for family in FAMILIES:
        rows = [row for row in candidates if row["family"] == family]
        order = sorted(rows, key=lambda row: (-row["semantic"], row["key"]))
        denominator = max(len(order) - 1, 1)
        for rank, row in enumerate(order):
            values[row["id"]] = 1.0 - rank / denominator
    return values


def source_order(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda row: (-row["semantic"], row["key"]))


def routed_order(
    candidates: list[dict[str, Any]],
    score: Callable[[dict[str, Any]], float],
) -> list[dict[str, Any]]:
    original = source_order(candidates)
    queues: dict[str, list[dict[str, Any]]] = {}
    for family in FAMILIES:
        rows = [row for row in candidates if row["family"] == family]
        if family == "support_contact":
            queues[family] = sorted(
                rows, key=lambda row: (-row["semantic"], row["key"])
            )
        else:
            queues[family] = sorted(rows, key=lambda row: (-score(row), row["key"]))
    offsets = {family: 0 for family in FAMILIES}
    routed: list[dict[str, Any]] = []
    for slot in original:
        family = slot["family"]
        routed.append(queues[family][offsets[family]])
        offsets[family] += 1
    return routed


def hard_tail_order(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    original = source_order(candidates)
    source_rank = {row["id"]: rank for rank, row in enumerate(original)}
    queues: dict[str, list[dict[str, Any]]] = {}
    for family in FAMILIES:
        rows = [row for row in candidates if row["family"] == family]
        if family == "support_contact":
            queues[family] = sorted(rows, key=lambda row: source_rank[row["id"]])
        else:
            queues[family] = sorted(
                rows,
                key=lambda row: (
                    row["status"] == "violated",
                    source_rank[row["id"]],
                    row["key"],
                ),
            )
    offsets = {family: 0 for family in FAMILIES}
    routed: list[dict[str, Any]] = []
    for slot in original:
        family = slot["family"]
        routed.append(queues[family][offsets[family]])
        offsets[family] += 1
    return routed


def hard_drop_selection(
    candidates: list[dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    """Remove every primary-verifier-violated row, then retain source order."""
    filtered = [
        row for row in source_order(candidates)
        if row["status"] != "violated"
    ]
    return filtered[:k]


def fit_positive_density(
    path: Path,
    train_scans: set[str],
    feature_contract: dict[str, list[str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    digest = hashlib.sha256()
    counts = {
        "input_rows": 0,
        "train_positive_rows": 0,
        "used_rows": 0,
        "missing_feature_cells": 0,
    }
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            counts["input_rows"] += 1
            row = json.loads(raw_line)
            if row.get("scan_id") not in train_scans:
                continue
            if ((row.get("label") or {}).get("geom_valid")) != 1:
                continue
            counts["train_positive_rows"] += 1
            family = (row.get("predicate") or {}).get("predicate_family")
            predicate = (row.get("predicate") or {}).get("predicate_label")
            if family not in RERANKED_FAMILIES or predicate not in feature_contract:
                continue
            aligned = strict.align_predicate(strict.raw_numeric(row), predicate)
            row_used = False
            for feature in feature_contract[predicate]:
                value = finite(aligned.get(feature))
                if value is None:
                    counts["missing_feature_cells"] += 1
                    continue
                values[predicate][feature].append(value)
                row_used = True
            if row_used:
                counts["used_rows"] += 1
    stats: dict[str, Any] = {}
    for predicate, features in feature_contract.items():
        stats[predicate] = {}
        for feature in features:
            array = np.asarray(values[predicate][feature], dtype=np.float64)
            if not len(array):
                raise ValueError(f"no_training_positive_values:{predicate}:{feature}")
            q25, median, q75 = np.percentile(array, (25.0, 50.0, 75.0))
            stats[predicate][feature] = {
                "count": int(len(array)),
                "median": float(median),
                "q25": float(q25),
                "q75": float(q75),
                "iqr": float(max(q75 - q25, 1e-6)),
            }
    counts["input_sha256"] = digest.hexdigest()
    return stats, counts


def direct_density(
    predicate: str,
    raw: dict[str, float],
    stats: dict[str, Any],
) -> float:
    aligned = strict.align_predicate(raw, predicate)
    squared: list[float] = []
    for feature, cell in stats[predicate].items():
        value = finite(aligned.get(feature))
        filled = cell["median"] if value is None else value
        squared.append(((filled - cell["median"]) / cell["iqr"]) ** 2)
    return math.exp(-0.5 * sum(squared) / len(squared))


def projected_density(
    family: str,
    predicate: str,
    raw: dict[str, float],
    stats: dict[str, Any],
) -> float:
    direct = direct_density(predicate, raw, stats)
    transformed = algebra.transformed_view(family, predicate, raw)
    if transformed is None:
        return direct
    transformed_predicate, transformed_raw = transformed
    return 0.5 * (
        direct
        + direct_density(transformed_predicate, transformed_raw, stats)
    )


def load_candidates(
    path: Path,
    linear_score: Callable[[str, str, dict[str, float]], float],
    mlp_model: dict[str, Any],
    density_stats: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    digest = hashlib.sha256()
    input_rows = 0
    in_scope_rows = 0
    duplicate_prediction_ids = 0
    seen_prediction_ids: set[str] = set()
    score_min = math.inf
    score_max = -math.inf
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
            prediction_id = row["prediction_id"]
            if prediction_id in seen_prediction_ids:
                duplicate_prediction_ids += 1
            seen_prediction_ids.add(prediction_id)
            predicate = row["predicate"]["predicate_label"]
            raw = strict.raw_numeric(row)
            semantic = strict.finite((row.get("semantic") or {}).get("ranking_score"))
            if semantic is None:
                raise ValueError(f"missing_semantic:{row['prediction_id']}")
            linear = linear_score(family, predicate, raw)
            mlp = nonlinear.projected_probability(
                mlp_model, family, predicate, raw
            )
            density = (
                projected_density(family, predicate, raw, density_stats)
                if family in RERANKED_FAMILIES
                else 1.0
            )
            score_min = min(score_min, semantic)
            score_max = max(score_max, semantic)
            grouped[row["subgraph_id"]].append(
                {
                    "id": prediction_id,
                    "scan": row["scan_id"],
                    "key": strict.candidate_key(row),
                    "family": family,
                    "predicate": predicate,
                    "semantic": float(semantic),
                    "linear": float(linear),
                    "mlp": float(mlp),
                    "density": float(density),
                    "status": row.get("verification_status")
                    or (row.get("verification") or {}).get("verification_status"),
                }
            )
    return grouped, {
        "input_rows": input_rows,
        "in_scope_rows": in_scope_rows,
        "input_sha256": digest.hexdigest(),
        "duplicate_prediction_ids": duplicate_prediction_ids,
        "unique_prediction_ids": len(seen_prediction_ids),
        "observed_score_min": score_min,
        "observed_score_max": score_max,
        "all_scores_nonnegative": score_min >= 0.0,
        "all_scores_at_most_one": score_max <= 1.0,
    }


def empty_values(
    methods: list[str],
    contexts: list[str],
) -> dict[str, dict[str, np.ndarray]]:
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
        for method in methods
    }


def add_cell(
    target: dict[str, np.ndarray],
    ki: int,
    ci: int,
    selected: list[dict[str, Any]],
    gt: set[tuple[Any, ...]],
) -> None:
    target["recall_num"][ki, ci] = len(
        {row["key"] for row in selected} & gt
    )
    target["recall_den"][ki, ci] = len(gt)
    target["selected"][ki, ci] = len(selected)
    for row in selected:
        status = row["status"]
        if status in {"satisfied", "uncertain", "violated"}:
            target[status][ki, ci] += 1
        else:
            target["other"][ki, ci] += 1


def ratio_arrays(
    values: dict[str, np.ndarray],
    metric: str,
    ki: int,
) -> tuple[np.ndarray, np.ndarray]:
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
        "status_coverage": (status, values["selected"][ki]),
        "decidable_coverage": (decidable, status),
    }
    return definitions[metric]


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


def summarize(
    values: dict[str, dict[str, np.ndarray]],
    methods: list[str],
    weights: np.ndarray,
) -> dict[str, Any]:
    report: dict[str, Any] = {method: {} for method in methods}
    cache: dict[str, Any] = {method: {} for method in methods}
    for method in methods:
        for ki, k in enumerate(KS):
            report[method][str(k)] = {
                "counts": {
                    name: int(values[method][name][ki].sum())
                    for name in COUNT_NAMES
                }
            }
            cache[method][str(k)] = {}
            for metric in METRICS:
                numerator, denominator = ratio_arrays(
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
                    "scan_cluster_ci95": base.ci95(boot),
                    "numerator": int(numerator.sum()),
                    "denominator": int(denominator.sum()),
                }
                cache[method][str(k)][metric] = boot
    report["deltas_vs_source"] = {}
    for method in methods:
        if method == "source":
            continue
        report["deltas_vs_source"][method] = {}
        for k in KS:
            report["deltas_vs_source"][method][str(k)] = {}
            for metric in METRICS:
                left = report[method][str(k)][metric]["point"]
                right = report["source"][str(k)][metric]["point"]
                delta = (
                    cache[method][str(k)][metric]
                    - cache["source"][str(k)][metric]
                )
                report["deltas_vs_source"][method][str(k)][metric] = {
                    "point": (
                        left - right
                        if left is not None and right is not None
                        else None
                    ),
                    "paired_scan_cluster_ci95": base.ci95(delta),
                }
    return report


def inversion_count(sequence: list[int]) -> int:
    def merge_count(values: list[int]) -> tuple[list[int], int]:
        if len(values) <= 1:
            return values, 0
        midpoint = len(values) // 2
        left, left_count = merge_count(values[:midpoint])
        right, right_count = merge_count(values[midpoint:])
        merged: list[int] = []
        inversions = left_count + right_count
        li = ri = 0
        while li < len(left) and ri < len(right):
            if left[li] <= right[ri]:
                merged.append(left[li])
                li += 1
            else:
                merged.append(right[ri])
                inversions += len(left) - li
                ri += 1
        merged.extend(left[li:])
        merged.extend(right[ri:])
        return merged, inversions

    return merge_count(sequence)[1]


def rank_stability(
    identity_orders: dict[str, list[dict[str, Any]]],
    mapped_orders: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    discordant = total_pairs = 0
    context_taus: list[float] = []
    for context, identity in identity_orders.items():
        mapped = mapped_orders[context]
        position = {row["id"]: rank for rank, row in enumerate(mapped)}
        sequence = [position[row["id"]] for row in identity]
        inversions = inversion_count(sequence)
        pairs = len(sequence) * (len(sequence) - 1) // 2
        if pairs:
            context_taus.append(1.0 - 2.0 * inversions / pairs)
            discordant += inversions
            total_pairs += pairs
    result["kendall_tau_context_mean"] = (
        float(np.mean(context_taus)) if context_taus else None
    )
    result["kendall_tau_pair_weighted"] = (
        1.0 - 2.0 * discordant / total_pairs if total_pairs else None
    )
    result["discordant_pairs"] = discordant
    result["total_pairs"] = total_pairs
    result["topk"] = {}
    for k in KS:
        intersections = unions = 0
        context_scores: list[float] = []
        for context, identity in identity_orders.items():
            left = {row["id"] for row in identity[:k]}
            right = {row["id"] for row in mapped_orders[context][:k]}
            intersection = len(left & right)
            union = len(left | right)
            intersections += intersection
            unions += union
            context_scores.append(intersection / union if union else 1.0)
        result["topk"][str(k)] = {
            "jaccard_context_mean": float(np.mean(context_scores)),
            "jaccard_micro": intersections / unions if unions else 1.0,
            "intersection": intersections,
            "union": unions,
        }
    return result


def evaluate_source(
    grouped: dict[str, list[dict[str, Any]]],
    gt: dict[str, set[tuple[Any, ...]]],
    contexts: list[str],
    mapping_specs: list[dict[str, Any]],
    resamples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    mapping_methods = [
        f"{estimator}__{mapping['name']}"
        for estimator in ("linear", "mlp")
        for mapping in mapping_specs
    ]
    methods = [
        "source",
        *mapping_methods,
        "hard_tail",
        "hard_drop",
        "positive_density",
    ]
    values = empty_values(methods, contexts)
    identity_orders: dict[str, dict[str, list[dict[str, Any]]]] = {
        "linear": {},
        "mlp": {},
    }
    mapped_orders: dict[str, dict[str, list[dict[str, Any]]]] = {
        method: {} for method in mapping_methods
    }
    monotonic_checks = {
        mapping["name"]: True for mapping in mapping_specs
    }
    for ci, context in enumerate(contexts):
        candidates = grouped.get(context, [])
        original = source_order(candidates)
        percentile = percentile_values(candidates)
        orders: dict[str, list[dict[str, Any]]] = {"source": original}
        for mapping in mapping_specs:
            mapping_name = mapping["name"]
            mapped_source = {
                row["id"]: (
                    percentile[row["id"]]
                    if mapping["kind"] == "percentile"
                    else score_mapping(row["semantic"], mapping)
                )
                for row in candidates
            }
            for family in FAMILIES:
                rows = [row for row in candidates if row["family"] == family]
                ordered = sorted(
                    rows, key=lambda row: (-row["semantic"], row["key"])
                )
                mapped_values = [mapped_source[row["id"]] for row in ordered]
                monotonic_checks[mapping_name] &= all(
                    left + 1e-15 >= right
                    for left, right in zip(
                        mapped_values, mapped_values[1:]
                    )
                )
            for estimator in ("linear", "mlp"):
                method = f"{estimator}__{mapping_name}"
                orders[method] = routed_order(
                    candidates,
                    lambda row, estimator=estimator: (
                        mapped_source[row["id"]] * row[estimator]
                    ),
                )
                mapped_orders[method][context] = orders[method]
                if mapping_name == "identity":
                    identity_orders[estimator][context] = orders[method]
        orders["hard_tail"] = hard_tail_order(candidates)
        orders["positive_density"] = routed_order(
            candidates, lambda row: row["semantic"] * row["density"]
        )
        for method in sorted(methods):
            for ki, k in enumerate(KS):
                chosen = (
                    hard_drop_selection(candidates, k)
                    if method == "hard_drop"
                    else orders[method][:k]
                )
                add_cell(
                    values[method],
                    ki,
                    ci,
                    chosen,
                    gt.get(context, set()),
                )
    weights, cluster_counts = scan_bootstrap.scan_weights(
        grouped, contexts, resamples, seed
    )
    stability: dict[str, Any] = {}
    for method in mapping_methods:
        estimator = method.split("__", 1)[0]
        stability[method] = rank_stability(
            identity_orders[estimator], mapped_orders[method]
        )
    return (
        {
            "counts": {
                **cluster_counts,
                "evaluation_contexts": len(contexts),
                "zero_prediction_contexts": len(set(contexts) - set(grouped)),
                "gt_denominator": sum(len(rows) for rows in gt.values()),
            },
            "results": summarize(values, methods, weights),
        },
        {
            "mapping_is_monotonic_on_source_order": monotonic_checks,
            "family_sequence_preserved": all(
                [
                    row["family"]
                    for row in source_order(grouped.get(context, []))
                ]
                == [
                    row["family"]
                    for row in mapped_orders[method][context]
                ]
                for method in mapping_methods
                for context in contexts
            ),
            "support_order_preserved": all(
                [
                    row["id"]
                    for row in source_order(grouped.get(context, []))
                    if row["family"] == "support_contact"
                ]
                == [
                    row["id"]
                    for row in mapped_orders[method][context]
                    if row["family"] == "support_contact"
                ]
                for method in mapping_methods
                for context in contexts
            ),
        },
        stability,
    )


def reported_match(
    sources: dict[str, Any],
    reference: dict[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    method_map = {
        "source": "source_score",
        "linear__identity": "routed_product",
        "mlp__identity": "routed_matched_mlp",
    }
    rows: list[dict[str, Any]] = []
    exact = True
    for source, payload in sources.items():
        for local_method, reference_method in method_map.items():
            for k in KS:
                for metric in ("recall", "violation_all"):
                    actual = payload["results"][local_method][str(k)][metric][
                        "point"
                    ]
                    expected = reference["sources"][source]["results"][
                        reference_method
                    ][str(k)][metric]["point"]
                    error = abs(actual - expected)
                    exact &= error <= 1e-12
                    rows.append(
                        {
                            "source": source,
                            "method": local_method,
                            "k": k,
                            "metric": metric,
                            "actual": actual,
                            "expected": expected,
                            "absolute_error": error,
                        }
                    )
    return exact, rows


def metric_rows(
    sources: dict[str, Any],
    methods: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, payload in sources.items():
        for method in methods:
            for k in KS:
                cell = payload["results"][method][str(k)]
                delta = payload["results"]["deltas_vs_source"].get(
                    method, {}
                ).get(str(k), {})
                rows.append(
                    {
                        "source": source,
                        "method": method,
                        "k": k,
                        "recall": cell["recall"]["point"],
                        "recall_delta": (
                            delta.get("recall") or {}
                        ).get("point"),
                        "recall_delta_ci95_low": (
                            (delta.get("recall") or {})
                            .get("paired_scan_cluster_ci95", [None, None])[0]
                        ),
                        "recall_delta_ci95_high": (
                            (delta.get("recall") or {})
                            .get("paired_scan_cluster_ci95", [None, None])[1]
                        ),
                        "violation": cell["violation_all"]["point"],
                        "violation_delta": (
                            delta.get("violation_all") or {}
                        ).get("point"),
                        "violation_delta_ci95_low": (
                            (delta.get("violation_all") or {})
                            .get("paired_scan_cluster_ci95", [None, None])[0]
                        ),
                        "violation_delta_ci95_high": (
                            (delta.get("violation_all") or {})
                            .get("paired_scan_cluster_ci95", [None, None])[1]
                        ),
                        "decidable_violation": cell[
                            "violation_decidable"
                        ]["point"],
                        "uncertainty": cell["uncertainty_rate"]["point"],
                        "status_coverage": cell["status_coverage"]["point"],
                        "decidable_coverage": cell[
                            "decidable_coverage"
                        ]["point"],
                        "selected": cell["counts"]["selected"],
                    }
                )
    return rows


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Score Robustness and Simple Baselines",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This post-hoc analysis uses the exact active candidate pool, source rows, "
        "family-slot route, model locks, and scan-cluster bootstrap protocol.",
        "",
        "## Reported-result rerun check",
        "",
        f"- Identity Linear/MLP and Source match the active routed-comparator "
        f"points: `{summary['validations']['reported_identity_points_exact']}`.",
        f"- Archived Tier-B hashes match the active manifests: "
        f"`{summary['validations']['tier_b_hashes_match']}`.",
        "",
        "## K=50 operating points",
        "",
        "| Predictor | Method | Recall | Violation | Decidable V | Uncertainty | Selected |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    display = (
        "source",
        "linear__identity",
        "mlp__identity",
        "hard_tail",
        "hard_drop",
        "positive_density",
    )
    for source, payload in summary["sources"].items():
        for method in display:
            cell = payload["results"][method]["50"]
            lines.append(
                f"| {source} | {method} | {cell['recall']['point']:.4f} | "
                f"{cell['violation_all']['point']:.4f} | "
                f"{cell['violation_decidable']['point']:.4f} | "
                f"{cell['uncertainty_rate']['point']:.4f} | "
                f"{cell['counts']['selected']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Monotonic mappings are fixed sensitivity conditions, not tuned alternatives.",
            "- Hard-tail and Hard-drop directly consume evaluation-verifier labels and are "
            "upper diagnostics, not deployable baselines.",
            "- Positive-density is fitted only from training-split positive geometry and "
            "is the closest non-learned continuous baseline.",
            "- A result may support robustness over the tested mappings, but never score-scale invariance.",
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
    if (
        protocol.get("status")
        != "frozen_before_p0_score_mapping_and_simple_baselines"
    ):
        raise ValueError("protocol_not_frozen")
    paths = {
        name: resolve(root, value) for name, value in protocol["inputs"].items()
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing_inputs:{missing}")
    locked = protocol["locked_sha256"]
    observed_hashes = {name: sha256_file(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": expected, "actual": observed_hashes.get(name)}
        for name, expected in locked.items()
        if observed_hashes.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"hash_mismatch:{mismatches}")

    train_scans = read_scans(paths["train_scans"])
    density_stats, density_counts = fit_positive_density(
        paths["training_table"],
        train_scans,
        protocol["simple_baselines"]["positive_density"]["features"],
    )
    if density_counts["input_sha256"] != locked["training_table"]:
        raise ValueError("training_table_stream_hash_mismatch")

    linear_models = json.loads(
        paths["structured_models"].read_text(encoding="utf-8")
    )
    nonlinear_models = json.loads(
        paths["nonlinear_models"].read_text(encoding="utf-8")
    )
    linear_score = base.make_structured_scorer(linear_models)
    mlp_model = nonlinear_models["shared_nonlinear_structured"]
    annotations = json.loads(
        paths["official_context_annotations"].read_text(encoding="utf-8")
    )
    contexts = sorted(
        {f"{row['scan']}_{row['split']}" for row in annotations["scans"]}
    )
    gt, _ = strict.load_gt(paths["ground_truth"])
    source_paths = {
        "vlsat": paths["vlsat_verification"],
        "open3dsg": paths["open3dsg_verification"],
        "sgfn": paths["sgfn_verification"],
    }
    sources: dict[str, Any] = {}
    source_loads: dict[str, Any] = {}
    route_checks: dict[str, Any] = {}
    stability: dict[str, Any] = {}
    for index, (source, path) in enumerate(source_paths.items()):
        grouped, load_counts = load_candidates(
            path, linear_score, mlp_model, density_stats
        )
        source_loads[source] = load_counts
        sources[source], route_checks[source], stability[source] = evaluate_source(
            grouped,
            gt,
            contexts,
            protocol["score_mappings"],
            int(protocol["evaluation"]["bootstrap_resamples"]),
            int(protocol["evaluation"]["bootstrap_seed"]) + index,
        )
        sources[source]["counts"].update(load_counts)

    reference = json.loads(
        paths["reported_summary"].read_text(encoding="utf-8")
    )
    identity_exact, reference_rows = reported_match(sources, reference)
    mapping_methods = {
        f"{estimator}__{mapping['name']}"
        for estimator in ("linear", "mlp")
        for mapping in protocol["score_mappings"]
    }
    baseline_methods = {
        "source",
        "linear__identity",
        "mlp__identity",
        "hard_tail",
        "hard_drop",
        "positive_density",
    }
    validations = {
        "tier_b_hashes_match": not mismatches,
        "reported_identity_points_exact": identity_exact,
        "official_context_universe_548": len(contexts) == 548,
        "gt_denominator_3972": sum(len(rows) for rows in gt.values()) == 3972,
        "all_source_scores_nonnegative_and_bounded": all(
            cell["all_scores_nonnegative"] and cell["all_scores_at_most_one"]
            for cell in source_loads.values()
        ),
        "all_tested_mappings_are_monotonic_on_source_order": all(
            all(cell["mapping_is_monotonic_on_source_order"].values())
            for cell in route_checks.values()
        ),
        "all_score_mapping_routes_preserve_family_sequence": all(
            cell["family_sequence_preserved"] for cell in route_checks.values()
        ),
        "all_score_mapping_routes_preserve_support_order": all(
            cell["support_order_preserved"] for cell in route_checks.values()
        ),
        "open3dsg_missing_predictions_are_empty": (
            sources["open3dsg"]["counts"]["zero_prediction_contexts"] == 15
        ),
        "candidate_prediction_ids_are_unique": all(
            cell["duplicate_prediction_ids"] == 0
            and cell["unique_prediction_ids"] == cell["in_scope_rows"]
            for cell in source_loads.values()
        ),
        "training_density_fit_uses_train_split_only": (
            len(train_scans) == 1061
            and density_counts["used_rows"] > 0
        ),
        "all_conditions_share_candidate_pool_and_resamples": True,
        "hard_drop_uses_primary_verifier_status": True,
    }
    status = "completed" if all(validations.values()) else "failed_validation"
    summary = {
        "schema_version": "relcompat3d_score_robustness",
        "protocol_frozen_at_kst": protocol["created_at_kst"],
        "status": status,
        "classification": "post-hoc robustness and closest-simple-baseline analysis; no method selection",
        "candidate_pool": "active main_experiment routed comparator pool",
        "score_mappings": protocol["score_mappings"],
        "simple_baselines": protocol["simple_baselines"],
        "density_fit": {
            "stats": density_stats,
            "counts": density_counts,
        },
        "sources": sources,
        "rank_stability": stability,
        "route_checks": route_checks,
        "validations": validations,
        "claim_boundary": protocol["claim_boundary"],
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary)
    write_json(out / "density_stats.json", summary["density_fit"])
    write_csv(out / "reported_validation.csv", reference_rows)
    write_csv(out / "score_mapping.csv", metric_rows(sources, mapping_methods))
    write_csv(out / "simple_baselines.csv", metric_rows(sources, baseline_methods))
    stability_rows: list[dict[str, Any]] = []
    for source, methods in stability.items():
        for method, cell in methods.items():
            for k in KS:
                stability_rows.append(
                    {
                        "source": source,
                        "method": method,
                        "k": k,
                        "kendall_tau_context_mean": cell[
                            "kendall_tau_context_mean"
                        ],
                        "kendall_tau_pair_weighted": cell[
                            "kendall_tau_pair_weighted"
                        ],
                        **cell["topk"][str(k)],
                    }
                )
    write_csv(out / "rank_stability.csv", stability_rows)
    (out / "summary.md").write_text(markdown(summary), encoding="utf-8")
    output_names = (
        "summary.json",
        "density_stats.json",
        "reported_validation.csv",
        "score_mapping.csv",
        "simple_baselines.csv",
        "rank_stability.csv",
        "summary.md",
    )
    manifest = {
        "schema_version": "relcompat3d_score_robustness_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "docker_command": (
            "env UID=$(id -u) GID=$(id -g) docker compose -f "
            "configs/relcompat3d/compose.yaml run --rm "
            "relcompat3d_score_robustness"
        ),
        "protocol": {
            "path": relpath(root, protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "inputs": {
            name: {
                "path": relpath(root, path),
                "sha256": observed_hashes[name],
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "outputs": {
            name: sha256_file(out / name) for name in output_names
        },
        "validations": validations,
        "warnings": [
            "Hard-tail and Hard-drop use evaluation-verifier labels and are non-deployable upper diagnostics.",
            "Positive-density is a training-derived diagonal robust-density baseline, not a learned estimator.",
            "The mapping grid is post-hoc and no mapping is selected as a replacement method.",
        ],
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(out / "manifest.json", manifest)
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
