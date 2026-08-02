#!/usr/bin/env python3
"""Shared utilities for RelCompat3D component and training-seed controls."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

import compatibility_features as calibration
import evaluate_main as base
import evaluate_train_only as strict
import fit_mlp as nonlinear
import relation_consistency as algebra


FAMILIES = base.FAMILIES
KS = base.KS
RERANKED_FAMILIES = {"proximity", "relative_vertical"}


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


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def validate_inputs(
    root: Path,
    protocol: dict[str, Any],
    deferred: set[str] | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    deferred = deferred or set()
    paths: dict[str, Path] = {}
    checks: dict[str, Any] = {}
    for name, spec in protocol["inputs"].items():
        path = resolve(root, spec["path"])
        if not path.exists():
            raise FileNotFoundError(f"missing_input:{name}:{path}")
        actual = None if name in deferred else sha256_file(path)
        if actual is not None and actual != spec["sha256"]:
            raise ValueError(f"input_hash_mismatch:{name}:{actual}")
        paths[name] = path
        checks[name] = {
            "path": relpath(root, path),
            "sha256": spec["sha256"] if actual is None else actual,
            "hash_checked_during_stream": name in deferred,
            "size_bytes": path.stat().st_size,
        }
    return paths, checks


def prepare_rows(paths: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_scans = read_scans(paths["train_scans"])
    dev_scans = read_scans(paths["internal_dev_scans"])
    final_scans = read_scans(paths["final_validation_scans"])
    if train_scans & dev_scans or train_scans & final_scans or dev_scans & final_scans:
        raise ValueError("split_overlap")
    rows = calibration.load_jsonl(paths["calibration_table"])
    leaked = sorted({str(row["scan_id"]) for row in rows} & final_scans)
    if leaked:
        raise ValueError(f"final_validation_rows_in_calibration:{leaked[:10]}")
    prepared, warnings = calibration.prepare_rows(
        rows, train_scans, dev_scans, set(FAMILIES)
    )
    counts = {
        "train_scans": len(train_scans),
        "internal_dev_scans": len(dev_scans),
        "final_validation_scans": len(final_scans),
        "train_rows": sum(row["_role"] == "train" for row in prepared),
        "internal_dev_rows": sum(row["_role"] == "dev" for row in prepared),
        "warnings": warnings,
        "zero_final_rows": not leaked,
    }
    return prepared, counts


def mlp_training_arrays(
    prepared: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    list[tuple[int, int]],
    dict[str, int],
]:
    train = [row for row in prepared if row["_role"] == "train"]
    original_values = [
        nonlinear.raw_feature_values(
            row["predicate"]["predicate_family"],
            row["predicate"]["predicate_label"],
            row["_raw_numeric"],
        )
        for row in train
    ]
    stats = nonlinear.fit_stats(original_values)
    original_y = np.asarray([row["_label"] for row in train], dtype=np.float64)
    id_to_index = {row["candidate_id"]: index for index, row in enumerate(train)}
    original_pairs: list[tuple[int, int]] = []
    for negative_index, row in enumerate(train):
        base_id = (row.get("candidate_source") or {}).get("base_candidate_id")
        if (
            row["_label"] == 0
            and base_id in id_to_index
            and train[id_to_index[base_id]]["_label"] == 1
        ):
            original_pairs.append((id_to_index[base_id], negative_index))

    orbit_values = list(original_values)
    orbit_labels = original_y.tolist()
    transform_index: dict[int, int] = {}
    for index, row in enumerate(train):
        family = row["predicate"]["predicate_family"]
        predicate = row["predicate"]["predicate_label"]
        transformed = algebra.transformed_view(family, predicate, row["_raw_numeric"])
        if transformed is None:
            continue
        transformed_predicate, transformed_raw = transformed
        transform_index[index] = len(orbit_values)
        orbit_values.append(
            nonlinear.raw_feature_values(
                family, transformed_predicate, transformed_raw
            )
        )
        orbit_labels.append(row["_label"])
    orbit_pairs = list(original_pairs)
    orbit_pairs.extend(
        (transform_index[pos], transform_index[neg])
        for pos, neg in original_pairs
        if pos in transform_index and neg in transform_index
    )
    orbit_x = nonlinear.normalize(orbit_values, stats)
    orbit_y = np.asarray(orbit_labels, dtype=np.float64)
    counts = {
        "train_rows": len(train),
        "linked_pairs": len(original_pairs),
        "orbit_rows": len(orbit_y),
        "orbit_pairs": len(orbit_pairs),
    }
    return stats, orbit_x, orbit_y, orbit_pairs, counts


def fit_mlp_condition(
    stats: dict[str, Any],
    orbit_x: np.ndarray,
    orbit_y: np.ndarray,
    orbit_pairs: list[tuple[int, int]],
    spec: dict[str, Any],
    pairwise: bool,
) -> dict[str, Any]:
    active_pairs = orbit_pairs if pairwise else []
    active_spec = dict(spec)
    if not pairwise:
        active_spec["pairwise_weight"] = 0.0
    params, trace = nonlinear.fit(orbit_x, orbit_y, active_pairs, active_spec)
    counts = {
        "train_rows": int(spec["expected_train_rows"]),
        "linked_pairs": int(spec["expected_linked_pairs"]),
        "orbit_rows": len(orbit_y),
        "orbit_pairs": len(orbit_pairs),
        "objective": (
            "BCE + linked pairwise margin + relation-preserving augmentation"
            if pairwise
            else "BCE + relation-preserving augmentation"
        ),
    }
    return nonlinear.serialize_model(params, stats, trace, counts)


class CompiledMLP:
    """Cache arrays used by the small NumPy MLP during row-level evaluation."""

    def __init__(self, model: dict[str, Any]) -> None:
        self.model = model
        self.mean = np.asarray(model["normalization"]["mean"], dtype=np.float64)
        self.std = np.asarray(model["normalization"]["std"], dtype=np.float64)
        self.continuous = np.asarray(
            model["normalization"]["continuous_indices"], dtype=int
        )
        self.params = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in model["parameters"].items()
        }
        self.predicate_start = len(nonlinear.FAMILIES)

    def direct(
        self, family: str, predicate: str, raw: dict[str, float]
    ) -> float:
        values = nonlinear.raw_feature_values(family, predicate, raw)
        x = np.asarray(
            [np.nan if value is None else value for value in values],
            dtype=np.float64,
        )
        x = np.where(np.isnan(x), self.mean, x)
        x[self.continuous] = (
            x[self.continuous] - self.mean[self.continuous]
        ) / self.std[self.continuous]
        pre = self.params["W"] @ x + self.params["b"]
        hidden = np.maximum(pre, 0.0)
        logit_value = (
            float(hidden @ self.params["v"])
            + float(self.params["out_b"][0])
            + float(
                x[
                    self.predicate_start : self.predicate_start
                    + len(nonlinear.PREDICATES)
                ]
                @ self.params["predicate_skip"]
            )
        )
        if logit_value >= 0:
            return 1.0 / (1.0 + math.exp(-logit_value))
        exp_value = math.exp(logit_value)
        return exp_value / (1.0 + exp_value)

    def projected(
        self, family: str, predicate: str, raw: dict[str, float]
    ) -> float:
        direct = self.direct(family, predicate, raw)
        transformed = algebra.transformed_view(family, predicate, raw)
        if transformed is None:
            return direct
        transformed_predicate, transformed_raw = transformed
        return 0.5 * (
            direct + self.direct(family, transformed_predicate, transformed_raw)
        )


class CompiledLinear:
    """Cache one family-specific linear compatibility head."""

    def __init__(self, models: dict[str, Any]) -> None:
        self.models = models
        self.weights = {
            family: np.asarray(model["weights"], dtype=np.float64)
            for family, model in models.items()
        }

    def direct(
        self, family: str, predicate: str, raw: dict[str, float]
    ) -> float:
        model = self.models[family]
        vector = algebra.existing_vector(model, family, predicate, raw)
        value = float(vector @ self.weights[family])
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    def projected(
        self, family: str, predicate: str, raw: dict[str, float]
    ) -> float:
        direct = self.direct(family, predicate, raw)
        transformed = algebra.transformed_view(family, predicate, raw)
        if transformed is None:
            return direct
        transformed_predicate, transformed_raw = transformed
        return 0.5 * (
            direct + self.direct(family, transformed_predicate, transformed_raw)
        )


def logit(value: float) -> float:
    clipped = min(max(float(value), 1e-12), 1.0 - 1e-12)
    return math.log(clipped / (1.0 - clipped))


def distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {
        "count": len(array),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def linked_pair_diagnostics(
    prepared: list[dict[str, Any]],
    score: Callable[[str, str, dict[str, float]], float],
    margin: float,
) -> dict[str, Any]:
    dev = [row for row in prepared if row["_role"] == "dev"]
    scores: dict[str, float] = {}
    for row in dev:
        scores[row["candidate_id"]] = score(
            row["predicate"]["predicate_family"],
            row["predicate"]["predicate_label"],
            row["_raw_numeric"],
        )
    margins: list[float] = []
    for row in dev:
        base_id = (row.get("candidate_source") or {}).get("base_candidate_id")
        if row["_label"] == 0 and base_id in scores:
            margins.append(logit(scores[base_id]) - logit(scores[row["candidate_id"]]))
    array = np.asarray(margins, dtype=np.float64)
    return {
        "pairs": len(margins),
        "positive_win_rate": float(np.mean(array > 0.0)),
        "margin_distribution": distribution(margins),
        "softplus_margin_loss": float(np.mean(np.logaddexp(0.0, margin - array))),
    }


def official_context_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        f"{row['scan']}_{row['split']}": str(row["scan"])
        for row in payload["scans"]
    }


def load_candidate_rows(
    path: Path,
    expected_sha256: str,
    scorers: dict[
        str,
        tuple[
            Callable[[str, str, dict[str, float]], float],
            Callable[[str, str, dict[str, float]], float],
        ],
    ],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    digest = hashlib.sha256()
    input_rows = 0
    in_scope_rows = 0
    transform_errors: dict[str, dict[str, list[float]]] = {
        condition: {family: [] for family in RERANKED_FAMILIES}
        for condition in scorers
    }
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
            raw = strict.raw_numeric(row)
            semantic = strict.finite((row.get("semantic") or {}).get("ranking_score"))
            if semantic is None:
                raise ValueError(f"missing_semantic:{row['prediction_id']}")
            compatibility: dict[str, float] = {}
            transformed_compatibility: dict[str, float] = {}
            transformed = algebra.transformed_view(family, predicate, raw)
            for condition, (score, direct_score) in scorers.items():
                compatibility[condition] = score(family, predicate, raw)
                if transformed is None:
                    transformed_compatibility[condition] = compatibility[condition]
                    continue
                transformed_predicate, transformed_raw = transformed
                transformed_compatibility[condition] = score(
                    family, transformed_predicate, transformed_raw
                )
                transform_errors[condition][family].append(
                    abs(
                        direct_score(family, predicate, raw)
                        - direct_score(family, transformed_predicate, transformed_raw)
                    )
                    if condition.endswith("no_averaging")
                    else abs(
                        compatibility[condition]
                        - transformed_compatibility[condition]
                    )
                )
            grouped[row["subgraph_id"]].append(
                {
                    "id": row["prediction_id"],
                    "scan": row["scan_id"],
                    "key": strict.candidate_key(row),
                    "family": family,
                    "semantic": float(semantic),
                    "status": row.get("verification_status")
                    or (row.get("verification") or {}).get("verification_status"),
                    "compatibility": compatibility,
                    "transformed_compatibility": transformed_compatibility,
                }
            )
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"stream_hash_mismatch:{path}:{actual_sha256}")
    transform_summary = {
        condition: {
            family: distribution(errors)
            for family, errors in family_errors.items()
        }
        for condition, family_errors in transform_errors.items()
    }
    return grouped, {
        "input_rows": input_rows,
        "in_scope_rows": in_scope_rows,
        "input_sha256": actual_sha256,
        "transformation_error": transform_summary,
    }


def routed_order(
    candidates: list[dict[str, Any]],
    condition: str,
    transformed: bool = False,
) -> list[dict[str, Any]]:
    compatibility_key = (
        "transformed_compatibility" if transformed else "compatibility"
    )
    source_order = sorted(candidates, key=lambda row: (-row["semantic"], row["key"]))
    queues: dict[str, list[dict[str, Any]]] = {}
    for family in FAMILIES:
        rows = [row for row in candidates if row["family"] == family]
        if family == "support_contact":
            queues[family] = sorted(
                rows, key=lambda row: (-row["semantic"], row["key"])
            )
        else:
            queues[family] = sorted(
                rows,
                key=lambda row: (
                    -row["semantic"] * row[compatibility_key][condition],
                    row["key"],
                ),
            )
    offsets = {family: 0 for family in FAMILIES}
    routed: list[dict[str, Any]] = []
    for source_row in source_order:
        family = source_row["family"]
        routed.append(queues[family][offsets[family]])
        offsets[family] += 1
    return routed


def evaluate_conditions(
    grouped: dict[str, list[dict[str, Any]]],
    gt: dict[str, set[tuple[Any, ...]]],
    contexts: list[str],
    conditions: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bool]]:
    totals = {
        condition: {
            str(k): {
                "recall_num": 0,
                "recall_den": 0,
                "violation_num": 0,
                "violation_den": 0,
            }
            for k in KS
        }
        for condition in conditions
    }
    membership = {
        condition: {
            str(k): {
                "intersection": 0,
                "union": 0,
                "exact_contexts": 0,
                "contexts": 0,
            }
            for k in KS
        }
        for condition in conditions
    }
    route_checks = {
        "family_sequence_exact": True,
        "support_subsequence_exact": True,
    }
    for context in contexts:
        candidates = grouped.get(context, [])
        source_order = sorted(
            candidates, key=lambda row: (-row["semantic"], row["key"])
        )
        for condition in conditions:
            routed = routed_order(candidates, condition, transformed=False)
            transformed_routed = routed_order(
                candidates, condition, transformed=True
            )
            route_checks["family_sequence_exact"] &= [
                row["family"] for row in source_order
            ] == [row["family"] for row in routed]
            route_checks["support_subsequence_exact"] &= [
                row["id"]
                for row in source_order
                if row["family"] == "support_contact"
            ] == [
                row["id"]
                for row in routed
                if row["family"] == "support_contact"
            ]
            for k in KS:
                selected = routed[:k]
                transformed_selected = transformed_routed[:k]
                exact = {row["key"] for row in selected}
                truth = gt.get(context, set())
                statuses = [
                    row["status"]
                    for row in selected
                    if row["status"] in {"satisfied", "uncertain", "violated"}
                ]
                cell = totals[condition][str(k)]
                cell["recall_num"] += len(exact & truth)
                cell["recall_den"] += len(truth)
                cell["violation_num"] += sum(
                    status == "violated" for status in statuses
                )
                cell["violation_den"] += len(statuses)
                left = {row["id"] for row in selected}
                right = {row["id"] for row in transformed_selected}
                member = membership[condition][str(k)]
                member["intersection"] += len(left & right)
                member["union"] += len(left | right)
                member["exact_contexts"] += int(left == right)
                member["contexts"] += 1
    results: dict[str, Any] = {}
    membership_summary: dict[str, Any] = {}
    for condition in conditions:
        results[condition] = {}
        membership_summary[condition] = {}
        for k in KS:
            cell = totals[condition][str(k)]
            results[condition][str(k)] = {
                "recall": (
                    cell["recall_num"] / cell["recall_den"]
                    if cell["recall_den"]
                    else None
                ),
                "violation": (
                    cell["violation_num"] / cell["violation_den"]
                    if cell["violation_den"]
                    else None
                ),
                **cell,
            }
            member = membership[condition][str(k)]
            membership_summary[condition][str(k)] = {
                "micro_jaccard": (
                    member["intersection"] / member["union"]
                    if member["union"]
                    else 1.0
                ),
                "exact_context_fraction": (
                    member["exact_contexts"] / member["contexts"]
                    if member["contexts"]
                    else None
                ),
                **member,
            }
    return results, membership_summary, route_checks


def summarize_seed_metrics(
    per_seed: dict[str, dict[str, Any]],
    source_metrics: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for estimator in ("linear", "mlp"):
        summary[estimator] = {}
        seeds = sorted(per_seed, key=int)
        for source in source_metrics:
            summary[estimator][source] = {}
            for k in KS:
                recall = np.asarray(
                    [
                        per_seed[seed][estimator][source][str(k)]["recall"]
                        for seed in seeds
                    ],
                    dtype=np.float64,
                )
                violation = np.asarray(
                    [
                        per_seed[seed][estimator][source][str(k)]["violation"]
                        for seed in seeds
                    ],
                    dtype=np.float64,
                )
                source_recall = source_metrics[source][str(k)]["recall"]
                source_violation = source_metrics[source][str(k)]["violation"]
                favorable = (recall >= source_recall - 1e-15) & (
                    violation <= source_violation + 1e-15
                )
                summary[estimator][source][str(k)] = {
                    "recall": {
                        "mean": float(np.mean(recall)),
                        "std": float(np.std(recall)),
                        "min": float(np.min(recall)),
                        "max": float(np.max(recall)),
                    },
                    "violation": {
                        "mean": float(np.mean(violation)),
                        "std": float(np.std(violation)),
                        "min": float(np.min(violation)),
                        "max": float(np.max(violation)),
                    },
                    "source": {
                        "recall": source_recall,
                        "violation": source_violation,
                    },
                    "favorable_seed_count": int(np.sum(favorable)),
                    "seed_count": len(seeds),
                    "source_direction_reversal_count": int(
                        len(seeds) - np.sum(favorable)
                    ),
                }
    return summary
