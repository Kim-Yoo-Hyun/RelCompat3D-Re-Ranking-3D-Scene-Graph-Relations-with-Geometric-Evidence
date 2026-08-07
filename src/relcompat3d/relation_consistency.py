#!/usr/bin/env python3
"""Run the fixed RelCompat3D relation-algebra compatibility development suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

import compatibility_features as base
import evaluate_base_models as model_eval


FAMILIES = ("support_contact", "proximity", "relative_vertical")
KS = (5, 10, 20, 50, 100)
COMPATIBILITIES = (
    "family",
    "orbit_projection",
    "orbit_augmented",
    "pairwise_margin",
    "orbit_pairwise",
    "orbit_pairwise_projected",
    "algebra_basis",
)
METHODS = (
    "semantic_only",
    "family_product",
    "orbit_projection_product",
    "orbit_augmented_product",
    "pairwise_margin_product",
    "orbit_pairwise_product",
    "orbit_pairwise_projected_product",
    "algebra_basis_product",
)
STRUCTURED_METHODS = METHODS[2:]
SOURCE_NAMES = ("vlsat", "open3dsg", "sgfn")


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


def read_scans(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def sigmoid_array(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def logit(value: float) -> float:
    clipped = min(max(float(value), 1e-12), 1.0 - 1e-12)
    return math.log(clipped / (1.0 - clipped))


def transformed_view(
    family: str,
    predicate: str,
    raw: dict[str, float],
) -> tuple[str, dict[str, float]] | None:
    """Apply the declared family relation algebra to an ordered-pair view."""
    if family == "support_contact":
        return None
    if family == "proximity":
        transformed_predicate = predicate
    elif family == "relative_vertical":
        if predicate == "higher than":
            transformed_predicate = "lower than"
        elif predicate == "lower than":
            transformed_predicate = "higher than"
        else:
            return None
    else:
        return None

    transformed = dict(raw)
    for name in (
        "predicate_aligned_center_delta_z",
        "predicate_aligned_normalized_center_delta_z",
    ):
        transformed.pop(name, None)
    if "center_delta_z" in raw:
        transformed["center_delta_z"] = -raw["center_delta_z"]
        transformed["abs_center_delta_z"] = abs(transformed["center_delta_z"])
    if "normalized_center_delta_z" in raw:
        transformed["normalized_center_delta_z"] = -raw["normalized_center_delta_z"]
        transformed["abs_normalized_center_delta_z"] = abs(
            transformed["normalized_center_delta_z"]
        )
    left = raw.get("projected_subject_overlap_ratio")
    right = raw.get("projected_object_overlap_ratio")
    if right is not None:
        transformed["projected_subject_overlap_ratio"] = right
    if left is not None:
        transformed["projected_object_overlap_ratio"] = left
    for left_name, right_name in (
        ("subject_bottom_z", "object_bottom_z"),
        ("subject_top_z", "object_top_z"),
    ):
        if right_name in raw:
            transformed[left_name] = raw[right_name]
        if left_name in raw:
            transformed[right_name] = raw[left_name]
    if "object_bottom_z" in raw and "subject_top_z" in raw:
        transformed["vertical_gap_subject_on_object"] = (
            raw["object_bottom_z"] - raw["subject_top_z"]
        )
        transformed["abs_vertical_gap_subject_on_object"] = abs(
            transformed["vertical_gap_subject_on_object"]
        )
    transformed = model_eval.align_predicate(transformed, transformed_predicate)
    return transformed_predicate, transformed


def existing_vector(
    model: dict[str, Any],
    family: str,
    predicate: str,
    raw: dict[str, float],
) -> np.ndarray:
    aligned = model_eval.align_predicate(raw, predicate)
    values: list[float] = []
    for feature in model["feature_names"]:
        if feature == "bias":
            values.append(1.0)
        elif feature.startswith("family:"):
            values.append(float(family == feature.split(":", 1)[1]))
        elif feature.startswith("predicate:"):
            values.append(float(predicate == feature.split(":", 1)[1]))
        elif feature.startswith("num:"):
            name = feature.split(":", 1)[1]
            stats = model["numeric_stats"][name]
            value = aligned.get(name, stats["mean"])
            values.append((value - stats["mean"]) / (stats["std"] or 1.0))
        else:
            raise ValueError(f"unsupported_feature:{feature}")
    return np.asarray(values, dtype=np.float64)


def existing_probability(
    model: dict[str, Any],
    family: str,
    predicate: str,
    raw: dict[str, float],
) -> float:
    vector = existing_vector(model, family, predicate, raw)
    return float(sigmoid_array(np.asarray([vector @ np.asarray(model["weights"])]))[0])


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    optimizer: dict[str, Any],
    pair_diffs: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    weights = np.zeros(x.shape[1], dtype=np.float64)
    epochs = int(optimizer["epochs"])
    learning_rate = float(optimizer["learning_rate"])
    l2 = float(optimizer["l2"])
    pair_weight = float(optimizer["pairwise_weight"]) if pair_diffs is not None else 0.0
    margin = float(optimizer["pairwise_margin"])
    trace: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        probabilities = sigmoid_array(x @ weights)
        gradient = (x.T @ (probabilities - y)) / len(y)
        gradient[1:] += l2 * weights[1:]
        pair_loss = 0.0
        if pair_diffs is not None and len(pair_diffs):
            differences = pair_diffs @ weights
            residual = margin - differences
            pair_sigmoid = sigmoid_array(residual)
            gradient += pair_weight * (-(pair_diffs.T @ pair_sigmoid) / len(pair_diffs))
            pair_loss = float(np.mean(np.logaddexp(0.0, residual)))
        weights -= learning_rate * gradient
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
            bce = float(np.mean(-(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))))
            trace.append(
                {
                    "epoch": epoch,
                    "bce": bce,
                    "pairwise_softplus": pair_loss,
                    "objective": bce + pair_weight * pair_loss,
                }
            )
    return weights, trace


def basis_names(family: str) -> tuple[str, ...]:
    shared = (
        "distance_3d",
        "distance_xy",
        "normalized_distance_3d",
        "normalized_distance_xy",
        "projected_iou_xy",
        "overlap_sum",
        "overlap_absdiff",
        "abs_center_delta_z",
        "abs_normalized_center_delta_z",
        "height_sum",
        "height_absdiff",
    )
    if family == "proximity":
        return shared
    if family == "relative_vertical":
        return shared + (
            "predicate_aligned_center_delta_z",
            "predicate_aligned_normalized_center_delta_z",
            "predicate_aligned_bottom_delta_z",
            "predicate_aligned_top_delta_z",
        )
    raise ValueError(f"unsupported_algebra_basis:{family}")


def basis_values(family: str, predicate: str, raw: dict[str, float]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in (
        "distance_3d",
        "distance_xy",
        "normalized_distance_3d",
        "normalized_distance_xy",
        "projected_iou_xy",
        "abs_center_delta_z",
        "abs_normalized_center_delta_z",
    ):
        if name in raw:
            values[name] = raw[name]
    subject_overlap = raw.get("projected_subject_overlap_ratio")
    object_overlap = raw.get("projected_object_overlap_ratio")
    if subject_overlap is not None and object_overlap is not None:
        values["overlap_sum"] = subject_overlap + object_overlap
        values["overlap_absdiff"] = abs(subject_overlap - object_overlap)
    if all(name in raw for name in ("subject_bottom_z", "subject_top_z", "object_bottom_z", "object_top_z")):
        subject_height = raw["subject_top_z"] - raw["subject_bottom_z"]
        object_height = raw["object_top_z"] - raw["object_bottom_z"]
        values["height_sum"] = subject_height + object_height
        values["height_absdiff"] = abs(subject_height - object_height)
        if family == "relative_vertical":
            direction = 1.0 if predicate == "higher than" else -1.0
            values["predicate_aligned_bottom_delta_z"] = direction * (
                raw["subject_bottom_z"] - raw["object_bottom_z"]
            )
            values["predicate_aligned_top_delta_z"] = direction * (
                raw["subject_top_z"] - raw["object_top_z"]
            )
    if family == "relative_vertical":
        direction = 1.0 if predicate == "higher than" else -1.0
        if "center_delta_z" in raw:
            values["predicate_aligned_center_delta_z"] = direction * raw["center_delta_z"]
        if "normalized_center_delta_z" in raw:
            values["predicate_aligned_normalized_center_delta_z"] = (
                direction * raw["normalized_center_delta_z"]
            )
    return values


def make_basis_model(
    family: str,
    train_rows: list[dict[str, Any]],
    optimizer: dict[str, Any],
) -> dict[str, Any]:
    names = basis_names(family)
    raw_values = [basis_values(family, row["predicate"]["predicate_label"], row["_raw_numeric"]) for row in train_rows]
    stats: dict[str, dict[str, float]] = {}
    for name in names:
        observed = [row[name] for row in raw_values if name in row]
        mean = float(np.mean(observed)) if observed else 0.0
        std = float(np.std(observed)) if observed else 1.0
        stats[name] = {"mean": mean, "std": std if std > 1e-12 else 1.0, "observed": len(observed)}
    x = np.asarray(
        [
            [1.0]
            + [
                (row.get(name, stats[name]["mean"]) - stats[name]["mean"]) / stats[name]["std"]
                for name in names
            ]
            for row in raw_values
        ],
        dtype=np.float64,
    )
    y = np.asarray([row["_label"] for row in train_rows], dtype=np.float64)
    weights, trace = fit_logistic(x, y, optimizer)
    return {
        "architecture": "relation_algebra_invariant_logistic",
        "family": family,
        "feature_names": ["bias", *names],
        "numeric_stats": stats,
        "weights": weights.tolist(),
        "training_trace": trace,
        "parameter_count": len(weights),
    }


def basis_probability(model: dict[str, Any], predicate: str, raw: dict[str, float]) -> float:
    family = model["family"]
    values = basis_values(family, predicate, raw)
    vector = [1.0]
    for name in model["feature_names"][1:]:
        stats = model["numeric_stats"][name]
        vector.append((values.get(name, stats["mean"]) - stats["mean"]) / stats["std"])
    return float(
        sigmoid_array(
            np.asarray([np.asarray(vector) @ np.asarray(model["weights"])])
        )[0]
    )


def calibration_metrics(probabilities: list[float], labels: list[int]) -> dict[str, Any]:
    return {
        "rows": len(labels),
        "positive": int(sum(labels)),
        "negative": int(len(labels) - sum(labels)),
        "auroc": base.auroc(probabilities, labels),
        "auprc": base.average_precision(probabilities, labels),
        "brier": base.brier_score(probabilities, labels),
        "nll": base.log_loss(probabilities, labels),
    }


def fit_attempts(
    prepared: list[dict[str, Any]],
    current_models: dict[str, Any],
    optimizer: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    train = [row for row in prepared if row["_role"] == "train"]
    dev = [row for row in prepared if row["_role"] == "dev"]
    attempts: dict[str, Any] = {
        name: {} for name in ("orbit_augmented", "pairwise_margin", "orbit_pairwise", "algebra_basis")
    }
    fit_counts: dict[str, Any] = {}
    for family in FAMILIES:
        family_train = [row for row in train if row["predicate"]["predicate_family"] == family]
        base_model = current_models["family_models"][family]
        x_original = np.asarray(
            [
                existing_vector(
                    base_model,
                    family,
                    row["predicate"]["predicate_label"],
                    row["_raw_numeric"],
                )
                for row in family_train
            ],
            dtype=np.float64,
        )
        y_original = np.asarray([row["_label"] for row in family_train], dtype=np.float64)
        id_to_index = {row["candidate_id"]: index for index, row in enumerate(family_train)}
        pairs: list[tuple[int, int]] = []
        for negative_index, row in enumerate(family_train):
            base_id = (row.get("candidate_source") or {}).get("base_candidate_id")
            if row["_label"] == 0 and base_id in id_to_index:
                positive_index = id_to_index[base_id]
                if family_train[positive_index]["_label"] == 1:
                    pairs.append((positive_index, negative_index))
        pair_diffs = np.asarray(
            [x_original[pos] - x_original[neg] for pos, neg in pairs], dtype=np.float64
        )

        transformed_vectors: list[np.ndarray] = []
        for row in family_train:
            transformed = transformed_view(
                family, row["predicate"]["predicate_label"], row["_raw_numeric"]
            )
            if transformed is not None:
                predicate, raw = transformed
                transformed_vectors.append(existing_vector(base_model, family, predicate, raw))
        if transformed_vectors:
            transformed_x = np.asarray(transformed_vectors, dtype=np.float64)
            orbit_x = np.concatenate((x_original, transformed_x), axis=0)
            orbit_y = np.concatenate((y_original, y_original), axis=0)
            transformed_pair_diffs = np.asarray(
                [transformed_x[pos] - transformed_x[neg] for pos, neg in pairs],
                dtype=np.float64,
            )
            orbit_pair_diffs = np.concatenate((pair_diffs, transformed_pair_diffs), axis=0)
        else:
            orbit_x, orbit_y, orbit_pair_diffs = x_original, y_original, pair_diffs

        if family == "support_contact":
            orbit_weights = np.asarray(base_model["weights"], dtype=np.float64)
            orbit_trace: list[dict[str, float]] = []
        else:
            orbit_weights, orbit_trace = fit_logistic(orbit_x, orbit_y, optimizer)
        pairwise_weights, pairwise_trace = fit_logistic(
            x_original, y_original, optimizer, pair_diffs=pair_diffs
        )
        combined_weights, combined_trace = fit_logistic(
            orbit_x, orbit_y, optimizer, pair_diffs=orbit_pair_diffs
        )
        for name, weights, trace, description in (
            (
                "orbit_augmented",
                orbit_weights,
                orbit_trace,
                "same-label relation-algebra augmentation; support/contact continuity model",
            ),
            (
                "pairwise_margin",
                pairwise_weights,
                pairwise_trace,
                "BCE plus linked-counterfactual pairwise margin",
            ),
            (
                "orbit_pairwise",
                combined_weights,
                combined_trace,
                "relation-algebra augmentation plus linked-counterfactual pairwise margin",
            ),
        ):
            attempts[name][family] = {
                "architecture": "family_logistic",
                "family": family,
                "description": description,
                "feature_names": base_model["feature_names"],
                "numeric_stats": base_model["numeric_stats"],
                "weights": weights.tolist(),
                "training_trace": trace,
                "parameter_count": len(weights),
            }
        if family == "support_contact":
            attempts["algebra_basis"][family] = {
                **base_model,
                "architecture": "support_contact_head_no_endpoint_transform",
                "parameter_count": len(base_model["weights"]),
            }
        else:
            attempts["algebra_basis"][family] = make_basis_model(
                family, family_train, optimizer
            )
        fit_counts[family] = {
            "train_rows": len(family_train),
            "linked_pairs": len(pairs),
            "orbit_rows": len(orbit_y),
            "parameters": len(base_model["weights"]),
        }

    def direct_score(
        condition: str, family: str, predicate: str, raw: dict[str, float]
    ) -> float:
        if condition == "family":
            return existing_probability(current_models["family_models"][family], family, predicate, raw)
        model = attempts[condition][family]
        if condition == "algebra_basis" and family != "support_contact":
            return basis_probability(model, predicate, raw)
        return existing_probability(model, family, predicate, raw)

    scorer = build_scorer(direct_score)
    diagnostics = diagnostics_for_rows(dev, scorer)
    diagnostics["fit_counts"] = fit_counts
    return attempts, diagnostics


def build_scorer(
    direct_score: Callable[[str, str, str, dict[str, float]], float]
) -> Callable[[str, str, str, dict[str, float]], float]:
    def score(condition: str, family: str, predicate: str, raw: dict[str, float]) -> float:
        if condition == "orbit_projection":
            base_score = direct_score("family", family, predicate, raw)
            transformed = transformed_view(family, predicate, raw)
            if transformed is None:
                return base_score
            transformed_predicate, transformed_raw = transformed
            return 0.5 * (
                base_score
                + direct_score("family", family, transformed_predicate, transformed_raw)
            )
        if condition == "orbit_pairwise_projected":
            base_score = direct_score("orbit_pairwise", family, predicate, raw)
            transformed = transformed_view(family, predicate, raw)
            if transformed is None:
                return base_score
            transformed_predicate, transformed_raw = transformed
            return 0.5 * (
                base_score
                + direct_score(
                    "orbit_pairwise", family, transformed_predicate, transformed_raw
                )
            )
        return direct_score(condition, family, predicate, raw)

    return score


def diagnostics_for_rows(
    rows: list[dict[str, Any]],
    score: Callable[[str, str, str, dict[str, float]], float],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "calibration": {},
        "linked_counterfactual": {},
        "transformation": {},
        "wrong_predicate": {},
    }
    for condition in COMPATIBILITIES:
        probabilities: list[float] = []
        labels: list[int] = []
        by_family: dict[str, tuple[list[float], list[int]]] = {
            family: ([], []) for family in FAMILIES
        }
        id_scores: dict[str, float] = {}
        transform_errors: dict[str, list[float]] = {
            "proximity": [], "relative_vertical": []
        }
        wrong_predicate: list[float] = []
        for row in rows:
            family = row["predicate"]["predicate_family"]
            predicate = row["predicate"]["predicate_label"]
            raw = row["_raw_numeric"]
            probability = score(condition, family, predicate, raw)
            probabilities.append(probability)
            labels.append(row["_label"])
            by_family[family][0].append(probability)
            by_family[family][1].append(row["_label"])
            id_scores[row["candidate_id"]] = probability
            transformed = transformed_view(family, predicate, raw)
            if transformed is not None:
                transformed_predicate, transformed_raw = transformed
                transformed_probability = score(
                    condition, family, transformed_predicate, transformed_raw
                )
                transform_errors[family].append(abs(probability - transformed_probability))
            if (
                family == "relative_vertical"
                and row["_label"] == 1
                and predicate in {"higher than", "lower than"}
            ):
                inverse = "lower than" if predicate == "higher than" else "higher than"
                wrong_predicate.append(probability - score(condition, family, inverse, raw))
        result["calibration"][condition] = {
            "overall": calibration_metrics(probabilities, labels),
            "by_family": {
                family: calibration_metrics(*by_family[family]) for family in FAMILIES
            },
        }
        margins: list[float] = []
        for row in rows:
            base_id = (row.get("candidate_source") or {}).get("base_candidate_id")
            if row["_label"] == 0 and base_id in id_scores:
                margins.append(logit(id_scores[base_id]) - logit(id_scores[row["candidate_id"]]))
        result["linked_counterfactual"][condition] = {
            "pairs": len(margins),
            "positive_win_rate": float(np.mean(np.asarray(margins) > 0.0)) if margins else None,
            "mean_logit_margin": float(np.mean(margins)) if margins else None,
            "median_logit_margin": float(np.median(margins)) if margins else None,
        }
        result["transformation"][condition] = {
            family: {
                "rows": len(errors),
                "mean_abs_error": float(np.mean(errors)) if errors else None,
                "p95_abs_error": float(np.percentile(errors, 95)) if errors else None,
                "max_abs_error": float(np.max(errors)) if errors else None,
            }
            for family, errors in transform_errors.items()
        }
        result["wrong_predicate"][condition] = {
            "positive_rows": len(wrong_predicate),
            "correct_win_rate": float(np.mean(np.asarray(wrong_predicate) > 0.0)) if wrong_predicate else None,
            "mean_correct_minus_wrong": float(np.mean(wrong_predicate)) if wrong_predicate else None,
        }
    return result


def load_ground_truth(path: Path) -> tuple[dict[str, set[tuple[Any, ...]]], dict[str, dict[str, set[tuple[Any, ...]]]]]:
    return model_eval.load_gt(path)


def empty_metric_arrays(contexts: list[str]) -> dict[str, dict[str, np.ndarray]]:
    return {
        method: {
            name: np.zeros((len(KS), len(contexts)), dtype=np.float64)
            for name in ("recall_num", "recall_den", "violation_num", "violation_den")
        }
        for method in METHODS
    }


def add_metric_cell(
    target: dict[str, np.ndarray],
    ki: int,
    ci: int,
    selected: list[dict[str, Any]],
    gt: set[tuple[Any, ...]],
) -> None:
    target["recall_num"][ki, ci] = len({row["key"] for row in selected} & gt)
    target["recall_den"][ki, ci] = len(gt)
    statuses = [
        row["status"]
        for row in selected
        if row["status"] in {"satisfied", "uncertain", "violated"}
    ]
    target["violation_num"][ki, ci] = sum(status == "violated" for status in statuses)
    target["violation_den"][ki, ci] = len(statuses)


def metric_contributions(
    grouped: dict[str, list[dict[str, Any]]],
    gt: dict[str, set[tuple[Any, ...]]],
    gt_family: dict[str, dict[str, set[tuple[Any, ...]]]],
    contexts: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    overall = empty_metric_arrays(contexts)
    within = {family: empty_metric_arrays(contexts) for family in FAMILIES}
    global_slice = {family: empty_metric_arrays(contexts) for family in FAMILIES}
    for ci, context in enumerate(contexts):
        candidates = grouped.get(context, [])
        for method in METHODS:
            ranked = sorted(candidates, key=lambda item: (-item["scores"][method], item["key"]))
            for ki, k in enumerate(KS):
                selected = ranked[:k]
                add_metric_cell(overall[method], ki, ci, selected, gt.get(context, set()))
                for family in FAMILIES:
                    family_ranked = [row for row in ranked if row["family"] == family]
                    add_metric_cell(
                        within[family][method],
                        ki,
                        ci,
                        family_ranked[:k],
                        gt_family.get(context, {}).get(family, set()),
                    )
                    add_metric_cell(
                        global_slice[family][method],
                        ki,
                        ci,
                        [row for row in selected if row["family"] == family],
                        gt_family.get(context, {}).get(family, set()),
                    )
    return overall, within, global_slice


def confidence_interval(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    return [float(value) for value in np.percentile(finite, (2.5, 97.5))]


def summarize_metric_arrays(
    values: dict[str, dict[str, np.ndarray]],
    samples: np.ndarray,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {method: {} for method in METHODS}
    bootstrap: dict[str, Any] = {method: {} for method in METHODS}
    for method in METHODS:
        for ki, k in enumerate(KS):
            metrics[method][str(k)] = {}
            bootstrap[method][str(k)] = {}
            for metric in ("recall", "violation"):
                numerator = values[method][f"{metric}_num"][ki]
                denominator = values[method][f"{metric}_den"][ki]
                point = float(numerator.sum() / denominator.sum()) if denominator.sum() else None
                boot_num = numerator[samples].sum(axis=1)
                boot_den = denominator[samples].sum(axis=1)
                boot = np.divide(
                    boot_num,
                    boot_den,
                    out=np.full_like(boot_num, np.nan),
                    where=boot_den > 0,
                )
                metrics[method][str(k)][metric] = {
                    "point": point,
                    "ci95": confidence_interval(boot),
                    "numerator": int(numerator.sum()),
                    "denominator": int(denominator.sum()),
                }
                bootstrap[method][str(k)][metric] = boot
    contrasts: dict[str, Any] = {}
    for left in STRUCTURED_METHODS:
        for right in ("semantic_only", "family_product"):
            name = f"{left}_minus_{right}"
            contrasts[name] = {}
            for k in KS:
                contrasts[name][str(k)] = {}
                for metric in ("recall", "violation"):
                    left_point = metrics[left][str(k)][metric]["point"]
                    right_point = metrics[right][str(k)][metric]["point"]
                    if left_point is None or right_point is None:
                        contrasts[name][str(k)][metric] = {
                            "point": None, "paired_ci95": [None, None]
                        }
                        continue
                    delta = left_point - right_point
                    boot = (
                        bootstrap[left][str(k)][metric]
                        - bootstrap[right][str(k)][metric]
                    )
                    contrasts[name][str(k)][metric] = {
                        "point": delta,
                        "paired_ci95": confidence_interval(boot),
                    }
    return {"metrics": metrics, "contrasts": contrasts}


def evaluate_source(
    source: str,
    path: Path,
    score: Callable[[str, str, str, dict[str, float]], float],
    gt: dict[str, set[tuple[Any, ...]]],
    gt_family: dict[str, dict[str, set[tuple[Any, ...]]]],
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    digest = hashlib.sha256()
    input_rows = 0
    in_scope_rows = 0
    transform_aggregate: dict[str, dict[str, dict[str, float | int]]] = {
        condition: {
            family: {"rows": 0, "sum_abs_error": 0.0, "max_abs_error": 0.0}
            for family in ("proximity", "relative_vertical")
        }
        for condition in COMPATIBILITIES
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
            raw = model_eval.raw_numeric(row)
            semantic = model_eval.finite((row.get("semantic") or {}).get("ranking_score"))
            if semantic is None:
                raise ValueError(f"missing_semantic:{source}:{row['prediction_id']}")
            compatibilities = {
                condition: score(condition, family, predicate, raw)
                for condition in COMPATIBILITIES
            }
            scores = {"semantic_only": semantic}
            for condition, compatibility in compatibilities.items():
                scores[f"{condition}_product" if condition != "family" else "family_product"] = (
                    semantic * compatibility
                )
            transformed = transformed_view(family, predicate, raw)
            if transformed is not None:
                transformed_predicate, transformed_raw = transformed
                for condition in COMPATIBILITIES:
                    error = abs(
                        compatibilities[condition]
                        - score(condition, family, transformed_predicate, transformed_raw)
                    )
                    cell = transform_aggregate[condition][family]
                    cell["rows"] = int(cell["rows"]) + 1
                    cell["sum_abs_error"] = float(cell["sum_abs_error"]) + error
                    cell["max_abs_error"] = max(float(cell["max_abs_error"]), error)
            grouped[row["subgraph_id"]].append(
                {
                    "key": model_eval.candidate_key(row),
                    "family": family,
                    "status": row.get("verification_status")
                    or (row.get("verification") or {}).get("verification_status"),
                    "scores": scores,
                }
            )
    contexts = sorted(set(grouped) | set(gt))
    samples = np.random.default_rng(seed).integers(
        0, len(contexts), size=(n_bootstrap, len(contexts))
    )
    overall, within, global_slice = metric_contributions(
        grouped, gt, gt_family, contexts
    )
    for condition in COMPATIBILITIES:
        for family in ("proximity", "relative_vertical"):
            cell = transform_aggregate[condition][family]
            rows = int(cell["rows"])
            cell["mean_abs_error"] = (
                float(cell["sum_abs_error"]) / rows if rows else None
            )
            cell.pop("sum_abs_error", None)
    return {
        "source": source,
        "input_path": str(path),
        "input_sha256": digest.hexdigest(),
        "input_rows": input_rows,
        "in_scope_rows": in_scope_rows,
        "contexts": len(contexts),
        "gt_denominator": sum(len(rows) for rows in gt.values()),
        "overall": summarize_metric_arrays(overall, samples),
        "within_family": {
            family: summarize_metric_arrays(within[family], samples)
            for family in FAMILIES
        },
        "global_topk_family_slice": {
            family: summarize_metric_arrays(global_slice[family], samples)
            for family in FAMILIES
        },
        "transformation": transform_aggregate,
    }


def gate_summary(
    diagnostics: dict[str, Any],
    source_results: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    baseline_pair_win = diagnostics["linked_counterfactual"]["family"]["positive_win_rate"]
    for compatibility in COMPATIBILITIES[1:]:
        method = f"{compatibility}_product"
        structural_errors: list[float] = []
        for family in ("proximity", "relative_vertical"):
            dev_error = diagnostics["transformation"][compatibility][family]["max_abs_error"]
            if dev_error is not None:
                structural_errors.append(float(dev_error))
            for source in SOURCE_NAMES:
                structural_errors.append(
                    float(source_results[source]["transformation"][compatibility][family]["max_abs_error"])
                )
        structural_pass = bool(structural_errors) and max(structural_errors) <= 1e-10
        source_gates: dict[str, Any] = {}
        all_source_vs_semantic = True
        all_source_recall_continuity = True
        for source in SOURCE_NAMES:
            overall = source_results[source]["overall"]["contrasts"]
            versus_semantic = overall[f"{method}_minus_semantic_only"]["100"]
            versus_family = overall[f"{method}_minus_family_product"]["100"]
            semantic_pass = (
                versus_semantic["recall"]["paired_ci95"][0] >= -0.01
                and versus_semantic["violation"]["paired_ci95"][1] < 0.0
            )
            continuity_pass = versus_family["recall"]["paired_ci95"][0] >= -0.01
            source_gates[source] = {
                "vs_semantic_joint_gate": semantic_pass,
                "vs_family_recall_continuity": continuity_pass,
                "vs_semantic": versus_semantic,
                "vs_family": versus_family,
            }
            all_source_vs_semantic &= semantic_pass
            all_source_recall_continuity &= continuity_pass
        pair_win = diagnostics["linked_counterfactual"][compatibility]["positive_win_rate"]
        pairwise_improved = pair_win is not None and baseline_pair_win is not None and pair_win > baseline_pair_win
        result[compatibility] = {
            "max_structural_error": max(structural_errors) if structural_errors else None,
            "structural_gate": structural_pass,
            "all_source_k100_joint_gate_vs_semantic": all_source_vs_semantic,
            "all_source_k100_recall_continuity_vs_family": all_source_recall_continuity,
            "linked_counterfactual_win_rate": pair_win,
            "linked_counterfactual_improves_over_family": pairwise_improved,
            "meets_selection_criteria": (
                structural_pass
                and all_source_vs_semantic
                and all_source_recall_continuity
                and pairwise_improved
            ),
            "by_source": source_gates,
        }
    return result


def make_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Relation-Algebra Compatibility Development",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is retrospective 3DSSG-only method development. It is not a prospective confirmation or dataset-generalization result.",
        "",
        "## K=100 Overall Results",
        "",
        "| Source | Method | Recall | verifier V |",
        "| --- | --- | ---: | ---: |",
    ]
    for source in SOURCE_NAMES:
        metrics = summary["sources"][source]["overall"]["metrics"]
        for method in METHODS:
            cell = metrics[method]["100"]
            lines.append(
                f"| {source} | {method} | {cell['recall']['point']:.4f} | {cell['violation']['point']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Fixed Attempt Gates",
            "",
            "| Compatibility | max algebra error | all-source joint vs source | recall continuity vs family | pair win improved | novelty gate |",
            "| --- | ---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for condition, gate in summary["gates"].items():
        lines.append(
            f"| {condition} | {gate['max_structural_error']:.3e} | "
            f"{'pass' if gate['all_source_k100_joint_gate_vs_semantic'] else 'fail'} | "
            f"{'pass' if gate['all_source_k100_recall_continuity_vs_family'] else 'fail'} | "
            f"{'pass' if gate['linked_counterfactual_improves_over_family'] else 'fail'} | "
            f"{'pass' if gate['meets_selection_criteria'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "A passing condition may strengthen RelCompat3D from generic calibration to relation-algebra-constrained compatibility. It does not establish formula optimality, independent physical validity, family-uniform improvement, or generalization beyond 3DSSG.",
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
    paths = {
        name: resolve(root, value) for name, value in protocol["inputs"].items()
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing_inputs:{missing}")
    train_scans = read_scans(paths["train_scans"])
    dev_scans = read_scans(paths["development_scans"])
    final_scans = read_scans(paths["final_validation_scans"])
    if train_scans & dev_scans or train_scans & final_scans or dev_scans & final_scans:
        raise ValueError("data_split_overlap")

    table_rows = base.load_jsonl(paths["calibration_table"])
    leaked = sorted({row["scan_id"] for row in table_rows} & final_scans)
    if leaked:
        raise ValueError(f"final_validation_rows_in_calibration:{leaked[:10]}")
    prepared, warnings = base.prepare_rows(
        table_rows, train_scans, dev_scans, set(FAMILIES)
    )
    current_models = json.loads(paths["current_models"].read_text(encoding="utf-8"))
    attempts, diagnostics = fit_attempts(
        prepared, current_models, protocol["optimizer"]
    )

    def direct_score(
        condition: str, family: str, predicate: str, raw: dict[str, float]
    ) -> float:
        if condition == "family":
            return existing_probability(
                current_models["family_models"][family], family, predicate, raw
            )
        model = attempts[condition][family]
        if condition == "algebra_basis" and family != "support_contact":
            return basis_probability(model, predicate, raw)
        return existing_probability(model, family, predicate, raw)

    scorer = build_scorer(direct_score)
    gt, gt_family = load_ground_truth(paths["ground_truth"])
    internal_gt, internal_gt_family = load_ground_truth(
        paths["development_ground_truth"]
    )
    development_source = evaluate_source(
        "development_sgfn",
        paths["development_verification"],
        scorer,
        internal_gt,
        internal_gt_family,
        int(protocol["evaluation"]["bootstrap_seed"]) - 1,
        int(protocol["evaluation"]["bootstrap_resamples"]),
    )
    source_paths = {
        "vlsat": paths["vlsat_verification"],
        "open3dsg": paths["open3dsg_verification"],
        "sgfn": paths["sgfn_verification"],
    }
    source_results: dict[str, Any] = {}
    for index, source in enumerate(SOURCE_NAMES):
        source_results[source] = evaluate_source(
            source,
            source_paths[source],
            scorer,
            gt,
            gt_family,
            int(protocol["evaluation"]["bootstrap_seed"]) + index,
            int(protocol["evaluation"]["bootstrap_resamples"]),
        )
    gates = gate_summary(diagnostics, source_results)
    validations = {
        "split_counts_1061_117_157": (len(train_scans), len(dev_scans), len(final_scans)) == (1061, 117, 157),
        "split_sets_pairwise_disjoint": not (train_scans & dev_scans or train_scans & final_scans or dev_scans & final_scans),
        "zero_final_rows_in_calibration": not leaked,
        "train_rows_60208": sum(row["_role"] == "train" for row in prepared) == 60208,
        "development_rows_6246": sum(row["_role"] == "dev" for row in prepared) == 6246,
        "source_contexts_548": all(result["contexts"] == 548 for result in source_results.values()),
        "gt_denominator_3972": all(result["gt_denominator"] == 3972 for result in source_results.values()),
        "vlsat_rows_220848": source_results["vlsat"]["in_scope_rows"] == 220848,
        "open3dsg_rows_160596": source_results["open3dsg"]["in_scope_rows"] == 160596,
        "sgfn_rows_220848": source_results["sgfn"]["in_scope_rows"] == 220848,
        "development_contexts_354": development_source["contexts"] == 354,
        "development_gt_denominator_2730": development_source["gt_denominator"] == 2730,
        "development_rows_139368": development_source["in_scope_rows"] == 139368,
        "all_attempts_reported": set(gates) == set(COMPATIBILITIES[1:]),
        "all_parameters_finite": all(
            math.isfinite(weight)
            for attempt in attempts.values()
            for model in attempt.values()
            for weight in model["weights"]
        ),
        "no_source_features_in_attempts": all(
            not any(token in feature.lower() for token in ("semantic", "source", "score", "rank", "baseline"))
            for attempt in attempts.values()
            for model in attempt.values()
            for feature in model["feature_names"]
        ),
    }
    status = "completed" if all(validations.values()) else "failed_validation"
    out.mkdir(parents=True, exist_ok=True)
    models_path = out / "models.json"
    diagnostics_path = out / "diagnostics.json"
    metrics_path = out / "source_metrics.json"
    summary_path = out / "summary.json"
    write_json(
        models_path,
        {
            "schema_version": "relcompat3d_relation_algebra_models_v1",
            "attempts": attempts,
            "source_score_used": False,
            "source_identity_used": False,
        },
    )
    write_json(
        diagnostics_path,
        {
            "schema_version": "relcompat3d_relation_algebra_diagnostics_v1",
            "role": "training_and_development_metrics",
            "diagnostics": diagnostics,
        },
    )
    write_json(metrics_path, source_results)
    summary = {
        "schema_version": "relcompat3d_relcompat3d_fit_summary_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "classification": protocol["classification"],
        "sources": source_results,
        "development_source": development_source,
        "gates": gates,
        "validations": validations,
        "warnings": warnings,
        "evaluation_scope": protocol["evaluation_scope"],
    }
    write_json(summary_path, summary)
    markdown_path = out / "summary.md"
    markdown_path.write_text(make_summary_markdown(summary), encoding="utf-8")
    compact_inputs: dict[str, Any] = {}
    source_input_keys = {
        "development_verification": "development_sgfn",
        "vlsat_verification": "vlsat",
        "open3dsg_verification": "open3dsg",
        "sgfn_verification": "sgfn",
    }
    for name, path in paths.items():
        if name in source_input_keys:
            source_key = source_input_keys[name]
            source_payload = (
                development_source
                if source_key == "development_sgfn"
                else source_results[source_key]
            )
            compact_inputs[name] = {
                "path": relpath(root, path),
                "size_bytes": path.stat().st_size,
                "sha256": source_payload["input_sha256"],
            }
        else:
            compact_inputs[name] = {
                "path": relpath(root, path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    manifest = {
        "schema_version": "relcompat3d_relcompat3d_fit_manifest_v1",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": relpath(root, protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "inputs": compact_inputs,
        "outputs": {
            path.name: {
                "path": relpath(root, path),
                "sha256": sha256_file(path),
            }
            for path in (models_path, diagnostics_path, metrics_path, summary_path, markdown_path)
        },
        "validations": validations,
        "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm relcompat3d_fit",
    }
    write_json(out / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": status,
                "gates": {
                    name: value["meets_selection_criteria"] for name, value in gates.items()
                },
                "out": relpath(root, out),
            },
            sort_keys=True,
        )
    )
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
