#!/usr/bin/env python3
"""RelCompat3D prediction-level metrics.

This computes semantic-only recall when prediction JSONL exists. Rule-verified
and probabilistic recalibration metrics are computed only if a verification
JSONL with geometry/calibration outputs is provided.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from paths import RelCompat3D_HYPOTHESIS_ROOT


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
RelCompat3D_ROOT = RelCompat3D_HYPOTHESIS_ROOT
DEFAULT_INPUT_DIR = RelCompat3D_ROOT / "artifacts" / "evaluation" / "vlsat_closed_set" / "mini"
DEFAULT_OUTPUT_DIR = RelCompat3D_ROOT / "artifacts" / "evaluation" / "vlsat_closed_set" / "mini_metrics"
DEFAULT_FAMILIES = ("support_contact", "proximity", "relative_vertical")
DEFAULT_KS = (50, 100)
DEFAULT_RULE_VARIANTS = ("obb_only", "point_subtype", "point_subtype_no_soft_support")
DEFAULT_ABLATION_CONTROLS = (
    "p_geom_valid_only",
    "distance_only",
    "family_specific_p_geom_valid",
    "shuffled_geometry",
    "wrong_pair_geometry",
)
CALIBRATION_NUMERIC_FEATURES = (
    "distance_3d",
    "distance_xy",
    "normalized_distance_3d",
    "normalized_distance_xy",
    "center_delta_z",
    "normalized_center_delta_z",
    "projected_iou_xy",
    "projected_subject_overlap_ratio",
    "projected_object_overlap_ratio",
    "vertical_gap_subject_on_object",
    "subject_bottom_z",
    "subject_top_z",
    "object_bottom_z",
    "object_top_z",
)
CONTROL_GEOMETRY_FEATURES = CALIBRATION_NUMERIC_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RelCompat3D prediction JSONL skeleton.")
    parser.add_argument("--predictions-jsonl", type=Path, default=DEFAULT_INPUT_DIR / "predictions.jsonl")
    parser.add_argument("--ground-truth-jsonl", type=Path, default=DEFAULT_INPUT_DIR / "ground_truth.jsonl")
    parser.add_argument("--verification-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    parser.add_argument("--policy", choices=["filter_safe", "filter_strict"], default="filter_safe")
    parser.add_argument(
        "--rule-variants",
        nargs="*",
        default=[],
        help=(
            "Optional verification_variants to evaluate as separate rule_verified_<variant> "
            "conditions. Example: obb_only point_subtype point_subtype_no_soft_support"
        ),
    )
    parser.add_argument(
        "--ablation-controls",
        nargs="*",
        default=[],
        help=(
            "Optional G3 controls to evaluate. Supported: "
            f"{', '.join(DEFAULT_ABLATION_CONTROLS)}"
        ),
    )
    parser.add_argument(
        "--family-specific-model-json",
        type=Path,
        help="Optional p_geom_valid family-specific calibrator from fit_family_calibration.py.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return records


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def prediction_key(row: dict[str, Any]) -> tuple[str, int, int, int, str]:
    return (
        row["scan_id"],
        int(row["subset_split_id"]),
        int(row["edge"]["subject_id"]),
        int(row["edge"]["object_id"]),
        row["predicate"]["predicate_label"],
    )


def gt_key(row: dict[str, Any]) -> tuple[str, int, int, int, str]:
    return (
        row["scan_id"],
        int(row["subset_split_id"]),
        int(row["subject_id"]),
        int(row["object_id"]),
        row["predicate_label"],
    )


def in_scope_prediction(row: dict[str, Any], families: set[str]) -> bool:
    return row["predicate"]["predicate_family"] in families


def in_scope_gt(row: dict[str, Any], families: set[str]) -> bool:
    return row["predicate_family"] in families


def sorted_by_subgraph(predictions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        groups[row["subgraph_id"]].append(row)
    for rows in groups.values():
        rows.sort(
            key=lambda row: (
                -float(row["scores"]["ranking_score"]),
                int(row["edge"]["subject_id"]),
                int(row["edge"]["object_id"]),
                row["predicate"]["predicate_label"],
            )
        )
    return groups


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def recall_at_k(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    ks: list[int],
    families: set[str],
) -> dict[str, Any]:
    scoped_gt = [row for row in ground_truth if in_scope_gt(row, families)]
    gt_keys = {gt_key(row) for row in scoped_gt}
    predictions = [row for row in predictions if in_scope_prediction(row, families)]
    grouped = sorted_by_subgraph(predictions)

    result: dict[str, Any] = {
        "denominator": len(gt_keys),
        "by_k": {},
        "by_family": {},
        "by_predicate_label": {},
    }
    for k in ks:
        selected_keys = set()
        for rows in grouped.values():
            selected_keys.update(prediction_key(row) for row in rows[:k])
        correct = len(selected_keys & gt_keys)
        result["by_k"][str(k)] = {
            "correct": correct,
            "recall": correct / len(gt_keys) if gt_keys else None,
            "selected_predictions": len(selected_keys),
        }

    for family in sorted(families):
        family_gt = [row for row in scoped_gt if row["predicate_family"] == family]
        family_gt_keys = {gt_key(row) for row in family_gt}
        family_predictions = [
            row for row in predictions if row["predicate"]["predicate_family"] == family
        ]
        family_grouped = sorted_by_subgraph(family_predictions)
        result["by_family"][family] = {}
        for k in ks:
            selected_keys = set()
            for rows in family_grouped.values():
                selected_keys.update(prediction_key(row) for row in rows[:k])
            correct = len(selected_keys & family_gt_keys)
            result["by_family"][family][str(k)] = {
                "denominator": len(family_gt_keys),
                "correct": correct,
                "recall": correct / len(family_gt_keys) if family_gt_keys else None,
            }

    labels = sorted({row["predicate_label"] for row in scoped_gt})
    for label in labels:
        label_gt_keys = {gt_key(row) for row in scoped_gt if row["predicate_label"] == label}
        label_predictions = [
            row for row in predictions if row["predicate"]["predicate_label"] == label
        ]
        label_grouped = sorted_by_subgraph(label_predictions)
        label_scores = []
        result["by_predicate_label"][label] = {}
        for k in ks:
            selected_keys = set()
            for rows in label_grouped.values():
                selected_keys.update(prediction_key(row) for row in rows[:k])
            correct = len(selected_keys & label_gt_keys)
            recall = correct / len(label_gt_keys) if label_gt_keys else None
            result["by_predicate_label"][label][str(k)] = {
                "denominator": len(label_gt_keys),
                "correct": correct,
                "recall": recall,
            }
            if recall is not None:
                label_scores.append(recall)
        if label_scores:
            result["by_predicate_label"][label]["mean_over_k"] = sum(label_scores) / len(label_scores)

    for k in ks:
        label_recalls = [
            data[str(k)]["recall"]
            for data in result["by_predicate_label"].values()
            if data[str(k)]["recall"] is not None
        ]
        result["by_k"][str(k)]["macro_predicate_recall"] = (
            sum(label_recalls) / len(label_recalls) if label_recalls else None
        )

    return result


def compact_verification(row: dict[str, Any]) -> dict[str, Any]:
    variants = {}
    for name, variant in (row.get("verification_variants") or {}).items():
        variants[name] = {
            "verification_status": variant.get("verification_status"),
            "consistency_score": variant.get("consistency_score"),
        }
    features = {}
    for name, value in ((row.get("geometry") or {}).get("features") or {}).items():
        if name in CONTROL_GEOMETRY_FEATURES:
            number = finite_float(value)
            if number is not None:
                features[name] = number
    return {
        "verification_status": row.get("verification_status"),
        "consistency_score": row.get("consistency_score"),
        "calibration": row.get("calibration", {}),
        "geometry_features": features,
        "verification_variants": variants,
    }


def load_verification(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            records[row["prediction_id"]] = compact_verification(row)
    return records


def verification_status(verification: dict[str, Any] | None, variant: str | None = None) -> str | None:
    if verification is None:
        return None
    if variant is None:
        return verification.get("verification_status")
    return (
        verification.get("verification_variants", {})
        .get(variant, {})
        .get("verification_status")
    )


def keep_under_policy(status: str | None, policy: str) -> bool:
    if policy == "filter_strict":
        return status == "satisfied"
    if policy == "filter_safe":
        return status in {"satisfied", "uncertain"}
    raise ValueError(policy)


def semantic_score(row: dict[str, Any]) -> float | None:
    score = finite_float(row.get("scores", {}).get("ranking_score"))
    if score is None:
        score = finite_float(row.get("scores", {}).get("predicate_score"))
    return score


def copy_with_ranking_score(
    row: dict[str, Any],
    score: float,
    score_type: str,
    extra_scores: dict[str, Any] | None = None,
) -> dict[str, Any]:
    copy = dict(row)
    scores = dict(row.get("scores", {}))
    scores["semantic_ranking_score"] = semantic_score(row)
    scores["ranking_score"] = score
    scores["ranking_score_type"] = score_type
    if extra_scores:
        scores.update(extra_scores)
    copy["scores"] = scores
    return copy


def p_geom_valid_from_verification(
    verification: dict[str, Any] | None,
    field_name: str = "p_geom_valid",
) -> float | None:
    if verification is None:
        return None
    return finite_float((verification.get("calibration") or {}).get(field_name))


def inverse_distance_score(verification: dict[str, Any] | None) -> float | None:
    if verification is None:
        return None
    features = verification.get("geometry_features", {})
    distance = finite_float(features.get("distance_3d"))
    if distance is None:
        distance = finite_float(features.get("normalized_distance_3d"))
    if distance is None:
        return None
    return 1.0 / (1.0 + max(distance, 0.0))


def raw_calibration_features(row: dict[str, Any], verification: dict[str, Any]) -> dict[str, float]:
    source = verification.get("geometry_features", {})
    predicate = row["predicate"]["predicate_label"]
    values: dict[str, float] = {}
    for name in CALIBRATION_NUMERIC_FEATURES:
        value = finite_float(source.get(name))
        if value is not None:
            values[name] = value
    if "center_delta_z" in values:
        values["abs_center_delta_z"] = abs(values["center_delta_z"])
    if "normalized_center_delta_z" in values:
        values["abs_normalized_center_delta_z"] = abs(values["normalized_center_delta_z"])
    if "vertical_gap_subject_on_object" in values:
        values["abs_vertical_gap_subject_on_object"] = abs(
            values["vertical_gap_subject_on_object"]
        )
    direction = 0.0
    if predicate == "higher than":
        direction = 1.0
    elif predicate == "lower than":
        direction = -1.0
    if direction and "center_delta_z" in values:
        values["predicate_aligned_center_delta_z"] = direction * values["center_delta_z"]
    if direction and "normalized_center_delta_z" in values:
        values["predicate_aligned_normalized_center_delta_z"] = (
            direction * values["normalized_center_delta_z"]
        )
    return values


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def dot(weights: list[float], vector: list[float]) -> float:
    return sum(weight * value for weight, value in zip(weights, vector))


def family_specific_p_geom_valid(
    row: dict[str, Any],
    verification: dict[str, Any] | None,
    family_specific_model: dict[str, Any] | None,
) -> float | None:
    if verification is None or family_specific_model is None:
        return None
    family = row["predicate"]["predicate_family"]
    family_model = family_specific_model.get("family_models", {}).get(family)
    if family_model is None:
        return None
    raw = raw_calibration_features(row, verification)
    vector = [1.0]
    for name in family_model["numeric_features"]:
        stats = family_model["numeric_stats"][name]
        value = raw.get(name, stats["mean"])
        std = stats["std"] or 1.0
        vector.append((value - stats["mean"]) / std)
    vector.extend(1.0 if family == name else 0.0 for name in family_model["families"])
    predicate = row["predicate"]["predicate_label"]
    vector.extend(1.0 if predicate == name else 0.0 for name in family_model["predicates"])
    return sigmoid(dot(family_model["weights"], vector))


def pair_key(row: dict[str, Any]) -> tuple[int, int]:
    return (int(row["edge"]["subject_id"]), int(row["edge"]["object_id"]))


def shifted_p_geom_by_family(
    predictions: list[dict[str, Any]],
    verification_by_id: dict[str, dict[str, Any]],
    families: set[str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        family = row["predicate"]["predicate_family"]
        if family in families and p_geom_valid_from_verification(
            verification_by_id.get(row["prediction_id"])
        ) is not None:
            grouped[family].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["prediction_id"])
        if len(rows) < 2:
            continue
        shift = max(1, len(rows) // 2)
        donor_values = [
            p_geom_valid_from_verification(verification_by_id[row["prediction_id"]])
            for row in rows
        ]
        for index, row in enumerate(rows):
            donor = donor_values[(index + shift) % len(rows)]
            if donor is not None:
                result[row["prediction_id"]] = donor
    return result


def shifted_p_geom_by_wrong_pair(
    predictions: list[dict[str, Any]],
    verification_by_id: dict[str, dict[str, Any]],
    families: set[str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        family = row["predicate"]["predicate_family"]
        if family in families and p_geom_valid_from_verification(
            verification_by_id.get(row["prediction_id"])
        ) is not None:
            key = (
                row["scan_id"],
                int(row["subset_split_id"]),
                row["subgraph_id"],
                family,
            )
            grouped[key].append(row)
    family_fallback = shifted_p_geom_by_family(predictions, verification_by_id, families)
    for rows in grouped.values():
        rows.sort(
            key=lambda row: (
                pair_key(row),
                row["predicate"]["predicate_label"],
                row["prediction_id"],
            )
        )
        if len({pair_key(row) for row in rows}) < 2:
            continue
        for index, row in enumerate(rows):
            for offset in range(1, len(rows)):
                donor_row = rows[(index + offset) % len(rows)]
                if pair_key(donor_row) == pair_key(row):
                    continue
                donor = p_geom_valid_from_verification(
                    verification_by_id.get(donor_row["prediction_id"])
                )
                if donor is not None:
                    result[row["prediction_id"]] = donor
                    break
    for prediction_id, donor in family_fallback.items():
        result.setdefault(prediction_id, donor)
    return result


def control_condition_name(control_name: str) -> str:
    return f"control_{control_name}"


def ablation_control_predictions(
    predictions: list[dict[str, Any]],
    verification_by_id: dict[str, dict[str, Any]],
    families: set[str],
    control_name: str,
    family_specific_model: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    if control_name not in DEFAULT_ABLATION_CONTROLS:
        return [], {"control": control_name}, [f"unknown_ablation_control:{control_name}"]

    summary: dict[str, Any] = {
        "control": control_name,
        "input_predictions": len(predictions),
        "in_scope_predictions": 0,
        "scored_predictions": 0,
        "missing_verification": 0,
        "missing_score": 0,
    }
    errors: list[str] = []
    output: list[dict[str, Any]] = []
    donor_map: dict[str, float] = {}
    if control_name == "shuffled_geometry":
        donor_map = shifted_p_geom_by_family(predictions, verification_by_id, families)
        summary["score_formula"] = "semantic_ranking_score*shuffled_family_p_geom_valid"
        summary["shuffle_policy"] = "deterministic_half_rotation_within_predicate_family"
    elif control_name == "wrong_pair_geometry":
        donor_map = shifted_p_geom_by_wrong_pair(predictions, verification_by_id, families)
        summary["score_formula"] = "semantic_ranking_score*wrong_pair_p_geom_valid"
        summary["shuffle_policy"] = "deterministic_rotation_within_subgraph_family_different_pair"
    elif control_name == "p_geom_valid_only":
        summary["score_formula"] = "p_geom_valid"
    elif control_name == "distance_only":
        summary["score_formula"] = "1/(1+distance_3d)"
    elif control_name == "family_specific_p_geom_valid":
        summary["score_formula"] = "semantic_ranking_score*p_geom_valid_family_specific"
        if family_specific_model is not None:
            summary["model_id"] = family_specific_model.get("model_id")
            summary["model_schema_version"] = family_specific_model.get("schema_version")

    for row in predictions:
        family = row["predicate"]["predicate_family"]
        if family not in families:
            output.append(row)
            continue

        summary["in_scope_predictions"] += 1
        prediction_id = row["prediction_id"]
        verification = verification_by_id.get(prediction_id)
        if verification is None:
            summary["missing_verification"] += 1
            output.append(row)
            continue

        semantic = semantic_score(row)
        score: float | None = None
        extra: dict[str, Any] = {}
        if control_name == "p_geom_valid_only":
            score = p_geom_valid_from_verification(verification)
            extra["p_geom_valid"] = score
        elif control_name == "distance_only":
            score = inverse_distance_score(verification)
            extra["distance_only_score"] = score
        elif control_name == "family_specific_p_geom_valid":
            p_family = p_geom_valid_from_verification(
                verification, field_name="p_geom_valid_family_specific"
            )
            if p_family is None:
                p_family = family_specific_p_geom_valid(row, verification, family_specific_model)
            if semantic is not None and p_family is not None:
                score = semantic * p_family
            extra["p_geom_valid_family_specific"] = p_family
        elif control_name in {"shuffled_geometry", "wrong_pair_geometry"}:
            donor = donor_map.get(prediction_id)
            if semantic is not None and donor is not None:
                score = semantic * donor
            extra[f"{control_name}_p_geom_valid"] = donor

        if score is None:
            summary["missing_score"] += 1
            output.append(row)
            continue
        output.append(
            copy_with_ranking_score(
                row,
                score,
                str(summary["score_formula"]),
                extra_scores=extra,
            )
        )
        summary["scored_predictions"] += 1

    if summary["missing_verification"]:
        errors.append(f"{control_name}:missing_verification:{summary['missing_verification']}")
    if summary["missing_score"]:
        errors.append(f"{control_name}:missing_score:{summary['missing_score']}")
    return output, summary, errors


def apply_rule_filter(
    predictions: list[dict[str, Any]],
    verification_by_id: dict[str, dict[str, Any]],
    policy: str,
    variant: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    missing = 0
    for row in predictions:
        verification = verification_by_id.get(row["prediction_id"])
        if verification is None:
            missing += 1
            continue
        status = verification_status(verification, variant)
        if status is None:
            missing += 1
            continue
        status_counts[str(status)] += 1
        if keep_under_policy(status, policy):
            kept.append(row)
    return kept, {
        "policy": policy,
        "verification_variant": variant,
        "input_predictions": len(predictions),
        "kept_predictions": len(kept),
        "missing_verification": missing,
        "status_counts": dict(sorted(status_counts.items())),
    }


def recalibrated_predictions(
    predictions: list[dict[str, Any]],
    verification_by_id: dict[str, dict[str, Any]],
    families: set[str],
    status_variant: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    recalibrated: list[dict[str, Any]] = []
    errors: list[str] = []
    p_geom_values: list[float] = []
    p_geom_by_family: dict[str, list[float]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    missing_verification = 0
    missing_p_geom_valid = 0
    missing_semantic_score = 0
    in_scope_count = 0
    scored_count = 0

    for row in predictions:
        family = row["predicate"]["predicate_family"]
        if family not in families:
            recalibrated.append(row)
            continue

        in_scope_count += 1
        prediction_id = row["prediction_id"]
        verification = verification_by_id.get(prediction_id)
        if verification is None:
            missing_verification += 1
            recalibrated.append(row)
            continue

        status_counts[str(verification_status(verification, status_variant))] += 1
        calibration = verification.get("calibration", {})
        p_geom_valid = finite_float(calibration.get("p_geom_valid"))
        semantic_score_value = semantic_score(row)
        if p_geom_valid is None:
            missing_p_geom_valid += 1
            recalibrated.append(row)
            continue
        if semantic_score_value is None:
            missing_semantic_score += 1
            recalibrated.append(row)
            continue

        recalibrated_score = semantic_score_value * p_geom_valid
        recalibrated.append(
            copy_with_ranking_score(
                row,
                recalibrated_score,
                "semantic_ranking_score*p_geom_valid",
                extra_scores={"p_geom_valid": p_geom_valid},
            )
        )
        scored_count += 1
        p_geom_values.append(p_geom_valid)
        p_geom_by_family[family].append(p_geom_valid)

    if missing_verification:
        errors.append(f"missing_verification_for_in_scope_predictions:{missing_verification}")
    if missing_p_geom_valid:
        errors.append(f"missing_p_geom_valid_for_in_scope_predictions:{missing_p_geom_valid}")
    if missing_semantic_score:
        errors.append(f"missing_semantic_score_for_in_scope_predictions:{missing_semantic_score}")

    summary = {
        "score_formula": "semantic_ranking_score*p_geom_valid",
        "verification_status_variant": status_variant,
        "input_predictions": len(predictions),
        "in_scope_predictions": in_scope_count,
        "scored_predictions": scored_count,
        "missing_verification": missing_verification,
        "missing_p_geom_valid": missing_p_geom_valid,
        "missing_semantic_score": missing_semantic_score,
        "verification_status_counts": dict(sorted(status_counts.items())),
        "p_geom_valid": summarize_values(p_geom_values),
        "p_geom_valid_by_family": {
            family: summarize_values(values)
            for family, values in sorted(p_geom_by_family.items())
        },
    }
    return recalibrated, summary, errors


def violation_rate_at_k(
    predictions: list[dict[str, Any]],
    verification_by_id: dict[str, dict[str, Any]],
    ks: list[int],
    families: set[str],
    variant: str | None = None,
) -> dict[str, Any]:
    scoped = [row for row in predictions if in_scope_prediction(row, families)]
    grouped = sorted_by_subgraph(scoped)
    result: dict[str, Any] = {"by_k": {}}
    for k in ks:
        considered = []
        for rows in grouped.values():
            considered.extend(rows[:k])
        statuses = [
            verification_status(verification_by_id[row["prediction_id"]], variant)
            for row in considered
            if row["prediction_id"] in verification_by_id
        ]
        denom = len([status for status in statuses if status in {"satisfied", "uncertain", "violated"}])
        violated = len([status for status in statuses if status == "violated"])
        result["by_k"][str(k)] = {
            "denominator": denom,
            "violated": violated,
            "violation_rate": violated / denom if denom else None,
            "geometry_coverage": len(statuses) / len(considered) if considered else None,
            "verification_variant": variant,
        }
    return result


def rule_condition_name(variant: str | None) -> str:
    return "rule_verified" if variant is None else f"rule_verified_{variant}"


def make_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Prediction Metrics",
        "",
        f"Created at: `{metrics['created_at']}`",
        f"Status: `{metrics['status']}`",
        f"Families: `{', '.join(metrics['families'])}`",
        f"K values: `{', '.join(str(k) for k in metrics['ks'])}`",
        "",
        "## Semantic Only",
        "",
    ]
    semantic = metrics["conditions"]["semantic_only"]
    for k, value in semantic["recall"]["by_k"].items():
        lines.append(
            f"- R@{k}: `{value['recall']}` "
            f"({value['correct']}/{semantic['recall']['denominator']})"
        )
    if "violation_rate" in semantic:
        for k, value in semantic["violation_rate"]["by_k"].items():
            lines.append(
                f"- Violation@{k}: `{value['violation_rate']}` "
                f"({value['violated']}/{value['denominator']})"
            )
    for condition_name in sorted(
        name for name in metrics["conditions"] if name.startswith("rule_verified")
    ):
        lines.extend(["", f"## {condition_name}", ""])
        rule = metrics["conditions"][condition_name]
        lines.append(f"- Policy: `{rule['filter']['policy']}`")
        if rule["filter"].get("verification_variant"):
            lines.append(f"- Variant: `{rule['filter']['verification_variant']}`")
        lines.append(f"- Kept: `{rule['filter']['kept_predictions']}` / `{rule['filter']['input_predictions']}`")
        for k, value in rule["recall"]["by_k"].items():
            lines.append(
                f"- R@{k}: `{value['recall']}` "
                f"({value['correct']}/{rule['recall']['denominator']})"
            )
        for k, value in rule["violation_rate"]["by_k"].items():
            lines.append(
                f"- Violation@{k}: `{value['violation_rate']}` "
                f"({value['violated']}/{value['denominator']})"
            )
    if "probabilistic_recalibrated" in metrics["conditions"]:
        lines.extend(["", "## Probabilistic Recalibrated", ""])
        probabilistic = metrics["conditions"]["probabilistic_recalibrated"]
        summary = probabilistic["score_summary"]
        lines.append(f"- Score formula: `{summary['score_formula']}`")
        lines.append(
            f"- Scored: `{summary['scored_predictions']}` / "
            f"`{summary['in_scope_predictions']}` in-scope predictions"
        )
        for k, value in probabilistic["recall"]["by_k"].items():
            lines.append(
                f"- R@{k}: `{value['recall']}` "
                f"({value['correct']}/{probabilistic['recall']['denominator']})"
            )
        for k, value in probabilistic["violation_rate"]["by_k"].items():
            lines.append(
                f"- Violation@{k}: `{value['violation_rate']}` "
                f"({value['violated']}/{value['denominator']})"
            )
    for condition_name in sorted(
        name for name in metrics["conditions"] if name.startswith("control_")
    ):
        lines.extend(["", f"## {condition_name}", ""])
        control = metrics["conditions"][condition_name]
        summary = control["score_summary"]
        lines.append(f"- Score formula: `{summary.get('score_formula')}`")
        lines.append(
            f"- Scored: `{summary['scored_predictions']}` / "
            f"`{summary['in_scope_predictions']}` in-scope predictions"
        )
        if control.get("blocked"):
            lines.append(f"- Blocked: `{control['blocked']}`")
            continue
        for k, value in control["recall"]["by_k"].items():
            lines.append(
                f"- R@{k}: `{value['recall']}` "
                f"({value['correct']}/{control['recall']['denominator']})"
            )
        for k, value in control["violation_rate"]["by_k"].items():
            lines.append(
                f"- Violation@{k}: `{value['violation_rate']}` "
                f"({value['violated']}/{value['denominator']})"
            )
    if metrics["blocked"]:
        lines.extend(["", "## Blocked", ""])
        for item in metrics["blocked"]:
            lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    blocked: list[str] = []
    for name, path in {
        "predictions_jsonl": args.predictions_jsonl,
        "ground_truth_jsonl": args.ground_truth_jsonl,
    }.items():
        if not path.exists():
            errors.append(f"missing_input:{name}:{relpath(path)}")
    if args.verification_jsonl is not None and not args.verification_jsonl.exists():
        errors.append(f"missing_input:verification_jsonl:{relpath(args.verification_jsonl)}")
    if args.family_specific_model_json is not None and not args.family_specific_model_json.exists():
        errors.append(
            f"missing_input:family_specific_model_json:{relpath(args.family_specific_model_json)}"
        )
    if errors:
        print(json.dumps({"status": "blocked", "errors": errors}, sort_keys=True))
        return 2

    families = set(args.families)
    predictions = load_jsonl(args.predictions_jsonl)
    ground_truth = load_jsonl(args.ground_truth_jsonl)
    verification_by_id = load_verification(args.verification_jsonl)
    family_specific_model = (
        load_json(args.family_specific_model_json)
        if args.family_specific_model_json is not None
        else None
    )

    metrics: dict[str, Any] = {
        "schema_version": "relcompat3d_prediction_metrics_v2",
        "created_at": date.today().isoformat(),
        "status": "ready",
        "families": sorted(families),
        "ks": args.ks,
        "inputs": {
            "predictions_jsonl": relpath(args.predictions_jsonl),
            "ground_truth_jsonl": relpath(args.ground_truth_jsonl),
            "verification_jsonl": relpath(args.verification_jsonl) if args.verification_jsonl else None,
            "family_specific_model_json": (
                relpath(args.family_specific_model_json)
                if args.family_specific_model_json
                else None
            ),
        },
        "counts": {
            "predictions": len(predictions),
            "ground_truth": len(ground_truth),
            "predictions_by_family": dict(
                sorted(Counter(row["predicate"]["predicate_family"] for row in predictions).items())
            ),
            "ground_truth_by_family": dict(
                sorted(Counter(row["predicate_family"] for row in ground_truth).items())
            ),
        },
        "conditions": {},
        "blocked": blocked,
    }

    metrics["conditions"]["semantic_only"] = {
        "recall": recall_at_k(predictions, ground_truth, args.ks, families)
    }

    if verification_by_id:
        metrics["conditions"]["semantic_only"]["violation_rate"] = violation_rate_at_k(
            predictions, verification_by_id, args.ks, families
        )
        rule_variants = args.rule_variants
        if rule_variants:
            available_variants = set()
            for verification in verification_by_id.values():
                available_variants.update(verification.get("verification_variants", {}).keys())
                if available_variants:
                    break
            missing_variants = sorted(set(rule_variants) - available_variants)
            if missing_variants:
                blocked.append(f"missing_rule_variants:{missing_variants}")
            for variant in rule_variants:
                rule_predictions, filter_summary = apply_rule_filter(
                    predictions, verification_by_id, args.policy, variant=variant
                )
                metrics["conditions"][rule_condition_name(variant)] = {
                    "filter": filter_summary,
                    "recall": recall_at_k(rule_predictions, ground_truth, args.ks, families),
                    "violation_rate": violation_rate_at_k(
                        rule_predictions,
                        verification_by_id,
                        args.ks,
                        families,
                        variant=variant,
                    ),
                }
        else:
            rule_predictions, filter_summary = apply_rule_filter(
                predictions, verification_by_id, args.policy
            )
            metrics["conditions"]["rule_verified"] = {
                "filter": filter_summary,
                "recall": recall_at_k(rule_predictions, ground_truth, args.ks, families),
                "violation_rate": violation_rate_at_k(
                    rule_predictions,
                    verification_by_id,
                    args.ks,
                    families,
                ),
            }
        probabilistic_predictions, score_summary, score_errors = recalibrated_predictions(
            predictions, verification_by_id, families
        )
        if score_errors:
            blocked.extend(score_errors)
        else:
            metrics["conditions"]["probabilistic_recalibrated"] = {
                "score_summary": score_summary,
                "recall": recall_at_k(probabilistic_predictions, ground_truth, args.ks, families),
                "violation_rate": violation_rate_at_k(
                    probabilistic_predictions,
                    verification_by_id,
                    args.ks,
                    families,
                ),
            }
        for control_name in args.ablation_controls:
            control_predictions, control_summary, control_errors = ablation_control_predictions(
                predictions,
                verification_by_id,
                families,
                control_name,
                family_specific_model=family_specific_model,
            )
            condition = {
                "score_summary": control_summary,
            }
            if control_errors:
                condition["blocked"] = control_errors
                blocked.extend(control_errors)
            else:
                condition.update(
                    {
                        "recall": recall_at_k(
                            control_predictions, ground_truth, args.ks, families
                        ),
                        "violation_rate": violation_rate_at_k(
                            control_predictions,
                            verification_by_id,
                            args.ks,
                            families,
                        ),
                    }
                )
            metrics["conditions"][control_condition_name(control_name)] = condition
    else:
        blocked.append("rule_verified requires verification_jsonl from geometry join/verifier")
        blocked.append("probabilistic_recalibrated requires p_geom_valid outputs")
        if args.ablation_controls:
            blocked.append("ablation_controls require verification_jsonl from geometry join/verifier")

    if blocked:
        metrics["status"] = "partial_ready" if len(metrics["conditions"]) > 1 else "semantic_only_ready"

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "metrics.json", metrics)
        (args.output_dir / "report.md").write_text(make_report(metrics), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": metrics["status"],
                "predictions": len(predictions),
                "ground_truth": len(ground_truth),
                "blocked": len(blocked),
                "dry_run": args.dry_run,
                "output_dir": relpath(args.output_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
