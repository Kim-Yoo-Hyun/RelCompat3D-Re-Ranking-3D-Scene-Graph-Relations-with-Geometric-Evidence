#!/usr/bin/env python3
"""Join standardized source predictions with ordered-pair geometry evidence.

The join is intentionally row-preserving: every prediction row receives one
verification row, including predicates outside the evaluated geometry scope.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_training_rows import compute_features, load_scan_geometries
from point_support import (
    DEFAULT_POINT_THRESHOLDS,
    assign_point_status,
    compute_object_stats,
    local_support_stats,
)
from compatibility_features import dot, raw_numeric_features, sigmoid, vectorize
from support_verifier import (
    DEFAULT_THRESHOLDS as V2_THRESHOLDS,
    read_target_points,
    support_v2_record,
)


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "local_dataset"

SCHEMA_VERSION = "relcompat3d_prediction_geometry_v1"
POLICY_NAME = "relcompat3d-geometry-verifier-obb-v1"
G2_POLICY_NAME = "relcompat3d-geometry-verifier-v2"
GEOMETRY_SOURCE = "semseg_obb_v0"
POINT_GEOMETRY_SOURCE = "ply_points_v1+subtype_rules_v2"
NO_SOFT_GEOMETRY_SOURCE = "semseg_obb_v0+subtype_ablation"
POINT_EVIDENCE_SOURCE = "ply_points_v1"
PRIMARY_FAMILIES = {"support_contact", "proximity", "relative_vertical"}
ALLOWED_STATUSES = {"satisfied", "uncertain", "violated", "unsupported"}
G2_VARIANTS = {"obb_only", "point_subtype", "point_subtype_no_soft_support"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--model-json",
        type=Path,
        help="Optional legacy calibration model; RelCompat3D fitting does not require it.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-scans", type=Path)
    parser.add_argument(
        "--verification-policy",
        choices=["obb_only", "point_subtype"],
        default="obb_only",
        help="Selected top-level verification policy. point_subtype also emits G2 variants.",
    )
    parser.add_argument("--point-cache-size", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def read_selected_scans(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


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


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def edge_ids(prediction: dict[str, Any]) -> tuple[int | None, int | None]:
    edge = prediction.get("edge", {})
    subject_id = edge.get("subject_id")
    object_id = edge.get("object_id")
    try:
        subject_id = int(subject_id) if subject_id is not None else None
        object_id = int(object_id) if object_id is not None else None
    except (TypeError, ValueError):
        return None, None
    return subject_id, object_id


def predicate_label(prediction: dict[str, Any]) -> str | None:
    predicate = prediction.get("predicate", {})
    label = predicate.get("predicate_label")
    return str(label) if label is not None else None


def predicate_family(prediction: dict[str, Any]) -> str:
    predicate = prediction.get("predicate", {})
    family = predicate.get("predicate_family")
    return str(family) if family else "unsupported_first_pass"


def bool_overlap(features: dict[str, Any]) -> bool:
    subject_overlap = finite_float(features.get("projected_subject_overlap_ratio")) or 0.0
    object_overlap = finite_float(features.get("projected_object_overlap_ratio")) or 0.0
    return subject_overlap > 0.0 or object_overlap > 0.0


def verify_proximity(features: dict[str, Any]) -> tuple[str, float, list[str]]:
    norm_xy = finite_float(features.get("normalized_distance_xy"))
    if norm_xy is None:
        return "uncertain", 0.5, ["missing_normalized_distance_xy"]

    overlap = bool_overlap(features)
    if overlap or norm_xy <= 2.5:
        score = 0.75 if overlap else clamp(1.0 - (norm_xy / 5.0), 0.55, 0.9)
        return "satisfied", score, ["near_in_xy_or_projected_overlap"]
    if norm_xy >= 3.5:
        return "violated", clamp(1.0 - (norm_xy / 5.0), 0.05, 0.35), ["far_in_normalized_xy"]
    return "uncertain", 0.5, ["proximity_margin_ambiguous"]


def verify_relative_vertical(label: str | None, features: dict[str, Any]) -> tuple[str, float, list[str]]:
    center_delta = finite_float(features.get("center_delta_z"))
    normalized_delta = finite_float(features.get("normalized_center_delta_z"))
    if center_delta is None or normalized_delta is None:
        return "uncertain", 0.5, ["missing_vertical_delta"]

    if label == "lower than":
        aligned = -center_delta
        aligned_norm = -normalized_delta
    else:
        aligned = center_delta
        aligned_norm = normalized_delta

    score = clamp((aligned_norm + 0.4) / 0.8, 0.05, 0.95)
    if aligned >= 0.25 and aligned_norm >= 0.15:
        return "satisfied", max(score, 0.75), ["vertical_order_matches_predicate"]
    if aligned <= -0.25 and aligned_norm <= -0.15:
        return "violated", min(score, 0.25), ["vertical_order_contradicts_predicate"]
    return "uncertain", 0.5, ["vertical_margin_ambiguous"]


def verify_support_contact(features: dict[str, Any]) -> tuple[str, float, list[str]]:
    norm_xy = finite_float(features.get("normalized_distance_xy"))
    vertical_gap = finite_float(features.get("vertical_gap_subject_on_object"))
    overlap = bool_overlap(features)

    if vertical_gap is None or norm_xy is None:
        return "uncertain", 0.5, ["missing_support_contact_geometry"]

    if overlap and abs(vertical_gap) <= 0.30:
        return "satisfied", 0.8, ["projected_overlap_and_small_vertical_gap"]
    if (not overlap) and norm_xy >= 2.0 and abs(vertical_gap) >= 0.30:
        return "violated", 0.15, ["no_projected_overlap_and_large_gap"]
    return "uncertain", 0.5, ["support_contact_obb_ambiguous"]


def verify_rule(prediction: dict[str, Any], features: dict[str, Any] | None) -> tuple[str, float | None, list[str]]:
    family = predicate_family(prediction)
    label = predicate_label(prediction)
    if family not in PRIMARY_FAMILIES:
        return "unsupported", None, [f"predicate_family_out_of_scope:{family}"]
    if features is None:
        return "uncertain", None, ["missing_geometry_features"]
    if family == "proximity":
        return verify_proximity(features)
    if family == "relative_vertical":
        return verify_relative_vertical(label, features)
    if family == "support_contact":
        return verify_support_contact(features)
    return "unsupported", None, [f"predicate_family_out_of_scope:{family}"]


def score_geometry_probability(
    prediction: dict[str, Any],
    features: dict[str, Any] | None,
    model: dict[str, Any] | None,
) -> tuple[float | None, str | None]:
    if model is None or features is None:
        return None, None

    row = {
        "predicate": prediction.get("predicate", {}),
        "geometry": {"features": features},
    }
    row["_raw_numeric"] = raw_numeric_features(row)
    try:
        vector = vectorize(row, model)
        probability = sigmoid(dot(model["weights"], vector))
    except Exception as exc:  # pragma: no cover - surfaced in output quality flags.
        return None, f"p_geom_valid_score_failed:{exc.__class__.__name__}"
    return probability, None


def semantic_payload(prediction: dict[str, Any]) -> dict[str, Any]:
    scores = prediction.get("scores", {})
    return {
        "predicate_score": finite_float(scores.get("predicate_score")),
        "ranking_score": finite_float(scores.get("ranking_score")),
        "ranks": prediction.get("ranks", {}),
    }


def edge_labels(prediction: dict[str, Any]) -> tuple[str | None, str | None]:
    edge = prediction.get("edge", {})
    return edge.get("subject_label"), edge.get("object_label")


def obb_variant(
    prediction: dict[str, Any],
    *,
    features: dict[str, Any] | None,
    missing_ids: list[str],
    policy_name: str = POLICY_NAME,
    policy_version: str = "v0",
    geometry_source: str = GEOMETRY_SOURCE,
) -> dict[str, Any]:
    family = predicate_family(prediction)
    status, consistency_score, reason_codes = verify_rule(prediction, features)

    if missing_ids and family in PRIMARY_FAMILIES:
        status = "uncertain"
        consistency_score = None
        reason_codes = [f"missing_geometry:{object_id}" for object_id in missing_ids]

    return {
        "verification_status": status,
        "policy_name": policy_name,
        "policy_version": policy_version,
        "geometry_source": geometry_source,
        "consistency_score": consistency_score,
        "reason_codes": reason_codes,
        "is_geometry_checkable": family in PRIMARY_FAMILIES,
    }


def variant_with_policy(
    variant: dict[str, Any],
    *,
    policy_name: str,
    policy_version: str,
    geometry_source: str,
    extra_reason_codes: list[str] | None = None,
) -> dict[str, Any]:
    copied = dict(variant)
    copied["policy_name"] = policy_name
    copied["policy_version"] = policy_version
    copied["geometry_source"] = geometry_source
    if extra_reason_codes:
        copied["reason_codes"] = sorted(set(copied.get("reason_codes", []) + extra_reason_codes))
    return copied


def support_fields(support_record: dict[str, Any] | None) -> dict[str, Any]:
    support_record = support_record or {}
    subject_stats = support_record.get("subject_point_stats") or {}
    object_stats = support_record.get("object_point_stats") or {}
    return {
        "support_subtype": support_record.get("subtype"),
        "support_evidence_source": POINT_EVIDENCE_SOURCE,
        "point_evidence_available": bool(support_record.get("point_evidence_available")),
        "subject_point_count": subject_stats.get("point_count"),
        "object_point_count": object_stats.get("point_count"),
        "support_points_under_subject_count": support_record.get(
            "support_points_under_subject_count"
        ),
        "local_vertical_gap_p05_p95": support_record.get("local_vertical_gap_p05_p95"),
        "local_vertical_gap_p01_p99": support_record.get("local_vertical_gap_p01_p99"),
        "xy_expansion_m": support_record.get("xy_expansion_m"),
        "geometry_quality_flags": support_record.get("geometry_quality_flags"),
        "subtype_reason_codes": support_record.get("subtype_reason_codes"),
    }


def support_edge_for_prediction(
    prediction: dict[str, Any],
    obb: dict[str, Any],
) -> dict[str, Any]:
    subject_id, object_id = edge_ids(prediction)
    subject_label, object_label = edge_labels(prediction)
    return {
        "edge_id": prediction.get("prediction_id"),
        "scan_id": prediction.get("scan_id"),
        "subject_id": subject_id,
        "object_id": object_id,
        "subject_label": subject_label,
        "predicate_label": predicate_label(prediction),
        "object_label": object_label,
        "predicate_family": predicate_family(prediction),
        "verification": {
            "status": obb.get("verification_status"),
            "rule_version": obb.get("policy_name"),
            "reason_codes": obb.get("reason_codes", []),
            "geometry_score": obb.get("consistency_score"),
        },
    }


def collect_support_object_ids(
    predictions_jsonl: Path,
    selected_scans: set[str] | None,
) -> tuple[dict[str, set[int]], list[str]]:
    by_scan: dict[str, set[int]] = defaultdict(set)
    warnings: list[str] = []
    for _, prediction in iter_jsonl(predictions_jsonl):
        scan_id = str(prediction.get("scan_id") or "")
        if selected_scans is not None and scan_id and scan_id not in selected_scans:
            warnings.append(f"prediction_scan_outside_selected_scope:{scan_id}")
        if predicate_family(prediction) != "support_contact":
            continue
        subject_id, object_id = edge_ids(prediction)
        if scan_id and subject_id is not None and object_id is not None:
            by_scan[scan_id].add(subject_id)
            by_scan[scan_id].add(object_id)
    return by_scan, sorted(set(warnings))


def make_point_record(
    prediction: dict[str, Any],
    *,
    object_stats: dict[int, dict[str, Any]],
    points_by_object: dict[int, dict[str, list[float]]],
    point_thresholds: dict[str, Any],
) -> dict[str, Any]:
    subject_id, object_id = edge_ids(prediction)
    subject_label, object_label = edge_labels(prediction)
    empty_stats = {"point_count": 0}
    subject_stats = object_stats.get(subject_id or -1, empty_stats)
    object_stats_record = object_stats.get(object_id or -1, empty_stats)
    object_points = points_by_object.get(object_id or -1, {"x": [], "y": [], "z": []})

    local_evidence: list[dict[str, Any]] = []
    point_status = "point_uncertain"
    point_reason_codes = ["missing_endpoint_points"]
    best_local_evidence = None
    if subject_stats.get("point_count", 0) and object_stats_record.get("point_count", 0):
        local_evidence = local_support_stats(subject_stats, object_points, point_thresholds)
        point_status, point_reason_codes, best_local_evidence = assign_point_status(
            local_evidence,
            point_thresholds,
        )

    return {
        "edge_id": prediction.get("prediction_id"),
        "scan_id": prediction.get("scan_id"),
        "subject_id": subject_id,
        "object_id": object_id,
        "subject_label": subject_label,
        "predicate_label": predicate_label(prediction),
        "object_label": object_label,
        "point_rule_version": point_thresholds["point_rule_version"],
        "subject_point_stats": subject_stats,
        "object_point_stats": object_stats_record,
        "local_support_evidence": local_evidence,
        "point_status": point_status,
        "point_reason_codes": point_reason_codes,
        "best_local_support_evidence": best_local_evidence,
        "point_evidence_available": best_local_evidence is not None,
    }


def load_point_context_cached(
    *,
    dataset_root: Path,
    scan_id: str,
    scan_object_ids: dict[str, set[int]],
    point_cache: OrderedDict[str, dict[str, Any]],
    point_cache_size: int,
    point_thresholds: dict[str, Any],
) -> dict[str, Any]:
    if scan_id in point_cache:
        point_cache.move_to_end(scan_id)
        return point_cache[scan_id]

    object_ids = scan_object_ids.get(scan_id, set())
    context: dict[str, Any] = {
        "scan_id": scan_id,
        "points_by_object": {},
        "object_stats": {},
        "pair_records": {},
        "ply_stats": {},
        "errors": [],
        "warnings": [],
        "point_thresholds": point_thresholds,
    }
    ply_path = dataset_root / "3RScan" / "scans" / scan_id / "labels.instances.annotated.v2.ply"
    if not object_ids:
        context["warnings"].append(f"no_support_object_ids:{scan_id}")
    elif not ply_path.exists():
        context["errors"].append(f"missing_point_ply:{scan_id}:{ply_path}")
    else:
        try:
            points_by_object, ply_stats = read_target_points(ply_path, set(object_ids))
            context["points_by_object"] = points_by_object
            context["object_stats"] = {
                object_id: compute_object_stats(points)
                for object_id, points in points_by_object.items()
            }
            context["ply_stats"] = ply_stats
            missing_point_ids = [
                object_id
                for object_id, stats in context["object_stats"].items()
                if not stats.get("point_count", 0)
            ]
            if missing_point_ids:
                context["warnings"].append(
                    f"support_object_ids_missing_points:{scan_id}:{len(missing_point_ids)}"
                )
        except Exception as exc:  # pragma: no cover - surfaced in manifest.
            context["errors"].append(f"point_context_failed:{scan_id}:{type(exc).__name__}:{exc}")

    point_cache[scan_id] = context
    point_cache.move_to_end(scan_id)
    while len(point_cache) > max(1, point_cache_size):
        point_cache.popitem(last=False)
    return context


def point_record_cached(
    prediction: dict[str, Any],
    point_context: dict[str, Any],
    point_thresholds: dict[str, Any],
) -> dict[str, Any] | None:
    subject_id, object_id = edge_ids(prediction)
    if subject_id is None or object_id is None or point_context.get("errors"):
        return None
    key = (subject_id, object_id)
    pair_records = point_context.setdefault("pair_records", {})
    if key not in pair_records:
        pair_records[key] = make_point_record(
            prediction,
            object_stats=point_context.get("object_stats", {}),
            points_by_object=point_context.get("points_by_object", {}),
            point_thresholds=point_thresholds,
        )
    return pair_records[key]


def point_subtype_variant(
    prediction: dict[str, Any],
    *,
    obb: dict[str, Any],
    point_context: dict[str, Any] | None,
    point_thresholds: dict[str, Any],
    v2_thresholds: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if predicate_family(prediction) != "support_contact":
        delegated = variant_with_policy(
            obb,
            policy_name=f"{G2_POLICY_NAME}:point_subtype",
            policy_version="g2",
            geometry_source=GEOMETRY_SOURCE,
            extra_reason_codes=["point_subtype_delegated_to_obb_for_family"],
        )
        return delegated, None

    edge = support_edge_for_prediction(prediction, obb)
    point_record = (
        point_record_cached(prediction, point_context, point_thresholds)
        if point_context is not None
        else None
    )
    support_record = support_v2_record(
        edge,
        point_record,
        None,
        (point_context or {}).get("points_by_object", {}),
        v2_thresholds,
    )
    if point_context and point_context.get("errors"):
        support_record["status"] = "uncertain"
        support_record["consistency_score"] = None
        support_record["reason_codes"] = sorted(
            set(support_record.get("reason_codes", []) + ["point_context_unavailable"])
        )
    if not support_record.get("point_evidence_available"):
        support_record["status"] = "uncertain"
        support_record["consistency_score"] = None
        support_record["reason_codes"] = sorted(
            set(support_record.get("reason_codes", []) + ["missing_point_evidence"])
        )

    variant = {
        "verification_status": support_record["status"],
        "policy_name": f"{G2_POLICY_NAME}:point_subtype",
        "policy_version": "g2",
        "geometry_source": POINT_GEOMETRY_SOURCE,
        "consistency_score": support_record["consistency_score"],
        "reason_codes": support_record["reason_codes"],
        "is_geometry_checkable": True,
    }
    variant.update(support_fields(support_record))
    return variant, support_record


def no_soft_variant(
    *,
    obb: dict[str, Any],
    point_variant: dict[str, Any],
    support_record: dict[str, Any] | None,
) -> dict[str, Any]:
    if support_record is None:
        return variant_with_policy(
            obb,
            policy_name=f"{G2_POLICY_NAME}:point_subtype_no_soft_support",
            policy_version="g2",
            geometry_source=GEOMETRY_SOURCE,
            extra_reason_codes=["no_soft_delegated_to_obb_for_family"],
        )

    subtype = support_record.get("subtype")
    if subtype in {"legged_floor_support", "soft_support_contact"}:
        variant = variant_with_policy(
            obb,
            policy_name=f"{G2_POLICY_NAME}:point_subtype_no_soft_support",
            policy_version="g2",
            geometry_source=NO_SOFT_GEOMETRY_SOURCE,
            extra_reason_codes=[f"special_support_recovery_disabled:{subtype}"],
        )
        variant.update(support_fields(support_record))
        return variant

    variant = variant_with_policy(
        point_variant,
        policy_name=f"{G2_POLICY_NAME}:point_subtype_no_soft_support",
        policy_version="g2",
        geometry_source=POINT_GEOMETRY_SOURCE,
        extra_reason_codes=["no_soft_same_as_point_subtype_for_rigid_support"],
    )
    variant.update(support_fields(support_record))
    return variant


def make_verification_row_g2(
    prediction: dict[str, Any],
    *,
    features: dict[str, Any] | None,
    geometry_available: bool,
    missing_ids: list[str],
    model: dict[str, Any] | None,
    model_path: Path | None,
    created_at: str,
    verification_policy: str,
    point_context: dict[str, Any] | None,
    point_thresholds: dict[str, Any],
    v2_thresholds: dict[str, Any],
) -> dict[str, Any]:
    obb = obb_variant(
        prediction,
        features=features,
        missing_ids=missing_ids,
        policy_name=f"{G2_POLICY_NAME}:obb_only",
        policy_version="g2",
        geometry_source=GEOMETRY_SOURCE,
    )
    point_variant, support_record = point_subtype_variant(
        prediction,
        obb=obb,
        point_context=point_context,
        point_thresholds=point_thresholds,
        v2_thresholds=v2_thresholds,
    )
    variants = {
        "obb_only": obb,
        "point_subtype": point_variant,
        "point_subtype_no_soft_support": no_soft_variant(
            obb=obb,
            point_variant=point_variant,
            support_record=support_record,
        ),
    }
    selected = variants[verification_policy]

    p_geom_valid, score_warning = score_geometry_probability(prediction, features, model)
    if score_warning:
        selected["reason_codes"] = sorted(set(selected.get("reason_codes", []) + [score_warning]))

    semantic = semantic_payload(prediction)
    predicate_score = semantic.get("predicate_score")
    p_final_product = None
    if predicate_score is not None and p_geom_valid is not None:
        p_final_product = predicate_score * p_geom_valid

    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "prediction_geometry_verification",
        "prediction_id": prediction.get("prediction_id"),
        "baseline_name": prediction.get("baseline_name"),
        "baseline_run_id": prediction.get("baseline_run_id"),
        "split_name": prediction.get("split_name"),
        "scan_id": prediction.get("scan_id"),
        "subset_split_id": prediction.get("subset_split_id"),
        "subgraph_id": prediction.get("subgraph_id"),
        "edge": prediction.get("edge", {}),
        "predicate": prediction.get("predicate", {}),
        "semantic": semantic,
        "geometry": {
            "geometry_available": geometry_available,
            "geometry_source": GEOMETRY_SOURCE,
            "features": features,
            "missing_object_ids": missing_ids,
        },
        "verification": selected,
        "verification_variants": variants,
        "verification_status": selected["verification_status"],
        "consistency_score": selected["consistency_score"],
        "calibration": {
            "model_id": model.get("model_id") if model else None,
            "model_path": str(model_path) if model_path else None,
            "p_geom_valid": p_geom_valid,
            "p_geom_invalid": (1.0 - p_geom_valid) if p_geom_valid is not None else None,
            "p_final_product": p_final_product,
        },
        "quality": {
            "row_preserved": True,
            "geometry_checkable": predicate_family(prediction) in PRIMARY_FAMILIES,
            "geometry_available": geometry_available,
            "abstain_reason": (
                selected["reason_codes"][0]
                if selected["verification_status"] in {"uncertain", "unsupported"}
                and selected.get("reason_codes")
                else None
            ),
        },
        "provenance": {
            "joiner": G2_POLICY_NAME,
            "created_at": created_at,
            "source_prediction_schema": prediction.get("schema_version"),
            "selected_verification_policy": verification_policy,
        },
    }


def make_verification_row(
    prediction: dict[str, Any],
    *,
    features: dict[str, Any] | None,
    geometry_available: bool,
    missing_ids: list[str],
    model: dict[str, Any] | None,
    model_path: Path | None,
    created_at: str,
) -> dict[str, Any]:
    family = predicate_family(prediction)
    status, consistency_score, reason_codes = verify_rule(prediction, features)

    if missing_ids and family in PRIMARY_FAMILIES:
        status = "uncertain"
        consistency_score = None
        reason_codes = [f"missing_geometry:{object_id}" for object_id in missing_ids]

    p_geom_valid, score_warning = score_geometry_probability(prediction, features, model)
    if score_warning:
        reason_codes.append(score_warning)

    semantic = semantic_payload(prediction)
    predicate_score = semantic.get("predicate_score")
    p_final_product = None
    if predicate_score is not None and p_geom_valid is not None:
        p_final_product = predicate_score * p_geom_valid

    verification = {
        "verification_status": status,
        "policy_name": POLICY_NAME,
        "policy_version": "v0",
        "geometry_source": GEOMETRY_SOURCE,
        "consistency_score": consistency_score,
        "reason_codes": reason_codes,
        "is_geometry_checkable": family in PRIMARY_FAMILIES,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "prediction_geometry_verification",
        "prediction_id": prediction.get("prediction_id"),
        "baseline_name": prediction.get("baseline_name"),
        "baseline_run_id": prediction.get("baseline_run_id"),
        "split_name": prediction.get("split_name"),
        "scan_id": prediction.get("scan_id"),
        "subset_split_id": prediction.get("subset_split_id"),
        "subgraph_id": prediction.get("subgraph_id"),
        "edge": prediction.get("edge", {}),
        "predicate": prediction.get("predicate", {}),
        "semantic": semantic,
        "geometry": {
            "geometry_available": geometry_available,
            "geometry_source": GEOMETRY_SOURCE,
            "features": features,
            "missing_object_ids": missing_ids,
        },
        "verification": verification,
        "verification_status": status,
        "consistency_score": consistency_score,
        "calibration": {
            "model_id": model.get("model_id") if model else None,
            "model_path": str(model_path) if model_path else None,
            "p_geom_valid": p_geom_valid,
            "p_geom_invalid": (1.0 - p_geom_valid) if p_geom_valid is not None else None,
            "p_final_product": p_final_product,
        },
        "quality": {
            "row_preserved": True,
            "geometry_checkable": family in PRIMARY_FAMILIES,
            "geometry_available": geometry_available,
            "abstain_reason": reason_codes[0] if status in {"uncertain", "unsupported"} and reason_codes else None,
        },
        "provenance": {
            "joiner": POLICY_NAME,
            "created_at": created_at,
            "source_prediction_schema": prediction.get("schema_version"),
        },
    }


def load_scan_geometry_cached(
    *,
    dataset_root: Path,
    scan_id: str,
    cache: dict[str, tuple[dict[int, Any], list[str], list[str]]],
) -> tuple[dict[int, Any], list[str], list[str]]:
    if scan_id not in cache:
        cache[scan_id] = load_scan_geometries(dataset_root, scan_id)
    return cache[scan_id]


def prediction_features(
    prediction: dict[str, Any],
    *,
    dataset_root: Path,
    geometry_cache: dict[str, tuple[dict[int, Any], list[str], list[str]]],
) -> tuple[dict[str, Any] | None, bool, list[str], list[str]]:
    family = predicate_family(prediction)
    if family not in PRIMARY_FAMILIES:
        return None, False, [], []

    scan_id = prediction.get("scan_id")
    subject_id, object_id = edge_ids(prediction)
    if not scan_id or subject_id is None or object_id is None:
        missing = []
        if subject_id is None:
            missing.append("<missing_subject_id>")
        if object_id is None:
            missing.append("<missing_object_id>")
        return None, False, missing, []

    geometries, warnings, errors = load_scan_geometry_cached(
        dataset_root=dataset_root,
        scan_id=str(scan_id),
        cache=geometry_cache,
    )
    if errors:
        return None, False, [str(subject_id), str(object_id)], warnings + errors
    missing_ids = [object_id for object_id in (subject_id, object_id) if object_id not in geometries]
    if missing_ids:
        return None, False, [str(object_id) for object_id in missing_ids], warnings

    features = compute_features(geometries[subject_id], geometries[object_id])
    return features, True, [], warnings


def render_report(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    lines = [
        "# Prediction Geometry Join Report",
        "",
        f"- created_at: `{manifest['created_at']}`",
        f"- joiner: `{manifest['joiner']}`",
        f"- predictions: `{counts['predictions']}`",
        f"- verification_rows: `{counts['verification_rows']}`",
        f"- rows_preserved: `{counts['rows_preserved']}`",
        f"- geometry_available_rows: `{counts['geometry_available_rows']}`",
        f"- calibration_scored_rows: `{counts['calibration_scored_rows']}`",
        "",
        "## Verification Status",
        "",
    ]
    for status, count in counts["by_status"].items():
        lines.append(f"- `{status}`: `{count}`")

    lines.extend(["", "## Predicate Family Status", ""])
    for family, status_counts in counts["by_family_status"].items():
        compact = ", ".join(f"{status}={count}" for status, count in status_counts.items())
        lines.append(f"- `{family}`: {compact}")

    if counts.get("by_variant_status"):
        lines.extend(["", "## Variant Status", ""])
        for variant, status_counts in counts["by_variant_status"].items():
            compact = ", ".join(f"{status}={count}" for status, count in status_counts.items())
            lines.append(f"- `{variant}`: {compact}")

    if counts.get("support_subtype_counts"):
        lines.extend(["", "## Support Subtypes", ""])
        for subtype, count in counts["support_subtype_counts"].items():
            lines.append(f"- `{subtype}`: `{count}`")

    if counts.get("obb_to_point_status_transitions"):
        lines.extend(["", "## OBB To Point/Subtype Transitions", ""])
        for transition, count in counts["obb_to_point_status_transitions"].items():
            lines.append(f"- `{transition}`: `{count}`")

    notes = manifest.get("notes", [])
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)

    warnings = manifest.get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings[:20]:
            lines.append(f"- `{warning}`")
        if len(warnings) > 20:
            lines.append(f"- ... `{len(warnings) - 20}` more")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    required_paths = [
        (args.predictions_jsonl, "predictions_jsonl"),
        (args.dataset_root, "dataset_root"),
    ]
    if args.model_json is not None:
        required_paths.append((args.model_json, "model_json"))
    for path, name in required_paths:
        if not path.exists():
            errors.append(f"missing_{name}:{path}")
    if errors:
        print("\n".join(errors))
        return 2

    model = load_json(args.model_json) if args.model_json is not None else None
    created_at = datetime.now(timezone.utc).isoformat()
    selected_scans = read_selected_scans(args.selected_scans)
    selected_scan_errors: set[str] = set()
    point_thresholds = dict(DEFAULT_POINT_THRESHOLDS)
    v2_thresholds = dict(V2_THRESHOLDS)
    scan_object_ids: dict[str, set[int]] = {}
    point_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
    point_context_error_scans: set[str] = set()
    point_context_warning_scans: set[str] = set()

    if args.verification_policy == "point_subtype":
        scan_object_ids, support_scan_warnings = collect_support_object_ids(
            args.predictions_jsonl,
            selected_scans,
        )
        warnings.extend(support_scan_warnings)

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    verification_path = args.output_dir / "verification.jsonl"

    counts = {
        "predictions": 0,
        "verification_rows": 0,
        "rows_preserved": False,
        "geometry_available_rows": 0,
        "calibration_scored_rows": 0,
        "primary_family_rows": 0,
        "unsupported_family_rows": 0,
    }
    by_status: Counter[str] = Counter()
    by_family: Counter[str] = Counter()
    by_family_status: dict[str, Counter[str]] = defaultdict(Counter)
    by_variant_status: dict[str, Counter[str]] = defaultdict(Counter)
    by_variant_family_status: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    support_subtype_counts: Counter[str] = Counter()
    obb_to_point_status_transitions: Counter[str] = Counter()
    seen_prediction_ids: set[str] = set()
    duplicate_prediction_ids: set[str] = set()
    geometry_cache: dict[str, tuple[dict[int, Any], list[str], list[str]]] = {}

    output_handle = None
    try:
        if not args.dry_run:
            output_handle = verification_path.open("w", encoding="utf-8")

        for _, prediction in iter_jsonl(args.predictions_jsonl):
            counts["predictions"] += 1
            prediction_id = prediction.get("prediction_id")
            if prediction_id in seen_prediction_ids:
                duplicate_prediction_ids.add(str(prediction_id))
            seen_prediction_ids.add(str(prediction_id))

            family = predicate_family(prediction)
            scan_id = str(prediction.get("scan_id") or "")
            if selected_scans is not None and scan_id not in selected_scans:
                selected_scan_errors.add(scan_id)
            by_family[family] += 1
            if family in PRIMARY_FAMILIES:
                counts["primary_family_rows"] += 1
            else:
                counts["unsupported_family_rows"] += 1

            features, geometry_available, missing_ids, scan_warnings = prediction_features(
                prediction,
                dataset_root=args.dataset_root,
                geometry_cache=geometry_cache,
            )
            if scan_warnings:
                for warning in scan_warnings:
                    if warning not in warnings:
                        warnings.append(warning)

            if args.verification_policy == "point_subtype":
                point_context = None
                if family == "support_contact":
                    point_context = load_point_context_cached(
                        dataset_root=args.dataset_root,
                        scan_id=scan_id,
                        scan_object_ids=scan_object_ids,
                        point_cache=point_cache,
                        point_cache_size=args.point_cache_size,
                        point_thresholds=point_thresholds,
                    )
                    if point_context.get("errors") and scan_id not in point_context_error_scans:
                        errors.extend(point_context["errors"])
                        point_context_error_scans.add(scan_id)
                    if point_context.get("warnings") and scan_id not in point_context_warning_scans:
                        warnings.extend(point_context["warnings"])
                        point_context_warning_scans.add(scan_id)

                row = make_verification_row_g2(
                    prediction,
                    features=features,
                    geometry_available=geometry_available,
                    missing_ids=missing_ids,
                    model=model,
                    model_path=args.model_json,
                    created_at=created_at,
                    verification_policy=args.verification_policy,
                    point_context=point_context,
                    point_thresholds=point_thresholds,
                    v2_thresholds=v2_thresholds,
                )
            else:
                row = make_verification_row(
                    prediction,
                    features=features,
                    geometry_available=geometry_available,
                    missing_ids=missing_ids,
                    model=model,
                    model_path=args.model_json,
                    created_at=created_at,
                )

            status = row["verification_status"]
            if status not in ALLOWED_STATUSES:
                errors.append(f"invalid_status:{status}:{prediction_id}")
            by_status[status] += 1
            by_family_status[family][status] += 1
            variants = row.get("verification_variants") or {}
            if variants:
                missing_variants = sorted(G2_VARIANTS - set(variants))
                if missing_variants:
                    errors.append(f"missing_variants:{prediction_id}:{missing_variants}")
                for variant_name, variant in variants.items():
                    variant_status = variant.get("verification_status")
                    if variant_status not in ALLOWED_STATUSES:
                        errors.append(
                            f"invalid_variant_status:{variant_name}:{variant_status}:{prediction_id}"
                        )
                        continue
                    by_variant_status[variant_name][variant_status] += 1
                    by_variant_family_status[variant_name][family][variant_status] += 1
                if family == "support_contact":
                    point_variant = variants.get("point_subtype", {})
                    obb_variant_row = variants.get("obb_only", {})
                    subtype = point_variant.get("support_subtype") or "<missing_subtype>"
                    support_subtype_counts[subtype] += 1
                    if point_variant.get("point_evidence_available") is False:
                        counts.setdefault("missing_point_evidence_rows", 0)
                        counts["missing_point_evidence_rows"] += 1
                        if point_variant.get("verification_status") == "violated":
                            errors.append(f"missing_point_evidence_marked_violated:{prediction_id}")
                    transition = (
                        f"{obb_variant_row.get('verification_status')}"
                        f"_to_{point_variant.get('verification_status')}"
                    )
                    obb_to_point_status_transitions[transition] += 1
            if geometry_available:
                counts["geometry_available_rows"] += 1
            if row["calibration"]["p_geom_valid"] is not None:
                counts["calibration_scored_rows"] += 1
            counts["verification_rows"] += 1

            if output_handle is not None:
                output_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    finally:
        if output_handle is not None:
            output_handle.close()

    counts["rows_preserved"] = counts["predictions"] == counts["verification_rows"]
    if not counts["rows_preserved"]:
        errors.append(
            f"row_count_mismatch:predictions={counts['predictions']}:verification_rows={counts['verification_rows']}"
        )
    if duplicate_prediction_ids:
        errors.append(f"duplicate_prediction_ids:{len(duplicate_prediction_ids)}")
    if selected_scan_errors:
        errors.append(f"prediction_scan_outside_selected_scope:{sorted(selected_scan_errors)[:10]}")
    if args.verification_policy == "point_subtype" and not by_variant_status:
        errors.append("g2_variants_not_emitted")

    joiner = G2_POLICY_NAME if args.verification_policy == "point_subtype" else POLICY_NAME
    manifest = {
        "schema_version": "relcompat3d_prediction_geometry_manifest_v1",
        "record_type": "prediction_geometry_join_manifest",
        "status": "blocked" if errors else "ready",
        "created_at": created_at,
        "joiner": joiner,
        "selected_verification_policy": args.verification_policy,
        "inputs": {
            "predictions_jsonl": str(args.predictions_jsonl),
            "dataset_root": str(args.dataset_root),
            "model_json": str(args.model_json) if args.model_json else None,
            "selected_scans": str(args.selected_scans) if args.selected_scans else None,
        },
        "outputs": {
            "verification_jsonl": str(verification_path),
            "manifest_json": str(args.output_dir / "manifest.json"),
            "report_md": str(args.output_dir / "report.md"),
        },
        "counts": {
            **counts,
            "by_status": dict(sorted(by_status.items())),
            "by_family": dict(sorted(by_family.items())),
            "by_family_status": {
                family: dict(sorted(status_counts.items()))
                for family, status_counts in sorted(by_family_status.items())
            },
            "by_variant_status": {
                variant: dict(sorted(status_counts.items()))
                for variant, status_counts in sorted(by_variant_status.items())
            },
            "by_variant_family_status": {
                variant: {
                    family: dict(sorted(status_counts.items()))
                    for family, status_counts in sorted(family_counts.items())
                }
                for variant, family_counts in sorted(by_variant_family_status.items())
            },
            "support_subtype_counts": dict(sorted(support_subtype_counts.items())),
            "obb_to_point_status_transitions": dict(
                sorted(obb_to_point_status_transitions.items())
            ),
            "point_context_error_scans": sorted(point_context_error_scans),
            "point_context_warning_scans": sorted(point_context_warning_scans),
        },
        "errors": errors,
        "warnings": warnings,
        "notes": [
            "This artifact preserves every prediction row.",
            "unsupported status means the predicate family is outside the evaluated geometry scope.",
            (
                "support_contact verification is OBB-only and conservative; point/contact subtype evidence is not applied in this join."
                if args.verification_policy == "obb_only"
                else "The join emits OBB-only, point/subtype, and no-soft-support variants."
            ),
        ],
    }

    if not args.dry_run:
        write_json(args.output_dir / "manifest.json", manifest)
        (args.output_dir / "report.md").write_text(render_report(manifest), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
