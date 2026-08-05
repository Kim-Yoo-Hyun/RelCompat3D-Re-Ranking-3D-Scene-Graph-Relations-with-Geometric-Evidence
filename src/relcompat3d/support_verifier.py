#!/usr/bin/env python3
"""Subtype-aware support/contact verifier used by RelCompat3D."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any


RULE_VERSION = "relcompat3d-support-verifier-v2"
SOFT_LABELS = {"pillow", "cushion", "blanket", "clothes", "towel"}
FURNITURE_SUPPORT_LABELS = {
    "table", "desk", "kitchen counter", "counter", "cabinet", "shelf",
    "sofa", "chair", "bed", "stool", "bench",
}
GEOMETRY_QUALITY_LABELS = {"segmentation_or_instance_issue"}
LOCAL_SURFACE_LABELS = {"local_surface_estimator_issue"}
RULE_TOO_STRICT_LABELS = {"rule_too_strict"}

DEFAULT_THRESHOLDS = {
    "rule_version": RULE_VERSION,
    "previous_rule_version": "relcompat3d-obb-rules-v1",
    "satisfied_score_min": 0.70,
    "uncertain_score_min": 0.40,
    "low_gap_pass_abs_m": 0.08,
    "low_gap_fail_abs_m": 0.18,
    "robust_gap_pass_abs_m": 0.10,
    "soft_penetration_pass_m": 0.15,
    "soft_penetration_max_m": 0.45,
    "positive_float_pass_m": 0.08,
    "positive_float_fail_m": 0.20,
    "support_density_good_count": 50,
    "plane_expansion_m": 0.20,
    "plane_bin_m": 0.04,
    "plane_search_below_m": 0.25,
    "plane_search_above_m": 0.12,
    "plane_gap_pass_abs_m": 0.08,
    "plane_gap_fail_abs_m": 0.22,
    "plane_min_inlier_count": 10,
}


def label(value: Any) -> str:
    return str(value or "").strip().lower()


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def score_abs(value: float | None, pass_abs: float, fail_abs: float) -> float:
    if value is None:
        return 0.0
    distance = abs(float(value))
    if distance <= pass_abs:
        return 1.0
    if distance >= fail_abs:
        return 0.0
    return clamp(1.0 - (distance - pass_abs) / (fail_abs - pass_abs))


def score_count(count: int | None, good_count: int) -> float:
    if count is None or good_count <= 0:
        return 0.0
    return clamp(float(count) / float(good_count))


def status_from_score(score: float | None, thresholds: dict[str, Any]) -> str:
    if score is None:
        return "uncertain"
    if score >= float(thresholds["satisfied_score_min"]):
        return "satisfied"
    if score >= float(thresholds["uncertain_score_min"]):
        return "uncertain"
    return "violated"


def parse_ply_header(path: Path) -> tuple[dict[str, Any], int]:
    properties: list[str] = []
    vertex_count: int | None = None
    face_count: int | None = None
    header_lines = 0
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline().strip()
        header_lines += 1
        if first_line != "ply":
            raise ValueError(f"expected_ply_header:{first_line!r}")
        for line in f:
            header_lines += 1
            stripped = line.strip()
            if stripped.startswith("format") and stripped != "format ascii 1.0":
                raise ValueError(f"unsupported_ply_format:{stripped}")
            if stripped.startswith("element vertex"):
                vertex_count = int(stripped.split()[-1])
                in_vertex = True
            elif stripped.startswith("element face"):
                face_count = int(stripped.split()[-1])
                in_vertex = False
            elif stripped.startswith("property") and in_vertex:
                properties.append(stripped.split()[-1])
            elif stripped == "end_header":
                break
    if vertex_count is None:
        raise ValueError("missing_vertex_count")
    return {
        "vertex_count": vertex_count,
        "face_count": face_count,
        "properties": properties,
    }, header_lines


def read_target_points(
    path: Path,
    object_ids: set[int],
) -> tuple[dict[int, dict[str, list[float]]], dict[str, Any]]:
    header, _ = parse_ply_header(path)
    properties = header["properties"]
    for required in ("x", "y", "z", "objectId"):
        if required not in properties:
            raise ValueError(f"missing_ply_property:{required}")
    x_idx = properties.index("x")
    y_idx = properties.index("y")
    z_idx = properties.index("z")
    object_id_idx = properties.index("objectId")
    max_idx = max(x_idx, y_idx, z_idx, object_id_idx)

    points = {object_id: {"x": [], "y": [], "z": []} for object_id in object_ids}
    rows_read = 0
    rows_kept = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip() == "end_header":
                break
        for _ in range(int(header["vertex_count"])):
            line = f.readline()
            if not line:
                break
            rows_read += 1
            parts = line.split()
            if len(parts) <= max_idx:
                continue
            object_id = int(parts[object_id_idx])
            if object_id not in object_ids:
                continue
            points[object_id]["x"].append(float(parts[x_idx]))
            points[object_id]["y"].append(float(parts[y_idx]))
            points[object_id]["z"].append(float(parts[z_idx]))
            rows_kept += 1

    stats = {
        "ply_vertex_count_header": header["vertex_count"],
        "ply_face_count_header": header["face_count"],
        "ply_vertex_rows_read": rows_read,
        "target_vertex_rows_kept": rows_kept,
        "target_object_ids": sorted(object_ids),
    }
    return points, stats


def local_support_z_values(
    subject_stats: dict[str, Any],
    support_points: dict[str, list[float]] | None,
    expansion_m: float,
) -> list[float]:
    if support_points is None:
        return []
    required = ("x_p05", "x_p95", "y_p05", "y_p95")
    if any(subject_stats.get(key) is None for key in required):
        return []
    x_min = float(subject_stats["x_p05"]) - expansion_m
    x_max = float(subject_stats["x_p95"]) + expansion_m
    y_min = float(subject_stats["y_p05"]) - expansion_m
    y_max = float(subject_stats["y_p95"]) + expansion_m
    values: list[float] = []
    for x, y, z in zip(support_points["x"], support_points["y"], support_points["z"]):
        if x_min <= x <= x_max and y_min <= y <= y_max:
            values.append(z)
    return values


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def estimate_horizontal_plane(
    local_z: list[float],
    subject_stats: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    if not local_z:
        return {
            "plane_available": False,
            "plane_z_m": None,
            "plane_inlier_count": 0,
            "plane_inlier_ratio": 0.0,
            "plane_residual_m": None,
            "plane_normal_z_abs": None,
            "plane_gap_m": None,
            "plane_confidence": 0.0,
        }

    subject_bottom = subject_stats.get("z_p05")
    if subject_bottom is None:
        return {
            "plane_available": False,
            "plane_z_m": None,
            "plane_inlier_count": 0,
            "plane_inlier_ratio": 0.0,
            "plane_residual_m": None,
            "plane_normal_z_abs": None,
            "plane_gap_m": None,
            "plane_confidence": 0.0,
        }

    bin_size = float(thresholds["plane_bin_m"])
    search_low = float(subject_bottom) - float(thresholds["plane_search_below_m"])
    search_high = float(subject_bottom) + float(thresholds["plane_search_above_m"])
    bins: dict[int, list[float]] = defaultdict(list)
    for z in local_z:
        if search_low <= z <= search_high:
            bins[math.floor(z / bin_size)].append(z)

    if not bins:
        # Fall back to all local support z values. This should usually produce
        # low confidence, but it prevents silent missing fields.
        for z in local_z:
            bins[math.floor(z / bin_size)].append(z)

    best_bin, best_values = max(
        bins.items(),
        key=lambda item: (
            len(item[1]),
            -abs((median(item[1]) or 0.0) - float(subject_bottom)),
        ),
    )
    _ = best_bin
    plane_z = median(best_values)
    if plane_z is None:
        plane_z = sum(best_values) / len(best_values)
    residual = sum(abs(z - plane_z) for z in best_values) / len(best_values)
    inlier_count = len(best_values)
    inlier_ratio = inlier_count / len(local_z)
    plane_gap = float(subject_bottom) - plane_z

    count_score = score_count(inlier_count, int(thresholds["plane_min_inlier_count"]))
    ratio_score = clamp(inlier_ratio / 0.15)
    residual_score = score_abs(residual, 0.015, 0.060)
    plane_confidence = 0.40 * count_score + 0.30 * ratio_score + 0.30 * residual_score

    return {
        "plane_available": inlier_count >= int(thresholds["plane_min_inlier_count"]),
        "plane_z_m": plane_z,
        "plane_inlier_count": inlier_count,
        "plane_inlier_ratio": inlier_ratio,
        "plane_residual_m": residual,
        "plane_normal_z_abs": 1.0,
        "plane_gap_m": plane_gap,
        "plane_confidence": clamp(plane_confidence),
    }


def assign_subtype(
    edge: dict[str, Any],
    point_record: dict[str, Any] | None,
    visual_label: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    subject_label = label(edge.get("subject_label") or (point_record or {}).get("subject_label"))
    object_label = label(edge.get("object_label") or (point_record or {}).get("object_label"))
    predicate_label = label(edge.get("predicate_label") or (point_record or {}).get("predicate_label"))
    visual_kind = label((visual_label or {}).get("inspection_label"))
    reasons: list[str] = []

    if visual_kind in GEOMETRY_QUALITY_LABELS or bool(
        (visual_label or {}).get("segmentation_or_instance_issue")
    ):
        reasons.append("subtype_geometry_quality_uncertain")
        return "geometry_quality_uncertain", reasons
    if predicate_label == "lying on" or subject_label in SOFT_LABELS or object_label in SOFT_LABELS:
        reasons.append("subtype_soft_support_contact")
        return "soft_support_contact", reasons
    if object_label == "floor":
        reasons.append("subtype_legged_floor_support")
        return "legged_floor_support", reasons
    reasons.append("subtype_rigid_object_on_furniture")
    if object_label in FURNITURE_SUPPORT_LABELS:
        reasons.append("furniture_support_object")
    return "rigid_object_on_furniture", reasons


def point_metrics(point_record: dict[str, Any] | None) -> dict[str, Any]:
    best = (point_record or {}).get("best_local_support_evidence") or {}
    return {
        "support_points_under_subject_count": best.get("support_points_under_subject_count"),
        "xy_expansion_m": best.get("xy_expansion_m"),
        "local_vertical_gap_p05_p95": best.get("local_vertical_gap_p05_p95"),
        "local_vertical_gap_p01_p99": best.get("local_vertical_gap_p01_p99"),
        "subject_point_stats": (point_record or {}).get("subject_point_stats"),
        "object_point_stats": (point_record or {}).get("object_point_stats"),
    }


def visual_reason_codes(visual_label: dict[str, Any] | None) -> list[str]:
    if not visual_label:
        return []
    kind = label(visual_label.get("inspection_label"))
    codes: list[str] = []
    if kind in RULE_TOO_STRICT_LABELS:
        codes.append("visual_rule_too_strict")
    if kind in LOCAL_SURFACE_LABELS:
        codes.append("visual_local_surface_issue")
    if kind in GEOMETRY_QUALITY_LABELS:
        codes.append("visual_geometry_quality_issue")
    return codes


def legged_score(
    point_record: dict[str, Any] | None,
    thresholds: dict[str, Any],
    visual_label: dict[str, Any] | None,
) -> tuple[float, dict[str, Any], list[str], list[str], list[str], list[str]]:
    metrics = point_metrics(point_record)
    low_gap = metrics["local_vertical_gap_p01_p99"]
    robust_gap = metrics["local_vertical_gap_p05_p95"]
    count = metrics["support_points_under_subject_count"]

    leg_contact_score = score_abs(
        low_gap,
        float(thresholds["low_gap_pass_abs_m"]),
        float(thresholds["low_gap_fail_abs_m"]),
    )
    support_density_score = score_count(
        count,
        int(thresholds["support_density_good_count"]),
    )
    contact_fraction_score = leg_contact_score
    score = clamp(0.70 * leg_contact_score + 0.30 * support_density_score)

    reason_codes = ["subtype_legged_floor_support"]
    passed: list[str] = []
    failed: list[str] = []
    uncertain: list[str] = []
    if leg_contact_score >= 0.70:
        passed.append("leg_contact_score")
        reason_codes.append("leg_contact_low_percentile_supported")
    else:
        uncertain.append("leg_contact_score")
    if support_density_score >= 0.70:
        passed.append("support_density_score")
    else:
        uncertain.append("support_density_score")
    if robust_gap is not None and abs(float(robust_gap)) > float(thresholds["robust_gap_pass_abs_m"]):
        reason_codes.append("robust_gap_too_strict_for_legs")

    # Low-percentile evidence alone is insufficient for a hard violation.
    if score < float(thresholds["uncertain_score_min"]):
        score = float(thresholds["uncertain_score_min"])
        uncertain.append("legged_floor_single_scan_violation")

    score_components = {
        "low_percentile_gap_m": low_gap,
        "robust_gap_m": robust_gap,
        "support_density_score": support_density_score,
        "contact_fraction_score": contact_fraction_score,
        "leg_contact_score": leg_contact_score,
        "visual_label": (visual_label or {}).get("inspection_label"),
    }
    return score, score_components, reason_codes, passed, failed, uncertain


def soft_score(
    point_record: dict[str, Any] | None,
    thresholds: dict[str, Any],
    visual_label: dict[str, Any] | None,
) -> tuple[float, dict[str, Any], list[str], list[str], list[str], list[str]]:
    metrics = point_metrics(point_record)
    signed_gap = metrics["local_vertical_gap_p05_p95"]
    count = metrics["support_points_under_subject_count"]
    penetration = max(0.0, -float(signed_gap)) if signed_gap is not None else None
    positive_float = max(0.0, float(signed_gap)) if signed_gap is not None else None
    soft_prior = 1.0

    if signed_gap is None:
        soft_gap_score = 0.0
    elif float(signed_gap) < 0:
        if penetration <= float(thresholds["soft_penetration_pass_m"]):
            soft_gap_score = 1.0
        elif penetration >= float(thresholds["soft_penetration_max_m"]):
            soft_gap_score = 0.55
        else:
            span = float(thresholds["soft_penetration_max_m"]) - float(
                thresholds["soft_penetration_pass_m"]
            )
            soft_gap_score = 1.0 - 0.45 * (
                (penetration - float(thresholds["soft_penetration_pass_m"])) / span
            )
    else:
        soft_gap_score = score_abs(
            positive_float,
            float(thresholds["positive_float_pass_m"]),
            float(thresholds["positive_float_fail_m"]),
        )

    support_density_score = score_count(
        count,
        int(thresholds["support_density_good_count"]),
    )
    score = clamp(0.55 * soft_gap_score + 0.30 * support_density_score + 0.15 * soft_prior)

    reason_codes = ["subtype_soft_support_contact"]
    passed: list[str] = []
    failed: list[str] = []
    uncertain: list[str] = []
    if signed_gap is not None and float(signed_gap) < 0:
        reason_codes.append("soft_penetration_allowed")
    if positive_float is not None and positive_float > float(thresholds["positive_float_fail_m"]):
        reason_codes.append("positive_float_gap_large")
        failed.append("positive_float_gap")
    else:
        passed.append("soft_gap_score")
    if support_density_score >= 0.70:
        passed.append("support_density_score")
    else:
        uncertain.append("support_density_score")

    score_components = {
        "signed_gap_m": signed_gap,
        "penetration_depth_m": penetration,
        "positive_float_gap_m": positive_float,
        "soft_prior": soft_prior,
        "soft_gap_score": soft_gap_score,
        "support_density_score": support_density_score,
        "visual_label": (visual_label or {}).get("inspection_label"),
    }
    return score, score_components, reason_codes, passed, failed, uncertain


def rigid_score(
    point_record: dict[str, Any] | None,
    points_by_object: dict[int, dict[str, list[float]]],
    thresholds: dict[str, Any],
    visual_label: dict[str, Any] | None,
) -> tuple[float | None, dict[str, Any], list[str], list[str], list[str], list[str]]:
    metrics = point_metrics(point_record)
    subject_stats = metrics.get("subject_point_stats") or {}
    object_id = int((point_record or {}).get("object_id") or -1)
    local_z = local_support_z_values(
        subject_stats,
        points_by_object.get(object_id),
        float(thresholds["plane_expansion_m"]),
    )
    plane = estimate_horizontal_plane(local_z, subject_stats, thresholds)
    plane_gap = plane["plane_gap_m"]
    plane_gap_score = score_abs(
        plane_gap,
        float(thresholds["plane_gap_pass_abs_m"]),
        float(thresholds["plane_gap_fail_abs_m"]),
    )
    support_density_score = score_count(
        len(local_z),
        int(thresholds["support_density_good_count"]),
    )
    score = clamp(
        0.55 * plane_gap_score
        + 0.35 * float(plane["plane_confidence"])
        + 0.10 * support_density_score
    )

    reason_codes = ["subtype_rigid_object_on_furniture"]
    passed: list[str] = []
    failed: list[str] = []
    uncertain: list[str] = []
    if plane["plane_available"] and plane["plane_confidence"] >= 0.40:
        reason_codes.append("horizontal_plane_found")
        passed.append("horizontal_plane")
    else:
        reason_codes.append("horizontal_plane_missing")
        uncertain.append("horizontal_plane")
    if plane_gap_score >= 0.70:
        reason_codes.append("plane_gap_supported")
        passed.append("plane_gap")
    elif plane_gap_score <= 0.20:
        reason_codes.append("plane_gap_large")
        failed.append("plane_gap")
    else:
        uncertain.append("plane_gap")

    if label((visual_label or {}).get("inspection_label")) in LOCAL_SURFACE_LABELS:
        reason_codes.append("surface_estimator_uncertain")
        # If a horizontal plane cannot be recovered, retain an uncertain status.
        if score < float(thresholds["uncertain_score_min"]):
            score = float(thresholds["uncertain_score_min"])

    score_components = {
        **plane,
        "plane_gap_score": plane_gap_score,
        "support_density_score": support_density_score,
        "local_support_point_count": len(local_z),
        "visual_label": (visual_label or {}).get("inspection_label"),
    }
    return score, score_components, reason_codes, passed, failed, uncertain


def geometry_quality_record(
    visual_label: dict[str, Any] | None,
) -> tuple[None, dict[str, Any], list[str], list[str], list[str], list[str]]:
    return (
        None,
        {
            "geometry_issue_source": "visual_inspection",
            "point_density_flag": None,
            "instance_completeness_flag": None,
            "visual_ambiguity_flag": True,
            "visual_label": (visual_label or {}).get("inspection_label"),
        },
        ["subtype_geometry_quality_uncertain", "visual_geometry_quality_issue"],
        [],
        [],
        ["geometry_quality"],
    )


def support_v2_record(
    edge: dict[str, Any],
    point_record: dict[str, Any] | None,
    visual_label: dict[str, Any] | None,
    points_by_object: dict[int, dict[str, list[float]]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    old_verification = edge.get("verification") or {}
    subtype, subtype_reason_codes = assign_subtype(edge, point_record, visual_label)

    if subtype == "geometry_quality_uncertain":
        score, components, reason_codes, passed, failed, uncertain = geometry_quality_record(
            visual_label
        )
        status = "uncertain"
    elif subtype == "legged_floor_support":
        score, components, reason_codes, passed, failed, uncertain = legged_score(
            point_record, thresholds, visual_label
        )
        status = status_from_score(score, thresholds)
    elif subtype == "soft_support_contact":
        score, components, reason_codes, passed, failed, uncertain = soft_score(
            point_record, thresholds, visual_label
        )
        status = status_from_score(score, thresholds)
    else:
        score, components, reason_codes, passed, failed, uncertain = rigid_score(
            point_record, points_by_object, thresholds, visual_label
        )
        if not components.get("plane_available"):
            status = "uncertain"
        else:
            status = status_from_score(score, thresholds)

    all_reason_codes = sorted(
        set(subtype_reason_codes + reason_codes + visual_reason_codes(visual_label))
    )
    geometry_quality_flags = {
        "visual_label": (visual_label or {}).get("inspection_label"),
        "relation_visually_plausible": (visual_label or {}).get("relation_visually_plausible"),
        "local_surface_correct": (visual_label or {}).get("local_surface_correct"),
        "segmentation_or_instance_issue": (visual_label or {}).get(
            "segmentation_or_instance_issue"
        ),
    }
    metrics = point_metrics(point_record)

    return {
        "edge_id": edge.get("edge_id"),
        "subject_id": edge.get("subject_id"),
        "object_id": edge.get("object_id"),
        "subject_label": edge.get("subject_label"),
        "predicate_label": edge.get("predicate_label"),
        "object_label": edge.get("object_label"),
        "previous_status": old_verification.get("status"),
        "previous_rule_version": old_verification.get("rule_version"),
        "subtype": subtype,
        "subtype_reason_codes": sorted(set(subtype_reason_codes)),
        "point_evidence_available": bool((point_record or {}).get("point_evidence_available")),
        "visual_label": (visual_label or {}).get("inspection_label"),
        "geometry_quality_flags": geometry_quality_flags,
        "consistency_score": score,
        "score_components": components,
        "status": status,
        "reason_codes": all_reason_codes,
        **metrics,
    }
