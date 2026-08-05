#!/usr/bin/env python3
"""Point-level support measurements used by the RelCompat3D verifier."""

from __future__ import annotations

from typing import Any


DEFAULT_POINT_THRESHOLDS = {
    "point_rule_version": "ply_points_v1",
    "local_vertical_gap_abs_max_m": 0.10,
    "local_vertical_gap_abs_relaxed_m": 0.15,
    "min_support_points_under_subject": 10,
    "max_expansion_for_primary_m": 0.10,
    "xy_expansion_steps_m": [0.00, 0.05, 0.10, 0.20],
}


def percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * pct / 100.0
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def axis_stats(values: list[float], percentiles: tuple[int, ...]) -> dict[str, float | None]:
    if not values:
        return {f"p{pct:02d}": None for pct in percentiles} | {"min": None, "max": None}
    sorted_values = sorted(values)
    stats: dict[str, float | None] = {
        "min": sorted_values[0],
        "max": sorted_values[-1],
    }
    for pct in percentiles:
        stats[f"p{pct:02d}"] = percentile(sorted_values, pct)
    return stats


def compute_object_stats(points: dict[str, list[float]]) -> dict[str, Any]:
    x_stats = axis_stats(points["x"], (5, 95))
    y_stats = axis_stats(points["y"], (5, 95))
    z_stats = axis_stats(points["z"], (1, 5, 50, 95, 99))

    def area(x_key_low: str, x_key_high: str, y_key_low: str, y_key_high: str) -> float | None:
        x_low = x_stats[x_key_low]
        x_high = x_stats[x_key_high]
        y_low = y_stats[y_key_low]
        y_high = y_stats[y_key_high]
        if None in (x_low, x_high, y_low, y_high):
            return None
        return max(0.0, float(x_high) - float(x_low)) * max(0.0, float(y_high) - float(y_low))

    return {
        "point_count": len(points["x"]),
        "x_min": x_stats["min"],
        "x_max": x_stats["max"],
        "y_min": y_stats["min"],
        "y_max": y_stats["max"],
        "z_min": z_stats["min"],
        "z_max": z_stats["max"],
        "x_p05": x_stats["p05"],
        "x_p95": x_stats["p95"],
        "y_p05": y_stats["p05"],
        "y_p95": y_stats["p95"],
        "z_p01": z_stats["p01"],
        "z_p05": z_stats["p05"],
        "z_p50": z_stats["p50"],
        "z_p95": z_stats["p95"],
        "z_p99": z_stats["p99"],
        "xy_footprint_area_p05_p95": area("p05", "p95", "p05", "p95"),
        "xy_footprint_area_min_max": area("min", "max", "min", "max"),
    }


def local_support_stats(
    subject_stats: dict[str, Any],
    support_points: dict[str, list[float]],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if any(subject_stats.get(key) is None for key in ("x_p05", "x_p95", "y_p05", "y_p95")):
        return results

    for expansion in thresholds["xy_expansion_steps_m"]:
        x_min = float(subject_stats["x_p05"]) - float(expansion)
        x_max = float(subject_stats["x_p95"]) + float(expansion)
        y_min = float(subject_stats["y_p05"]) - float(expansion)
        y_max = float(subject_stats["y_p95"]) + float(expansion)
        local_z: list[float] = []
        for x, y, z in zip(support_points["x"], support_points["y"], support_points["z"]):
            if x_min <= x <= x_max and y_min <= y <= y_max:
                local_z.append(z)
        local_z_sorted = sorted(local_z)
        support_z_p50 = percentile(local_z_sorted, 50)
        support_z_p95 = percentile(local_z_sorted, 95)
        support_z_p99 = percentile(local_z_sorted, 99)
        subject_z_p05 = subject_stats.get("z_p05")
        subject_z_p01 = subject_stats.get("z_p01")
        gap_p05_p95 = (
            float(subject_z_p05) - float(support_z_p95)
            if subject_z_p05 is not None and support_z_p95 is not None
            else None
        )
        gap_p01_p99 = (
            float(subject_z_p01) - float(support_z_p99)
            if subject_z_p01 is not None and support_z_p99 is not None
            else None
        )
        results.append(
            {
                "xy_expansion_m": expansion,
                "support_points_under_subject_count": len(local_z),
                "support_points_under_subject_z_p50": support_z_p50,
                "support_points_under_subject_z_p95": support_z_p95,
                "support_points_under_subject_z_p99": support_z_p99,
                "local_vertical_gap_p05_p95": gap_p05_p95,
                "local_vertical_gap_p01_p99": gap_p01_p99,
            }
        )
    return results


def assign_point_status(
    local_evidence: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> tuple[str, list[str], dict[str, Any] | None]:
    min_points = int(thresholds["min_support_points_under_subject"])
    max_primary_expansion = float(thresholds["max_expansion_for_primary_m"])
    max_gap = float(thresholds["local_vertical_gap_abs_max_m"])
    relaxed_gap = float(thresholds["local_vertical_gap_abs_relaxed_m"])

    enough_points = [
        record
        for record in local_evidence
        if int(record["support_points_under_subject_count"]) >= min_points
        and record["local_vertical_gap_p05_p95"] is not None
    ]
    if not enough_points:
        return "point_uncertain", ["sparse_local_support_points"], None

    best = enough_points[0]
    gap = abs(float(best["local_vertical_gap_p05_p95"]))
    expansion = float(best["xy_expansion_m"])
    if expansion <= max_primary_expansion and gap <= max_gap:
        return "point_satisfied", ["local_support_gap_within_threshold"], best
    if expansion <= max_primary_expansion and gap <= relaxed_gap:
        return "point_uncertain", ["local_support_gap_relaxed_band"], best
    if expansion > max_primary_expansion and gap <= max_gap:
        return "point_uncertain", ["support_points_only_after_large_expansion"], best
    return "point_violated", ["local_support_gap_too_large"], best
