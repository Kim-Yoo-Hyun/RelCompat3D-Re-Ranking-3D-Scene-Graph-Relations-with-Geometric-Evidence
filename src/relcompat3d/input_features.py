"""Feature definitions shared by the RelCompat3D estimators."""

from __future__ import annotations

import math
from typing import Any


FAMILIES = ("proximity", "relative_vertical", "support_contact")
PREDICATES = (
    "close by",
    "higher than",
    "lower than",
    "lying on",
    "standing on",
    "supported by",
)
GEOMETRY_FEATURES = (
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
    "abs_center_delta_z",
    "abs_normalized_center_delta_z",
    "abs_vertical_gap_subject_on_object",
)
SIGNED_HEIGHT_FEATURES = (
    "predicate_aligned_center_delta_z",
    "predicate_aligned_normalized_center_delta_z",
)
FEATURE_SETS = {
    "M_T": {"use_t": True, "numeric": ()},
    "M_G": {"use_t": False, "numeric": GEOMETRY_FEATURES},
    "M_add": {"use_t": True, "numeric": GEOMETRY_FEATURES},
    "M_int": {
        "use_t": True,
        "numeric": GEOMETRY_FEATURES + SIGNED_HEIGHT_FEATURES,
    },
}


def numeric_stats(
    rows: list[dict[str, Any]], names: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    """Compute training-split normalization statistics."""
    stats: dict[str, dict[str, float]] = {}
    for name in names:
        values = [
            row["_raw_numeric"][name]
            for row in rows
            if name in row["_raw_numeric"]
        ]
        mean = sum(values) / len(values) if values else 0.0
        variance = (
            sum((value - mean) ** 2 for value in values) / len(values)
            if values
            else 0.0
        )
        stats[name] = {
            "mean": mean,
            "std": math.sqrt(variance) if variance > 0.0 else 1.0,
            "observed_train_rows": len(values),
        }
    return stats


def feature_names(spec: dict[str, Any]) -> list[str]:
    """Return the ordered feature names for a feature specification."""
    names = ["bias"]
    if spec["use_t"]:
        names.extend(f"family:{value}" for value in FAMILIES)
        names.extend(f"predicate:{value}" for value in PREDICATES)
    names.extend(f"num:{value}" for value in spec["numeric"])
    return names


def vectorize(
    row: dict[str, Any],
    spec: dict[str, Any],
    stats: dict[str, dict[str, float]],
) -> list[float]:
    """Convert one training row to the requested feature vector."""
    vector = [1.0]
    if spec["use_t"]:
        family = row["predicate"]["predicate_family"]
        predicate = row["predicate"]["predicate_label"]
        vector.extend(1.0 if family == value else 0.0 for value in FAMILIES)
        vector.extend(1.0 if predicate == value else 0.0 for value in PREDICATES)
    raw = row["_raw_numeric"]
    for name in spec["numeric"]:
        mean = stats[name]["mean"]
        value = raw.get(name, mean)
        vector.append((value - mean) / stats[name]["std"])
    return vector
