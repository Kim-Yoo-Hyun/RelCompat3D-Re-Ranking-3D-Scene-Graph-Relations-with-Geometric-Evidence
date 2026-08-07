#!/usr/bin/env python3
"""Export RelCompat3D calibration rows and high-margin counterfactual negatives."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from paths import RelCompat3D_HYPOTHESIS_ROOT


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
RelCompat3D_ROOT = RelCompat3D_HYPOTHESIS_ROOT
DEFAULT_DATASET_ROOT = REPO_ROOT / "local_dataset"
DEFAULT_SUBSET_JSON = DEFAULT_DATASET_ROOT / "3DSSG_subset" / "relationships_validation.json"
DEFAULT_RELATIONSHIPS_FILE = DEFAULT_DATASET_ROOT / "3DSSG_subset" / "relationships.txt"
DEFAULT_SELECTED_SCANS = RelCompat3D_ROOT / "artifacts" / "subset" / "relcompat3d_mini" / "scans.txt"
DEFAULT_OUTPUT_DIR = RelCompat3D_ROOT / "artifacts" / "calibration" / "mini_schema_smoke"

ROW_SCHEMA_VERSION = "relcompat3d_calibration_row_v1"
MANIFEST_SCHEMA_VERSION = "relcompat3d_calibration_manifest_v1"
NEGATIVE_SCHEMA_VERSION = "relcompat3d_counterfactual_negative_v1"
EXPORTER_VERSION = "v0"
NEGATIVE_POLICY_VERSION = "relcompat3d-counterfactual-v1"

SUPPORT_CONTACT = {"standing on", "lying on", "supported by"}
PROXIMITY = {"close by"}
RELATIVE_VERTICAL = {"higher than", "lower than"}
ALLOWED_FAMILIES = {"support_contact", "proximity", "relative_vertical"}

SUPPORT_NORM_XY_MIN = 2.0
SUPPORT_ABS_VERTICAL_GAP_MIN = 0.30
PROXIMITY_NORM_XY_MIN = 2.5
VERTICAL_ABS_DELTA_Z_MIN = 0.25
VERTICAL_ABS_NORM_DELTA_Z_MIN = 0.15
ZERO_OVERLAP_EPS = 1e-9


@dataclass(frozen=True)
class SubgraphContext:
    scan_id: str
    subset_split_id: int
    subgraph_id: str
    object_labels: dict[int, str]
    geometries: dict[int, dict[str, Any]]
    positive_edges: set[tuple[int, int, str]]
    support_contact_pairs: set[frozenset[int]]
    source_relation_ids: set[str]


@dataclass(frozen=True)
class PositiveSpec:
    row: dict[str, Any]
    context: SubgraphContext
    relation_index: int
    source_relation_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export RelCompat3D calibration table rows for schema smoke checks."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--subset-json", type=Path, default=DEFAULT_SUBSET_JSON)
    parser.add_argument("--relationships-file", type=Path, default=DEFAULT_RELATIONSHIPS_FILE)
    parser.add_argument("--selected-scans", type=Path, default=DEFAULT_SELECTED_SCANS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-name", default="mini_schema_smoke")
    parser.add_argument(
        "--include-counterfactuals",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-negatives-per-positive", type=int, default=2)
    parser.add_argument("--max-negatives-per-subgraph-family", type=int, default=200)
    parser.add_argument("--max-negative-to-positive-ratio-per-family", type=float, default=3.0)
    parser.add_argument(
        "--allow-selected-scans-without-positive-rows",
        action="store_true",
        help=(
            "Keep zero-positive scans in the data split separation without treating their absence "
            "from the row table as an export failure. The selected scan list is unchanged."
        ),
    )
    parser.add_argument("--fail-on-warnings", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def as_float_list(values: list[Any] | None, expected_len: int) -> list[float] | None:
    if values is None or len(values) != expected_len:
        return None
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        return None


def vector_sub(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def euclidean(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def safe_div(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return num / den


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def interval_intersection(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    return max(0.0, min(a_max, b_max) - max(a_min, b_min))


def volume(aabb_min: list[float], aabb_max: list[float], dims: tuple[int, ...]) -> float:
    result = 1.0
    for dim in dims:
        result *= max(0.0, aabb_max[dim] - aabb_min[dim])
    return result


def intersection_volume(
    a_min: list[float],
    a_max: list[float],
    b_min: list[float],
    b_max: list[float],
    dims: tuple[int, ...],
) -> float:
    result = 1.0
    for dim in dims:
        result *= interval_intersection(a_min[dim], a_max[dim], b_min[dim], b_max[dim])
    return result


def iou(
    a_min: list[float],
    a_max: list[float],
    b_min: list[float],
    b_max: list[float],
    dims: tuple[int, ...],
) -> float | None:
    inter = intersection_volume(a_min, a_max, b_min, b_max, dims)
    a_vol = volume(a_min, a_max, dims)
    b_vol = volume(b_min, b_max, dims)
    union = a_vol + b_vol - inter
    return safe_div(inter, union) if union > 0 else None


def predicate_family(label: str) -> str:
    if label in SUPPORT_CONTACT:
        return "support_contact"
    if label in PROXIMITY:
        return "proximity"
    if label in RELATIVE_VERTICAL:
        return "relative_vertical"
    return "unsupported_first_pass"


def derive_aabb_from_obb(obb: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    centroid = as_float_list(obb.get("centroid"), 3)
    axes_lengths = as_float_list(obb.get("axesLengths"), 3)
    normalized_axes = as_float_list(obb.get("normalizedAxes"), 9)

    if centroid is None:
        return None, ["missing_or_invalid_centroid"]
    if axes_lengths is None:
        return None, ["missing_or_invalid_axes_lengths"]
    if normalized_axes is None:
        return None, ["missing_or_invalid_normalized_axes"]
    if any(length <= 0 for length in axes_lengths):
        return None, ["non_positive_axes_length"]

    rows = [normalized_axes[0:3], normalized_axes[3:6], normalized_axes[6:9]]
    for idx, row in enumerate(rows):
        norm = euclidean(row)
        if abs(norm - 1.0) > 0.05:
            warnings.append(f"obb_axis_{idx}_norm_{norm:.4f}")
    for i in range(3):
        for j in range(i + 1, 3):
            axis_dot = abs(dot(rows[i], rows[j]))
            if axis_dot > 0.05:
                warnings.append(f"obb_axes_{i}_{j}_dot_{axis_dot:.4f}")

    half_lengths = [length / 2.0 for length in axes_lengths]
    half_extent_world = [
        sum(abs(rows[i][j]) * half_lengths[j] for j in range(3))
        for i in range(3)
    ]
    aabb_min = [centroid[i] - half_extent_world[i] for i in range(3)]
    aabb_max = [centroid[i] + half_extent_world[i] for i in range(3)]
    size_xyz = [aabb_max[i] - aabb_min[i] for i in range(3)]
    if any(size <= 0 for size in size_xyz):
        return None, warnings + ["non_positive_aabb_extent"]

    diag_3d = euclidean(size_xyz)
    diag_xy = math.sqrt(size_xyz[0] * size_xyz[0] + size_xyz[1] * size_xyz[1])
    return {
        "center_xyz": centroid,
        "aabb_min_xyz": aabb_min,
        "aabb_max_xyz": aabb_max,
        "size_xyz": size_xyz,
        "height_z": size_xyz[2],
        "diag_3d": diag_3d,
        "diag_xy": diag_xy,
    }, warnings


def compute_features(subject_geom: dict[str, Any], object_geom: dict[str, Any]) -> dict[str, Any]:
    s_min = subject_geom["aabb_min_xyz"]
    s_max = subject_geom["aabb_max_xyz"]
    o_min = object_geom["aabb_min_xyz"]
    o_max = object_geom["aabb_max_xyz"]
    s_center = subject_geom["center_xyz"]
    o_center = object_geom["center_xyz"]
    delta = vector_sub(s_center, o_center)

    distance_3d = euclidean(delta)
    distance_xy = math.sqrt(delta[0] * delta[0] + delta[1] * delta[1])
    mean_diag_3d = (subject_geom["diag_3d"] + object_geom["diag_3d"]) / 2.0
    mean_diag_xy = (subject_geom["diag_xy"] + object_geom["diag_xy"]) / 2.0
    mean_height = (subject_geom["height_z"] + object_geom["height_z"]) / 2.0

    inter_xy = intersection_volume(s_min, s_max, o_min, o_max, (0, 1))
    subject_xy_area = volume(s_min, s_max, (0, 1))
    object_xy_area = volume(o_min, o_max, (0, 1))

    return {
        "distance_3d": distance_3d,
        "distance_xy": distance_xy,
        "normalized_distance_3d": safe_div(distance_3d, mean_diag_3d),
        "normalized_distance_xy": safe_div(distance_xy, mean_diag_xy),
        "center_delta_z": delta[2],
        "normalized_center_delta_z": safe_div(delta[2], mean_height),
        "projected_iou_xy": iou(s_min, s_max, o_min, o_max, (0, 1)),
        "projected_subject_overlap_ratio": safe_div(inter_xy, subject_xy_area),
        "projected_object_overlap_ratio": safe_div(inter_xy, object_xy_area),
        "vertical_gap_subject_on_object": s_min[2] - o_max[2],
        "subject_bottom_z": s_min[2],
        "subject_top_z": s_max[2],
        "object_bottom_z": o_min[2],
        "object_top_z": o_max[2],
    }


def load_relationship_id_map(path: Path) -> dict[str, int]:
    labels = path.read_text(encoding="utf-8").splitlines()
    return {label.strip(): idx for idx, label in enumerate(labels) if label.strip()}


def load_selected_scans(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    selected = {
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    }
    return selected


def load_scan_geometries(
    dataset_root: Path,
    scan_id: str,
) -> tuple[dict[int, dict[str, Any]], list[str], list[str]]:
    semseg_path = dataset_root / "3RScan" / "scans" / scan_id / "semseg.v2.json"
    errors: list[str] = []
    warnings: list[str] = []
    if not semseg_path.exists():
        return {}, [], [f"missing_semseg:{scan_id}:{relpath(semseg_path)}"]

    semseg_data = load_json(semseg_path)
    geometries: dict[int, dict[str, Any]] = {}
    for semseg_obj in semseg_data.get("segGroups", []):
        object_id = int(semseg_obj["objectId"])
        geometry, obb_warnings = derive_aabb_from_obb(semseg_obj.get("obb", {}))
        if geometry is None:
            warnings.append(f"invalid_obb:{scan_id}:{object_id}:{','.join(obb_warnings)}")
            continue
        geometries[object_id] = geometry
        for warning in obb_warnings:
            warnings.append(f"obb_warning:{scan_id}:{object_id}:{warning}")
    if not geometries:
        errors.append(f"zero_valid_geometries:{scan_id}")
    return geometries, warnings, errors


def relation_parts(row: list[Any]) -> tuple[int, int, int, str]:
    return int(row[0]), int(row[1]), int(row[2]), str(row[3])


def source_relation_id(scan_id: str, subset_split_id: int, relation_index: int) -> str:
    return f"3dssg_subset:{scan_id}:{subset_split_id}:{relation_index}"


def candidate_id(
    split_name: str,
    scan_id: str,
    subset_split_id: int,
    subject_id: int,
    object_id: int,
    predicate_label: str,
    suffix: str,
) -> str:
    return (
        f"calib:{split_name}:{scan_id}:{subset_split_id}:"
        f"{subject_id}:{object_id}:{predicate_label}:{suffix}"
    )


def negative_id(
    split_name: str,
    scan_id: str,
    subset_split_id: int,
    subject_id: int,
    object_id: int,
    predicate_label: str,
    strategy: str,
    base_relation_index: int,
) -> str:
    return (
        f"neg:{split_name}:{scan_id}:{subset_split_id}:"
        f"{subject_id}:{object_id}:{predicate_label}:{strategy}:{base_relation_index}"
    )


def calibration_role_for_split(split_name: str) -> str:
    if split_name == "mini_schema_smoke":
        return "validation_smoke"
    if split_name == "train_dev_calib":
        return "train_dev_candidate"
    return "candidate_export"


def make_row(
    *,
    split_name: str,
    subset_source: str,
    context: SubgraphContext,
    subject_id: int,
    object_id: int,
    predicate_label: str,
    raw_predicate_id: int,
    relation_index: int,
    relationship_id_map: dict[str, int],
    candidate_suffix: str,
    candidate_source: dict[str, Any],
    label: dict[str, Any],
    negative: dict[str, Any] | None,
    features: dict[str, Any],
    created_at: str,
) -> dict[str, Any]:
    family = predicate_family(predicate_label)
    cid = candidate_id(
        split_name,
        context.scan_id,
        context.subset_split_id,
        subject_id,
        object_id,
        predicate_label,
        candidate_suffix,
    )
    raw_id_from_file = relationship_id_map.get(predicate_label)
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "record_type": "calibration_candidate",
        "candidate_id": cid,
        "split_name": split_name,
        "subset_source": subset_source,
        "scan_id": context.scan_id,
        "subset_split_id": context.subset_split_id,
        "subgraph_id": context.subgraph_id,
        "edge": {
            "subject_id": subject_id,
            "object_id": object_id,
            "subject_label": context.object_labels.get(subject_id),
            "object_label": context.object_labels.get(object_id),
        },
        "predicate": {
            "predicate_label": predicate_label,
            "predicate_family": family,
            "raw_3dssg_predicate_id": raw_predicate_id,
            "vlsat_predicate_index": raw_predicate_id - 1 if raw_predicate_id > 0 else None,
            "relationships_txt_index": raw_id_from_file,
        },
        "candidate_source": candidate_source,
        "semantic": {
            "baseline_name": None,
            "prediction_id": None,
            "p_semantic": None,
            "rank": None,
            "predicate_rank_for_pair": None,
            "semantic_rank_in_subgraph": None,
        },
        "geometry": {
            "geometry_available": True,
            "geometry_source": "semseg_obb_v0",
            "verifier_version": None,
            "verification_status": None,
            "consistency_score": None,
            "support_subtype": None,
            "reason_codes": [],
            "features": {
                "predicate_family": family,
                "predicate_label": predicate_label,
                "subject_label": context.object_labels.get(subject_id),
                "object_label": context.object_labels.get(object_id),
                "geometry_available": True,
                "consistency_score": None,
                "verification_status": None,
                "geometry_quality_flags": {},
                "support_points_under_subject_count": None,
                "local_vertical_gap_p05_p95": None,
                "local_vertical_gap_p01_p99": None,
                "support_density_score": None,
                "low_percentile_gap_m": None,
                "robust_gap_m": None,
                "leg_contact_score": None,
                "soft_gap_score": None,
                "plane_available": None,
                "plane_inlier_ratio": None,
                "plane_residual_m": None,
                "plane_gap_m": None,
                "plane_confidence": None,
                **features,
            },
        },
        "label": label,
        "negative": negative,
        "quality": {
            "abstain_reason": None,
            "geometry_quality_flags": {},
            "leakage_group": "scan_id",
            "calibration_role": calibration_role_for_split(split_name),
        },
        "provenance": {
            "exporter": "build_training_rows",
            "exporter_version": EXPORTER_VERSION,
            "created_at": created_at,
            "source_relation_index": relation_index,
        },
    }


def zero_overlap(features: dict[str, Any]) -> bool:
    subject_overlap = features.get("projected_subject_overlap_ratio")
    object_overlap = features.get("projected_object_overlap_ratio")
    return (
        subject_overlap is not None
        and object_overlap is not None
        and subject_overlap <= ZERO_OVERLAP_EPS
        and object_overlap <= ZERO_OVERLAP_EPS
    )


def support_margin(features: dict[str, Any]) -> tuple[bool, dict[str, Any], list[str]]:
    checks: list[str] = []
    normalized_distance_xy = features.get("normalized_distance_xy")
    vertical_gap = features.get("vertical_gap_subject_on_object")
    if normalized_distance_xy is not None and normalized_distance_xy >= SUPPORT_NORM_XY_MIN:
        checks.append("normalized_distance_xy")
    if zero_overlap(features):
        checks.append("zero_projected_overlap")
    if vertical_gap is not None and abs(vertical_gap) >= SUPPORT_ABS_VERTICAL_GAP_MIN:
        checks.append("abs_vertical_gap")
    margin_fields = {
        "min_normalized_xy_distance": SUPPORT_NORM_XY_MIN,
        "max_projected_overlap_xy": 0.0,
        "min_abs_vertical_gap_m": SUPPORT_ABS_VERTICAL_GAP_MIN,
        "observed_normalized_distance_xy": normalized_distance_xy,
        "observed_projected_subject_overlap_ratio": features.get("projected_subject_overlap_ratio"),
        "observed_projected_object_overlap_ratio": features.get("projected_object_overlap_ratio"),
        "observed_vertical_gap_subject_on_object": vertical_gap,
        "passed_checks": checks,
    }
    return len(checks) >= 2, margin_fields, checks


def proximity_margin(features: dict[str, Any]) -> tuple[bool, dict[str, Any], list[str]]:
    checks: list[str] = []
    normalized_distance_xy = features.get("normalized_distance_xy")
    if normalized_distance_xy is not None and normalized_distance_xy >= PROXIMITY_NORM_XY_MIN:
        checks.append("normalized_distance_xy")
    if zero_overlap(features):
        checks.append("zero_projected_overlap")
    margin_fields = {
        "min_normalized_xy_distance": PROXIMITY_NORM_XY_MIN,
        "max_projected_overlap_xy": 0.0,
        "observed_normalized_distance_xy": normalized_distance_xy,
        "observed_projected_subject_overlap_ratio": features.get("projected_subject_overlap_ratio"),
        "observed_projected_object_overlap_ratio": features.get("projected_object_overlap_ratio"),
        "passed_checks": checks,
    }
    return len(checks) == 2, margin_fields, checks


def vertical_inversion_margin(
    positive_predicate: str,
    features: dict[str, Any],
) -> tuple[bool, dict[str, Any], list[str]]:
    center_delta_z = features.get("center_delta_z")
    normalized_center_delta_z = features.get("normalized_center_delta_z")
    checks: list[str] = []
    if center_delta_z is None or normalized_center_delta_z is None:
        pass
    elif positive_predicate == "higher than":
        if center_delta_z >= VERTICAL_ABS_DELTA_Z_MIN:
            checks.append("center_delta_z_direction")
        if normalized_center_delta_z >= VERTICAL_ABS_NORM_DELTA_Z_MIN:
            checks.append("normalized_center_delta_z_direction")
    elif positive_predicate == "lower than":
        if center_delta_z <= -VERTICAL_ABS_DELTA_Z_MIN:
            checks.append("center_delta_z_direction")
        if normalized_center_delta_z <= -VERTICAL_ABS_NORM_DELTA_Z_MIN:
            checks.append("normalized_center_delta_z_direction")

    margin_fields = {
        "min_abs_center_delta_z_m": VERTICAL_ABS_DELTA_Z_MIN,
        "min_abs_normalized_center_delta_z": VERTICAL_ABS_NORM_DELTA_Z_MIN,
        "observed_center_delta_z": center_delta_z,
        "observed_normalized_center_delta_z": normalized_center_delta_z,
        "passed_checks": checks,
    }
    return len(checks) == 2, margin_fields, checks


def geometry_features_for_pair(
    context: SubgraphContext,
    subject_id: int,
    object_id: int,
) -> dict[str, Any] | None:
    subject_geom = context.geometries.get(subject_id)
    object_geom = context.geometries.get(object_id)
    if subject_geom is None or object_geom is None:
        return None
    return compute_features(subject_geom, object_geom)


def build_contexts_and_positives(
    *,
    subset_data: dict[str, Any],
    selected_scans: set[str] | None,
    dataset_root: Path,
    subset_source: str,
    split_name: str,
    relationship_id_map: dict[str, int],
    created_at: str,
    allow_selected_scans_without_positive_rows: bool = False,
) -> tuple[list[PositiveSpec], dict[str, SubgraphContext], Counter[str], list[str], list[str]]:
    positives: list[PositiveSpec] = []
    contexts: dict[str, SubgraphContext] = {}
    geometry_cache: dict[str, dict[int, dict[str, Any]]] = {}
    warnings: list[str] = []
    errors: list[str] = []
    skipped_positive_counts: Counter[str] = Counter()

    entries = subset_data.get("scans", [])
    for entry in entries:
        scan_id = str(entry.get("scan"))
        if selected_scans is not None and scan_id not in selected_scans:
            continue
        subset_split_id = int(entry.get("split"))
        subgraph_id = f"{scan_id}_{subset_split_id}"
        object_labels = {int(key): str(value) for key, value in entry.get("objects", {}).items()}
        relation_rows = entry.get("relationships", [])

        if scan_id not in geometry_cache:
            geometries, geom_warnings, geom_errors = load_scan_geometries(dataset_root, scan_id)
            geometry_cache[scan_id] = geometries
            warnings.extend(geom_warnings)
            errors.extend(geom_errors)

        positive_edges: set[tuple[int, int, str]] = set()
        support_contact_pairs: set[frozenset[int]] = set()
        source_relation_ids: set[str] = set()
        for relation_index, row in enumerate(relation_rows):
            subject_id, object_id, _, predicate_label = relation_parts(row)
            positive_edges.add((subject_id, object_id, predicate_label))
            source_relation_ids.add(source_relation_id(scan_id, subset_split_id, relation_index))
            if predicate_family(predicate_label) == "support_contact":
                support_contact_pairs.add(frozenset((subject_id, object_id)))

        context = SubgraphContext(
            scan_id=scan_id,
            subset_split_id=subset_split_id,
            subgraph_id=subgraph_id,
            object_labels=object_labels,
            geometries=geometry_cache[scan_id],
            positive_edges=positive_edges,
            support_contact_pairs=support_contact_pairs,
            source_relation_ids=source_relation_ids,
        )
        contexts[subgraph_id] = context

        for relation_index, row in enumerate(relation_rows):
            subject_id, object_id, raw_predicate_id, predicate_label = relation_parts(row)
            family = predicate_family(predicate_label)
            if family not in ALLOWED_FAMILIES:
                continue
            if subject_id == object_id:
                skipped_positive_counts["same_endpoint"] += 1
                continue
            if subject_id not in object_labels or object_id not in object_labels:
                skipped_positive_counts["missing_subgraph_object"] += 1
                continue
            features = geometry_features_for_pair(context, subject_id, object_id)
            if features is None:
                skipped_positive_counts["missing_geometry"] += 1
                continue

            src_id = source_relation_id(scan_id, subset_split_id, relation_index)
            row_record = make_row(
                split_name=split_name,
                subset_source=subset_source,
                context=context,
                subject_id=subject_id,
                object_id=object_id,
                predicate_label=predicate_label,
                raw_predicate_id=raw_predicate_id,
                relation_index=relation_index,
                relationship_id_map=relationship_id_map,
                candidate_suffix=f"positive:{relation_index}",
                candidate_source={
                    "type": "gt_positive",
                    "source_relation_id": src_id,
                    "base_candidate_id": None,
                },
                label={
                    "geom_valid": 1,
                    "label_status": "positive",
                    "label_source": "gt_geometry_checkable_positive",
                    "label_confidence": "default_positive",
                    "manual_label_id": None,
                },
                negative=None,
                features=features,
                created_at=created_at,
            )
            positives.append(
                PositiveSpec(
                    row=row_record,
                    context=context,
                    relation_index=relation_index,
                    source_relation_id=src_id,
                )
            )

    if skipped_positive_counts:
        warnings.append(f"skipped_positive_counts:{dict(sorted(skipped_positive_counts.items()))}")
    if selected_scans is not None:
        matched_scans = {spec.context.scan_id for spec in positives}
        missing_selected = sorted(selected_scans - matched_scans)
        if missing_selected:
            message = f"selected_scans_without_positive_rows:{missing_selected}"
            if allow_selected_scans_without_positive_rows:
                warnings.append(message)
            else:
                errors.append(message)

    return positives, contexts, skipped_positive_counts, warnings, errors


def make_negative_payload(
    *,
    negative_record_id: str,
    strategy: str,
    base_candidate_id: str,
    replacement_type: str,
    margin_fields: dict[str, Any],
    filters_applied: list[str],
) -> dict[str, Any]:
    return {
        "negative_id": negative_record_id,
        "strategy": strategy,
        "base_candidate_id": base_candidate_id,
        "replacement_type": replacement_type,
        "margin_fields": margin_fields,
        "filters_applied": filters_applied,
    }


def make_negative_record(
    *,
    negative_record_id: str,
    base_candidate_id: str,
    strategy: str,
    split_name: str,
    context: SubgraphContext,
    subject_id: int,
    object_id: int,
    predicate_label: str,
    margin_fields: dict[str, Any],
    filters_applied: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": NEGATIVE_SCHEMA_VERSION,
        "record_type": "counterfactual_negative",
        "negative_id": negative_record_id,
        "base_candidate_id": base_candidate_id,
        "strategy": strategy,
        "split_name": split_name,
        "scan_id": context.scan_id,
        "subset_split_id": context.subset_split_id,
        "subject_id": subject_id,
        "object_id": object_id,
        "predicate_label": predicate_label,
        "predicate_family": predicate_family(predicate_label),
        "margin_fields": margin_fields,
        "filters_applied": filters_applied,
        "emit_to_table": True,
        "skip_reason": None,
    }


def invert_vertical_predicate(predicate_label: str) -> str:
    if predicate_label == "higher than":
        return "lower than"
    if predicate_label == "lower than":
        return "higher than"
    raise ValueError(f"Cannot invert vertical predicate {predicate_label!r}")


def generate_negatives(
    *,
    positives: list[PositiveSpec],
    split_name: str,
    subset_source: str,
    relationship_id_map: dict[str, int],
    created_at: str,
    max_negatives_per_positive: int,
    max_negatives_per_subgraph_family: int,
    max_negative_to_positive_ratio_per_family: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    positive_counts_by_family = Counter(
        spec.row["predicate"]["predicate_family"] for spec in positives
    )
    negative_rows: list[dict[str, Any]] = []
    negative_records: list[dict[str, Any]] = []
    skipped_attempt_counts: Counter[str] = Counter()
    emitted_tuple_keys = {
        (
            spec.context.subgraph_id,
            spec.row["edge"]["subject_id"],
            spec.row["edge"]["object_id"],
            spec.row["predicate"]["predicate_label"],
        )
        for spec in positives
    }
    negative_counts_by_family: Counter[str] = Counter()
    negative_counts_by_subgraph_family: Counter[tuple[str, str]] = Counter()

    def can_emit(
        context: SubgraphContext,
        subject_id: int,
        object_id: int,
        predicate_label: str,
        family: str,
        per_positive_count: int,
    ) -> tuple[bool, str | None]:
        if per_positive_count >= max_negatives_per_positive:
            return False, "max_negatives_per_positive"
        if subject_id == object_id:
            return False, "same_endpoint"
        if subject_id not in context.object_labels or object_id not in context.object_labels:
            return False, "missing_subgraph_object"
        if (subject_id, object_id, predicate_label) in context.positive_edges:
            return False, "same_predicate_positive_exists"
        tuple_key = (context.subgraph_id, subject_id, object_id, predicate_label)
        if tuple_key in emitted_tuple_keys:
            return False, "duplicate_candidate_tuple"
        if negative_counts_by_subgraph_family[(context.subgraph_id, family)] >= max_negatives_per_subgraph_family:
            return False, "max_negatives_per_subgraph_family"
        max_for_family = int(positive_counts_by_family[family] * max_negative_to_positive_ratio_per_family)
        if negative_counts_by_family[family] >= max_for_family:
            return False, "max_negative_to_positive_ratio_per_family"
        return True, None

    def emit_negative(
        *,
        base: PositiveSpec,
        subject_id: int,
        object_id: int,
        predicate_label: str,
        raw_predicate_id: int,
        strategy: str,
        replacement_type: str,
        margin_fields: dict[str, Any],
        filters_applied: list[str],
        features: dict[str, Any],
        per_positive_count: int,
    ) -> int:
        context = base.context
        family = predicate_family(predicate_label)
        ok, skip_reason = can_emit(
            context,
            subject_id,
            object_id,
            predicate_label,
            family,
            per_positive_count,
        )
        if not ok:
            skipped_attempt_counts[skip_reason or "unknown"] += 1
            return per_positive_count

        base_candidate_id = base.row["candidate_id"]
        neg_id = negative_id(
            split_name,
            context.scan_id,
            context.subset_split_id,
            subject_id,
            object_id,
            predicate_label,
            strategy,
            base.relation_index,
        )
        negative_payload = make_negative_payload(
            negative_record_id=neg_id,
            strategy=strategy,
            base_candidate_id=base_candidate_id,
            replacement_type=replacement_type,
            margin_fields=margin_fields,
            filters_applied=filters_applied,
        )
        row_record = make_row(
            split_name=split_name,
            subset_source=subset_source,
            context=context,
            subject_id=subject_id,
            object_id=object_id,
            predicate_label=predicate_label,
            raw_predicate_id=raw_predicate_id,
            relation_index=base.relation_index,
            relationship_id_map=relationship_id_map,
            candidate_suffix=f"neg:{strategy}:{base.relation_index}",
            candidate_source={
                "type": "counterfactual_negative",
                "source_relation_id": base.source_relation_id,
                "base_candidate_id": base_candidate_id,
            },
            label={
                "geom_valid": 0,
                "label_status": "negative",
                "label_source": "counterfactual_high_margin",
                "label_confidence": "high_margin",
                "manual_label_id": None,
            },
            negative=negative_payload,
            features=features,
            created_at=created_at,
        )
        negative_record = make_negative_record(
            negative_record_id=neg_id,
            base_candidate_id=base_candidate_id,
            strategy=strategy,
            split_name=split_name,
            context=context,
            subject_id=subject_id,
            object_id=object_id,
            predicate_label=predicate_label,
            margin_fields=margin_fields,
            filters_applied=filters_applied,
        )
        negative_rows.append(row_record)
        negative_records.append(negative_record)
        emitted_tuple_keys.add((context.subgraph_id, subject_id, object_id, predicate_label))
        negative_counts_by_family[family] += 1
        negative_counts_by_subgraph_family[(context.subgraph_id, family)] += 1
        return per_positive_count + 1

    for base in positives:
        context = base.context
        predicate_label = base.row["predicate"]["predicate_label"]
        family = base.row["predicate"]["predicate_family"]
        raw_predicate_id = int(base.row["predicate"]["raw_3dssg_predicate_id"])
        base_subject_id = int(base.row["edge"]["subject_id"])
        base_object_id = int(base.row["edge"]["object_id"])
        emitted_for_positive = 0

        if family == "support_contact":
            sorted_object_ids = sorted(context.object_labels)
            for candidate_object_id in sorted_object_ids:
                if emitted_for_positive >= max_negatives_per_positive:
                    break
                if candidate_object_id in {base_subject_id, base_object_id}:
                    continue
                if frozenset((base_subject_id, candidate_object_id)) in context.support_contact_pairs:
                    skipped_attempt_counts["support_contact_positive_pair_exists"] += 1
                    continue
                if context.object_labels.get(base_subject_id, "").lower() == "floor":
                    skipped_attempt_counts["support_negative_floor_subject"] += 1
                    break
                features = geometry_features_for_pair(context, base_subject_id, candidate_object_id)
                if features is None:
                    skipped_attempt_counts["missing_geometry"] += 1
                    continue
                margin_ok, margin_fields, checks = support_margin(features)
                if not margin_ok:
                    skipped_attempt_counts["support_margin_failed"] += 1
                    continue
                emitted_for_positive = emit_negative(
                    base=base,
                    subject_id=base_subject_id,
                    object_id=candidate_object_id,
                    predicate_label=predicate_label,
                    raw_predicate_id=raw_predicate_id,
                    strategy="support_replace_object_far_or_incompatible",
                    replacement_type="object_id",
                    margin_fields=margin_fields,
                    filters_applied=[
                        "same_subgraph_objects",
                        "not_same_predicate_positive",
                        "no_support_contact_positive_either_direction",
                        "exclude_floor_subject",
                        *checks,
                    ],
                    features=features,
                    per_positive_count=emitted_for_positive,
                )

            for candidate_subject_id in sorted_object_ids:
                if emitted_for_positive >= max_negatives_per_positive:
                    break
                if candidate_subject_id in {base_subject_id, base_object_id}:
                    continue
                if frozenset((candidate_subject_id, base_object_id)) in context.support_contact_pairs:
                    skipped_attempt_counts["support_contact_positive_pair_exists"] += 1
                    continue
                if context.object_labels.get(candidate_subject_id, "").lower() == "floor":
                    skipped_attempt_counts["support_negative_floor_subject"] += 1
                    continue
                features = geometry_features_for_pair(context, candidate_subject_id, base_object_id)
                if features is None:
                    skipped_attempt_counts["missing_geometry"] += 1
                    continue
                margin_ok, margin_fields, checks = support_margin(features)
                if not margin_ok:
                    skipped_attempt_counts["support_margin_failed"] += 1
                    continue
                emitted_for_positive = emit_negative(
                    base=base,
                    subject_id=candidate_subject_id,
                    object_id=base_object_id,
                    predicate_label=predicate_label,
                    raw_predicate_id=raw_predicate_id,
                    strategy="support_replace_subject_floating",
                    replacement_type="subject_id",
                    margin_fields=margin_fields,
                    filters_applied=[
                        "same_subgraph_objects",
                        "not_same_predicate_positive",
                        "no_support_contact_positive_either_direction",
                        "exclude_floor_subject",
                        *checks,
                    ],
                    features=features,
                    per_positive_count=emitted_for_positive,
                )

        elif family == "proximity":
            candidate_pairs: list[tuple[float, int, int, dict[str, Any], dict[str, Any], list[str]]] = []
            object_ids = sorted(context.object_labels)
            for subject_id in object_ids:
                for object_id in object_ids:
                    if subject_id == object_id:
                        continue
                    features = geometry_features_for_pair(context, subject_id, object_id)
                    if features is None:
                        skipped_attempt_counts["missing_geometry"] += 1
                        continue
                    margin_ok, margin_fields, checks = proximity_margin(features)
                    if not margin_ok:
                        continue
                    rank_value = features.get("normalized_distance_xy") or 0.0
                    candidate_pairs.append((rank_value, subject_id, object_id, features, margin_fields, checks))
            candidate_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
            for _, subject_id, object_id, features, margin_fields, checks in candidate_pairs:
                if emitted_for_positive >= max_negatives_per_positive:
                    break
                emitted_for_positive = emit_negative(
                    base=base,
                    subject_id=subject_id,
                    object_id=object_id,
                    predicate_label=predicate_label,
                    raw_predicate_id=raw_predicate_id,
                    strategy="proximity_far_pair",
                    replacement_type="pair",
                    margin_fields=margin_fields,
                    filters_applied=[
                        "same_subgraph_objects",
                        "not_same_predicate_positive",
                        *checks,
                    ],
                    features=features,
                    per_positive_count=emitted_for_positive,
                )

        elif family == "relative_vertical":
            inverted_predicate = invert_vertical_predicate(predicate_label)
            inverted_raw_predicate_id = relationship_id_map[inverted_predicate]
            features = geometry_features_for_pair(context, base_subject_id, base_object_id)
            if features is None:
                skipped_attempt_counts["missing_geometry"] += 1
                continue
            margin_ok, margin_fields, checks = vertical_inversion_margin(predicate_label, features)
            if not margin_ok:
                skipped_attempt_counts["vertical_margin_failed"] += 1
                continue
            emitted_for_positive = emit_negative(
                base=base,
                subject_id=base_subject_id,
                object_id=base_object_id,
                predicate_label=inverted_predicate,
                raw_predicate_id=inverted_raw_predicate_id,
                strategy="vertical_invert_higher_lower",
                replacement_type="predicate_label",
                margin_fields=margin_fields,
                filters_applied=[
                    "same_directed_pair",
                    "not_same_predicate_positive",
                    *checks,
                ],
                features=features,
                per_positive_count=emitted_for_positive,
            )

    return negative_rows, negative_records, skipped_attempt_counts


def validate_export(
    *,
    table_rows: list[dict[str, Any]],
    negative_records: list[dict[str, Any]],
    contexts: dict[str, SubgraphContext],
    selected_scans: set[str] | None,
    split_name: str,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    candidate_counts = Counter(row["candidate_id"] for row in table_rows)
    duplicate_candidates = sorted(key for key, count in candidate_counts.items() if count > 1)
    if duplicate_candidates:
        errors.append(f"duplicate_candidate_ids:{duplicate_candidates[:10]}")

    negative_counts = Counter(record["negative_id"] for record in negative_records)
    duplicate_negatives = sorted(key for key, count in negative_counts.items() if count > 1)
    if duplicate_negatives:
        errors.append(f"duplicate_negative_ids:{duplicate_negatives[:10]}")

    table_negative_ids = {
        row["negative"]["negative_id"]
        for row in table_rows
        if row.get("negative") is not None
    }
    missing_negative_rows = sorted(
        record["negative_id"]
        for record in negative_records
        if record["negative_id"] not in table_negative_ids
    )
    if missing_negative_rows:
        errors.append(f"negative_record_without_table_row:{missing_negative_rows[:10]}")

    candidate_ids = set(candidate_counts)
    positive_candidate_ids = {
        row["candidate_id"]
        for row in table_rows
        if row["candidate_source"]["type"] == "gt_positive"
    }
    for record in negative_records:
        if record["base_candidate_id"] not in positive_candidate_ids:
            errors.append(f"negative_missing_positive_base:{record['negative_id']}")
        if record["emit_to_table"] and record["skip_reason"] is not None:
            errors.append(f"emitted_negative_has_skip_reason:{record['negative_id']}")

    seen_row_tuples: set[tuple[str, int, int, str]] = set()
    positive_edge_tuples = {
        (
            row["subgraph_id"],
            int(row["edge"]["subject_id"]),
            int(row["edge"]["object_id"]),
            row["predicate"]["predicate_label"],
        )
        for row in table_rows
        if row["candidate_source"]["type"] == "gt_positive"
    }
    for row in table_rows:
        cid = row["candidate_id"]
        if row["schema_version"] != ROW_SCHEMA_VERSION:
            errors.append(f"bad_row_schema:{cid}:{row['schema_version']}")
        if row["record_type"] != "calibration_candidate":
            errors.append(f"bad_record_type:{cid}:{row['record_type']}")
        if selected_scans is not None and row["scan_id"] not in selected_scans:
            errors.append(f"row_outside_selected_scans:{cid}")
        context = contexts.get(row["subgraph_id"])
        if context is None:
            errors.append(f"row_missing_context:{cid}:{row['subgraph_id']}")
            continue
        subject_id = int(row["edge"]["subject_id"])
        object_id = int(row["edge"]["object_id"])
        predicate_label = row["predicate"]["predicate_label"]
        family = row["predicate"]["predicate_family"]
        row_tuple = (row["subgraph_id"], subject_id, object_id, predicate_label)
        if row_tuple in seen_row_tuples:
            errors.append(f"duplicate_candidate_tuple:{cid}")
        seen_row_tuples.add(row_tuple)
        if subject_id == object_id:
            errors.append(f"same_endpoint:{cid}")
        if subject_id not in context.object_labels:
            errors.append(f"missing_subject_in_subgraph:{cid}")
        if object_id not in context.object_labels:
            errors.append(f"missing_object_in_subgraph:{cid}")
        if predicate_label == "none":
            errors.append(f"none_predicate_emitted:{cid}")
        if family not in ALLOWED_FAMILIES:
            errors.append(f"unsupported_family_emitted:{cid}:{family}")
        if row["geometry"]["geometry_available"] is not True:
            errors.append(f"geometry_unavailable_row:{cid}")
        if split_name == "mini_schema_smoke" and row["quality"].get("calibration_role") == "train":
            errors.append(f"mini_schema_smoke_marked_train:{cid}")
        if row["candidate_source"]["type"] == "counterfactual_negative":
            if row["negative"] is None:
                errors.append(f"negative_row_missing_payload:{cid}")
            elif row["negative"]["base_candidate_id"] not in candidate_ids:
                errors.append(f"negative_row_missing_base:{cid}")
            if row_tuple in positive_edge_tuples:
                errors.append(f"negative_duplicates_positive_tuple:{cid}")
            if row["label"]["geom_valid"] != 0:
                errors.append(f"negative_row_bad_label:{cid}")
        elif row["candidate_source"]["type"] == "gt_positive":
            if row["negative"] is not None:
                errors.append(f"positive_row_has_negative_payload:{cid}")
            if row["label"]["geom_valid"] != 1:
                errors.append(f"positive_row_bad_label:{cid}")
        else:
            errors.append(f"unknown_candidate_source:{cid}:{row['candidate_source']['type']}")

    if not table_rows:
        errors.append("zero_table_rows")
    if not negative_records:
        warnings.append("zero_negative_records")

    return errors, warnings


def count_rows(table_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows_by_family = Counter(row["predicate"]["predicate_family"] for row in table_rows)
    rows_by_label_source = Counter(row["label"]["label_source"] for row in table_rows)
    rows_by_candidate_source = Counter(row["candidate_source"]["type"] for row in table_rows)
    rows_by_scan = Counter(row["scan_id"] for row in table_rows)
    return {
        "scans": len(rows_by_scan),
        "subgraphs": len({row["subgraph_id"] for row in table_rows}),
        "positive_rows": rows_by_candidate_source["gt_positive"],
        "negative_rows": rows_by_candidate_source["counterfactual_negative"],
        "uncertain_rows": sum(
            1 for row in table_rows if row["label"].get("label_status") == "uncertain"
        ),
        "rows_by_family": dict(sorted(rows_by_family.items())),
        "rows_by_label_source": dict(sorted(rows_by_label_source.items())),
        "rows_by_candidate_source": dict(sorted(rows_by_candidate_source.items())),
        "rows_by_scan": dict(sorted(rows_by_scan.items())),
    }


def notes_for_split(split_name: str) -> list[str]:
    notes = [
        "Semantic scores remain null until the prediction adapter exists.",
        "Counterfactual negatives are high-margin synthetic candidates, not absent-edge negatives.",
    ]
    if split_name == "mini_schema_smoke":
        return [
            "RelCompat3D-Mini export is schema smoke validation only; do not use it as calibration train/dev data.",
            *notes,
        ]
    if split_name == "train_dev_calib":
        return [
            "RelCompat3D-Calib-Pilot train/dev export is fitting input; assign train/dev roles from 25_pilot.md scan lists.",
            "Do not mix this export with held-out RelCompat3D-Mini validation scans.",
            *notes,
        ]
    return [
        "Use the declared scan split policy before fitting or reporting calibration metrics.",
        *notes,
    ]


def next_action_for_manifest(manifest: dict[str, Any]) -> str:
    if manifest["status"] != "ready":
        return "resolve validation errors or missing geometry before fitting `p_geom_valid`."
    split_name = manifest["split_name"]
    if split_name == "train_dev_calib":
        return (
            "fit/evaluate `p_geom_valid` using the train/dev scan lists in `25_pilot.md`; "
            "keep RelCompat3D-Mini held out."
        )
    if split_name == "mini_schema_smoke":
        return (
            "keep this artifact as schema smoke only; use a validated train/dev export "
            "before fitting `p_geom_valid`."
        )
    return "apply the split policy in `24_calibration_data.md` before fitting any calibrator."


def make_report(manifest: dict[str, Any]) -> str:
    validation = manifest["validation"]
    counts = manifest["counts"]
    lines = [
        "# Calibration Export",
        "",
        f"Created at: `{manifest['created_at']}`",
        f"Split name: `{manifest['split_name']}`",
        f"Status: `{manifest['status']}`",
        "",
        "## Inputs",
        "",
        f"- Subset source: `{manifest['subset_source']}`",
        f"- Selected scans file: `{manifest['selected_scans_file']}`",
        f"- Geometry sources: `{', '.join(manifest['geometry_sources'])}`",
        "",
        "## Outputs",
        "",
        f"- Table: `{manifest['table_file']}`",
        f"- Negatives: `{manifest['negatives_file']}`",
        f"- Manifest: `manifest.json`",
        "",
        "## Counts",
        "",
        f"- Scans: `{counts['scans']}`",
        f"- Subgraphs: `{counts['subgraphs']}`",
        f"- Positive rows: `{counts['positive_rows']}`",
        f"- Negative rows: `{counts['negative_rows']}`",
        f"- Uncertain rows: `{counts['uncertain_rows']}`",
        "",
        "## Families",
        "",
    ]
    for family, count in counts["rows_by_family"].items():
        lines.append(f"- `{family}`: `{count}`")
    lines.extend(["", "## Candidate Sources", ""])
    for source, count in counts["rows_by_candidate_source"].items():
        lines.append(f"- `{source}`: `{count}`")
    lines.extend(["", "## Negative Strategies", ""])
    for strategy, count in manifest["counts"]["negative_rows_by_strategy"].items():
        lines.append(f"- `{strategy}`: `{count}`")
    lines.extend([
        "",
        "## Validation",
        "",
        f"- Passed: `{validation['passed']}`",
        f"- Errors: `{len(validation['errors'])}`",
        f"- Warnings: `{len(validation['warnings'])}`",
    ])
    if validation["errors"]:
        lines.extend(["", "### Errors", ""])
        for error in validation["errors"][:20]:
            lines.append(f"- `{error}`")
    if validation["warnings"]:
        lines.extend(["", "### Warnings", ""])
        for warning in validation["warnings"][:20]:
            lines.append(f"- `{warning}`")
    if manifest["notes"]:
        lines.extend(["", "## Notes", ""])
        for note in manifest["notes"]:
            lines.append(f"- {note}")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Use the split policy in `24_calibration_data.md` before fitting any calibrator.",
        "",
        f"Next action: {next_action_for_manifest(manifest)}",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"nonempty_output:{args.output_dir}")
    created_at = date.today().isoformat()
    errors: list[str] = []
    warnings: list[str] = []

    required_paths = {
        "dataset_root": args.dataset_root,
        "subset_json": args.subset_json,
        "relationships_file": args.relationships_file,
    }
    if args.selected_scans is not None:
        required_paths["selected_scans"] = args.selected_scans
    for name, path in required_paths.items():
        if not path.exists():
            errors.append(f"missing_input:{name}:{relpath(path)}")
    if errors:
        print("\n".join(errors))
        return 2

    relationship_id_map = load_relationship_id_map(args.relationships_file)
    selected_scans = load_selected_scans(args.selected_scans)
    subset_data = load_json(args.subset_json)
    subset_source = relpath(args.subset_json)

    positives, contexts, skipped_positive_counts, build_warnings, build_errors = build_contexts_and_positives(
        subset_data=subset_data,
        selected_scans=selected_scans,
        dataset_root=args.dataset_root,
        subset_source=subset_source,
        split_name=args.split_name,
        relationship_id_map=relationship_id_map,
        created_at=created_at,
        allow_selected_scans_without_positive_rows=(
            args.allow_selected_scans_without_positive_rows
        ),
    )
    warnings.extend(build_warnings)
    errors.extend(build_errors)

    negative_rows: list[dict[str, Any]] = []
    negative_records: list[dict[str, Any]] = []
    skipped_negative_attempts: Counter[str] = Counter()
    if args.include_counterfactuals and not errors:
        negative_rows, negative_records, skipped_negative_attempts = generate_negatives(
            positives=positives,
            split_name=args.split_name,
            subset_source=subset_source,
            relationship_id_map=relationship_id_map,
            created_at=created_at,
            max_negatives_per_positive=args.max_negatives_per_positive,
            max_negatives_per_subgraph_family=args.max_negatives_per_subgraph_family,
            max_negative_to_positive_ratio_per_family=args.max_negative_to_positive_ratio_per_family,
        )

    table_rows = [spec.row for spec in positives] + negative_rows
    validation_errors, validation_warnings = validate_export(
        table_rows=table_rows,
        negative_records=negative_records,
        contexts=contexts,
        selected_scans=selected_scans,
        split_name=args.split_name,
    )
    errors.extend(validation_errors)
    warnings.extend(validation_warnings)

    negative_rows_by_strategy = Counter(record["strategy"] for record in negative_records)
    counts = count_rows(table_rows)
    counts["negative_rows_by_strategy"] = dict(sorted(negative_rows_by_strategy.items()))
    counts["skipped_positive_rows"] = dict(sorted(skipped_positive_counts.items()))
    counts["skipped_negative_attempts"] = dict(sorted(skipped_negative_attempts.items()))

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "split_name": args.split_name,
        "subset_source": subset_source,
        "selected_scans_file": relpath(args.selected_scans) if args.selected_scans else None,
        "table_file": "table.jsonl",
        "negatives_file": "negatives.jsonl",
        "prediction_manifest": None,
        "geometry_sources": ["semseg_obb_v0"],
        "negative_policy_version": NEGATIVE_POLICY_VERSION,
        "row_schema_version": ROW_SCHEMA_VERSION,
        "created_at": created_at,
        "status": "ready" if not errors else "blocked",
        "counts": counts,
        "negative_caps": {
            "max_negatives_per_positive": args.max_negatives_per_positive,
            "max_negatives_per_subgraph_family": args.max_negatives_per_subgraph_family,
            "max_negative_to_positive_ratio_per_family": args.max_negative_to_positive_ratio_per_family,
        },
        "validation": {
            "passed": not errors and not (args.fail_on_warnings and warnings),
            "errors": errors,
            "warnings": warnings,
        },
        "notes": notes_for_split(args.split_name),
    }
    manifest["report_file"] = "report.md"

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "manifest.json", manifest)
        write_jsonl(args.output_dir / "table.jsonl", table_rows)
        write_jsonl(args.output_dir / "negatives.jsonl", negative_records)
        (args.output_dir / "report.md").write_text(make_report(manifest), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_dir": relpath(args.output_dir),
                "positive_rows": counts["positive_rows"],
                "negative_rows": counts["negative_rows"],
                "errors": len(errors),
                "warnings": len(warnings),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    if errors or (args.fail_on_warnings and warnings):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
