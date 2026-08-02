#!/usr/bin/env python3
"""Run the frozen raw-surface construct-validity audit for RelCompat3D."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import evaluate_main as base
import evaluate_train_only as strict


FAMILIES = ("proximity", "relative_vertical")
METHODS = ("source", "relcompat3d")
AUDITS = ("point", "mesh", "consensus")
KS = (5, 10, 20, 50, 100)
STATUSES = ("satisfied", "uncertain", "violated", "unsupported")


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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def stable_order_key(*parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_scans(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def percentile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        raise ValueError(f"empty_percentile_input:{q}")
    return float(np.percentile(array, q))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if not len(values):
        raise ValueError("empty_weighted_quantile")
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values[order], dtype=np.float64)
    sorted_weights = np.asarray(weights[order], dtype=np.float64)
    positive = sorted_weights > 0
    sorted_values, sorted_weights = sorted_values[positive], sorted_weights[positive]
    if not len(sorted_values):
        return float(np.quantile(values, q))
    cumulative = np.cumsum(sorted_weights)
    target = min(max(float(q), 0.0), 1.0) * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, target, side="left")), len(sorted_values) - 1)
    return float(sorted_values[index])


def deterministic_sample(
    values: np.ndarray,
    maximum: int,
    seed_parts: tuple[Any, ...],
) -> tuple[np.ndarray, np.ndarray]:
    count = len(values)
    if count <= maximum:
        indices = np.arange(count, dtype=np.int64)
        return values, indices
    rng = np.random.default_rng(stable_seed(*seed_parts))
    indices = np.sort(rng.choice(count, size=maximum, replace=False))
    return values[indices], indices


def parse_ply_header(handle: Any, path: Path) -> tuple[int, int, list[str]]:
    if handle.readline().strip() != "ply":
        raise ValueError(f"not_ply:{path}")
    vertex_count: int | None = None
    face_count: int | None = None
    vertex_properties: list[str] = []
    current_element: str | None = None
    for line in handle:
        stripped = line.strip()
        if stripped.startswith("format") and stripped != "format ascii 1.0":
            raise ValueError(f"unsupported_ply_format:{path}:{stripped}")
        if stripped.startswith("element "):
            parts = stripped.split()
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
            elif current_element == "face":
                face_count = int(parts[2])
        elif stripped.startswith("property ") and current_element == "vertex":
            vertex_properties.append(stripped.split()[-1])
        elif stripped == "end_header":
            break
    if vertex_count is None or face_count is None:
        raise ValueError(f"missing_ply_counts:{path}")
    return vertex_count, face_count, vertex_properties


def load_object_surfaces(
    path: Path,
    object_ids: set[int],
    point_maximum: int,
    triangle_maximum: int,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[str, int]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        vertex_count, face_count, properties = parse_ply_header(handle, path)
        indices = {name: index for index, name in enumerate(properties)}
        for required in ("x", "y", "z", "objectId"):
            if required not in indices:
                raise ValueError(f"missing_ply_property:{path}:{required}")
        max_index = max(indices[name] for name in ("x", "y", "z", "objectId"))
        coordinates = np.empty((vertex_count, 3), dtype=np.float32)
        labels = np.empty(vertex_count, dtype=np.int32)
        for row_index in range(vertex_count):
            parts = handle.readline().split()
            if len(parts) <= max_index:
                raise ValueError(f"short_vertex_row:{path}:{row_index}")
            coordinates[row_index] = (
                float(parts[indices["x"]]),
                float(parts[indices["y"]]),
                float(parts[indices["z"]]),
            )
            labels[row_index] = int(parts[indices["objectId"]])

        triangle_centroids: dict[int, list[np.ndarray]] = defaultdict(list)
        triangle_areas: dict[int, list[float]] = defaultdict(list)
        mixed_faces = 0
        degenerate_faces = 0
        for face_index in range(face_count):
            parts = handle.readline().split()
            if not parts:
                continue
            width = int(parts[0])
            face = [int(value) for value in parts[1 : width + 1]]
            if width < 3:
                continue
            anchor = face[0]
            for offset in range(1, width - 1):
                tri = np.asarray((anchor, face[offset], face[offset + 1]), dtype=np.int64)
                tri_labels = labels[tri]
                if not np.all(tri_labels == tri_labels[0]):
                    mixed_faces += 1
                    continue
                object_id = int(tri_labels[0])
                if object_id not in object_ids:
                    continue
                xyz = coordinates[tri].astype(np.float64)
                area = 0.5 * float(np.linalg.norm(np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])))
                if not math.isfinite(area) or area <= 1e-12:
                    degenerate_faces += 1
                    continue
                triangle_centroids[object_id].append(xyz.mean(axis=0))
                triangle_areas[object_id].append(area)

    points: dict[int, dict[str, Any]] = {}
    meshes: dict[int, dict[str, Any]] = {}
    for object_id in sorted(object_ids):
        object_points = coordinates[labels == object_id].astype(np.float64)
        if len(object_points):
            median = np.median(object_points, axis=0)
            radii = np.linalg.norm(object_points - median[None, :], axis=1)
            scale = float(np.percentile(radii, 90.0)) if len(radii) else 0.0
            sampled, _ = deterministic_sample(
                object_points,
                point_maximum,
                ("orthogonal-point", path.parent.name, object_id),
            )
            points[object_id] = {
                "count": int(len(object_points)),
                "sample": sampled,
                "median": median,
                "scale": scale,
            }
        centroids = np.asarray(triangle_centroids.get(object_id, []), dtype=np.float64)
        areas = np.asarray(triangle_areas.get(object_id, []), dtype=np.float64)
        if len(centroids):
            sampled_centroids, sample_indices = deterministic_sample(
                centroids,
                triangle_maximum,
                ("orthogonal-mesh", path.parent.name, object_id),
            )
            sampled_areas = areas[sample_indices]
            total_area = float(areas.sum())
            meshes[object_id] = {
                "count": int(len(centroids)),
                "sample": sampled_centroids,
                "sample_weights": sampled_areas,
                "area": total_area,
                "median_z": weighted_quantile(centroids[:, 2], areas, 0.5),
                "scale": math.sqrt(max(total_area, 0.0) / math.pi),
            }
    return points, meshes, {
        "vertices": vertex_count,
        "faces": face_count,
        "mixed_triangles_skipped": mixed_faces,
        "degenerate_triangles_skipped": degenerate_faces,
    }


def nearest_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not len(left) or not len(right):
        return np.asarray([], dtype=np.float64)
    distances = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.min(distances, axis=1))


def point_pair_measurement(
    subject: dict[str, Any] | None,
    object_: dict[str, Any] | None,
    minimum_scale: float,
) -> dict[str, Any]:
    if subject is None or object_ is None:
        return {"available": False}
    scale = max(minimum_scale, 0.5 * (float(subject["scale"]) + float(object_["scale"])))
    left = nearest_distances(subject["sample"], object_["sample"])
    right = nearest_distances(object_["sample"], subject["sample"])
    proximity = float(np.percentile(np.concatenate((left, right)), 10.0) / scale)
    vertical = float((subject["median"][2] - object_["median"][2]) / scale)
    return {
        "available": True,
        "subject_count": int(subject["count"]),
        "object_count": int(object_["count"]),
        "subject_scale_m": float(subject["scale"]),
        "object_scale_m": float(object_["scale"]),
        "pair_scale_m": scale,
        "proximity": proximity,
        "vertical_delta": vertical,
    }


def mesh_pair_measurement(
    subject: dict[str, Any] | None,
    object_: dict[str, Any] | None,
    minimum_scale: float,
) -> dict[str, Any]:
    if subject is None or object_ is None:
        return {"available": False}
    scale = max(minimum_scale, 0.5 * (float(subject["scale"]) + float(object_["scale"])))
    left = nearest_distances(subject["sample"], object_["sample"])
    right = nearest_distances(object_["sample"], subject["sample"])
    values = np.concatenate((left, right))
    weights = np.concatenate((subject["sample_weights"], object_["sample_weights"]))
    proximity = weighted_quantile(values, weights, 0.10) / scale
    vertical = float((subject["median_z"] - object_["median_z"]) / scale)
    return {
        "available": True,
        "subject_count": int(subject["count"]),
        "object_count": int(object_["count"]),
        "subject_area_m2": float(subject["area"]),
        "object_area_m2": float(object_["area"]),
        "subject_scale_m": float(subject["scale"]),
        "object_scale_m": float(object_["scale"]),
        "pair_scale_m": scale,
        "proximity": float(proximity),
        "vertical_delta": vertical,
    }


def load_training_cases(path: Path, train_scans: set[str]) -> list[dict[str, Any]]:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            scan = str(row["scan_id"])
            family = str(row["predicate"]["predicate_family"])
            if scan not in train_scans or family not in FAMILIES:
                continue
            if (row.get("label") or {}).get("label_status") != "positive":
                continue
            if (row.get("candidate_source") or {}).get("type") != "gt_positive":
                continue
            case = {
                "scan_id": scan,
                "subject_id": int(row["edge"]["subject_id"]),
                "object_id": int(row["edge"]["object_id"]),
                "family": family,
                "predicate": str(row["predicate"]["predicate_label"]),
            }
            key = (
                case["scan_id"],
                case["subject_id"],
                case["object_id"],
                case["family"],
                case["predicate"],
            )
            unique[key] = case
    return [unique[key] for key in sorted(unique)]


def group_pairs(cases: Iterable[dict[str, Any]]) -> dict[str, set[tuple[int, int]]]:
    grouped: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for case in cases:
        grouped[str(case["scan_id"])].add((int(case["subject_id"]), int(case["object_id"])))
    return grouped


def measure_scan_task(
    task: tuple[
        str,
        Path,
        set[tuple[int, int]],
        int,
        int,
        float,
        set[tuple[str, int, int]],
    ]
) -> tuple[str, list[dict[str, Any]], dict[tuple[str, int, int], dict[str, Any]], dict[str, Any], tuple[str, int] | None]:
    scan, raw_scan_root, pairs, point_maximum, triangle_maximum, minimum_scale, mechanism_pairs = task
    path = raw_scan_root / scan / "labels.instances.annotated.v2.ply"
    if not path.is_file():
        missing = [
            {
                "scan_id": scan,
                "subject_id": subject_id,
                "object_id": object_id,
                "point": {"available": False},
                "mesh": {"available": False},
                "missing_reason": "missing_annotated_ply",
            }
            for subject_id, object_id in sorted(pairs)
        ]
        return scan, missing, {}, {}, None
    object_ids = {value for pair in pairs for value in pair}
    points, meshes, stats = load_object_surfaces(path, object_ids, point_maximum, triangle_maximum)
    measured: list[dict[str, Any]] = []
    retained_surfaces: dict[tuple[str, int, int], dict[str, Any]] = {}
    for subject_id, object_id in sorted(pairs):
        key = (scan, subject_id, object_id)
        measured.append({
            "scan_id": scan,
            "subject_id": subject_id,
            "object_id": object_id,
            "point": point_pair_measurement(points.get(subject_id), points.get(object_id), minimum_scale),
            "mesh": mesh_pair_measurement(meshes.get(subject_id), meshes.get(object_id), minimum_scale),
        })
        if key in mechanism_pairs:
            retained_surfaces[key] = {
                "point_subject": points.get(subject_id),
                "point_object": points.get(object_id),
                "mesh_subject": meshes.get(subject_id),
                "mesh_object": meshes.get(object_id),
            }
    return scan, measured, retained_surfaces, stats, (relpath(raw_scan_root, path), path.stat().st_size)


def measure_pairs(
    raw_scan_root: Path,
    grouped_pairs: dict[str, set[tuple[int, int]]],
    point_maximum: int,
    triangle_maximum: int,
    minimum_scale: float,
    mechanism_pairs: set[tuple[str, int, int]] | None = None,
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], dict[tuple[str, int, int], dict[str, Any]], dict[str, Any]]:
    measurements: dict[tuple[str, int, int], dict[str, Any]] = {}
    surfaces: dict[tuple[str, int, int], dict[str, Any]] = {}
    inventory: list[tuple[str, int]] = []
    parser_totals: Counter[str] = Counter()
    mechanism_pairs = mechanism_pairs or set()
    tasks = [
        (
            scan,
            raw_scan_root,
            grouped_pairs[scan],
            point_maximum,
            triangle_maximum,
            minimum_scale,
            {key for key in mechanism_pairs if key[0] == scan},
        )
        for scan in sorted(grouped_pairs)
    ]
    workers = min(4, max(1, len(tasks)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(measure_scan_task, tasks, chunksize=1)
        for scan_index, (_, measured, retained, stats, inventory_row) in enumerate(iterator, 1):
            for row in measured:
                key = (row["scan_id"], int(row["subject_id"]), int(row["object_id"]))
                measurements[key] = row
            surfaces.update(retained)
            parser_totals.update(stats)
            if inventory_row is not None:
                inventory.append(inventory_row)
            if scan_index % 50 == 0:
                print(json.dumps({"surface_scan_progress": scan_index, "surface_scan_total": len(grouped_pairs)}), flush=True)
    inventory_payload = "\n".join(f"{path}\t{size}" for path, size in sorted(inventory)).encode("utf-8")
    return measurements, surfaces, {
        "scans_requested": len(grouped_pairs),
        "scans_available": len(inventory),
        "inventory_sha256": hashlib.sha256(inventory_payload).hexdigest(),
        "inventory_bytes": sum(size for _, size in inventory),
        "parser_totals": dict(parser_totals),
    }


def signed_vertical(predicate: str, delta: float) -> float:
    if predicate == "higher than":
        return float(delta)
    if predicate == "lower than":
        return float(-delta)
    raise ValueError(f"unsupported_vertical_predicate:{predicate}")


def derive_thresholds(
    cases: list[dict[str, Any]],
    measurements: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[str, Any]:
    thresholds: dict[str, Any] = {}
    for audit, hard_floor in (("point", 32), ("mesh", 16)):
        endpoint_counts: list[float] = []
        for case in cases:
            cell = measurements[(case["scan_id"], case["subject_id"], case["object_id"])][audit]
            if cell.get("available"):
                endpoint_counts.extend((float(cell["subject_count"]), float(cell["object_count"])))
        support_p01 = percentile(endpoint_counts, 1.0)
        minimum_count = max(hard_floor, int(math.floor(support_p01)))
        proximity_values: list[float] = []
        vertical_values: list[float] = []
        for case in cases:
            cell = measurements[(case["scan_id"], case["subject_id"], case["object_id"])][audit]
            if not cell.get("available"):
                continue
            if min(int(cell["subject_count"]), int(cell["object_count"])) < minimum_count:
                continue
            if case["family"] == "proximity":
                proximity_values.append(float(cell["proximity"]))
            else:
                vertical_values.append(signed_vertical(case["predicate"], float(cell["vertical_delta"])))
        vertical_margin = max(1e-6, percentile((abs(value) for value in vertical_values), 10.0))
        thresholds[audit] = {
            "minimum_endpoint_count": minimum_count,
            "support_count_p01": support_p01,
            "proximity": {
                "satisfied_max_p90": percentile(proximity_values, 90.0),
                "violated_min_p99": percentile(proximity_values, 99.0),
                "training_rows": len(proximity_values),
            },
            "relative_vertical": {
                "absolute_signed_margin_p10": vertical_margin,
                "training_rows": len(vertical_values),
                "positive_direction_fraction": float(np.mean(np.asarray(vertical_values) > 0.0)),
                "signed_median": float(np.median(vertical_values)),
            },
            "available_training_endpoints": len(endpoint_counts),
        }
    return thresholds


def audit_status(
    measurement: dict[str, Any],
    audit: str,
    family: str,
    predicate: str,
    thresholds: dict[str, Any],
) -> str:
    cell = measurement[audit]
    threshold = thresholds[audit]
    if not cell.get("available"):
        return "unsupported"
    if min(int(cell["subject_count"]), int(cell["object_count"])) < int(threshold["minimum_endpoint_count"]):
        return "unsupported"
    if family == "proximity":
        value = float(cell["proximity"])
        if value <= float(threshold["proximity"]["satisfied_max_p90"]):
            return "satisfied"
        if value > float(threshold["proximity"]["violated_min_p99"]):
            return "violated"
        return "uncertain"
    value = signed_vertical(predicate, float(cell["vertical_delta"]))
    margin = float(threshold["relative_vertical"]["absolute_signed_margin_p10"])
    if value >= margin:
        return "satisfied"
    if value <= -margin:
        return "violated"
    return "uncertain"


def all_audit_statuses(
    measurement: dict[str, Any],
    family: str,
    predicate: str,
    thresholds: dict[str, Any],
) -> dict[str, str]:
    point = audit_status(measurement, "point", family, predicate, thresholds)
    mesh = audit_status(measurement, "mesh", family, predicate, thresholds)
    if "unsupported" in (point, mesh):
        consensus = "unsupported"
    elif point == mesh and point in {"satisfied", "violated"}:
        consensus = point
    else:
        consensus = "uncertain"
    return {"point": point, "mesh": mesh, "consensus": consensus}


def add_primary_ranking(grouped: dict[str, list[dict[str, Any]]]) -> None:
    for candidates in grouped.values():
        source_order = sorted(candidates, key=lambda row: (-row["scores"]["source_score"], row["key"]))
        queues: dict[str, list[dict[str, Any]]] = {}
        for family in base.FAMILIES:
            rows = [row for row in candidates if row["family"] == family]
            score_name = "source_score" if family == "support_contact" else "structured_product"
            queues[family] = sorted(rows, key=lambda row: (-row["scores"][score_name], row["key"]))
        offsets = {family: 0 for family in base.FAMILIES}
        output: list[dict[str, Any]] = []
        for row in source_order:
            family = row["family"]
            output.append(queues[family][offsets[family]])
            offsets[family] += 1
        for rank, row in enumerate(output, 1):
            row["scores"]["relcompat3d"] = float(len(output) - rank + 1)


def lightweight(row: dict[str, Any], context: str) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "context": context,
        "scan_id": str(row["scan"]),
        "key": tuple(row["key"]),
        "subject_id": int(row["key"][2]),
        "object_id": int(row["key"][3]),
        "family": str(row["family"]),
        "predicate": str(row["predicate"]),
    }


def load_source_rankings(
    path: Path,
    contexts: list[str],
    scorer: Any,
    strict_models: dict[str, Any],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    grouped, load_info = base.load_candidates(path, scorer, strict_models)
    add_primary_ranking(grouped)
    rankings: dict[str, dict[str, list[dict[str, Any]]]] = {method: {} for method in METHODS}
    for context in contexts:
        candidates = grouped.get(context, [])
        source = sorted(candidates, key=lambda row: (-row["scores"]["source_score"], row["key"]))[:100]
        reranked = sorted(candidates, key=lambda row: (-row["scores"]["relcompat3d"], row["key"]))[:100]
        rankings["source"][context] = [lightweight(row, context) for row in source]
        rankings["relcompat3d"][context] = [lightweight(row, context) for row in reranked]
    counts = {
        **load_info,
        "contexts": len(contexts),
        "prediction_contexts": len(grouped),
        "zero_prediction_contexts": len(set(contexts) - set(grouped)),
    }
    del grouped
    gc.collect()
    return rankings, counts


def selected_candidates(rankings: dict[str, Any]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for source, source_rankings in rankings.items():
        for method in METHODS:
            for rows in source_rankings[method].values():
                for row in rows:
                    if row["family"] in FAMILIES:
                        unique[(source, row["id"])] = {**row, "source": source}
    return list(unique.values())


def mechanism_sample(candidates: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for family in FAMILIES:
        rows = [row for row in candidates if row["family"] == family]
        rows.sort(key=lambda row: stable_order_key("mechanism", row["source"], row["id"]))
        output.extend(rows[:maximum])
    return output


def load_raw_features(path: Path, target_ids: set[str]) -> dict[str, dict[str, float]]:
    features: dict[str, dict[str, float]] = {}
    if not target_ids:
        return features
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prediction_id = str(row["prediction_id"])
            if prediction_id in target_ids:
                features[prediction_id] = strict.raw_numeric(row)
                if len(features) == len(target_ids):
                    break
    return features


def load_ground_truth_scope(path: Path) -> tuple[dict[str, set[tuple[Any, ...]]], dict[str, set[tuple[Any, ...]]]]:
    all_gt: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    scope_gt: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            context = str(row["subgraph_id"])
            key = strict.gt_key(row)
            family = str(row["predicate_family"])
            if family in base.FAMILIES:
                all_gt[context].add(key)
            if family in FAMILIES:
                scope_gt[context].add(key)
    return all_gt, scope_gt


def empty_scan_values(scan_count: int) -> dict[str, np.ndarray]:
    names = (
        "recall_all_num",
        "recall_all_den",
        "recall_scope_num",
        "recall_scope_den",
        "selected_scope",
        "satisfied",
        "uncertain",
        "violated",
        "unsupported",
    )
    return {name: np.zeros(scan_count, dtype=np.float64) for name in names}


def build_contributions(
    rankings: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
    contexts: list[str],
    scans: list[str],
    all_gt: dict[str, set[tuple[Any, ...]]],
    scope_gt: dict[str, set[tuple[Any, ...]]],
    measurements: dict[tuple[str, int, int], dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    scan_index = {scan: index for index, scan in enumerate(scans)}
    values: dict[str, Any] = {
        source: {
            audit: {
                method: {k: empty_scan_values(len(scans)) for k in KS}
                for method in METHODS
            }
            for audit in AUDITS
        }
        for source in rankings
    }
    status_cache: dict[tuple[Any, ...], dict[str, str]] = {}
    for source, source_rankings in rankings.items():
        for method in METHODS:
            for context in contexts:
                ranked = source_rankings[method][context]
                scan = context.rsplit("_", 1)[0]
                si = scan_index[scan]
                gt_all = all_gt.get(context, set())
                gt_scope = scope_gt.get(context, set())
                for k in KS:
                    chosen = ranked[:k]
                    chosen_keys = {row["key"] for row in chosen}
                    for audit in AUDITS:
                        cell = values[source][audit][method][k]
                        cell["recall_all_num"][si] += len(chosen_keys & gt_all)
                        cell["recall_all_den"][si] += len(gt_all)
                        cell["recall_scope_num"][si] += len(chosen_keys & gt_scope)
                        cell["recall_scope_den"][si] += len(gt_scope)
                    for row in chosen:
                        if row["family"] not in FAMILIES:
                            continue
                        pair_key = (row["scan_id"], row["subject_id"], row["object_id"])
                        relation_key = (*pair_key, row["family"], row["predicate"])
                        if relation_key not in status_cache:
                            status_cache[relation_key] = all_audit_statuses(
                                measurements[pair_key], row["family"], row["predicate"], thresholds
                            )
                        for audit in AUDITS:
                            cell = values[source][audit][method][k]
                            cell["selected_scope"][si] += 1
                            cell[status_cache[relation_key][audit]][si] += 1
    return values


def ratio_parts(values: dict[str, np.ndarray], metric: str) -> tuple[np.ndarray, np.ndarray]:
    supported = values["satisfied"] + values["uncertain"] + values["violated"]
    definitions = {
        "recall_all": (values["recall_all_num"], values["recall_all_den"]),
        "recall_scope": (values["recall_scope_num"], values["recall_scope_den"]),
        "violation": (values["violated"], supported),
        "uncertainty": (values["uncertain"], supported),
        "coverage": (supported, values["selected_scope"]),
        "decidable_coverage": (values["satisfied"] + values["violated"], values["selected_scope"]),
    }
    return definitions[metric]


def bootstrap_ratio(numerator: np.ndarray, denominator: np.ndarray, samples: np.ndarray) -> np.ndarray:
    boot_num = numerator[samples].sum(axis=1)
    boot_den = denominator[samples].sum(axis=1)
    return np.divide(
        boot_num,
        boot_den,
        out=np.full(boot_num.shape, np.nan, dtype=np.float64),
        where=boot_den > 0,
    )


def ci95(values: np.ndarray) -> list[float | None]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return [None, None]
    return [float(value) for value in np.percentile(finite, (2.5, 97.5))]


def summarize_contributions(values: dict[str, Any], samples: np.ndarray) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for source, source_values in values.items():
        report[source] = {"audits": {}, "recall": {method: {} for method in METHODS}}
        recall_cache: dict[str, Any] = {method: {} for method in METHODS}
        reference_audit = "point"
        for method in METHODS:
            for k in KS:
                report[source]["recall"][method][str(k)] = {}
                recall_cache[method][str(k)] = {}
                reference = source_values[reference_audit][method][k]
                for metric in ("recall_all", "recall_scope"):
                    numerator, denominator = ratio_parts(reference, metric)
                    point = float(numerator.sum() / denominator.sum()) if denominator.sum() else None
                    boot = bootstrap_ratio(numerator, denominator, samples)
                    report[source]["recall"][method][str(k)][metric] = {
                        "point": point,
                        "scan_cluster_ci95": ci95(boot),
                        "numerator": int(numerator.sum()),
                        "denominator": int(denominator.sum()),
                    }
                    recall_cache[method][str(k)][metric] = boot
        report[source]["recall"]["relcompat3d_minus_source"] = {}
        for k in KS:
            report[source]["recall"]["relcompat3d_minus_source"][str(k)] = {}
            for metric in ("recall_all", "recall_scope"):
                left = report[source]["recall"]["relcompat3d"][str(k)][metric]["point"]
                right = report[source]["recall"]["source"][str(k)][metric]["point"]
                delta = recall_cache["relcompat3d"][str(k)][metric] - recall_cache["source"][str(k)][metric]
                report[source]["recall"]["relcompat3d_minus_source"][str(k)][metric] = {
                    "point": left - right if left is not None and right is not None else None,
                    "paired_scan_cluster_ci95": ci95(delta),
                }
        for audit in AUDITS:
            report[source]["audits"][audit] = {method: {} for method in METHODS}
            cache: dict[str, Any] = {method: {} for method in METHODS}
            for method in METHODS:
                for k in KS:
                    report[source]["audits"][audit][method][str(k)] = {}
                    cache[method][str(k)] = {}
                    cell = source_values[audit][method][k]
                    for metric in ("violation", "uncertainty", "coverage", "decidable_coverage"):
                        numerator, denominator = ratio_parts(cell, metric)
                        point = float(numerator.sum() / denominator.sum()) if denominator.sum() else None
                        boot = bootstrap_ratio(numerator, denominator, samples)
                        report[source]["audits"][audit][method][str(k)][metric] = {
                            "point": point,
                            "scan_cluster_ci95": ci95(boot),
                            "numerator": int(numerator.sum()),
                            "denominator": int(denominator.sum()),
                        }
                        cache[method][str(k)][metric] = boot
                    report[source]["audits"][audit][method][str(k)]["counts"] = {
                        name: int(cell[name].sum())
                        for name in ("selected_scope", "satisfied", "uncertain", "violated", "unsupported")
                    }
            report[source]["audits"][audit]["relcompat3d_minus_source"] = {}
            for k in KS:
                report[source]["audits"][audit]["relcompat3d_minus_source"][str(k)] = {}
                for metric in ("violation", "uncertainty", "coverage", "decidable_coverage"):
                    left = report[source]["audits"][audit]["relcompat3d"][str(k)][metric]["point"]
                    right = report[source]["audits"][audit]["source"][str(k)][metric]["point"]
                    delta = cache["relcompat3d"][str(k)][metric] - cache["source"][str(k)][metric]
                    report[source]["audits"][audit]["relcompat3d_minus_source"][str(k)][metric] = {
                        "point": left - right if left is not None and right is not None else None,
                        "paired_scan_cluster_ci95": ci95(delta),
                    }
    return report


def monotone(sequence: list[float], direction: str, tolerance: float = 1e-10) -> bool:
    if direction == "nondecreasing":
        return all(right + tolerance >= left for left, right in zip(sequence, sequence[1:]))
    if direction == "nonincreasing":
        return all(right <= left + tolerance for left, right in zip(sequence, sequence[1:]))
    raise ValueError(direction)


def translated_proximity(
    subject: dict[str, Any],
    object_: dict[str, Any],
    shift: float,
    audit: str,
    fixed_scale: float,
) -> float:
    subject_points = subject["sample"]
    object_points = object_["sample"]
    direction = np.asarray(subject_points[:, :2].mean(axis=0) - object_points[:, :2].mean(axis=0))
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction = np.asarray((1.0, 0.0))
    else:
        direction = direction / norm
    translated = subject_points.copy()
    translated[:, :2] += shift * direction[None, :]
    left = nearest_distances(translated, object_points)
    right = nearest_distances(object_points, translated)
    if audit == "point":
        return float(np.percentile(np.concatenate((left, right)), 10.0) / fixed_scale)
    weights = np.concatenate((subject["sample_weights"], object_["sample_weights"]))
    return weighted_quantile(np.concatenate((left, right)), weights, 0.10) / fixed_scale


def intervention_raw(raw: dict[str, float], family: str, predicate: str, level: float, scale_m: float) -> dict[str, float]:
    values = dict(raw)
    if family == "proximity":
        shift = level * scale_m
        old_xy = float(values.get("distance_xy", 0.0))
        old_3d = float(values.get("distance_3d", math.hypot(old_xy, values.get("center_delta_z", 0.0))))
        new_xy = old_xy + shift
        center_z = float(values.get("center_delta_z", 0.0))
        new_3d = math.hypot(new_xy, center_z)
        values["distance_xy"] = new_xy
        values["distance_3d"] = new_3d
        if "normalized_distance_xy" in values:
            ratio = values["normalized_distance_xy"] / old_xy if old_xy > 1e-9 else 1.0 / max(scale_m, 1e-6)
            values["normalized_distance_xy"] = values["normalized_distance_xy"] + shift * ratio
        if "normalized_distance_3d" in values:
            ratio = values["normalized_distance_3d"] / old_3d if old_3d > 1e-9 else 1.0 / max(scale_m, 1e-6)
            values["normalized_distance_3d"] = values["normalized_distance_3d"] + (new_3d - old_3d) * ratio
        decay = math.exp(-2.0 * level)
        for name in ("projected_iou_xy", "projected_subject_overlap_ratio", "projected_object_overlap_ratio"):
            if name in values:
                values[name] *= decay
    else:
        direction = 1.0 if predicate == "higher than" else -1.0
        shift = direction * level * scale_m
        if "center_delta_z" in values:
            values["center_delta_z"] += shift
        if "normalized_center_delta_z" in values:
            values["normalized_center_delta_z"] += direction * level
        for name in ("subject_bottom_z", "subject_top_z"):
            if name in values:
                values[name] += shift
        if "object_bottom_z" in values and "subject_top_z" in values:
            values["vertical_gap_subject_on_object"] = values["object_bottom_z"] - values["subject_top_z"]
    for source_name, target_name in (
        ("center_delta_z", "abs_center_delta_z"),
        ("normalized_center_delta_z", "abs_normalized_center_delta_z"),
        ("vertical_gap_subject_on_object", "abs_vertical_gap_subject_on_object"),
    ):
        if source_name in values:
            values[target_name] = abs(values[source_name])
    return values


def run_mechanism_test(
    cases: list[dict[str, Any]],
    surfaces: dict[tuple[str, int, int], dict[str, Any]],
    measurements: dict[tuple[str, int, int], dict[str, Any]],
    raw_features: dict[tuple[str, str], dict[str, float]],
    scorer: Any,
    levels: list[float],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        pair_key = (case["scan_id"], case["subject_id"], case["object_id"])
        surface = surfaces.get(pair_key)
        measure = measurements.get(pair_key)
        raw = raw_features.get((case["source"], case["id"]))
        row: dict[str, Any] = {
            "source": case["source"],
            "prediction_id": case["id"],
            "family": case["family"],
            "predicate": case["predicate"],
            "point_monotone": None,
            "mesh_monotone": None,
            "compatibility_monotone": None,
            "point_endpoint_change": None,
            "mesh_endpoint_change": None,
            "compatibility_endpoint_change": None,
        }
        if surface is not None and measure is not None:
            for audit in ("point", "mesh"):
                subject = surface.get(f"{audit}_subject")
                object_ = surface.get(f"{audit}_object")
                cell = measure[audit]
                if subject is None or object_ is None or not cell.get("available"):
                    continue
                scale = float(cell["pair_scale_m"])
                if case["family"] == "proximity":
                    sequence = [translated_proximity(subject, object_, level * scale, audit, scale) for level in levels]
                else:
                    direction = 1.0 if case["predicate"] == "higher than" else -1.0
                    base_delta = float(cell["vertical_delta"])
                    sequence = [direction * (base_delta + direction * level) for level in levels]
                row[f"{audit}_monotone"] = monotone(sequence, "nondecreasing")
                row[f"{audit}_endpoint_change"] = float(sequence[-1] - sequence[0])
        if raw is not None and measure is not None and measure["point"].get("available"):
            scale = float(measure["point"]["pair_scale_m"])
            sequence = [
                scorer(case["family"], case["predicate"], intervention_raw(raw, case["family"], case["predicate"], level, scale))
                for level in levels
            ]
            direction = "nonincreasing" if case["family"] == "proximity" else "nondecreasing"
            row["compatibility_monotone"] = monotone(sequence, direction)
            row["compatibility_endpoint_change"] = float(sequence[-1] - sequence[0])
        rows.append(row)

    summary: dict[str, Any] = {"levels_in_pair_scale_units": levels, "families": {}, "rows": rows}
    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        cell: dict[str, Any] = {"selected_cases": len(family_rows)}
        for measure_name in ("point", "mesh", "compatibility"):
            flags = [row[f"{measure_name}_monotone"] for row in family_rows if row[f"{measure_name}_monotone"] is not None]
            changes = [row[f"{measure_name}_endpoint_change"] for row in family_rows if row[f"{measure_name}_endpoint_change"] is not None]
            cell[measure_name] = {
                "covered_cases": len(flags),
                "monotonicity_rate": float(np.mean(flags)) if flags else None,
                "mean_endpoint_change": float(np.mean(changes)) if changes else None,
                "median_endpoint_change": float(np.median(changes)) if changes else None,
            }
        summary["families"][family] = cell
    return summary


def training_rows_for_output(
    cases: list[dict[str, Any]],
    measurements: dict[tuple[str, int, int], dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    for case in cases:
        yield {
            **case,
            "measurement": measurements[(case["scan_id"], case["subject_id"], case["object_id"])],
        }


def measurement_rows(measurements: dict[tuple[str, int, int], dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for key in sorted(measurements):
        yield measurements[key]


def metrics_csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, payload in summary.items():
        for audit in AUDITS:
            for method in METHODS:
                for k in KS:
                    cell = payload["audits"][audit][method][str(k)]
                    delta = payload["audits"][audit]["relcompat3d_minus_source"][str(k)]
                    recall = payload["recall"][method][str(k)]
                    rows.append({
                        "source": source,
                        "audit": audit,
                        "method": method,
                        "k": k,
                        "recall_all": recall["recall_all"]["point"],
                        "recall_scope": recall["recall_scope"]["point"],
                        "violation": cell["violation"]["point"],
                        "violation_ci_low": cell["violation"]["scan_cluster_ci95"][0],
                        "violation_ci_high": cell["violation"]["scan_cluster_ci95"][1],
                        "coverage": cell["coverage"]["point"],
                        "uncertainty": cell["uncertainty"]["point"],
                        "delta_violation": delta["violation"]["point"] if method == "relcompat3d" else None,
                        "delta_violation_ci_low": delta["violation"]["paired_scan_cluster_ci95"][0] if method == "relcompat3d" else None,
                        "delta_violation_ci_high": delta["violation"]["paired_scan_cluster_ci95"][1] if method == "relcompat3d" else None,
                        **cell["counts"],
                    })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.4f}"


def build_markdown(
    status: str,
    thresholds: dict[str, Any],
    results: dict[str, Any],
    mechanism: dict[str, Any],
    coverage: dict[str, Any],
    claim_boundary: str,
) -> str:
    lines = [
        "# Orthogonal Geometry Audit v1",
        "",
        f"Status: `{status}`",
        "",
        "The primary audit covers proximity and relative-vertical selections only. Point and mesh labels are derived from raw instance surfaces without reading OBB inputs, source scores, compatibility scores, or the existing verifier status.",
        "",
        "## Frozen train-only thresholds",
        "",
        "| Audit | Minimum endpoint support | Proximity P90 / P99 | Vertical absolute-signed P10 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for audit in ("point", "mesh"):
        cell = thresholds[audit]
        lines.append(
            f"| {audit} | {cell['minimum_endpoint_count']} | {cell['proximity']['satisfied_max_p90']:.4f} / {cell['proximity']['violated_min_p99']:.4f} | {cell['relative_vertical']['absolute_signed_margin_p10']:.4f} |"
        )
    lines += [
        "",
        "## Exact-label Recall",
        "",
        "| Source | K | Source R | RelCompat3D R | delta R (paired scan CI) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for source, payload in results.items():
        for k in KS:
            left = payload["recall"]["source"][str(k)]["recall_all"]["point"]
            right = payload["recall"]["relcompat3d"][str(k)]["recall_all"]["point"]
            delta = payload["recall"]["relcompat3d_minus_source"][str(k)]["recall_all"]
            ci = delta["paired_scan_cluster_ci95"]
            lines.append(f"| {source} | {k} | {fmt(left)} | {fmt(right)} | {delta['point']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    lines += ["", "## Independent audit results", ""]
    for audit in AUDITS:
        lines += [
            f"### {audit.title()} audit",
            "",
            "| Source | K | Source V | RelCompat3D V | delta V (paired scan CI) | Source / RelCompat3D coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for source, payload in results.items():
            cell = payload["audits"][audit]
            for k in KS:
                source_cell = cell["source"][str(k)]
                method_cell = cell["relcompat3d"][str(k)]
                delta = cell["relcompat3d_minus_source"][str(k)]["violation"]
                ci = delta["paired_scan_cluster_ci95"]
                lines.append(
                    f"| {source} | {k} | {fmt(source_cell['violation']['point'])} | {fmt(method_cell['violation']['point'])} | "
                    f"{delta['point']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                    f"{fmt(source_cell['coverage']['point'])} / {fmt(method_cell['coverage']['point'])} |"
                )
        lines.append("")
    lines += [
        "## Synthetic-intervention monotonicity",
        "",
        "| Family | Measure | Cases | Monotonicity | Mean endpoint change |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for family, cell in mechanism["families"].items():
        for measure_name in ("point", "mesh", "compatibility"):
            measure = cell[measure_name]
            lines.append(
                f"| {family} | {measure_name} | {measure['covered_cases']} | {fmt(measure['monotonicity_rate'])} | {fmt(measure['mean_endpoint_change'])} |"
            )
    lines += [
        "",
        "## Coverage and interpretation",
        "",
        f"Raw-surface inventory: {coverage['evaluation']['scans_available']}/{coverage['evaluation']['scans_requested']} evaluation scans and {coverage['training']['scans_available']}/{coverage['training']['scans_requested']} training scans.",
        "",
        claim_boundary,
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    protocol_path = resolve(root, args.protocol)
    out = resolve(root, args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"nonempty_output:{out}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_orthogonal_audit_execution":
        raise ValueError("protocol_not_frozen")
    if tuple(protocol["scope"]["ks"]) != KS:
        raise ValueError("k_contract_mismatch")
    paths = {name: resolve(root, value) for name, value in protocol["inputs"].items()}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing_inputs:{missing}")
    for name, expected in protocol["locked_sha256"].items():
        actual = sha256_file(paths[name])
        if actual != expected:
            raise ValueError(f"hash_mismatch:{name}:{actual}")

    temp = out.with_name(out.name + ".tmp")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)

    train_scans = read_scans(paths["train_scans"])
    validation_scans = read_scans(paths["final_validation_scans"])
    annotations = json.loads(paths["official_context_annotations"].read_text(encoding="utf-8"))
    contexts = sorted({f"{row['scan']}_{row['split']}" for row in annotations["scans"]})
    context_scans = {context.rsplit("_", 1)[0] for context in contexts}
    structured_models = json.loads(paths["structured_models"].read_text(encoding="utf-8"))
    strict_models = json.loads(paths["strict_models"].read_text(encoding="utf-8"))
    scorer = base.make_structured_scorer(structured_models)

    point_maximum = int(protocol["raw_surface_contract"]["point_audit"]["maximum_vertices_per_object"])
    triangle_maximum = int(protocol["raw_surface_contract"]["mesh_audit"]["maximum_triangles_per_object"])
    minimum_scale = float(protocol["raw_surface_contract"]["minimum_metric_scale_m"])

    training_cases = load_training_cases(paths["training_table"], train_scans)
    training_measurements, _, train_inventory = measure_pairs(
        paths["raw_scan_root"],
        group_pairs(training_cases),
        point_maximum,
        triangle_maximum,
        minimum_scale,
    )
    thresholds = derive_thresholds(training_cases, training_measurements)
    write_json(temp / "thresholds.json", thresholds)
    write_jsonl(temp / "training_measurements.jsonl", training_rows_for_output(training_cases, training_measurements))
    del training_measurements
    gc.collect()

    source_paths = {
        "vlsat": paths["vlsat_verification"],
        "open3dsg": paths["open3dsg_verification"],
        "sgfn": paths["sgfn_verification"],
    }
    rankings: dict[str, Any] = {}
    source_counts: dict[str, Any] = {}
    for source, path in source_paths.items():
        print(json.dumps({"loading_source": source}), flush=True)
        rankings[source], source_counts[source] = load_source_rankings(path, contexts, scorer, strict_models)

    candidates = selected_candidates(rankings)
    maximum_cases = int(protocol["synthetic_intervention"]["maximum_cases_per_family"])
    mechanism_cases = mechanism_sample(candidates, maximum_cases)
    mechanism_pair_keys = {
        (row["scan_id"], row["subject_id"], row["object_id"])
        for row in mechanism_cases
    }
    evaluation_pairs = group_pairs(candidates)
    for scan in validation_scans:
        evaluation_pairs.setdefault(scan, set())
    evaluation_measurements, mechanism_surfaces, eval_inventory = measure_pairs(
        paths["raw_scan_root"],
        evaluation_pairs,
        point_maximum,
        triangle_maximum,
        minimum_scale,
        mechanism_pair_keys,
    )
    write_jsonl(temp / "evaluation_measurements.jsonl", measurement_rows(evaluation_measurements))

    all_gt, scope_gt = load_ground_truth_scope(paths["ground_truth"])
    all_gt_denominator = sum(len(rows) for rows in all_gt.values())
    scope_gt_denominator = sum(len(rows) for rows in scope_gt.values())
    scans = sorted(validation_scans)
    rng = np.random.default_rng(int(protocol["uncertainty"]["seed"]))
    samples = rng.integers(0, len(scans), size=(int(protocol["uncertainty"]["resamples"]), len(scans)))
    contributions = build_contributions(
        rankings,
        contexts,
        scans,
        all_gt,
        scope_gt,
        evaluation_measurements,
        thresholds,
    )
    result_summary = summarize_contributions(contributions, samples)

    mechanism_features: dict[tuple[str, str], dict[str, float]] = {}
    for source, path in source_paths.items():
        ids = {row["id"] for row in mechanism_cases if row["source"] == source}
        for prediction_id, raw in load_raw_features(path, ids).items():
            mechanism_features[(source, prediction_id)] = raw
    mechanism = run_mechanism_test(
        mechanism_cases,
        mechanism_surfaces,
        evaluation_measurements,
        mechanism_features,
        scorer,
        [float(value) for value in protocol["synthetic_intervention"]["levels_in_pair_scale_units"]],
    )
    mechanism_rows = mechanism.pop("rows")
    write_json(temp / "mechanism.json", mechanism)
    write_csv(temp / "mechanism_rows.csv", mechanism_rows)

    validations = {
        "protocol_frozen_before_execution": True,
        "locked_model_hashes_match": True,
        "train_scans_1061": len(train_scans) == 1061,
        "validation_scans_157": len(validation_scans) == 157,
        "official_contexts_548": len(contexts) == 548,
        "paper_scope_gt_denominator_3972": all_gt_denominator == 3972,
        "audit_scope_gt_denominator_2156": scope_gt_denominator == 2156,
        "context_scans_match_validation_split": context_scans == validation_scans,
        "training_cases_train_only": all(row["scan_id"] in train_scans for row in training_cases),
        "training_and_validation_disjoint": not bool(train_scans & validation_scans),
        "all_training_surface_scans_available": train_inventory["scans_available"] == train_inventory["scans_requested"],
        "all_validation_surface_scans_available": eval_inventory["scans_available"] == eval_inventory["scans_requested"] == 157,
        "all_three_sources_evaluated": set(rankings) == {"vlsat", "open3dsg", "sgfn"},
        "all_ks_evaluated": tuple(protocol["scope"]["ks"]) == KS,
        "point_mesh_consensus_reported": set(AUDITS) == {"point", "mesh", "consensus"},
        "no_obb_or_existing_verifier_input": (
            not protocol["firewall"]["obb_measurements_in_audit"]
            and not protocol["firewall"]["main_verifier_status_in_audit"]
        ),
        "mechanism_cases_present_for_both_families": all(
            mechanism["families"][family]["selected_cases"] > 0 for family in FAMILIES
        ),
    }
    status = "completed" if all(validations.values()) else "failed_validation"
    coverage = {"training": train_inventory, "evaluation": eval_inventory, "sources": source_counts}
    summary = {
        "schema_version": "relcompat3d_orthogonal_geometry_audit_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": protocol["scope"],
        "thresholds": thresholds,
        "coverage": coverage,
        "results": result_summary,
        "mechanism": mechanism,
        "validations": validations,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(temp / "summary.json", summary)
    metric_rows = metrics_csv_rows(result_summary)
    write_csv(temp / "metrics.csv", metric_rows)
    (temp / "summary.md").write_text(
        build_markdown(status, thresholds, result_summary, mechanism, coverage, protocol["claim_boundary"]),
        encoding="utf-8",
    )

    output_names = (
        "thresholds.json",
        "training_measurements.jsonl",
        "evaluation_measurements.jsonl",
        "mechanism.json",
        "mechanism_rows.csv",
        "metrics.csv",
        "summary.json",
        "summary.md",
    )
    input_manifest: dict[str, Any] = {}
    for name, path in paths.items():
        if path.is_file():
            input_manifest[name] = {"path": relpath(root, path), "sha256": sha256_file(path)}
        else:
            input_manifest[name] = {
                "path": relpath(root, path),
                "training_inventory_sha256": train_inventory["inventory_sha256"],
                "evaluation_inventory_sha256": eval_inventory["inventory_sha256"],
            }
    manifest = {
        "schema_version": "relcompat3d_orthogonal_geometry_audit_manifest_v1",
        "status": status,
        "protocol": {"path": relpath(root, protocol_path), "sha256": sha256_file(protocol_path)},
        "inputs": input_manifest,
        "outputs": {name: sha256_file(temp / name) for name in output_names},
        "validations": validations,
        "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm orthogonal_geometry_audit",
    }
    write_json(temp / "manifest.json", manifest)

    if out.exists():
        out.rmdir()
    temp.rename(out)
    print(json.dumps({"status": status, "validations": validations}), flush=True)
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
