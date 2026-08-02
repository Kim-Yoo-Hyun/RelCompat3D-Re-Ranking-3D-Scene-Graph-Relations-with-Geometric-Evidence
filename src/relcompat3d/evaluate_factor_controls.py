#!/usr/bin/env python3
"""Apply every frozen factor condition and metamorphic control to one source."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np


FAMILIES = ("support_contact", "proximity", "relative_vertical")
KS = (5, 10, 20, 50, 100)
MODEL_CONDITIONS = ("M_T", "M_G", "M_add", "M_int", "M_existing")
RANKING_CONDITIONS = (
    "semantic_only",
    "product_M_T", "product_M_G", "product_M_add", "product_M_int", "product_M_existing",
    "rank_average_M_T", "rank_average_M_G", "rank_average_M_add", "rank_average_M_int", "rank_average_M_existing",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260710)
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def relpath(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_eval_module(root: Path) -> Any:
    path = root / "src/relcompat3d/evaluate_metrics.py"
    spec = importlib.util.spec_from_file_location("relcompat3d_eval_factor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot_import:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def raw_numeric(verification: dict[str, Any], predicate: str) -> dict[str, float]:
    source = (verification.get("geometry") or {}).get("features") or {}
    base_names = (
        "distance_3d", "distance_xy", "normalized_distance_3d", "normalized_distance_xy",
        "center_delta_z", "normalized_center_delta_z", "projected_iou_xy",
        "projected_subject_overlap_ratio", "projected_object_overlap_ratio",
        "vertical_gap_subject_on_object", "subject_bottom_z", "subject_top_z",
        "object_bottom_z", "object_top_z",
    )
    values = {name: value for name in base_names if (value := finite(source.get(name))) is not None}
    for source_name, target_name in (
        ("center_delta_z", "abs_center_delta_z"),
        ("normalized_center_delta_z", "abs_normalized_center_delta_z"),
        ("vertical_gap_subject_on_object", "abs_vertical_gap_subject_on_object"),
    ):
        if source_name in values:
            values[target_name] = abs(values[source_name])
    direction = 1.0 if predicate == "higher than" else -1.0 if predicate == "lower than" else 0.0
    if direction and "center_delta_z" in values:
        values["predicate_aligned_center_delta_z"] = direction * values["center_delta_z"]
    if direction and "normalized_center_delta_z" in values:
        values["predicate_aligned_normalized_center_delta_z"] = direction * values["normalized_center_delta_z"]
    return values


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def factor_probability(model: dict[str, Any], family: str, predicate: str, raw: dict[str, float]) -> float:
    vector: list[float] = []
    for feature in model["feature_names"]:
        if feature == "bias":
            vector.append(1.0)
        elif feature.startswith("family:"):
            vector.append(1.0 if family == feature.split(":", 1)[1] else 0.0)
        elif feature.startswith("predicate:"):
            vector.append(1.0 if predicate == feature.split(":", 1)[1] else 0.0)
        elif feature.startswith("num:"):
            name = feature.split(":", 1)[1]
            stat = model["numeric_stats"][name]
            vector.append((raw.get(name, stat["mean"]) - stat["mean"]) / (stat["std"] or 1.0))
        else:
            raise ValueError(f"unsupported_factor_feature:{feature}")
    if len(vector) != len(model["weights"]):
        raise ValueError(f"factor_vector_width_mismatch:{model.get('condition', 'existing')}:{len(vector)}:{len(model['weights'])}")
    return sigmoid(sum(weight * value for weight, value in zip(model["weights"], vector)))


def pred_key(row: dict[str, Any]) -> tuple[str, int, int, int, str]:
    return (
        row["scan_id"], int(row["subset_split_id"]), int(row["edge"]["subject_id"]),
        int(row["edge"]["object_id"]), row["predicate"]["predicate_label"],
    )


def gt_key(row: dict[str, Any]) -> tuple[str, int, int, int, str]:
    return (
        row["scan_id"], int(row["subset_split_id"]), int(row["subject_id"]),
        int(row["object_id"]), row["predicate_label"],
    )


def load_rows(
    prediction_path: Path,
    verification_path: Path,
    models: dict[str, Any],
    evalmod: Any,
    family_model: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_rows = 0
    with prediction_path.open("r", encoding="utf-8") as pred_handle, verification_path.open("r", encoding="utf-8") as ver_handle:
        for line_no, pair in enumerate(zip_longest(pred_handle, ver_handle), 1):
            pred_line, ver_line = pair
            if pred_line is None or ver_line is None:
                raise ValueError(f"prediction_verification_length_mismatch:{line_no}")
            pred, verification = json.loads(pred_line), json.loads(ver_line)
            input_rows += 1
            if pred["prediction_id"] != verification["prediction_id"]:
                raise ValueError(f"prediction_verification_id_mismatch:{line_no}")
            family = pred["predicate"]["predicate_family"]
            if family not in FAMILIES:
                continue
            predicate = pred["predicate"]["predicate_label"]
            semantic = evalmod.semantic_score(pred)
            compact = evalmod.compact_verification(verification)
            existing = evalmod.family_specific_p_geom_valid(pred, compact, family_model)
            if semantic is None or existing is None:
                raise ValueError(f"missing_existing_score:{pred['prediction_id']}")
            raw = raw_numeric(verification, predicate)
            compat = {name: factor_probability(model, family, predicate, raw) for name, model in models.items()}
            compat["M_existing"] = float(existing)
            grouped[pred["subgraph_id"]].append(
                {
                    "id": pred["prediction_id"],
                    "key": pred_key(pred),
                    "subgraph": pred["subgraph_id"],
                    "family": family,
                    "predicate": predicate,
                    "subject": int(pred["edge"]["subject_id"]),
                    "object": int(pred["edge"]["object_id"]),
                    "semantic": float(semantic),
                    "compat": compat,
                    "raw": raw,
                    "status": compact.get("verification_status"),
                    "scores": {},
                }
            )
    for candidates in grouped.values():
        semantic_order = sorted(candidates, key=lambda row: (-row["semantic"], row["key"]))
        semantic_rank = {row["id"]: rank for rank, row in enumerate(semantic_order, 1)}
        denominator = max(len(candidates) - 1, 1)
        for name in MODEL_CONDITIONS:
            compat_order = sorted(candidates, key=lambda row: (-row["compat"][name], row["key"]))
            compat_rank = {row["id"]: rank for rank, row in enumerate(compat_order, 1)}
            for row in candidates:
                sem_pct = 1.0 - (semantic_rank[row["id"]] - 1) / denominator
                comp_pct = 1.0 - (compat_rank[row["id"]] - 1) / denominator
                row["scores"][f"product_{name}"] = row["semantic"] * row["compat"][name]
                row["scores"][f"rank_average_{name}"] = 0.5 * (sem_pct + comp_pct)
        for row in candidates:
            row["scores"]["semantic_only"] = row["semantic"]
    return grouped, input_rows


def load_gt(path: Path) -> tuple[dict[str, set[tuple[Any, ...]]], dict[str, dict[str, set[tuple[Any, ...]]]]]:
    overall: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    by_family: dict[str, dict[str, set[tuple[Any, ...]]]] = defaultdict(lambda: defaultdict(set))
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            family = row["predicate_family"]
            if family in FAMILIES:
                key = gt_key(row)
                overall[row["subgraph_id"]].add(key)
                by_family[row["subgraph_id"]][family].add(key)
    return overall, by_family


def arrays(subgraphs: list[str]) -> dict[str, dict[str, np.ndarray]]:
    return {
        condition: {
            field: np.zeros((len(KS), len(subgraphs)), dtype=np.float64)
            for field in ("recall_num", "recall_den", "violation_num", "violation_den")
        }
        for condition in RANKING_CONDITIONS
    }


def add_metrics(target: dict[str, np.ndarray], k_index: int, s_index: int, selected: list[dict[str, Any]], gt: set[tuple[Any, ...]]) -> None:
    target["recall_num"][k_index, s_index] = len({row["key"] for row in selected} & gt)
    target["recall_den"][k_index, s_index] = len(gt)
    statuses = [row["status"] for row in selected if row["status"] in {"satisfied", "uncertain", "violated"}]
    target["violation_num"][k_index, s_index] = sum(value == "violated" for value in statuses)
    target["violation_den"][k_index, s_index] = len(statuses)


def contributions(grouped: dict[str, list[dict[str, Any]]], gt: dict[str, set[tuple[Any, ...]]], gt_family: dict[str, dict[str, set[tuple[Any, ...]]]], subgraphs: list[str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    overall = arrays(subgraphs)
    family = {name: arrays(subgraphs) for name in FAMILIES}
    global_family_slice = {name: arrays(subgraphs) for name in FAMILIES}
    for s_index, subgraph in enumerate(subgraphs):
        candidates = grouped.get(subgraph, [])
        for condition in RANKING_CONDITIONS:
            ranked = sorted(candidates, key=lambda row: (-row["scores"][condition], row["key"]))
            for k_index, k in enumerate(KS):
                add_metrics(overall[condition], k_index, s_index, ranked[:k], gt.get(subgraph, set()))
                for family_name in FAMILIES:
                    ranked_family = [row for row in ranked if row["family"] == family_name]
                    add_metrics(family[family_name][condition], k_index, s_index, ranked_family[:k], gt_family.get(subgraph, {}).get(family_name, set()))
                    global_selected_family = [row for row in ranked[:k] if row["family"] == family_name]
                    add_metrics(global_family_slice[family_name][condition], k_index, s_index, global_selected_family, gt_family.get(subgraph, {}).get(family_name, set()))
    return overall, family, global_family_slice


def ratio_samples(values: dict[str, np.ndarray], metric: str, k_index: int, samples: np.ndarray) -> tuple[float | None, np.ndarray, int, int]:
    num = values[f"{metric}_num"][k_index]
    den = values[f"{metric}_den"][k_index]
    point = float(num.sum() / den.sum()) if den.sum() else None
    boot_num, boot_den = num[samples].sum(axis=1), den[samples].sum(axis=1)
    boot = np.divide(boot_num, boot_den, out=np.full_like(boot_num, np.nan), where=boot_den > 0)
    return point, boot, int(num.sum()), int(den.sum())


def summarize_scope(values: dict[str, Any], samples: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    report: dict[str, Any] = {}
    sample_cache: dict[str, Any] = {}
    for condition in RANKING_CONDITIONS:
        report[condition], sample_cache[condition] = {}, {}
        for k_index, k in enumerate(KS):
            report[condition][str(k)], sample_cache[condition][str(k)] = {}, {}
            for metric in ("recall", "violation"):
                point, boot, numerator, denominator = ratio_samples(values[condition], metric, k_index, samples)
                finite_boot = boot[np.isfinite(boot)]
                report[condition][str(k)][metric] = {
                    "point": point,
                    "ci95": [float(value) for value in np.percentile(finite_boot, [2.5, 97.5])] if len(finite_boot) else [None, None],
                    "numerator": numerator,
                    "denominator": denominator,
                }
                sample_cache[condition][str(k)][metric] = boot
    report["deltas_vs_semantic_only"] = {}
    for condition in RANKING_CONDITIONS[1:]:
        report["deltas_vs_semantic_only"][condition] = {}
        for k in KS:
            report["deltas_vs_semantic_only"][condition][str(k)] = {}
            for metric in ("recall", "violation"):
                left_point = report[condition][str(k)][metric]["point"]
                right_point = report["semantic_only"][str(k)][metric]["point"]
                point = left_point - right_point if left_point is not None and right_point is not None else None
                boot = sample_cache[condition][str(k)][metric] - sample_cache["semantic_only"][str(k)][metric]
                finite_boot = boot[np.isfinite(boot)]
                report["deltas_vs_semantic_only"][condition][str(k)][metric] = {
                    "point": point,
                    "ci95": [float(value) for value in np.percentile(finite_boot, [2.5, 97.5])] if len(finite_boot) else [None, None],
                }
    return report, sample_cache


def add_fixed_contrasts(report: dict[str, Any], sample_cache: dict[str, Any]) -> None:
    contrast_models = {
        "M_int_minus_M_T": ("M_int", "M_T"),
        "M_int_minus_M_G": ("M_int", "M_G"),
        "M_int_minus_M_add": ("M_int", "M_add"),
        "M_existing_minus_semantic_only_for_continuity_only": ("M_existing", None),
    }
    report["fixed_contrasts"] = {}
    for fusion in ("product", "rank_average"):
        report["fixed_contrasts"][fusion] = {}
        for contrast, (left_model, right_model) in contrast_models.items():
            left = f"{fusion}_{left_model}"
            right = "semantic_only" if right_model is None else f"{fusion}_{right_model}"
            report["fixed_contrasts"][fusion][contrast] = {}
            for k in KS:
                report["fixed_contrasts"][fusion][contrast][str(k)] = {}
                for metric in ("recall", "violation"):
                    left_point = report[left][str(k)][metric]["point"]
                    right_point = report[right][str(k)][metric]["point"]
                    point = left_point - right_point if left_point is not None and right_point is not None else None
                    boot = sample_cache[left][str(k)][metric] - sample_cache[right][str(k)][metric]
                    finite_boot = boot[np.isfinite(boot)]
                    report["fixed_contrasts"][fusion][contrast][str(k)][metric] = {
                        "point": point,
                        "ci95": [float(value) for value in np.percentile(finite_boot, [2.5, 97.5])] if len(finite_boot) else [None, None],
                    }


def add_simultaneous_family_ci(family_report: dict[str, Any], family_samples: dict[str, Any]) -> None:
    for condition in RANKING_CONDITIONS[1:]:
        for k in KS:
            for metric in ("recall", "violation"):
                deltas = []
                points = []
                active_families = []
                for family in FAMILIES:
                    current = family_samples[family][condition][str(k)][metric]
                    reference = family_samples[family]["semantic_only"][str(k)][metric]
                    delta = current - reference
                    point = family_report[family]["deltas_vs_semantic_only"][condition][str(k)][metric]["point"]
                    if point is not None and np.any(np.isfinite(delta)):
                        deltas.append(delta)
                        points.append(point)
                        active_families.append(family)
                if not deltas:
                    for family in FAMILIES:
                        family_report[family]["deltas_vs_semantic_only"][condition][str(k)][metric]["simultaneous_familywise_ci95"] = [None, None]
                    continue
                matrix = np.column_stack(deltas)
                point_array = np.asarray(points, dtype=np.float64)
                max_error = np.nanmax(np.abs(matrix - point_array[None, :]), axis=1)
                radius = float(np.nanpercentile(max_error, 95.0))
                for family_index, family in enumerate(FAMILIES):
                    item = family_report[family]["deltas_vs_semantic_only"][condition][str(k)][metric]
                    item["simultaneous_familywise_ci95"] = (
                        [item["point"] - radius, item["point"] + radius]
                        if family in active_families and item["point"] is not None
                        else [None, None]
                    )
    for fusion in ("product", "rank_average"):
        for contrast in family_report[FAMILIES[0]]["fixed_contrasts"][fusion]:
            if contrast == "M_existing_minus_semantic_only_for_continuity_only":
                left, right = f"{fusion}_M_existing", "semantic_only"
            else:
                left_model, right_model = contrast.split("_minus_")
                left, right = f"{fusion}_{left_model}", f"{fusion}_{right_model}"
            for k in KS:
                for metric in ("recall", "violation"):
                    matrices, points, active_families = [], [], []
                    for family in FAMILIES:
                        delta = (
                            family_samples[family][left][str(k)][metric]
                            - family_samples[family][right][str(k)][metric]
                        )
                        point = family_report[family]["fixed_contrasts"][fusion][contrast][str(k)][metric]["point"]
                        if point is not None and np.any(np.isfinite(delta)):
                            matrices.append(delta)
                            points.append(point)
                            active_families.append(family)
                    if not matrices:
                        for family in FAMILIES:
                            family_report[family]["fixed_contrasts"][fusion][contrast][str(k)][metric]["simultaneous_familywise_ci95"] = [None, None]
                        continue
                    matrix = np.column_stack(matrices)
                    point_array = np.asarray(points, dtype=np.float64)
                    radius = float(
                        np.nanpercentile(
                            np.nanmax(np.abs(matrix - point_array[None, :]), axis=1),
                            95.0,
                        )
                    )
                    for family in FAMILIES:
                        item = family_report[family]["fixed_contrasts"][fusion][contrast][str(k)][metric]
                        item["simultaneous_familywise_ci95"] = (
                            [item["point"] - radius, item["point"] + radius]
                            if family in active_families and item["point"] is not None
                            else [None, None]
                        )


def clustered_control_ci(rows: list[tuple[str, float]], subgraphs: list[str], samples: np.ndarray) -> dict[str, Any]:
    sums = np.zeros(len(subgraphs), dtype=np.float64)
    counts = np.zeros(len(subgraphs), dtype=np.float64)
    index = {value: offset for offset, value in enumerate(subgraphs)}
    for subgraph, value in rows:
        sums[index[subgraph]] += value
        counts[index[subgraph]] += 1
    boot_sums, boot_counts = sums[samples].sum(axis=1), counts[samples].sum(axis=1)
    boot = np.divide(boot_sums, boot_counts, out=np.full_like(boot_sums, np.nan), where=boot_counts > 0)
    finite_boot = boot[np.isfinite(boot)]
    raw_values = np.asarray([value for _, value in rows], dtype=np.float64)
    return {
        "rows": int(counts.sum()),
        "subgraphs": int(np.sum(counts > 0)),
        "mean": float(sums.sum() / counts.sum()) if counts.sum() else None,
        "median": float(np.median(raw_values)) if len(raw_values) else None,
        "p95": float(np.percentile(raw_values, 95.0)) if len(raw_values) else None,
        "paired_subgraph_bootstrap_ci95": [float(value) for value in np.percentile(finite_boot, [2.5, 97.5])],
    }


def controls(grouped: dict[str, list[dict[str, Any]]], models: dict[str, Any], family_model: dict[str, Any], subgraphs: list[str], samples: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {"relative_vertical_wrong_T": {}, "close_by_swap_invariance": {}, "relative_vertical_inverse_equivariance": {}}
    by_tuple: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for candidates in grouped.values():
        for row in candidates:
            by_tuple[(row["subgraph"], row["subject"], row["object"], row["predicate"])] = row
    wrong_t_models = dict(models)
    wrong_t_models["M_existing"] = family_model["family_models"]["relative_vertical"]
    for model_name in ("M_T", "M_add", "M_int", "M_existing"):
        differences, wins = [], []
        for candidates in grouped.values():
            for row in candidates:
                if row["family"] != "relative_vertical":
                    continue
                wrong = "lower than" if row["predicate"] == "higher than" else "higher than"
                wrong_c = factor_probability(wrong_t_models[model_name], row["family"], wrong, raw_numeric({"geometry": {"features": row["raw"]}}, wrong))
                difference = row["compat"][model_name] - wrong_c
                differences.append((row["subgraph"], difference))
                wins.append((row["subgraph"], float(difference > 0.0)))
        result["relative_vertical_wrong_T"][model_name] = {
            "correct_minus_wrong": clustered_control_ci(differences, subgraphs, samples),
            "correct_above_wrong_rate": clustered_control_ci(wins, subgraphs, samples),
        }
    for model_name in MODEL_CONDITIONS:
        close_diffs, vertical_diffs = [], []
        for candidates in grouped.values():
            for row in candidates:
                if row["predicate"] == "close by":
                    swapped = by_tuple.get((row["subgraph"], row["object"], row["subject"], "close by"))
                    if swapped is not None:
                        close_diffs.append((row["subgraph"], abs(row["compat"][model_name] - swapped["compat"][model_name])))
                if row["family"] == "relative_vertical":
                    inverse = "lower than" if row["predicate"] == "higher than" else "higher than"
                    swapped = by_tuple.get((row["subgraph"], row["object"], row["subject"], inverse))
                    if swapped is not None:
                        vertical_diffs.append((row["subgraph"], abs(row["compat"][model_name] - swapped["compat"][model_name])))
        result["close_by_swap_invariance"][model_name] = {"absolute_difference": clustered_control_ci(close_diffs, subgraphs, samples)}
        result["relative_vertical_inverse_equivariance"][model_name] = {"absolute_difference": clustered_control_ci(vertical_diffs, subgraphs, samples)}
    result["support_contact_endpoint_swap"] = {"status": "not_run_prohibited_by_frozen_protocol"}
    return result


def make_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RelCompat3D Fresh-Source Factor-Isolation Evaluation",
        "",
        f"Status: `{report['status']}`",
        "",
        "All factor models were fit on calibration train only; dev and fresh-source results selected nothing.",
        "",
        "## Global K=100",
        "",
        "| condition | Recall | delta Recall | V | delta V |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    scope = report["overall_global"]
    for condition in RANKING_CONDITIONS:
        value = scope[condition]["100"]
        delta = scope["deltas_vs_semantic_only"].get(condition, {}).get("100", {})
        lines.append(
            f"| {condition} | {value['recall']['point']:.6f} | {delta.get('recall', {}).get('point', 0.0):.6f} | {value['violation']['point']:.6f} | {delta.get('violation', {}).get('point', 0.0):.6f} |"
        )
    lines.extend(["", "Family-wise marginal and simultaneous CIs and frozen metamorphic controls are in `summary.json`.", ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    paths = {
        "predictions": resolve(root, args.predictions),
        "verification": resolve(root, args.verification),
        "ground_truth": resolve(root, args.ground_truth),
        "models": resolve(root, args.models),
        "protocol": resolve(root, args.protocol),
    }
    out = resolve(root, args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"nonempty_output:{out}")
    protocol = read_json(paths["protocol"])
    if protocol.get("schema_version") != "relcompat3d_factor_controls_v1":
        raise ValueError("protocol_not_frozen_v1")
    model_payload = read_json(paths["models"])
    models = model_payload["models"]
    evalmod = load_eval_module(root)
    family_model_path = root / "archive/hypothesis_records/hypothesis/CAND-001/RelCompat3D_geometry-grounded-verification/artifacts/calibration/p_geom_valid_family/model.json"
    family_model = read_json(family_model_path)
    grouped, input_rows = load_rows(paths["predictions"], paths["verification"], models, evalmod, family_model)
    gt, gt_family = load_gt(paths["ground_truth"])
    subgraphs = sorted(set(grouped) | set(gt))
    rng = np.random.default_rng(args.seed)
    sample_indices = rng.integers(0, len(subgraphs), size=(args.n_bootstrap, len(subgraphs)))
    overall_contrib, family_contrib, slice_contrib = contributions(grouped, gt, gt_family, subgraphs)
    overall_report, overall_samples = summarize_scope(overall_contrib, sample_indices)
    add_fixed_contrasts(overall_report, overall_samples)
    family_report, family_samples, slice_report, slice_samples = {}, {}, {}, {}
    for family in FAMILIES:
        family_report[family], family_samples[family] = summarize_scope(family_contrib[family], sample_indices)
        add_fixed_contrasts(family_report[family], family_samples[family])
        slice_report[family], slice_samples[family] = summarize_scope(slice_contrib[family], sample_indices)
        add_fixed_contrasts(slice_report[family], slice_samples[family])
    add_simultaneous_family_ci(family_report, family_samples)
    add_simultaneous_family_ci(slice_report, slice_samples)
    validations = {
        "evaluation_contexts_548": len(subgraphs) == 548,
        "ground_truth_denominator_3972": sum(len(value) for value in gt.values()) == 3972,
        "all_four_frozen_factor_models_present": set(models) == {"M_T", "M_G", "M_add", "M_int"},
        "all_conditions_reported": set(RANKING_CONDITIONS) == set(overall_report) - {"deltas_vs_semantic_only", "fixed_contrasts"},
        "fixed_contrasts_reported": set(overall_report["fixed_contrasts"]["product"]) == {
            "M_int_minus_M_T", "M_int_minus_M_G", "M_int_minus_M_add",
            "M_existing_minus_semantic_only_for_continuity_only",
        },
        "global_topk_family_slice_reported": set(slice_report) == set(FAMILIES),
        "bootstrap_1000_seed_20260710": args.n_bootstrap == 1000 and args.seed == 20260710,
    }
    report = {
        "schema_version": "relcompat3d_factor_isolation_metrics_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "factor_isolation_fresh_source_ready" if all(validations.values()) else "blocked_factor_isolation_metrics",
        "source": "3dssg_official_full_l160_confirmatory",
        "input_rows": input_rows,
        "in_scope_rows": sum(len(value) for value in grouped.values()),
        "subgraphs": len(subgraphs),
        "conditions": list(RANKING_CONDITIONS),
        "overall_global": overall_report,
        "within_family": family_report,
        "global_topk_family_slice": slice_report,
        "controls": controls(grouped, models, family_model, subgraphs, sample_indices),
        "validations": validations,
        "limitations": [
            "Calibration target y_cal is constructed rather than independent human physical-validity labels.",
            "Violation is frozen verifier-derived and remains diagnostic.",
            "No factor condition is promoted based on this target.",
        ],
        "inputs": {
            **{name: {"path": relpath(root, path), "sha256": sha256_file(path)} for name, path in paths.items()},
            "existing_family_model": {"path": relpath(root, family_model_path), "sha256": sha256_file(family_model_path)},
        },
        "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm factor_isolation_metrics_3dssg",
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", report)
    (out / "summary.md").write_text(make_markdown(report), encoding="utf-8")
    write_json(out / "manifest.json", {
        "schema_version": report["schema_version"],
        "created_at_utc": report["created_at_utc"],
        "status": report["status"],
        "outputs": {
            "summary_json": {"path": relpath(root, out / "summary.json"), "sha256": sha256_file(out / "summary.json")},
            "summary_md": {"path": relpath(root, out / "summary.md"), "sha256": sha256_file(out / "summary.md")},
        },
        "docker_command": report["docker_command"],
    })
    print(json.dumps({"status": report["status"], "input_rows": input_rows, "in_scope_rows": report["in_scope_rows"], "out": relpath(root, out)}))
    return 0 if report["status"] == "factor_isolation_fresh_source_ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
