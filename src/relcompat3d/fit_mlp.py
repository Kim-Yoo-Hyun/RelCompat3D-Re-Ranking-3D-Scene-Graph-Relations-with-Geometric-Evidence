#!/usr/bin/env python3
"""Compare RelCompat3D with source-excluded, supervision-matched MLP models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import compatibility_features as calibration
import relation_consistency as algebra
import evaluate_all_families as base
import evaluate_base_models as model_eval


FAMILIES = base.FAMILIES
PREDICATES = ("close by", "higher than", "lower than", "lying on", "standing on", "supported by")
RAW_FEATURES = (
    "distance_3d", "distance_xy", "normalized_distance_3d", "normalized_distance_xy",
    "center_delta_z", "normalized_center_delta_z", "projected_iou_xy",
    "projected_subject_overlap_ratio", "projected_object_overlap_ratio",
    "vertical_gap_subject_on_object", "subject_bottom_z", "subject_top_z",
    "object_bottom_z", "object_top_z", "abs_center_delta_z",
    "abs_normalized_center_delta_z", "abs_vertical_gap_subject_on_object",
)
INTERACTIONS = (
    "predicate_aligned_center_delta_z",
    "predicate_aligned_normalized_center_delta_z",
    "overlap_sum",
)
INPUT_DIM = len(FAMILIES) + len(PREDICATES) + len(RAW_FEATURES) + len(INTERACTIONS)
HIDDEN = 2
PARAMETER_COUNT = HIDDEN * INPUT_DIM + HIDDEN + HIDDEN + 1 + len(PREDICATES)
METHODS = (
    "source",
    "all_family_product",
    "shared_mlp_bce_product",
    "shared_mlp_pairwise_product",
)


def add_reference_deltas(
    report: dict[str, Any],
    cache: dict[str, Any],
    reference: str,
    methods: tuple[str, ...],
) -> None:
    """Add paired bootstrap contrasts against a named method."""
    key = f"deltas_vs_{reference}"
    report[key] = {}
    for method in methods:
        report[key][method] = {}
        for k in KS:
            report[key][method][str(k)] = {}
            for metric in base.RATIO_METRICS:
                left = report[method][str(k)][metric]["point"]
                right = report[reference][str(k)][metric]["point"]
                delta = cache[method][str(k)][metric] - cache[reference][str(k)][metric]
                report[key][method][str(k)][metric] = {
                    "point": left - right if left is not None and right is not None else None,
                    "paired_ci95": base.ci95(delta),
                }
KS = base.KS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--calibration-table",
        type=Path,
        help="Override the protocol calibration-table path with a local regenerated table.",
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="Fit and export the two shared MLP estimators without source evaluation.",
    )
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def relpath(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_scans(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def raw_feature_values(family: str, predicate: str, raw: dict[str, float]) -> list[float | None]:
    aligned = model_eval.align_predicate(raw, predicate)
    overlap_a = aligned.get("projected_subject_overlap_ratio")
    overlap_b = aligned.get("projected_object_overlap_ratio")
    overlap_sum = overlap_a + overlap_b if overlap_a is not None and overlap_b is not None else None
    values: list[float | None] = []
    values.extend(float(family == item) for item in FAMILIES)
    values.extend(float(predicate == item) for item in PREDICATES)
    values.extend(aligned.get(name) for name in RAW_FEATURES)
    values.extend((
        aligned.get("predicate_aligned_center_delta_z"),
        aligned.get("predicate_aligned_normalized_center_delta_z"),
        overlap_sum,
    ))
    if len(values) != INPUT_DIM:
        raise AssertionError(f"feature_width:{len(values)}:{INPUT_DIM}")
    return values


def fit_stats(rows: list[list[float | None]]) -> dict[str, Any]:
    raw = np.asarray([[np.nan if value is None else value for value in row] for row in rows], dtype=np.float64)
    means = np.nanmean(raw, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    filled = np.where(np.isnan(raw), means[None, :], raw)
    continuous = np.arange(len(FAMILIES) + len(PREDICATES), INPUT_DIM)
    std = np.ones(INPUT_DIM, dtype=np.float64)
    std[continuous] = filled[:, continuous].std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return {"mean": means.tolist(), "std": std.tolist(), "continuous_indices": continuous.tolist()}


def normalize(rows: list[list[float | None]], stats: dict[str, Any]) -> np.ndarray:
    means, std = np.asarray(stats["mean"]), np.asarray(stats["std"])
    raw = np.asarray([[np.nan if value is None else value for value in row] for row in rows], dtype=np.float64)
    filled = np.where(np.isnan(raw), means[None, :], raw)
    continuous = np.asarray(stats["continuous_indices"], dtype=int)
    filled[:, continuous] = (filled[:, continuous] - means[continuous]) / std[continuous]
    return filled


def initialize(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "W": rng.normal(0.0, math.sqrt(2.0 / INPUT_DIM), size=(HIDDEN, INPUT_DIM)),
        "b": np.zeros(HIDDEN),
        "v": rng.normal(0.0, math.sqrt(2.0 / HIDDEN), size=HIDDEN),
        "out_b": np.zeros(1),
        "predicate_skip": np.zeros(len(PREDICATES)),
    }


def sigmoid(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def forward(params: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pre = x @ params["W"].T + params["b"]
    hidden = np.maximum(pre, 0.0)
    predicate_start = len(FAMILIES)
    logits = hidden @ params["v"] + params["out_b"][0] + x[:, predicate_start:predicate_start + len(PREDICATES)] @ params["predicate_skip"]
    return sigmoid(logits), logits, hidden, pre


def fit(
    x: np.ndarray,
    y: np.ndarray,
    pairs: list[tuple[int, int]],
    spec: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    params = initialize(int(spec["seed"]))
    first = {name: np.zeros_like(value) for name, value in params.items()}
    second = {name: np.zeros_like(value) for name, value in params.items()}
    pair_array = np.asarray(pairs, dtype=int) if pairs else np.empty((0, 2), dtype=int)
    trace: list[dict[str, float]] = []
    lr, l2 = float(spec["learning_rate"]), float(spec["l2"])
    pair_weight, margin = float(spec["pairwise_weight"]), float(spec["pairwise_margin"])
    for epoch in range(1, int(spec["epochs"]) + 1):
        probs, logits, hidden, pre = forward(params, x)
        dlogit = (probs - y) / len(y)
        pair_loss = 0.0
        if len(pair_array):
            residual = margin - (logits[pair_array[:, 0]] - logits[pair_array[:, 1]])
            pair_prob = sigmoid(residual)
            scale = pair_weight / len(pair_array)
            np.add.at(dlogit, pair_array[:, 0], -scale * pair_prob)
            np.add.at(dlogit, pair_array[:, 1], scale * pair_prob)
            pair_loss = float(np.mean(np.logaddexp(0.0, residual)))
        predicate_start = len(FAMILIES)
        grads = {
            "v": hidden.T @ dlogit + l2 * params["v"],
            "out_b": np.asarray([dlogit.sum()]),
            "predicate_skip": x[:, predicate_start:predicate_start + len(PREDICATES)].T @ dlogit + l2 * params["predicate_skip"],
        }
        dpre = (dlogit[:, None] * params["v"][None, :]) * (pre > 0.0)
        grads["W"] = dpre.T @ x + l2 * params["W"]
        grads["b"] = dpre.sum(axis=0)
        for name in params:
            first[name] = 0.9 * first[name] + 0.1 * grads[name]
            second[name] = 0.999 * second[name] + 0.001 * np.square(grads[name])
            mhat = first[name] / (1.0 - 0.9 ** epoch)
            vhat = second[name] / (1.0 - 0.999 ** epoch)
            params[name] -= lr * mhat / (np.sqrt(vhat) + 1e-8)
        if epoch in {1, 20, 60, int(spec["epochs"])}:
            clipped = np.clip(probs, 1e-12, 1.0 - 1e-12)
            bce = float(np.mean(-(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))))
            trace.append({"epoch": epoch, "bce": bce, "pairwise_softplus": pair_loss, "objective": bce + pair_weight * pair_loss})
    return params, trace


def serialize_model(params: dict[str, np.ndarray], stats: dict[str, Any], trace: list[dict[str, float]], training: dict[str, Any]) -> dict[str, Any]:
    return {
        "architecture": "source-excluded two-hidden-unit ReLU MLP with predicate-linear skip",
        "input_dim": INPUT_DIM,
        "hidden_width": HIDDEN,
        "parameter_count": int(sum(value.size for value in params.values())),
        "feature_spec": {
            "family_one_hot": list(FAMILIES),
            "predicate_one_hot": list(PREDICATES),
            "raw_geometry": list(RAW_FEATURES),
            "interactions": list(INTERACTIONS),
            "source_score_input": False,
            "source_identity_input": False
        },
        "normalization": stats,
        "parameters": {name: value.tolist() for name, value in params.items()},
        "training_trace": trace,
        "training": training,
    }


def params_from(model: dict[str, Any]) -> dict[str, np.ndarray]:
    return {name: np.asarray(value, dtype=np.float64) for name, value in model["parameters"].items()}


def probability(model: dict[str, Any], family: str, predicate: str, raw: dict[str, float]) -> float:
    values = raw_feature_values(family, predicate, raw)
    x = normalize([values], model["normalization"])
    return float(forward(params_from(model), x)[0][0])


def projected_probability(model: dict[str, Any], family: str, predicate: str, raw: dict[str, float]) -> float:
    direct = probability(model, family, predicate, raw)
    transformed = algebra.transformed_view(family, predicate, raw)
    if transformed is None:
        return direct
    transformed_predicate, transformed_raw = transformed
    return 0.5 * (direct + probability(model, family, transformed_predicate, transformed_raw))


def prepare_training(prepared: list[dict[str, Any]], spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    train = [row for row in prepared if row["_role"] == "train"]
    original_values = [raw_feature_values(row["predicate"]["predicate_family"], row["predicate"]["predicate_label"], row["_raw_numeric"]) for row in train]
    stats = fit_stats(original_values)
    original_x = normalize(original_values, stats)
    original_y = np.asarray([row["_label"] for row in train], dtype=np.float64)
    id_to_index = {row["candidate_id"]: index for index, row in enumerate(train)}
    original_pairs = []
    for neg_index, row in enumerate(train):
        base_id = (row.get("candidate_source") or {}).get("base_candidate_id")
        if row["_label"] == 0 and base_id in id_to_index and train[id_to_index[base_id]]["_label"] == 1:
            original_pairs.append((id_to_index[base_id], neg_index))

    orbit_values = list(original_values)
    orbit_labels = original_y.tolist()
    transform_index: dict[int, int] = {}
    for index, row in enumerate(train):
        family, predicate, raw = row["predicate"]["predicate_family"], row["predicate"]["predicate_label"], row["_raw_numeric"]
        transformed = algebra.transformed_view(family, predicate, raw)
        if transformed is None:
            continue
        transformed_predicate, transformed_raw = transformed
        transform_index[index] = len(orbit_values)
        orbit_values.append(raw_feature_values(family, transformed_predicate, transformed_raw))
        orbit_labels.append(row["_label"])
    orbit_pairs = list(original_pairs)
    orbit_pairs.extend((transform_index[pos], transform_index[neg]) for pos, neg in original_pairs if pos in transform_index and neg in transform_index)
    bce_params, bce_trace = fit(original_x, original_y, [], {**spec, "pairwise_weight": 0.0})
    pairwise_params, pairwise_trace = fit(normalize(orbit_values, stats), np.asarray(orbit_labels, dtype=np.float64), orbit_pairs, spec)
    common = {"train_rows": len(train), "linked_pairs": len(original_pairs), "orbit_rows": len(orbit_values), "orbit_pairs": len(orbit_pairs)}
    return (
        serialize_model(bce_params, stats, bce_trace, {**common, "objective": "BCE"}),
        serialize_model(pairwise_params, stats, pairwise_trace, {**common, "objective": "BCE + linked pairwise margin + relation-algebra augmentation"}),
    )


def load_candidates(
    path: Path,
    linear_score: Any,
    bce_model: dict[str, Any],
    pairwise_model: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    digest = hashlib.sha256()
    input_rows = in_scope_rows = 0
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
            predicate, raw = row["predicate"]["predicate_label"], model_eval.raw_numeric(row)
            semantic = model_eval.finite((row.get("semantic") or {}).get("ranking_score"))
            if semantic is None:
                raise ValueError(f"missing_semantic:{row['prediction_id']}")
            c_linear = linear_score(family, predicate, raw)
            c_bce = probability(bce_model, family, predicate, raw)
            c_mlp_pairwise = projected_probability(pairwise_model, family, predicate, raw)
            grouped[row["subgraph_id"]].append({
                "id": row["prediction_id"], "scan": row["scan_id"], "key": model_eval.candidate_key(row),
                "family": family, "predicate": predicate, "semantic": float(semantic),
                "linear": float(c_linear), "previous": 0.0, "pooled": 0.0,
                "status": row.get("verification_status") or (row.get("verification") or {}).get("verification_status"),
                "point_status": base.point_status(row),
                "scores": {
                    "source": float(semantic),
                    "all_family_product": float(semantic) * c_linear,
                    "shared_mlp_bce_product": float(semantic) * c_bce,
                    "shared_mlp_pairwise_product": float(semantic) * c_mlp_pairwise,
                },
            })
    return grouped, {"input_rows": input_rows, "in_scope_rows": in_scope_rows, "input_sha256": digest.hexdigest()}


def evaluate_source(path: Path, gt_path: Path, linear_score: Any, bce_model: dict[str, Any], pairwise_model: dict[str, Any], seed: int, resamples: int) -> dict[str, Any]:
    grouped, counts = load_candidates(path, linear_score, bce_model, pairwise_model)
    gt, gt_family = model_eval.load_gt(gt_path)
    contexts = sorted(set(grouped) | set(gt))
    samples = np.random.default_rng(seed).integers(0, len(contexts), size=(resamples, len(contexts)))
    original_methods = base.METHODS
    base.METHODS = METHODS
    try:
        overall_values, within_values, global_values = base.contributions(grouped, gt, gt_family, contexts)
        overall, overall_cache = base.summarize(overall_values, samples)
        add_reference_deltas(
            overall,
            overall_cache,
            "all_family_product",
            ("shared_mlp_bce_product", "shared_mlp_pairwise_product"),
        )
        within, global_slice = {}, {}
        for family in FAMILIES:
            within[family], within_cache = base.summarize(within_values[family], samples)
            global_slice[family], global_cache = base.summarize(global_values[family], samples)
            add_reference_deltas(
                within[family],
                within_cache,
                "all_family_product",
                ("shared_mlp_bce_product", "shared_mlp_pairwise_product"),
            )
            add_reference_deltas(
                global_slice[family],
                global_cache,
                "all_family_product",
                ("shared_mlp_bce_product", "shared_mlp_pairwise_product"),
            )
    finally:
        base.METHODS = original_methods
    return {"counts": {**counts, "contexts": len(contexts), "gt_denominator": sum(len(rows) for rows in gt.values())}, "overall": overall, "within_family": within, "global_topk_family_slice": global_slice}


def calibration_diagnostic(prepared: list[dict[str, Any]], model: dict[str, Any], projected: bool) -> dict[str, Any]:
    dev = [row for row in prepared if row["_role"] == "dev"]
    probs, labels = [], []
    for row in dev:
        family, predicate, raw = row["predicate"]["predicate_family"], row["predicate"]["predicate_label"], row["_raw_numeric"]
        probs.append(projected_probability(model, family, predicate, raw) if projected else probability(model, family, predicate, raw))
        labels.append(row["_label"])
    return {
        "rows": len(labels), "positive": int(sum(labels)),
        "brier": calibration.brier_score(probs, labels),
        "auroc": calibration.auroc(probs, labels),
        "auprc": calibration.average_precision(probs, labels),
    }


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    protocol_path, out = resolve(root, args.protocol), resolve(root, args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"nonempty_output:{out}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "ready_for_mlp_fit":
        raise ValueError("protocol_version_mismatch")
    paths = {name: resolve(root, value) for name, value in protocol["inputs"].items()}
    if args.calibration_table is not None:
        paths["calibration_table"] = resolve(root, args.calibration_table)
    fit_inputs = {
        name: paths[name]
        for name in (
            "calibration_table",
            "train_scans",
            "development_scans",
            "final_validation_scans",
        )
    }
    required_paths = fit_inputs if args.fit_only else paths
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing_inputs:{missing}")
    train_scans, dev_scans, final_scans = (read_scans(paths[name]) for name in ("train_scans", "development_scans", "final_validation_scans"))
    if train_scans & dev_scans or train_scans & final_scans or dev_scans & final_scans:
        raise ValueError("split_overlap")
    table_rows = calibration.load_jsonl(paths["calibration_table"])
    prepared, warnings = calibration.prepare_rows(table_rows, train_scans, dev_scans, set(FAMILIES))
    bce_model, pairwise_model = prepare_training(prepared, protocol["optimizer"])
    if args.fit_only:
        validations = {
            "split_counts_1061_117_157": (
                len(train_scans), len(dev_scans), len(final_scans)
            ) == (1061, 117, 157),
            "split_sets_pairwise_disjoint": not (
                train_scans & dev_scans
                or train_scans & final_scans
                or dev_scans & final_scans
            ),
            "zero_final_rows_in_training_table": not (
                {row["scan_id"] for row in table_rows} & final_scans
            ),
            "train_rows_60208": sum(
                row["_role"] == "train" for row in prepared
            ) == 60208,
            "dev_rows_6246": sum(
                row["_role"] == "dev" for row in prepared
            ) == 6246,
            "parameter_count_69": (
                bce_model["parameter_count"]
                == pairwise_model["parameter_count"]
                == PARAMETER_COUNT
                == 69
            ),
            "source_score_excluded": (
                not bce_model["feature_spec"]["source_score_input"]
                and not pairwise_model["feature_spec"]["source_score_input"]
            ),
            "all_parameters_finite": all(
                math.isfinite(float(value))
                for model in (bce_model, pairwise_model)
                for array in model["parameters"].values()
                for value in np.asarray(array).ravel()
            ),
        }
        status = "completed" if all(validations.values()) else "failed_validation"
        out.mkdir(parents=True, exist_ok=True)
        models_path = out / "models.json"
        summary_path = out / "summary.json"
        write_json(
            models_path,
            {
                "shared_mlp_bce": bce_model,
                "shared_mlp_pairwise": pairwise_model,
            },
        )
        write_json(
            summary_path,
            {
                "schema_version": "relcompat3d_mlp_fit_v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "training_warnings": warnings,
                "parameter_count": PARAMETER_COUNT,
                "development_calibration": {
                    "shared_mlp_bce": calibration_diagnostic(
                        prepared, bce_model, False
                    ),
                    "shared_mlp_pairwise": calibration_diagnostic(
                        prepared, pairwise_model, True
                    ),
                },
                "validations": validations,
            },
        )
        write_json(
            out / "manifest.json",
            {
                "schema_version": "relcompat3d_mlp_fit_manifest_v1",
                "status": status,
                "protocol": {
                    "path": relpath(root, protocol_path),
                    "sha256": sha256(protocol_path),
                },
                "inputs": {
                    name: {"path": relpath(root, path), "sha256": sha256(path)}
                    for name, path in fit_inputs.items()
                },
                "outputs": {
                    path.name: {"path": relpath(root, path), "sha256": sha256(path)}
                    for path in (models_path, summary_path)
                },
                "validations": validations,
            },
        )
        print(json.dumps({"status": status, "validations": validations}))
        return 0 if status == "completed" else 2
    linear_models = json.loads(paths["linear_models"].read_text(encoding="utf-8"))
    linear_score = base.make_linear_scorer(linear_models)
    sources = {
        "development": (paths["development_verification"], paths["development_ground_truth"]),
        "vlsat": (paths["vlsat_verification"], paths["final_ground_truth"]),
        "open3dsg": (paths["open3dsg_verification"], paths["final_ground_truth"]),
        "sgfn": (paths["sgfn_verification"], paths["final_ground_truth"]),
    }
    seed, resamples = int(protocol["evaluation"]["bootstrap_seed"]), int(protocol["evaluation"]["bootstrap_resamples"])
    results = {source: evaluate_source(path, gt_path, linear_score, bce_model, pairwise_model, seed + index, resamples) for index, (source, (path, gt_path)) in enumerate(sources.items())}
    exact_label_summaries = {source: json.loads(paths[f"exact_label_{source}_summary"].read_text(encoding="utf-8")) for source in ("vlsat", "open3dsg", "sgfn")}
    comparison_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for source in ("vlsat", "open3dsg", "sgfn"):
        for method in METHODS:
            for k in KS:
                cell = results[source]["overall"][method][str(k)]
                comparison_rows.append({"source": source, "supervision": "shared constructed compatibility target", "method": method, "k": k, "recall": cell["recall"]["point"], "violation": cell["violation_all"]["point"]})
        for k in KS:
            cell = exact_label_summaries[source]["metrics"]["mlp_reranker"][str(k)]
            comparison_rows.append({"source": source, "supervision": "SGFN-specific exact-label correctness", "method": "source_specific_exact_label_mlp", "k": k, "recall": cell["recall"]["point"], "violation": cell["violation"]["point"]})
        for method in ("shared_mlp_bce_product", "shared_mlp_pairwise_product"):
            for k in KS:
                contrasts = results[source]["overall"]["deltas_vs_all_family_product"][method][str(k)]
                paired_rows.append({
                    "source": source,
                    "method": method,
                    "reference": "all_family_product",
                    "k": k,
                    "delta_recall": contrasts["recall"]["point"],
                    "delta_recall_ci95_low": contrasts["recall"]["paired_ci95"][0],
                    "delta_recall_ci95_high": contrasts["recall"]["paired_ci95"][1],
                    "delta_violation": contrasts["violation_all"]["point"],
                    "delta_violation_ci95_low": contrasts["violation_all"]["paired_ci95"][0],
                    "delta_violation_ci95_high": contrasts["violation_all"]["paired_ci95"][1],
                })
    validations = {
        "split_counts_1061_117_157": (len(train_scans), len(dev_scans), len(final_scans)) == (1061, 117, 157),
        "zero_final_rows_in_training_table": not ({row["scan_id"] for row in table_rows} & final_scans),
        "train_rows_60208": sum(row["_role"] == "train" for row in prepared) == 60208,
        "dev_rows_6246": sum(row["_role"] == "dev" for row in prepared) == 6246,
        "parameter_count_69": bce_model["parameter_count"] == pairwise_model["parameter_count"] == PARAMETER_COUNT == 69,
        "source_score_excluded": not bce_model["feature_spec"]["source_score_input"] and not pairwise_model["feature_spec"]["source_score_input"],
        "all_sources_context_count_matches": results["development"]["counts"]["contexts"] == 354 and all(results[source]["counts"]["contexts"] == 548 for source in ("vlsat", "open3dsg", "sgfn")),
        "all_final_gt_denominator_3972": all(results[source]["counts"]["gt_denominator"] == 3972 for source in ("vlsat", "open3dsg", "sgfn")),
        "all_parameters_finite": all(math.isfinite(float(value)) for model in (bce_model, pairwise_model) for array in model["parameters"].values() for value in np.asarray(array).ravel()),
    }
    status = "completed" if all(validations.values()) else "failed_validation"
    summary = {
        "schema_version": "relcompat3d_relcompat3d_fit_mlp_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "training_warnings": warnings,
        "parameter_count": PARAMETER_COUNT,
        "development_calibration": {
            "shared_mlp_bce": calibration_diagnostic(prepared, bce_model, False),
            "shared_mlp_pairwise": calibration_diagnostic(prepared, pairwise_model, True),
        },
        "sources": results,
        "exact_label_comparator_provenance": {source: relpath(root, paths[f"exact_label_{source}_summary"]) for source in exact_label_summaries},
        "validations": validations,
        "evaluation_scope": protocol["evaluation_scope"],
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "models.json", {"shared_mlp_bce": bce_model, "shared_mlp_pairwise": pairwise_model})
    write_json(out / "summary.json", summary)
    with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0])); writer.writeheader(); writer.writerows(comparison_rows)
    with (out / "paired_contrasts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0])); writer.writeheader(); writer.writerows(paired_rows)
    lines = ["# MLP Compatibility Comparison", "", f"Status: `{status}`", "", "| Source | Method | Supervision | R@100 | V@100 |", "| --- | --- | --- | ---: | ---: |"]
    for source in ("vlsat", "open3dsg", "sgfn"):
        for method in ("all_family_product", "shared_mlp_bce_product", "shared_mlp_pairwise_product"):
            cell = results[source]["overall"][method]["100"]
            lines.append(f"| {source} | {method} | shared compatibility target | {cell['recall']['point']:.4f} | {cell['violation_all']['point']:.4f} |")
        cell = exact_label_summaries[source]["metrics"]["mlp_reranker"]["100"]
        lines.append(f"| {source} | source-specific MLP | SGFN exact-label correctness | {cell['recall']['point']:.4f} | {cell['violation']['point']:.4f} |")
    lines.extend(["", "## Paired K=100 contrast against the linear product", "", "| Source | Matched MLP method | delta Recall (95% CI) | delta V (95% CI) |", "| --- | --- | ---: | ---: |"])
    for source in ("vlsat", "open3dsg", "sgfn"):
        for method in ("shared_mlp_bce_product", "shared_mlp_pairwise_product"):
            contrast = results[source]["overall"]["deltas_vs_all_family_product"][method]["100"]
            dr, dv = contrast["recall"], contrast["violation_all"]
            lines.append(
                f"| {source} | {method} | {dr['point']:+.4f} [{dr['paired_ci95'][0]:+.4f}, {dr['paired_ci95'][1]:+.4f}] | "
                f"{dv['point']:+.4f} [{dv['paired_ci95'][0]:+.4f}, {dv['paired_ci95'][1]:+.4f}] |"
            )
    lines.extend(["", "The shared MLP compatibility models use no source score or predictor identity and are applied unchanged to all three predictors. The exact-label rescorer is reported separately because it uses stronger, SGFN-specific correctness supervision.", ""])
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    outputs = [out / name for name in ("models.json", "summary.json", "summary.md", "comparison.csv", "paired_contrasts.csv")]
    write_json(out / "manifest.json", {"schema_version": "relcompat3d_relcompat3d_fit_mlp_manifest_v1", "status": status, "protocol": {"path": relpath(root, protocol_path), "sha256": sha256(protocol_path)}, "inputs": {name: {"path": relpath(root, path), "sha256": sha256(path)} for name, path in paths.items()}, "outputs": {path.name: {"path": relpath(root, path), "sha256": sha256(path)} for path in outputs}, "validations": validations, "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm relcompat3d_fit_mlp"})
    print(json.dumps({"status": status, "validations": validations}))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
