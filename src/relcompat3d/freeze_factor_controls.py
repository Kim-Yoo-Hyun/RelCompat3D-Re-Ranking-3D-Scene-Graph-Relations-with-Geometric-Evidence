#!/usr/bin/env python3
"""Freeze RelCompat3D factor-isolation diagnostics and audit current-score equivalence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from collections import Counter
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "relcompat3d_factor_isolation_protocol_v1"
FAMILIES = ("support_contact", "proximity", "relative_vertical")
PREDICATES = (
    "close by",
    "higher than",
    "lower than",
    "lying on",
    "standing on",
    "supported by",
)
KS = (5, 10, 20, 50, 100)

BASE_RAW_G_FEATURES = (
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
DERIVED_RAW_G_FEATURES = (
    "abs_center_delta_z",
    "abs_normalized_center_delta_z",
    "abs_vertical_gap_subject_on_object",
)
RAW_G_FEATURES = BASE_RAW_G_FEATURES + DERIVED_RAW_G_FEATURES
INTERACTION_FEATURES = (
    "predicate_aligned_center_delta_z",
    "predicate_aligned_normalized_center_delta_z",
)
FORBIDDEN_FIELDS = (
    "scores.ranking_score",
    "scores.predicate_score",
    "scores.semantic_ranking_score",
    "scores.triplet_score",
    "scores.subject_score",
    "scores.object_score",
    "ranks.predicate_rank",
    "ranks.triplet_rank",
    "baseline_name",
    "baseline_run_id",
    "adapter.source_name",
    "source",
)

FAMILY_MODEL_PATH = Path(
    "archive/hypothesis_records/hypothesis/CAND-001/"
    "RelCompat3D_geometry-grounded-verification/artifacts/calibration/"
    "p_geom_valid_family/model.json"
)
POOLED_MODEL_PATH = Path(
    "archive/hypothesis_records/hypothesis/CAND-001/"
    "RelCompat3D_geometry-grounded-verification/artifacts/calibration/"
    "p_geom_valid_smoke/model.json"
)
CALIBRATION_MANIFEST_PATH = Path(
    "archive/hypothesis_records/hypothesis/CAND-001/"
    "RelCompat3D_geometry-grounded-verification/artifacts/calibration/"
    "train_dev_calib/manifest.json"
)
TRAIN_SCANS_PATH = Path(
    "archive/hypothesis_records/hypothesis/CAND-001/"
    "RelCompat3D_geometry-grounded-verification/artifacts/subset/"
    "relcompat3d_calib_pilot/train_scans.txt"
)
DEV_SCANS_PATH = Path(
    "archive/hypothesis_records/hypothesis/CAND-001/"
    "RelCompat3D_geometry-grounded-verification/artifacts/subset/"
    "relcompat3d_calib_pilot/dev_scans.txt"
)
GROUND_TRUTH_PATH = Path(
    "experiments/RelCompat3D_geom_reliability/sources/vlsat/full_validation/"
    "adapter/ground_truth.jsonl"
)

SOURCE_SPECS = {
    "vlsat_closed_set_full_validation": {
        "role": "retrospective_controlled_anchor_post_hoc_mechanism_diagnostic",
        "predictions": Path(
            "experiments/RelCompat3D_geom_reliability/sources/vlsat/full_validation/"
            "adapter/predictions.jsonl"
        ),
        "verification": Path(
            "experiments/RelCompat3D_geom_reliability/sources/vlsat/full_validation/"
            "geometry/verification.jsonl"
        ),
        "expected_rows": 957008,
        "expected_in_scope_rows": 220848,
        "expected_subgraphs": 548,
    },
    "open3dsg_recovery_full_validation": {
        "role": "retrospective_open_vocabulary_case_post_hoc_mechanism_diagnostic",
        "predictions": Path(
            "experiments/RelCompat3D_geom_reliability/sources/open3dsg/full_validation/"
            "recovery_relaxed_views_min2/adapter/predictions.jsonl"
        ),
        "verification": Path(
            "experiments/RelCompat3D_geom_reliability/sources/open3dsg/full_validation/"
            "recovery_relaxed_views_min2/geometry/verification.jsonl"
        ),
        "expected_rows": 695916,
        "expected_in_scope_rows": 160596,
        "expected_subgraphs": 548,
    },
    "sgfn_official_full_l160": {
        "role": "originally_confirmatory_source_but_new_controls_strictly_post_hoc",
        "predictions": Path(
            "experiments/RelCompat3D_geom_reliability/sources/sgfn/adapter/predictions.jsonl"
        ),
        "verification": Path(
            "experiments/RelCompat3D_geom_reliability/sources/sgfn/geometry/verification.jsonl"
        ),
        "expected_rows": 957008,
        "expected_in_scope_rows": 220848,
        "expected_subgraphs": 548,
    },
}

SMALL_PROVENANCE_PATHS = {
    "family_model": FAMILY_MODEL_PATH,
    "pooled_model": POOLED_MODEL_PATH,
    "calibration_manifest": CALIBRATION_MANIFEST_PATH,
    "train_scans": TRAIN_SCANS_PATH,
    "dev_scans": DEV_SCANS_PATH,
    "ground_truth": GROUND_TRUTH_PATH,
    "vlsat_metrics": Path(
        "experiments/RelCompat3D_geom_reliability/sources/vlsat/full_validation/"
        "metrics_k_sweep/metrics.json"
    ),
    "open3dsg_metrics": Path(
        "experiments/RelCompat3D_geom_reliability/sources/open3dsg/full_validation/"
        "recovery_relaxed_views_min2/metrics_k_sweep/metrics.json"
    ),
    "sgfn_metrics": Path(
        "experiments/RelCompat3D_geom_reliability/sources/sgfn/"
        "confirmatory_metrics/summary.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def relpath(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_scan_set(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def load_eval_module(root: Path) -> Any:
    path = root / "src/relcompat3d/evaluate_metrics.py"
    spec = importlib.util.spec_from_file_location("relcompat3d_factor_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot_import:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def feature_factor(name: str) -> tuple[str, str]:
    if name == "bias":
        return "T", "constant_intercept_present_in_all_learned_conditions"
    if name.startswith("family:"):
        return "T", "predicate_family_one_hot"
    if name.startswith("predicate:"):
        return "T", "predicate_label_one_hot"
    if name.startswith("num:"):
        numeric = name.split(":", 1)[1]
        if numeric in RAW_G_FEATURES:
            return "raw_G", "predicate_independent_same_pair_geometry"
        if numeric in INTERACTION_FEATURES:
            return "T_x_G", "predicate_aligned_geometry_interaction"
    raise ValueError(f"unclassified_model_feature:{name}")


def feature_ledger(
    pooled_model: dict[str, Any], family_model: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, bool]]:
    specs: dict[str, dict[str, Any]] = {"pooled": pooled_model}
    for family, spec in family_model["family_models"].items():
        specs[f"family:{family}"] = spec

    union = sorted({name for spec in specs.values() for name in spec["feature_names"]})
    entries: list[dict[str, Any]] = []
    errors: list[str] = []
    factor_counts: Counter[str] = Counter()
    for name in union:
        try:
            factor, subrole = feature_factor(name)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        factor_counts[factor] += 1
        entries.append(
            {
                "feature": name,
                "factor": factor,
                "subrole": subrole,
                "present_in_models": [key for key, spec in specs.items() if name in spec["feature_names"]],
            }
        )

    forbidden_tokens = (
        "semantic",
        "ranking_score",
        "predicate_score",
        "triplet_score",
        "subject_score",
        "object_score",
        "baseline_name",
        "baseline_run_id",
        "source_name",
    )
    forbidden_hits = [
        name for name in union if any(token in name.lower() for token in forbidden_tokens)
    ]
    expected_numeric = set(RAW_G_FEATURES + INTERACTION_FEATURES)
    numeric_sets_match = all(
        set(spec["numeric_features"]) == expected_numeric for spec in specs.values()
    )
    validations = {
        "all_current_model_features_classified": not errors and len(entries) == len(union),
        "all_model_numeric_sets_match_frozen_raw_plus_interaction_set": numeric_sets_match,
        "forbidden_Z_or_source_features_absent_from_models": not forbidden_hits,
        "family_models_exactly_three": set(family_model["family_models"]) == set(FAMILIES),
        "pooled_predicate_set_exact": set(pooled_model["predicates"]) == set(PREDICATES),
        "pooled_family_set_exact": set(pooled_model["families"]) == set(FAMILIES),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "allowed_factor_classes": ["T", "raw_G", "T_x_G", "forbidden_Z_or_source"],
        "classification_rule": {
            "T": "bias plus predicate-family and exact-predicate one-hot features",
            "raw_G": "predicate-independent geometry of the same ordered object pair",
            "T_x_G": "explicit predicate-aligned transforms derived from T and raw G",
            "forbidden_Z_or_source": "source confidence, ranks, source identity, baseline identity, or adapters",
        },
        "entries": entries,
        "factor_counts_over_union": dict(sorted(factor_counts.items())),
        "forbidden_fields": [
            {
                "field": field,
                "factor": "forbidden_Z_or_source",
                "allowed_use": "final_F_Z_C_fusion_or_provenance_only",
                "calibrator_input": False,
            }
            for field in FORBIDDEN_FIELDS
        ],
        "raw_G_ordered_endpoint_contract": {
            "invariant_under_endpoint_swap": [
                "distance_3d",
                "distance_xy",
                "normalized_distance_3d",
                "normalized_distance_xy",
                "projected_iou_xy",
            ],
            "sign_flip_under_endpoint_swap": [
                "center_delta_z",
                "normalized_center_delta_z",
            ],
            "exchange_under_endpoint_swap": [
                ["projected_subject_overlap_ratio", "projected_object_overlap_ratio"],
                ["subject_bottom_z", "object_bottom_z"],
                ["subject_top_z", "object_top_z"],
            ],
            "recompute_under_endpoint_swap": {
                "vertical_gap_subject_on_object": "object_bottom_z - subject_top_z",
                "abs_center_delta_z": "abs(swapped center_delta_z)",
                "abs_normalized_center_delta_z": "abs(swapped normalized_center_delta_z)",
                "abs_vertical_gap_subject_on_object": "abs(swapped vertical gap)",
            },
        },
        "errors": errors,
        "forbidden_hits": forbidden_hits,
        "validations": validations,
    }
    return payload, validations


def independent_raw_features(
    prediction: dict[str, Any], verification: dict[str, Any]
) -> dict[str, float]:
    source = (verification.get("geometry") or {}).get("features") or {}
    values: dict[str, float] = {}
    for name in BASE_RAW_G_FEATURES:
        value = finite_float(source.get(name))
        if value is not None:
            values[name] = value
    if "center_delta_z" in values:
        values["abs_center_delta_z"] = abs(values["center_delta_z"])
    if "normalized_center_delta_z" in values:
        values["abs_normalized_center_delta_z"] = abs(
            values["normalized_center_delta_z"]
        )
    if "vertical_gap_subject_on_object" in values:
        values["abs_vertical_gap_subject_on_object"] = abs(
            values["vertical_gap_subject_on_object"]
        )
    predicate = prediction["predicate"]["predicate_label"]
    direction = 1.0 if predicate == "higher than" else -1.0 if predicate == "lower than" else 0.0
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


def independent_family_score(
    prediction: dict[str, Any],
    verification: dict[str, Any],
    family_model: dict[str, Any],
) -> float:
    family = prediction["predicate"]["predicate_family"]
    spec = family_model["family_models"][family]
    raw = independent_raw_features(prediction, verification)
    vector = [1.0]
    for name in spec["numeric_features"]:
        stats = spec["numeric_stats"][name]
        value = raw.get(name, stats["mean"])
        vector.append((value - stats["mean"]) / (stats["std"] or 1.0))
    vector.extend(1.0 if family == name else 0.0 for name in spec["families"])
    predicate = prediction["predicate"]["predicate_label"]
    vector.extend(1.0 if predicate == name else 0.0 for name in spec["predicates"])
    return sigmoid(sum(weight * value for weight, value in zip(spec["weights"], vector)))


def semantic_score(prediction: dict[str, Any]) -> float | None:
    scores = prediction.get("scores") or {}
    value = finite_float(scores.get("ranking_score"))
    return value if value is not None else finite_float(scores.get("predicate_score"))


def update_score_hash(
    digest: Any, prediction_id: str, compatibility: float, product: float
) -> None:
    digest.update(
        f"{prediction_id}\t{compatibility.hex()}\t{product.hex()}\n".encode("utf-8")
    )


def equivalence_for_source(
    root: Path,
    name: str,
    spec: dict[str, Any],
    family_model: dict[str, Any],
    eval_module: Any,
) -> tuple[dict[str, Any], dict[str, bool]]:
    prediction_path = resolve(root, spec["predictions"])
    verification_path = resolve(root, spec["verification"])
    prediction_digest = hashlib.sha256()
    verification_digest = hashlib.sha256()
    canonical_digest = hashlib.sha256()
    independent_digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    exact_c_matches = 0
    exact_product_matches = 0
    max_c_error = 0.0
    max_product_error = 0.0
    mismatch_examples: list[dict[str, Any]] = []
    input_subgraphs: set[str] = set()
    in_scope_subgraphs: set[str] = set()

    with prediction_path.open("r", encoding="utf-8") as pred_handle, verification_path.open(
        "r", encoding="utf-8"
    ) as ver_handle:
        for line_no, pair in enumerate(zip_longest(pred_handle, ver_handle), 1):
            pred_line, ver_line = pair
            if pred_line is None or ver_line is None:
                raise ValueError(f"prediction_verification_length_mismatch:{name}:{line_no}")
            prediction_digest.update(pred_line.encode("utf-8"))
            verification_digest.update(ver_line.encode("utf-8"))
            prediction = json.loads(pred_line)
            verification = json.loads(ver_line)
            counts["input_rows"] += 1
            input_subgraphs.add(prediction["subgraph_id"])
            if prediction["prediction_id"] != verification["prediction_id"]:
                raise ValueError(f"prediction_verification_id_mismatch:{name}:{line_no}")
            family = prediction["predicate"]["predicate_family"]
            if family not in FAMILIES:
                continue
            counts["in_scope_rows"] += 1
            in_scope_subgraphs.add(prediction["subgraph_id"])
            counts[f"family:{family}"] += 1
            compact = eval_module.compact_verification(verification)
            canonical_c = eval_module.family_specific_p_geom_valid(
                prediction, compact, family_model
            )
            independent_c = independent_family_score(prediction, verification, family_model)
            semantic = semantic_score(prediction)
            if canonical_c is None or semantic is None:
                counts["missing_required_score"] += 1
                continue
            canonical_product = semantic * canonical_c
            independent_product = semantic * independent_c
            c_error = abs(canonical_c - independent_c)
            product_error = abs(canonical_product - independent_product)
            max_c_error = max(max_c_error, c_error)
            max_product_error = max(max_product_error, product_error)
            if canonical_c == independent_c:
                exact_c_matches += 1
            elif len(mismatch_examples) < 10:
                mismatch_examples.append(
                    {
                        "prediction_id": prediction["prediction_id"],
                        "canonical_c": canonical_c,
                        "independent_c": independent_c,
                    }
                )
            if canonical_product == independent_product:
                exact_product_matches += 1
            update_score_hash(
                canonical_digest,
                prediction["prediction_id"],
                canonical_c,
                canonical_product,
            )
            update_score_hash(
                independent_digest,
                prediction["prediction_id"],
                independent_c,
                independent_product,
            )

    counts["input_subgraphs"] = len(input_subgraphs)
    counts["in_scope_subgraphs"] = len(in_scope_subgraphs)
    in_scope = counts["in_scope_rows"]
    validations = {
        "input_row_count_exact": counts["input_rows"] == spec["expected_rows"],
        "in_scope_row_count_exact": in_scope == spec["expected_in_scope_rows"],
        "evaluation_subgraph_count_exact": len(input_subgraphs)
        == spec["expected_subgraphs"],
        "in_scope_rows_cover_all_evaluation_subgraphs": len(in_scope_subgraphs)
        == spec["expected_subgraphs"],
        "all_in_scope_rows_scored": counts["missing_required_score"] == 0,
        "compatibility_bit_exact_all_rows": exact_c_matches == in_scope,
        "product_bit_exact_all_rows": exact_product_matches == in_scope,
        "score_stream_sha256_equal": canonical_digest.hexdigest()
        == independent_digest.hexdigest(),
    }
    result = {
        "source": name,
        "role": spec["role"],
        "input_paths": {
            "predictions": relpath(root, prediction_path),
            "verification": relpath(root, verification_path),
        },
        "input_sha256": {
            "predictions": prediction_digest.hexdigest(),
            "verification": verification_digest.hexdigest(),
        },
        "counts": dict(sorted(counts.items())),
        "expected": {
            "input_rows": spec["expected_rows"],
            "in_scope_rows": spec["expected_in_scope_rows"],
            "evaluation_subgraphs": spec["expected_subgraphs"],
        },
        "equivalence": {
            "compatibility_exact_matches": exact_c_matches,
            "product_exact_matches": exact_product_matches,
            "max_abs_compatibility_error": max_c_error,
            "max_abs_product_error": max_product_error,
            "canonical_score_stream_sha256": canonical_digest.hexdigest(),
            "independent_score_stream_sha256": independent_digest.hexdigest(),
            "mismatch_examples": mismatch_examples,
            "rank_average_invariance": {
                "status": "proven_by_identical_semantic_and_compatibility_operands",
                "semantic_operand": "unchanged_source_score_Z",
                "compatibility_operand_bit_exact": exact_c_matches == in_scope,
                "tie_breaker": "unchanged_prediction_key",
            },
        },
        "validations": validations,
    }
    return result, validations


def ground_truth_contract(root: Path) -> tuple[dict[str, Any], dict[str, bool]]:
    path = resolve(root, GROUND_TRUTH_PATH)
    keys: set[tuple[Any, ...]] = set()
    family_counts: Counter[str] = Counter()
    subgraphs: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            family = row["predicate_family"]
            if family not in FAMILIES:
                continue
            key = (
                row["scan_id"],
                int(row["subset_split_id"]),
                int(row["subject_id"]),
                int(row["object_id"]),
                row["predicate_label"],
            )
            if key not in keys:
                family_counts[family] += 1
            keys.add(key)
            subgraphs.add(row["subgraph_id"])
    expected = {
        "support_contact": 1816,
        "proximity": 1766,
        "relative_vertical": 390,
    }
    validations = {
        "total_denominator_3972": len(keys) == 3972,
        "family_denominators_exact": dict(family_counts) == expected,
        "in_scope_GT_bearing_subgraphs_538": len(subgraphs) == 538,
    }
    return {
        "path": relpath(root, path),
        "sha256": sha256_file(path),
        "exact_label_denominator": len(keys),
        "by_family": dict(sorted(family_counts.items())),
        "in_scope_GT_bearing_subgraphs": len(subgraphs),
        "self_relation_rows_remain_in_denominator": 11,
        "validations": validations,
    }, validations


def calibration_contract(
    root: Path, family_model: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, bool]]:
    manifest = read_json(resolve(root, CALIBRATION_MANIFEST_PATH))
    train_scans = read_scan_set(resolve(root, TRAIN_SCANS_PATH))
    dev_scans = read_scan_set(resolve(root, DEV_SCANS_PATH))
    by_family: dict[str, Any] = {}
    train_rows = 0
    dev_rows = 0
    train_labels: Counter[str] = Counter()
    dev_labels: Counter[str] = Counter()
    for family, spec in family_model["family_models"].items():
        counts = spec["counts"]
        by_family[family] = counts
        train_rows += counts["train_rows"]
        dev_rows += counts["dev_rows"]
        train_labels.update(counts["train_label_counts"])
        dev_labels.update(counts["dev_label_counts"])
    validations = {
        "train_dev_scan_disjoint": not (train_scans & dev_scans),
        "train_scan_count_24": len(train_scans) == 24,
        "dev_scan_count_8": len(dev_scans) == 8,
        "train_rows_4616": train_rows == 4616,
        "dev_rows_1193": dev_rows == 1193,
        "total_positive_rows_2565": manifest["counts"]["positive_rows"] == 2565,
        "total_negative_rows_3244": manifest["counts"]["negative_rows"] == 3244,
        "family_model_source_split_train_dev": family_model.get("source_split")
        == "train_dev_calib",
    }
    return {
        "target": "y_cal_constructed_not_human_physical_validity",
        "train_scans": len(train_scans),
        "dev_scans": len(dev_scans),
        "scan_overlap": sorted(train_scans & dev_scans),
        "train_rows": train_rows,
        "dev_rows": dev_rows,
        "train_labels": dict(sorted(train_labels.items())),
        "dev_labels": dict(sorted(dev_labels.items())),
        "total_counts": manifest["counts"],
        "by_family": by_family,
        "negative_policy_version": manifest.get("negative_policy_version"),
        "validations": validations,
    }, validations


def condition_contract() -> dict[str, Any]:
    common = {
        "architecture": "single_pooled_logistic_regression",
        "fit_rows": "calibration_train_only_4616_rows_24_scans",
        "selection_rows": "none_no_condition_selection_from_source_metrics",
        "diagnostic_eval_rows": "calibration_dev_1193_rows_8_disjoint_scans",
        "standardization": "train_only_mean_std_missing_values_imputed_to_train_mean",
        "optimizer": {
            "epochs": 800,
            "learning_rate": 0.2,
            "l2": 0.0001,
            "initial_weights": 0.0,
            "batching": "deterministic_full_batch",
        },
    }
    return {
        "M_existing": {
            "role": "read_only_continuity_reference_not_factor_fairness_baseline",
            "architecture": "frozen_per_family_logistic_models",
            "features": ["T", "raw_G", "T_x_G"],
            "model_path": str(FAMILY_MODEL_PATH),
            "refit": False,
            "paper_score": "family_conditional_risk_equals_Z_times_C_existing",
        },
        "M_T": {
            **common,
            "features": ["bias", "family_one_hot", "predicate_one_hot"],
            "forbidden": ["raw_G", "T_x_G", "Z", "source_identity"],
            "parameter_count": 10,
        },
        "M_G": {
            **common,
            "features": ["bias", *RAW_G_FEATURES],
            "forbidden": [
                "family_routing",
                "family_one_hot",
                "predicate_one_hot",
                "T_x_G",
                "Z",
                "source_identity",
            ],
            "parameter_count": 18,
            "true_G_only_guard": "one pooled model across all families; no family-specific model selection",
        },
        "M_add": {
            **common,
            "features": [
                "bias",
                "family_one_hot",
                "predicate_one_hot",
                *RAW_G_FEATURES,
            ],
            "forbidden": ["T_x_G", "Z", "source_identity"],
            "parameter_count": 27,
            "additivity_guard": "one shared coefficient vector; no family-specific G surfaces",
        },
        "M_int": {
            **common,
            "features": [
                "bias",
                "family_one_hot",
                "predicate_one_hot",
                *RAW_G_FEATURES,
                *INTERACTION_FEATURES,
            ],
            "forbidden": ["Z", "source_identity"],
            "parameter_count": 29,
            "interaction_scope": [
                "higher_or_lower_sign_times_center_delta_z",
                "higher_or_lower_sign_times_normalized_center_delta_z",
            ],
            "support_specific_interactions_added": False,
        },
        "fair_comparison_rule": (
            "M_T_M_G_M_add_M_int share one pooled architecture, optimizer, split, "
            "and train-only normalization; M_existing is reported separately"
        ),
    }


def control_contract() -> dict[str, Any]:
    return {
        "relative_vertical_wrong_T": {
            "status": "frozen",
            "eligible_family": "relative_vertical",
            "endpoint_operation": "none_keep_ordered_G_fixed",
            "predicate_map": {"higher than": "lower than", "lower than": "higher than"},
            "recompute": ["T", "predicate_aligned_T_x_G"],
            "primary_pairing": "same_row_correct_T_vs_wrong_T",
            "diagnostics": [
                "mean_and_median_C_correct_minus_C_wrong",
                "paired_win_rate_C_correct_above_C_wrong",
                "paired_subgraph_bootstrap_CI",
            ],
            "not_defined_for": ["proximity", "support_contact"],
        },
        "close_by_swap_invariance": {
            "status": "frozen",
            "eligible_predicate": "close by",
            "endpoint_operation": "exact_subject_object_swap",
            "predicate_operation": "unchanged",
            "raw_G_transform": "feature_ledger.raw_G_ordered_endpoint_contract",
            "diagnostics": [
                "absolute_C_original_minus_C_swapped",
                "mean_and_p95_absolute_difference",
                "paired_subgraph_bootstrap_CI",
            ],
            "interpretation": "diagnostic_invariance_test_not_assumed_pass",
        },
        "relative_vertical_inverse_equivariance": {
            "status": "frozen",
            "eligible_family": "relative_vertical",
            "endpoint_operation": "exact_subject_object_swap",
            "predicate_map": {"higher than": "lower than", "lower than": "higher than"},
            "raw_G_transform": "feature_ledger.raw_G_ordered_endpoint_contract",
            "recompute": ["T", "raw_G", "T_x_G"],
            "diagnostics": [
                "absolute_C_original_minus_C_inverse_swapped",
                "mean_and_p95_absolute_difference",
                "paired_subgraph_bootstrap_CI",
            ],
            "interpretation": "diagnostic_equivariance_test_not_assumed_pass",
        },
        "support_contact_endpoint_swap": {
            "status": "prohibited_in_v1",
            "rule": "no_blanket_endpoint_swap",
            "reason": (
                "standing_on and lying_on encode subject roles; supported_by lacks "
                "its inverse supports in the frozen predicate vocabulary"
            ),
            "reopen_requirement": (
                "pre-frozen exact predicate inverse map plus exact geometry and "
                "support-subtype transform; otherwise no score or claim"
            ),
        },
    }


def evaluation_contract() -> dict[str, Any]:
    return {
        "classification": "post_hoc_mechanism_diagnostic_not_original_sgfn_confirmatory_gate",
        "sources": {
            name: {
                "role": spec["role"],
                "predictions": str(spec["predictions"]),
                "verification": str(spec["verification"]),
            }
            for name, spec in SOURCE_SPECS.items()
        },
        "families": list(FAMILIES),
        "ks": list(KS),
        "primary_k": 100,
        "denominator": {
            "matching": "exact_predicate_label",
            "global": 3972,
            "evaluation_contexts": 548,
            "in_scope_GT_bearing_contexts": 538,
            "by_family": {
                "support_contact": 1816,
                "proximity": 1766,
                "relative_vertical": 390,
            },
            "self_supported_by_rows_retained": 11,
            "missing_prediction_policy": "no_synthetic_edges_zero_recall_credit",
        },
        "metrics": [
            "R_at_K",
            "verifier_V_at_K",
            "calibration_dev_AUROC_AUPRC_Brier_NLL",
            "control_pairwise_compatibility_differences",
        ],
        "scopes": ["global_top_K", "within_family_top_K", "global_top_K_family_slice"],
        "paired_uncertainty": {
            "unit": "subgraph_cluster_with_same_indices_for_all_conditions_in_contrast",
            "resamples": 1000,
            "seed": 20260710,
            "marginal_interval": "two_sided_percentile_95",
            "family_wise_interval": (
                "simultaneous_95_percent_max_absolute_centered_bootstrap_band_"
                "over_three_families_per_source_x_K_x_contrast_x_metric"
            ),
            "report": [
                "point_estimate",
                "marginal_CI95",
                "simultaneous_family_wise_CI95",
                "numerator",
                "denominator",
            ],
        },
        "fixed_contrasts": [
            "M_int_minus_M_T",
            "M_int_minus_M_G",
            "M_int_minus_M_add",
            "M_existing_minus_semantic_only_for_continuity_only",
        ],
        "fusion": {
            "product": "Z_times_C_condition",
            "rank_average": "equal_mean_within_subgraph_percentile_rank_of_Z_and_C_condition",
            "tie_breaker": "prediction_key_scan_subgraph_subject_object_predicate",
            "no_weight_tuning": True,
        },
        "docker": {
            "freeze_command": (
                "env UID=$(id -u) GID=$(id -g) docker compose -f "
                "configs/relcompat3d/compose.yaml run --rm factor_isolation_protocol_freeze"
            ),
            "future_metric_command_reserved": (
                "env UID=$(id -u) GID=$(id -g) docker compose -f "
                "configs/relcompat3d/compose.yaml run --rm factor_isolation_metrics"
            ),
            "future_metric_command_status": "reserved_not_executable_until_next_stage_implementation",
        },
    }


def frozen_contract_validations(
    ledger: dict[str, Any],
    conditions: dict[str, Any],
    controls: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, bool]:
    swap = ledger["raw_G_ordered_endpoint_contract"]
    swap_sequence = list(swap["invariant_under_endpoint_swap"])
    swap_sequence.extend(swap["sign_flip_under_endpoint_swap"])
    for pair in swap["exchange_under_endpoint_swap"]:
        swap_sequence.extend(pair)
    swap_sequence.extend(swap["recompute_under_endpoint_swap"])
    swap_covered = set(swap_sequence)
    return {
        "factor_union_counts_T10_G17_interaction2": ledger["factor_counts_over_union"]
        == {"T": 10, "T_x_G": 2, "raw_G": 17},
        "endpoint_swap_contract_covers_each_raw_G_feature_once": swap_covered
        == set(RAW_G_FEATURES)
        and len(swap_sequence) == len(swap_covered) == len(RAW_G_FEATURES),
        "condition_names_exact": set(conditions)
        == {"M_existing", "M_T", "M_G", "M_add", "M_int", "fair_comparison_rule"},
        "M_T_parameter_count_10": conditions["M_T"]["parameter_count"] == 10,
        "M_G_true_pooled_no_family_route": conditions["M_G"]["parameter_count"] == 18
        and "family_routing" in conditions["M_G"]["forbidden"]
        and "predicate_one_hot" in conditions["M_G"]["forbidden"],
        "M_add_parameter_count_27_no_interaction": conditions["M_add"]["parameter_count"]
        == 27
        and "T_x_G" in conditions["M_add"]["forbidden"],
        "M_int_parameter_count_29_interactions_exact": conditions["M_int"][
            "parameter_count"
        ]
        == 29
        and conditions["M_int"]["interaction_scope"]
        == [
            "higher_or_lower_sign_times_center_delta_z",
            "higher_or_lower_sign_times_normalized_center_delta_z",
        ],
        "all_new_conditions_forbid_Z_and_source": all(
            "Z" in conditions[name]["forbidden"]
            and "source_identity" in conditions[name]["forbidden"]
            for name in ("M_T", "M_G", "M_add", "M_int")
        ),
        "wrong_T_vertical_inverse_map_exact": controls["relative_vertical_wrong_T"][
            "predicate_map"
        ]
        == {"higher than": "lower than", "lower than": "higher than"},
        "close_by_swap_invariance_exact_endpoint_operation": controls[
            "close_by_swap_invariance"
        ]["endpoint_operation"]
        == "exact_subject_object_swap"
        and controls["close_by_swap_invariance"]["predicate_operation"] == "unchanged",
        "vertical_inverse_equivariance_exact": controls[
            "relative_vertical_inverse_equivariance"
        ]["endpoint_operation"]
        == "exact_subject_object_swap"
        and controls["relative_vertical_inverse_equivariance"]["predicate_map"]
        == {"higher than": "lower than", "lower than": "higher than"},
        "support_contact_blanket_swap_prohibited": controls[
            "support_contact_endpoint_swap"
        ]["status"]
        == "prohibited_in_v1",
        "sources_exactly_three": set(evaluation["sources"]) == set(SOURCE_SPECS),
        "K_grid_exact": evaluation["ks"] == list(KS) and evaluation["primary_k"] == 100,
        "denominator_exact": evaluation["denominator"]["global"] == 3972
        and evaluation["denominator"]["evaluation_contexts"] == 548
        and evaluation["denominator"]["in_scope_GT_bearing_contexts"] == 538,
        "paired_family_wise_CI_frozen": evaluation["paired_uncertainty"]["resamples"]
        == 1000
        and evaluation["paired_uncertainty"]["seed"] == 20260710
        and "simultaneous_95_percent" in evaluation["paired_uncertainty"]["family_wise_interval"],
        "post_hoc_classification_exact": evaluation["classification"]
        == "post_hoc_mechanism_diagnostic_not_original_sgfn_confirmatory_gate",
        "Docker_freeze_command_fixed": evaluation["docker"]["freeze_command"].endswith(
            "factor_isolation_protocol_freeze"
        ),
    }


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    out = resolve(root, args.out)
    required = {
        name: resolve(root, path) for name, path in SMALL_PROVENANCE_PATHS.items()
    }
    for source_name, spec in SOURCE_SPECS.items():
        required[f"{source_name}_predictions"] = resolve(root, spec["predictions"])
        required[f"{source_name}_verification"] = resolve(root, spec["verification"])
    missing = [relpath(root, path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing_inputs:{missing}")

    family_model = read_json(resolve(root, FAMILY_MODEL_PATH))
    pooled_model = read_json(resolve(root, POOLED_MODEL_PATH))
    ledger, ledger_checks = feature_ledger(pooled_model, family_model)
    calibration, calibration_checks = calibration_contract(root, family_model)
    gt, gt_checks = ground_truth_contract(root)
    eval_module = load_eval_module(root)
    equivalence_sources: dict[str, Any] = {}
    equivalence_checks: dict[str, bool] = {}
    for source_name, spec in SOURCE_SPECS.items():
        result, checks = equivalence_for_source(
            root, source_name, spec, family_model, eval_module
        )
        equivalence_sources[source_name] = result
        for check_name, value in checks.items():
            equivalence_checks[f"{source_name}:{check_name}"] = value

    conditions = condition_contract()
    controls = control_contract()
    evaluation = evaluation_contract()
    contract_checks = frozen_contract_validations(
        ledger, conditions, controls, evaluation
    )
    all_checks = {
        **{f"feature_ledger:{key}": value for key, value in ledger_checks.items()},
        **{f"calibration:{key}": value for key, value in calibration_checks.items()},
        **{f"denominator:{key}": value for key, value in gt_checks.items()},
        **equivalence_checks,
        **{f"frozen_contract:{key}": value for key, value in contract_checks.items()},
    }
    status = (
        "frozen_ready_for_post_hoc_mechanism_implementation"
        if all(all_checks.values())
        else "blocked_validation_failed"
    )
    created_at = datetime.now(timezone.utc).isoformat()
    equivalence = {
        "schema_version": SCHEMA_VERSION,
        "status": "bit_exact" if all(equivalence_checks.values()) else "failed",
        "definition": (
            "independent factor-ledger scorer versus canonical frozen family scorer; "
            "semantic Z is read only for product continuity"
        ),
        "sources": equivalence_sources,
        "validations": equivalence_checks,
        "no_overwrite_rule": {
            "family_model_refit": False,
            "existing_prediction_or_verification_mutation": False,
            "existing_metric_mutation": False,
            "new_factor_conditions_may_not_replace_existing_score_on_seen_sources": True,
        },
    }

    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "feature_ledger.json", ledger)
    write_json(out / "conditions.json", conditions)
    write_json(out / "controls.json", controls)
    write_json(out / "evaluation.json", evaluation)
    write_json(out / "equivalence_audit.json", equivalence)

    generated = {
        name: {
            "path": relpath(root, out / name),
            "sha256": sha256_file(out / name),
        }
        for name in (
            "feature_ledger.json",
            "conditions.json",
            "controls.json",
            "evaluation.json",
            "equivalence_audit.json",
        )
    }
    provenance = {
        name: {"path": relpath(root, path), "sha256": sha256_file(path)}
        for name, path in required.items()
        if "_predictions" not in name and "_verification" not in name
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": created_at,
        "status": status,
        "classification": "post_hoc_mechanism_diagnostic_not_original_sgfn_confirmatory_gate",
        "claim_lock": {
            "existing_framework_score_changed": False,
            "existing_sgfn_gate_changed": False,
            "H002_metrics_imported": False,
            "factor_necessity_claim_authorized": False,
            "support_contact_endpoint_swap_authorized": False,
        },
        "calibration_contract": calibration,
        "denominator_contract": gt,
        "generated_artifacts": generated,
        "provenance_inputs": provenance,
        "validations": all_checks,
        "validation_errors": [key for key, value in all_checks.items() if not value],
        "next_stage": (
            "implement frozen M_T_M_G_M_add_M_int and metamorphic diagnostics in "
            "the reserved Docker metric service; do not change this protocol after metrics"
        ),
    }
    write_json(out / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": status,
                "out": relpath(root, out),
                "validation_errors": manifest["validation_errors"],
                "equivalence_rows": {
                    name: row["counts"]["in_scope_rows"]
                    for name, row in equivalence_sources.items()
                },
            },
            sort_keys=True,
        )
    )
    return 0 if status.startswith("frozen_ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
