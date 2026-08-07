#!/usr/bin/env python3
"""Fit RelCompat3D-Linear without a constant family-indicator input.

This stage reads only training and development data. It writes the fitted
models and score definition used by the separate validation stage.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import compatibility_features as calibration
import fit_base_models as base_fit
import relation_consistency as algebra


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
        "--base-models",
        type=Path,
        help=(
            "Override the protocol feature-template path. This lets a fresh run use "
            "base models fitted from the same training rows."
        ),
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="Fit and export the Linear estimator without development evaluation.",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_scans(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def strip_family_indicator(model: dict[str, Any]) -> dict[str, Any]:
    """Remove the single constant family feature from a family-specific head."""
    feature_names = list(model["feature_names"])
    family_indices = [
        index for index, name in enumerate(feature_names) if name.startswith("family:")
    ]
    if len(family_indices) != 1:
        raise ValueError(
            f"expected_one_family_indicator:{model.get('family')}:{family_indices}"
        )
    removed = feature_names[family_indices[0]]
    expected = f"family:{model['family']}"
    if removed != expected:
        raise ValueError(f"wrong_family_indicator:{removed}:{expected}")
    keep = [index for index in range(len(feature_names)) if index not in family_indices]
    result = copy.deepcopy(model)
    result["feature_names"] = [feature_names[index] for index in keep]
    result["weights"] = [float(model["weights"][index]) for index in keep]
    result["families"] = []
    result["parameterization"] = {
        "id": "relcompat3d_linear",
        "removed_feature": removed,
        "family_selected_by": "family-specific head",
    }
    return result


def fit_linear_family_models(
    prepared: list[dict[str, Any]], current_models: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    train = [row for row in prepared if row["_role"] == "train"]
    dev = [row for row in prepared if row["_role"] == "dev"]
    family_models: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for family in algebra.FAMILIES:
        family_train = [
            row for row in train if row["predicate"]["predicate_family"] == family
        ]
        family_dev = [
            row for row in dev if row["predicate"]["predicate_family"] == family
        ]
        template = strip_family_indicator(current_models["family_models"][family])
        train_x = [
            algebra.existing_vector(
                template,
                family,
                row["predicate"]["predicate_label"],
                row["_raw_numeric"],
            )
            for row in family_train
        ]
        dev_x = [
            algebra.existing_vector(
                template,
                family,
                row["predicate"]["predicate_label"],
                row["_raw_numeric"],
            )
            for row in family_dev
        ]
        train_y = [int(row["_label"]) for row in family_train]
        dev_y = [int(row["_label"]) for row in family_dev]
        weights, trace = base_fit.fit_numpy(train_x, train_y)
        model = {
            **template,
            "weights": weights,
            "train_prior": sum(train_y) / len(train_y),
            "fit_split": "training_split_1061",
            "training_trace": trace,
            "counts": {
                "train_rows": len(train_y),
                "development_rows": len(dev_y),
                "train_labels": {
                    "0": len(train_y) - sum(train_y),
                    "1": sum(train_y),
                },
                "development_labels": {
                    "0": len(dev_y) - sum(dev_y),
                    "1": sum(dev_y),
                },
            },
        }
        family_models[family] = model
        diagnostics[family] = {
            "train": base_fit.metrics(base_fit.predict_numpy(train_x, weights), train_y),
            "development": base_fit.metrics(
                base_fit.predict_numpy(dev_x, weights), dev_y
            ),
        }
    base_models = copy.deepcopy(current_models)
    base_models["family_models"] = family_models
    base_models["parameterization"] = {
        "id": "relcompat3d_linear",
        "change": "remove the constant family one-hot from each family-specific head",
        "factor_models": "unchanged; family indicators remain meaningful in pooled heads",
    }
    return base_models, diagnostics


def model_features(models: dict[str, Any]) -> list[str]:
    return [
        feature
        for attempt in models["attempts"].values()
        for model in attempt.values()
        for feature in model["feature_names"]
    ]


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    protocol_path = resolve(root, args.protocol)
    out = resolve(root, args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"nonempty_output:{out}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "ready_for_linear_fit":
        raise ValueError("protocol_version_mismatch")
    paths = {name: resolve(root, value) for name, value in protocol["inputs"].items()}
    if args.calibration_table is not None:
        paths["calibration_table"] = resolve(root, args.calibration_table)
    if args.base_models is not None:
        paths["current_base_models"] = resolve(root, args.base_models)
    fit_inputs = {
        name: paths[name]
        for name in (
            "calibration_table",
            "train_scans",
            "development_scans",
            "final_validation_scans",
            "current_base_models",
        )
    }
    required_paths = fit_inputs if args.fit_only else paths
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing_inputs:{missing}")

    train_scans = read_scans(paths["train_scans"])
    dev_scans = read_scans(paths["development_scans"])
    final_scans = read_scans(paths["final_validation_scans"])
    if train_scans & dev_scans or train_scans & final_scans or dev_scans & final_scans:
        raise ValueError("data_split_overlap")

    rows = calibration.load_jsonl(paths["calibration_table"])
    leaked = sorted({str(row["scan_id"]) for row in rows} & final_scans)
    if leaked:
        raise ValueError(f"final_validation_rows_in_calibration:{leaked[:10]}")
    prepared, warnings = calibration.prepare_rows(
        rows, train_scans, dev_scans, set(algebra.FAMILIES)
    )
    current_base = json.loads(paths["current_base_models"].read_text(encoding="utf-8"))
    fitted_base, base_diagnostics = fit_linear_family_models(prepared, current_base)
    attempts, diagnostics = algebra.fit_attempts(
        prepared, fitted_base, protocol["optimizer"]
    )
    linear_models = {
        "schema_version": "relcompat3d_relation_algebra_models_v1",
        "attempts": attempts,
        "source_score_used": False,
        "source_identity_used": False,
        "parameterization": {
            "id": "relcompat3d_linear",
            "family_indicator_input": False,
            "family_selected_by": "family-specific head",
        },
    }

    def direct_score(
        condition: str, family: str, predicate: str, raw: dict[str, float]
    ) -> float:
        if condition == "family":
            return algebra.existing_probability(
                fitted_base["family_models"][family], family, predicate, raw
            )
        model = attempts[condition][family]
        if condition == "algebra_basis" and family != "support_contact":
            return algebra.basis_probability(model, predicate, raw)
        return algebra.existing_probability(model, family, predicate, raw)

    scorer = algebra.build_scorer(direct_score)
    if args.fit_only:
        expected_optimizer = {
            "epochs": 800,
            "learning_rate": 0.2,
            "l2": 0.0001,
            "initial_weights": 0.0,
            "batching": "deterministic_full_batch",
            "pairwise_loss": "softplus(margin - (logit_positive - logit_negative))",
            "pairwise_margin": 1.0,
            "pairwise_weight": 0.25,
        }
        old_family_features = {
            family: current_base["family_models"][family]["feature_names"]
            for family in algebra.FAMILIES
        }
        new_family_features = {
            family: fitted_base["family_models"][family]["feature_names"]
            for family in algebra.FAMILIES
        }
        validations = {
            "split_counts_1061_117_157": (
                len(train_scans), len(dev_scans), len(final_scans)
            ) == (1061, 117, 157),
            "split_sets_pairwise_disjoint": not (
                train_scans & dev_scans
                or train_scans & final_scans
                or dev_scans & final_scans
            ),
            "zero_final_rows_in_calibration": not leaked,
            "train_rows_60208": sum(
                row["_role"] == "train" for row in prepared
            ) == 60208,
            "development_rows_6246": sum(
                row["_role"] == "dev" for row in prepared
            ) == 6246,
            "optimizer_configuration_matches": protocol["optimizer"] == expected_optimizer,
            "one_family_feature_removed_per_head": all(
                len(old_family_features[family])
                == len(new_family_features[family]) + 1
                and [
                    name
                    for name in old_family_features[family]
                    if not name.startswith("family:")
                ]
                == new_family_features[family]
                for family in algebra.FAMILIES
            ),
            "linear_family_features_absent": not any(
                name.startswith("family:")
                for name in model_features(linear_models)
            ),
            "all_parameters_finite": all(
                math.isfinite(weight)
                for attempt in attempts.values()
                for model in attempt.values()
                for weight in model["weights"]
            ),
            "source_score_and_identity_excluded": (
                not linear_models["source_score_used"]
                and not linear_models["source_identity_used"]
            ),
        }
        status = "completed" if all(validations.values()) else "failed_validation"
        out.mkdir(parents=True, exist_ok=True)
        base_path = out / "base_models.json"
        linear_path = out / "linear_models.json"
        diagnostics_path = out / "training_diagnostics.json"
        write_json(base_path, fitted_base)
        write_json(linear_path, linear_models)
        write_json(
            diagnostics_path,
            {
                "schema_version": "relcompat3d_relation_algebra_diagnostics_v1",
                "role": "training_split_fit",
                "diagnostics": diagnostics,
                "base_family_models": base_diagnostics,
                "validations": validations,
            },
        )
        outputs = [base_path, linear_path, diagnostics_path]
        write_json(
            out / "manifest.json",
            {
                "schema_version": "relcompat3d_linear_fit_manifest_v1",
                "status": status,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "protocol": {
                    "path": relpath(root, protocol_path),
                    "sha256": sha256_file(protocol_path),
                },
                "inputs": {
                    name: {"path": relpath(root, path), "sha256": sha256_file(path)}
                    for name, path in fit_inputs.items()
                },
                "outputs": {
                    path.name: {"path": relpath(root, path), "sha256": sha256_file(path)}
                    for path in outputs
                },
                "validations": validations,
                "warnings": warnings,
            },
        )
        print(json.dumps({"status": status, "validations": validations}, sort_keys=True))
        return 0 if status == "completed" else 2
    development_gt, development_gt_family = algebra.load_ground_truth(
        paths["development_ground_truth"]
    )
    development_source = algebra.evaluate_source(
        "development_sgfn",
        paths["development_verification"],
        scorer,
        development_gt,
        development_gt_family,
        int(protocol["development"]["bootstrap_seed"]),
        int(protocol["development"]["bootstrap_resamples"]),
    )

    out.mkdir(parents=True, exist_ok=True)
    base_path = out / "base_models.json"
    linear_path = out / "linear_models.json"
    diagnostics_path = out / "development_metrics.json"
    development_path = out / "development_predictions.json"
    score_path = out / "score_definition.json"
    write_json(base_path, fitted_base)
    write_json(linear_path, linear_models)
    write_json(
        diagnostics_path,
        {
            "schema_version": "relcompat3d_relation_algebra_diagnostics_v1",
            "role": "training_and_development_diagnostics",
            "diagnostics": diagnostics,
            "base_family_models": base_diagnostics,
        },
    )
    write_json(development_path, development_source)
    write_json(
        score_path,
        {
            "schema_version": "relcompat3d_score_definition_v1",
            "compatibility": "orbit_pairwise heads followed by transformation averaging",
            "inputs": {
                "predicate": True,
                "predicate_independent_geometry": True,
                "predicate_signed_vertical_interactions": True,
                "family_indicator": False,
                "predictor_score": False,
                "predictor_identity": False,
            },
            "family_selection": "a_i selects the family head, normalization statistics, transformation set, and ranking scope",
            "ranking": {
                "proximity_and_vertical": "u_i = Z_i * C_i^tr within the source family-label sequence",
                "support_contact": "exact source-ranking subsequence",
                "ties": "exact relation-candidate identity in re-ranked families",
            },
            "ks": [5, 10, 20, 50, 100],
        },
    )

    expected_optimizer = {
        "epochs": 800,
        "learning_rate": 0.2,
        "l2": 0.0001,
        "initial_weights": 0.0,
        "batching": "deterministic_full_batch",
        "pairwise_loss": "softplus(margin - (logit_positive - logit_negative))",
        "pairwise_margin": 1.0,
        "pairwise_weight": 0.25,
    }
    old_family_features = {
        family: current_base["family_models"][family]["feature_names"]
        for family in algebra.FAMILIES
    }
    new_family_features = {
        family: fitted_base["family_models"][family]["feature_names"]
        for family in algebra.FAMILIES
    }
    validations = {
        "split_counts_1061_117_157": (len(train_scans), len(dev_scans), len(final_scans))
        == (1061, 117, 157),
        "split_sets_pairwise_disjoint": not (
            train_scans & dev_scans or train_scans & final_scans or dev_scans & final_scans
        ),
        "zero_final_rows_in_calibration": not leaked,
        "train_rows_60208": sum(row["_role"] == "train" for row in prepared) == 60208,
        "development_rows_6246": sum(row["_role"] == "dev" for row in prepared) == 6246,
        "optimizer_configuration_matches": protocol["optimizer"] == expected_optimizer,
        "one_family_feature_removed_per_head": all(
            len(old_family_features[family]) == len(new_family_features[family]) + 1
            and [name for name in old_family_features[family] if not name.startswith("family:")]
            == new_family_features[family]
            for family in algebra.FAMILIES
        ),
        "base_family_features_absent": all(
            not any(name.startswith("family:") for name in names)
            for names in new_family_features.values()
        ),
        "linear_family_features_absent": not any(
            name.startswith("family:") for name in model_features(linear_models)
        ),
        "normalization_statistics_unchanged": all(
            current_base["family_models"][family]["numeric_stats"]
            == fitted_base["family_models"][family]["numeric_stats"]
            for family in algebra.FAMILIES
        ),
        "feature_templates_unchanged": current_base["factor_models"]
        == fitted_base["factor_models"],
        "all_parameters_finite": all(
            math.isfinite(weight)
            for attempt in attempts.values()
            for model in attempt.values()
            for weight in model["weights"]
        ),
        "source_score_and_identity_excluded": not linear_models["source_score_used"]
        and not linear_models["source_identity_used"],
        "validation_not_used_for_model_selection": development_source["source"]
        == "development_sgfn",
        "development_contexts_354": development_source["contexts"] == 354,
        "development_gt_denominator_2730": development_source["gt_denominator"]
        == 2730,
    }
    status = "completed" if all(validations.values()) else "failed_validation"
    selection_path = out / "model_selection.json"
    write_json(
        selection_path,
        {
            "schema_version": "relcompat3d_model_selection_v1",
            "status": "selected" if status == "completed" else "failed_validation",
            "selected_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": sha256_file(protocol_path),
            "linear_model_sha256": sha256_file(linear_path),
            "base_model_sha256": sha256_file(base_path),
            "score_definition_sha256": sha256_file(score_path),
            "validation_data_used_for_fitting": False,
        },
    )
    manifest_path = out / "manifest.json"
    outputs = [
        base_path,
        linear_path,
        diagnostics_path,
        development_path,
        score_path,
        selection_path,
    ]
    write_json(
        manifest_path,
        {
            "schema_version": "relcompat3d_linear_fit_manifest_v1",
            "status": status,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": {
                "path": relpath(root, protocol_path),
                "sha256": sha256_file(protocol_path),
            },
            "inputs": {
                name: {"path": relpath(root, path), "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
            "outputs": {
                path.name: {"path": relpath(root, path), "sha256": sha256_file(path)}
                for path in outputs
            },
            "validations": validations,
            "warnings": warnings,
            "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm relcompat3d_fit",
        },
    )
    print(
        json.dumps(
            {
                "status": status,
                "linear_model_sha256": sha256_file(linear_path),
                "base_model_sha256": sha256_file(base_path),
                "score_definition_sha256": sha256_file(score_path),
                "validations": validations,
            },
            sort_keys=True,
        )
    )
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
