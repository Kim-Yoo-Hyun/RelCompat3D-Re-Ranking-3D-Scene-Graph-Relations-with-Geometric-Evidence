#!/usr/bin/env python3
"""Fit the strict RelCompat3D family heads without their constant family indicator.

This stage reads only training and internal-development artifacts.  It writes a
model/score lock for a later, separately invoked official-validation stage.
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
import fit_train_only as strict_fit
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
        "--fit-only",
        action="store_true",
        help="Fit and export the Linear estimator without internal-development evaluation.",
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
        "id": "main_experiment",
        "removed_feature": removed,
        "family_selected_by": "family-specific head",
    }
    return result


def refit_strict_family_models(
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
        weights, trace = strict_fit.fit_numpy(train_x, train_y)
        model = {
            **template,
            "weights": weights,
            "train_prior": sum(train_y) / len(train_y),
            "fit_split": "strict_train_1061_only",
            "training_trace": trace,
            "counts": {
                "train_rows": len(train_y),
                "internal_dev_rows": len(dev_y),
                "train_labels": {
                    "0": len(train_y) - sum(train_y),
                    "1": sum(train_y),
                },
                "internal_dev_labels": {
                    "0": len(dev_y) - sum(dev_y),
                    "1": sum(dev_y),
                },
            },
        }
        family_models[family] = model
        diagnostics[family] = {
            "train": strict_fit.metrics(strict_fit.predict_numpy(train_x, weights), train_y),
            "internal_dev_no_selection": strict_fit.metrics(
                strict_fit.predict_numpy(dev_x, weights), dev_y
            ),
        }
    strict_models = copy.deepcopy(current_models)
    strict_models["family_models"] = family_models
    strict_models["parameterization"] = {
        "id": "main_experiment",
        "change": "remove the constant family one-hot from each family-specific head",
        "factor_models": "unchanged; family indicators remain meaningful in pooled heads",
    }
    return strict_models, diagnostics


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
    if protocol.get("status") != "frozen_before_relcompat3d_fit":
        raise ValueError("protocol_not_frozen")
    paths = {name: resolve(root, value) for name, value in protocol["inputs"].items()}
    if args.calibration_table is not None:
        paths["calibration_table"] = resolve(root, args.calibration_table)
    fit_inputs = {
        name: paths[name]
        for name in (
            "calibration_table",
            "train_scans",
            "internal_dev_scans",
            "final_validation_scans",
            "current_strict_models",
        )
    }
    required_paths = fit_inputs if args.fit_only else paths
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing_inputs:{missing}")

    train_scans = read_scans(paths["train_scans"])
    dev_scans = read_scans(paths["internal_dev_scans"])
    final_scans = read_scans(paths["final_validation_scans"])
    if train_scans & dev_scans or train_scans & final_scans or dev_scans & final_scans:
        raise ValueError("split_firewall_overlap")

    rows = calibration.load_jsonl(paths["calibration_table"])
    leaked = sorted({str(row["scan_id"]) for row in rows} & final_scans)
    if leaked:
        raise ValueError(f"final_validation_rows_in_calibration:{leaked[:10]}")
    prepared, warnings = calibration.prepare_rows(
        rows, train_scans, dev_scans, set(algebra.FAMILIES)
    )
    current_strict = json.loads(paths["current_strict_models"].read_text(encoding="utf-8"))
    new_strict, strict_diagnostics = refit_strict_family_models(prepared, current_strict)
    attempts, diagnostics = algebra.fit_attempts(
        prepared, new_strict, protocol["optimizer"]
    )
    structured_models = {
        "schema_version": "relcompat3d_relation_algebra_models_v1",
        "attempts": attempts,
        "source_score_used": False,
        "source_identity_used": False,
        "parameterization": {
            "id": "main_experiment",
            "family_indicator_input": False,
            "family_selected_by": "family-specific head",
        },
    }

    def direct_score(
        condition: str, family: str, predicate: str, raw: dict[str, float]
    ) -> float:
        if condition == "family":
            return algebra.existing_probability(
                new_strict["family_models"][family], family, predicate, raw
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
            family: current_strict["family_models"][family]["feature_names"]
            for family in algebra.FAMILIES
        }
        new_family_features = {
            family: new_strict["family_models"][family]["feature_names"]
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
            "internal_dev_rows_6246": sum(
                row["_role"] == "dev" for row in prepared
            ) == 6246,
            "optimizer_exactly_preserved": protocol["optimizer"] == expected_optimizer,
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
            "structured_family_features_absent": not any(
                name.startswith("family:")
                for name in model_features(structured_models)
            ),
            "all_parameters_finite": all(
                math.isfinite(weight)
                for attempt in attempts.values()
                for model in attempt.values()
                for weight in model["weights"]
            ),
            "source_score_and_identity_excluded": (
                not structured_models["source_score_used"]
                and not structured_models["source_identity_used"]
            ),
        }
        status = "completed" if all(validations.values()) else "failed_validation"
        out.mkdir(parents=True, exist_ok=True)
        strict_path = out / "strict_models.json"
        structured_path = out / "structured_models.json"
        diagnostics_path = out / "training_diagnostics.json"
        write_json(strict_path, new_strict)
        write_json(structured_path, structured_models)
        write_json(
            diagnostics_path,
            {
                "schema_version": "relcompat3d_relation_algebra_diagnostics_v1",
                "role": "train_only_fit",
                "diagnostics": diagnostics,
                "strict_family_models": strict_diagnostics,
                "validations": validations,
            },
        )
        outputs = [strict_path, structured_path, diagnostics_path]
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
    internal_gt, internal_gt_family = algebra.load_ground_truth(
        paths["internal_dev_ground_truth"]
    )
    internal_dev_source = algebra.evaluate_source(
        "internal_dev_sgfn",
        paths["internal_dev_verification"],
        scorer,
        internal_gt,
        internal_gt_family,
        int(protocol["internal_dev"]["bootstrap_seed"]),
        int(protocol["internal_dev"]["bootstrap_resamples"]),
    )

    out.mkdir(parents=True, exist_ok=True)
    strict_path = out / "strict_models.json"
    structured_path = out / "structured_models.json"
    diagnostics_path = out / "internal_dev_diagnostics.json"
    internal_path = out / "internal_dev_source.json"
    score_path = out / "score_contract.json"
    write_json(strict_path, new_strict)
    write_json(structured_path, structured_models)
    write_json(
        diagnostics_path,
        {
            "schema_version": "relcompat3d_relation_algebra_diagnostics_v1",
            "role": "train_only_fit_internal_dev_sanity_no_selection",
            "diagnostics": diagnostics,
            "strict_family_models": strict_diagnostics,
        },
    )
    write_json(internal_path, internal_dev_source)
    write_json(
        score_path,
        {
            "schema_version": "relcompat3d_main_experiment_score_contract_v1",
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
        family: current_strict["family_models"][family]["feature_names"]
        for family in algebra.FAMILIES
    }
    new_family_features = {
        family: new_strict["family_models"][family]["feature_names"]
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
        "internal_dev_rows_6246": sum(row["_role"] == "dev" for row in prepared) == 6246,
        "optimizer_exactly_preserved": protocol["optimizer"] == expected_optimizer,
        "one_family_feature_removed_per_head": all(
            len(old_family_features[family]) == len(new_family_features[family]) + 1
            and [name for name in old_family_features[family] if not name.startswith("family:")]
            == new_family_features[family]
            for family in algebra.FAMILIES
        ),
        "strict_family_features_absent": all(
            not any(name.startswith("family:") for name in names)
            for names in new_family_features.values()
        ),
        "structured_family_features_absent": not any(
            name.startswith("family:") for name in model_features(structured_models)
        ),
        "normalization_statistics_unchanged": all(
            current_strict["family_models"][family]["numeric_stats"]
            == new_strict["family_models"][family]["numeric_stats"]
            for family in algebra.FAMILIES
        ),
        "factor_models_unchanged": current_strict["factor_models"]
        == new_strict["factor_models"],
        "all_parameters_finite": all(
            math.isfinite(weight)
            for attempt in attempts.values()
            for model in attempt.values()
            for weight in model["weights"]
        ),
        "source_score_and_identity_excluded": not structured_models["source_score_used"]
        and not structured_models["source_identity_used"],
        "internal_dev_only_before_lock": internal_dev_source["source"]
        == "internal_dev_sgfn",
        "internal_dev_contexts_354": internal_dev_source["contexts"] == 354,
        "internal_dev_gt_denominator_2730": internal_dev_source["gt_denominator"]
        == 2730,
    }
    status = "completed" if all(validations.values()) else "failed_validation"
    lock_path = out / "final_lock.json"
    write_json(
        lock_path,
        {
            "schema_version": "relcompat3d_main_experiment_final_lock_v1",
            "status": "locked_before_official_validation" if status == "completed" else "not_locked",
            "locked_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": sha256_file(protocol_path),
            "structured_model_sha256": sha256_file(structured_path),
            "strict_model_sha256": sha256_file(strict_path),
            "score_contract_sha256": sha256_file(score_path),
            "official_validation_read_by_fit_stage": False,
        },
    )
    manifest_path = out / "manifest.json"
    outputs = [
        strict_path,
        structured_path,
        diagnostics_path,
        internal_path,
        score_path,
        lock_path,
    ]
    write_json(
        manifest_path,
        {
            "schema_version": "relcompat3d_relcompat3d_fit_manifest_v1",
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
                "structured_model_sha256": sha256_file(structured_path),
                "strict_model_sha256": sha256_file(strict_path),
                "score_contract_sha256": sha256_file(score_path),
                "validations": validations,
            },
            sort_keys=True,
        )
    )
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
