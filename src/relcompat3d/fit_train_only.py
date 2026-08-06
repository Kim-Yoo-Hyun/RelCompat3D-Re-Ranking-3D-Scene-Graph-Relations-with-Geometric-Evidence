#!/usr/bin/env python3
"""Fit all frozen RelCompat3D calibrators under the strict 1061/117 train firewall."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import compatibility_features as base
from fit_factor_controls import CONDITIONS, FAMILIES, PREDICATES, feature_names, numeric_stats, vectorize


SCHEMA = "relcompat3d_strict_train_only_calibrators_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--calibration-table", type=Path, required=True)
    parser.add_argument("--train-scans", type=Path, required=True)
    parser.add_argument("--dev-scans", type=Path, required=True)
    parser.add_argument("--final-scans", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
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


def read_scans(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def fit_numpy(vectors: list[list[float]], labels: list[int]) -> tuple[list[float], list[dict[str, float]]]:
    x = np.asarray(vectors, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    trace: list[dict[str, float]] = []
    for epoch in range(1, 801):
        probabilities = stable_sigmoid(x @ weights)
        gradient = (x.T @ (probabilities - y)) / len(y)
        gradient[1:] += 1e-4 * weights[1:]
        weights -= 0.2 * gradient
        if epoch == 1 or epoch % 50 == 0 or epoch == 800:
            clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
            nll = float(np.mean(-(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))))
            penalty = float(0.5e-4 * np.sum(weights[1:] ** 2))
            trace.append({"epoch": epoch, "train_nll": nll + penalty})
    return weights.tolist(), trace


def predict_numpy(vectors: list[list[float]], weights: list[float]) -> list[float]:
    return stable_sigmoid(np.asarray(vectors, dtype=np.float64) @ np.asarray(weights, dtype=np.float64)).tolist()


def metrics(probabilities: list[float], labels: list[int]) -> dict[str, Any]:
    return {
        "rows": len(labels),
        "positive": sum(labels),
        "negative": len(labels) - sum(labels),
        "auroc": base.auroc(probabilities, labels),
        "auprc": base.average_precision(probabilities, labels),
        "brier": base.brier_score(probabilities, labels),
        "nll": base.log_loss(probabilities, labels),
    }


def fit_factor_models(train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    models: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    train_y = [int(row["_label"]) for row in train_rows]
    dev_y = [int(row["_label"]) for row in dev_rows]
    for condition, spec in CONDITIONS.items():
        stats = numeric_stats(train_rows, spec["numeric"])
        names = feature_names(spec)
        train_x = [vectorize(row, spec, stats) for row in train_rows]
        dev_x = [vectorize(row, spec, stats) for row in dev_rows]
        weights, trace = fit_numpy(train_x, train_y)
        models[condition] = {
            "condition": condition,
            "architecture": "single_pooled_logistic_regression",
            "feature_names": names,
            "numeric_features": list(spec["numeric"]),
            "numeric_stats": stats,
            "families": list(FAMILIES) if spec["use_t"] else [],
            "predicates": list(PREDICATES) if spec["use_t"] else [],
            "weights": weights,
            "optimizer": {"epochs": 800, "learning_rate": 0.2, "l2": 1e-4, "batching": "deterministic_full_batch", "initial_weights": 0.0},
            "fit_split": "strict_train_1061_only",
            "trace": trace,
        }
        diagnostics[condition] = {
            "train": metrics(predict_numpy(train_x, weights), train_y),
            "internal_dev_no_selection": metrics(predict_numpy(dev_x, weights), dev_y),
        }
    return models, diagnostics


def fit_family_models(train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    models: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for family in FAMILIES:
        train = [row for row in train_rows if row["predicate"]["predicate_family"] == family]
        dev = [row for row in dev_rows if row["predicate"]["predicate_family"] == family]
        spec = base.build_model_spec(train)
        train_x, dev_x = [base.vectorize(row, spec) for row in train], [base.vectorize(row, spec) for row in dev]
        train_y, dev_y = [int(row["_label"]) for row in train], [int(row["_label"]) for row in dev]
        if set(train_y) != {0, 1} or set(dev_y) != {0, 1}:
            raise ValueError(f"family_binary_label_missing:{family}")
        weights, trace = fit_numpy(train_x, train_y)
        models[family] = {
            "family": family,
            "feature_names": spec["feature_names"],
            "numeric_features": spec["numeric_features"],
            "numeric_stats": spec["numeric_stats"],
            "families": spec["families"],
            "predicates": spec["predicates"],
            "weights": weights,
            "train_prior": sum(train_y) / len(train_y),
            "fit_split": "strict_train_1061_only",
            "training_trace": trace,
            "counts": {"train_rows": len(train), "internal_dev_rows": len(dev), "train_labels": dict(sorted(Counter(train_y).items())), "internal_dev_labels": dict(sorted(Counter(dev_y).items()))},
        }
        diagnostics[family] = {
            "train": metrics(predict_numpy(train_x, weights), train_y),
            "internal_dev_no_selection": metrics(predict_numpy(dev_x, weights), dev_y),
        }
    return models, diagnostics


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    paths = {name: resolve(root, value) for name, value in {
        "protocol": args.protocol,
        "calibration_table": args.calibration_table,
        "train_scans": args.train_scans,
        "dev_scans": args.dev_scans,
        "final_scans": args.final_scans,
    }.items()}
    out = resolve(root, args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"nonempty_output:{out}")
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    if protocol.get("status") != "protocol_frozen_before_strict_calibration_and_internal_dev_inference":
        raise ValueError("train_only_protocol_not_frozen")
    train_scans, dev_scans, final_scans = (read_scans(paths[name]) for name in ("train_scans", "dev_scans", "final_scans"))
    if train_scans & dev_scans or train_scans & final_scans or dev_scans & final_scans:
        raise ValueError("split_firewall_overlap")
    rows = base.load_jsonl(paths["calibration_table"])
    prepared, warnings = base.prepare_rows(rows, train_scans, dev_scans, set(FAMILIES))
    train_rows = [row for row in prepared if row["_role"] == "train"]
    dev_rows = [row for row in prepared if row["_role"] == "dev"]
    leaked = sorted({row["scan_id"] for row in rows} & final_scans)
    if leaked:
        raise ValueError(f"final_validation_rows_in_calibration:{leaked[:10]}")
    factor_models, factor_diagnostics = fit_factor_models(train_rows, dev_rows)
    family_models, family_diagnostics = fit_family_models(train_rows, dev_rows)
    validations = {
        "split_counts_1061_117_157": (len(train_scans), len(dev_scans), len(final_scans)) == (1061, 117, 157),
        "zero_final_validation_rows": not leaked,
        "all_export_rows_assigned_train_or_dev": len(prepared) == len(rows),
        "train_and_dev_nonempty": bool(train_rows) and bool(dev_rows),
        "factor_conditions_exact": set(factor_models) == {"M_T", "M_G", "M_add", "M_int"},
        "family_models_exact": set(family_models) == set(FAMILIES),
        "no_Z_or_source_features": all(not any(token in feature.lower() for token in ("semantic", "source", "rank", "baseline", "score")) for model in list(factor_models.values()) + list(family_models.values()) for feature in model["feature_names"]),
        "all_weights_finite": all(math.isfinite(weight) for model in list(factor_models.values()) + list(family_models.values()) for weight in model["weights"]),
    }
    out.mkdir(parents=True, exist_ok=True)
    models_path = out / "models.json"
    diagnostics_path = out / "internal_dev_calibration_diagnostics.json"
    write_json(models_path, {
        "schema_version": SCHEMA,
        "default_compatibility": "family_specific",
        "family_models": family_models,
        "factor_models": factor_models,
    })
    write_json(diagnostics_path, {
        "schema_version": SCHEMA,
        "role": "train_only_internal_dev_diagnostic_no_final_validation_access",
        "factor": factor_diagnostics,
        "family": family_diagnostics,
    })
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "strict_train_only_models_ready_pre_internal_dev_source_metrics" if all(validations.values()) else "blocked_strict_train_only_fit",
        "counts": {
            "input_rows": len(rows), "train_rows": len(train_rows), "internal_dev_rows": len(dev_rows),
            "train_scans_with_rows": len({row["scan_id"] for row in train_rows}),
            "internal_dev_scans_with_rows": len({row["scan_id"] for row in dev_rows}),
            "train_labels": dict(sorted(Counter(row["_label"] for row in train_rows).items())),
            "internal_dev_labels": dict(sorted(Counter(row["_label"] for row in dev_rows).items())),
        },
        "warnings": warnings,
        "validations": validations,
        "inputs": {name: {"path": relpath(root, path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "outputs": {
            "models": {"path": relpath(root, models_path), "sha256": sha256_file(models_path)},
            "diagnostics": {"path": relpath(root, diagnostics_path), "sha256": sha256_file(diagnostics_path)},
        },
        "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm relcompat3d_fit_base",
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"], "validations": validations, "out": relpath(root, out)}))
    return 0 if manifest["status"].startswith("strict_train_only_models_ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
