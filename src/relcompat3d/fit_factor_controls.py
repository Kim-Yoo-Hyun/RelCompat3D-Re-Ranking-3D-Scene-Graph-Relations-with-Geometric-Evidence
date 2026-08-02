#!/usr/bin/env python3
"""Fit the four pre-registered RelCompat3D factor-isolation models on train rows only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import compatibility_features as base


SCHEMA_VERSION = "relcompat3d_factor_isolation_fitted_models_v1"
FAMILIES = ("proximity", "relative_vertical", "support_contact")
PREDICATES = ("close by", "higher than", "lower than", "lying on", "standing on", "supported by")
RAW_G = (
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
INTERACTIONS = (
    "predicate_aligned_center_delta_z",
    "predicate_aligned_normalized_center_delta_z",
)
CONDITIONS = {
    "M_T": {"use_t": True, "numeric": ()},
    "M_G": {"use_t": False, "numeric": RAW_G},
    "M_add": {"use_t": True, "numeric": RAW_G},
    "M_int": {"use_t": True, "numeric": RAW_G + INTERACTIONS},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--calibration-table", type=Path, required=True)
    parser.add_argument("--train-scans", type=Path, required=True)
    parser.add_argument("--dev-scans", type=Path, required=True)
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_scan_list(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def metric_block(probabilities: list[float], labels: list[int]) -> dict[str, Any]:
    return {
        "rows": len(labels),
        "positive": sum(labels),
        "negative": len(labels) - sum(labels),
        "auroc": base.auroc(probabilities, labels),
        "auprc": base.average_precision(probabilities, labels),
        "brier": base.brier_score(probabilities, labels),
        "nll": base.log_loss(probabilities, labels),
    }


def numeric_stats(rows: list[dict[str, Any]], names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for name in names:
        values = [row["_raw_numeric"][name] for row in rows if name in row["_raw_numeric"]]
        mean = sum(values) / len(values) if values else 0.0
        variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
        stats[name] = {
            "mean": mean,
            "std": math.sqrt(variance) if variance > 0.0 else 1.0,
            "observed_train_rows": len(values),
        }
    return stats


def feature_names(spec: dict[str, Any]) -> list[str]:
    names = ["bias"]
    if spec["use_t"]:
        names.extend(f"family:{value}" for value in FAMILIES)
        names.extend(f"predicate:{value}" for value in PREDICATES)
    names.extend(f"num:{value}" for value in spec["numeric"])
    return names


def vectorize(row: dict[str, Any], spec: dict[str, Any], stats: dict[str, dict[str, float]]) -> list[float]:
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


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    paths = {
        "protocol": resolve(root, args.protocol),
        "calibration_table": resolve(root, args.calibration_table),
        "train_scans": resolve(root, args.train_scans),
        "dev_scans": resolve(root, args.dev_scans),
    }
    out = resolve(root, args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"nonempty_output:{out}")
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "relcompat3d_factor_controls_v1":
        raise ValueError("factor_protocol_not_frozen_v1")
    train_scans = read_scan_list(paths["train_scans"])
    dev_scans = read_scan_list(paths["dev_scans"])
    if train_scans & dev_scans:
        raise ValueError("train_dev_scan_overlap")
    rows = base.load_jsonl(paths["calibration_table"])
    prepared, warnings = base.prepare_rows(rows, train_scans, dev_scans, set(FAMILIES))
    train_rows = [row for row in prepared if row["_role"] == "train"]
    dev_rows = [row for row in prepared if row["_role"] == "dev"]
    if len(train_rows) != 4616 or len(dev_rows) != 1193:
        raise ValueError(f"frozen_row_count_mismatch:{len(train_rows)}:{len(dev_rows)}")

    models: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    validations: dict[str, bool] = {}
    for condition, spec in CONDITIONS.items():
        stats = numeric_stats(train_rows, spec["numeric"])
        names = feature_names(spec)
        train_x = [vectorize(row, spec, stats) for row in train_rows]
        dev_x = [vectorize(row, spec, stats) for row in dev_rows]
        train_y = [int(row["_label"]) for row in train_rows]
        dev_y = [int(row["_label"]) for row in dev_rows]
        weights, trace = base.fit_logistic(
            train_x,
            train_y,
            epochs=800,
            learning_rate=0.2,
            l2=1e-4,
        )
        train_prob = base.predict(train_x, weights)
        dev_prob = base.predict(dev_x, weights)
        models[condition] = {
            "condition": condition,
            "architecture": "single_pooled_logistic_regression",
            "feature_names": names,
            "numeric_features": list(spec["numeric"]),
            "numeric_stats": stats,
            "families": list(FAMILIES) if spec["use_t"] else [],
            "predicates": list(PREDICATES) if spec["use_t"] else [],
            "weights": weights,
            "optimizer": {
                "epochs": 800,
                "learning_rate": 0.2,
                "l2": 1e-4,
                "batching": "deterministic_full_batch",
                "initial_weights": 0.0,
            },
            "fit_split": "calibration_train_only",
            "trace": trace,
        }
        diagnostics[condition] = {
            "train": metric_block(train_prob, train_y),
            "dev_no_selection": metric_block(dev_prob, dev_y),
        }
        expected_count = int(protocol["generated_artifacts"] and json.loads((paths["protocol"].parent / "conditions.json").read_text(encoding="utf-8"))[condition]["parameter_count"])
        validations[f"{condition}:parameter_count_{expected_count}"] = len(names) == expected_count == len(weights)
        validations[f"{condition}:finite_weights"] = all(math.isfinite(value) for value in weights)
        validations[f"{condition}:train_only_stats"] = all(
            int(value["observed_train_rows"]) <= len(train_rows) for value in stats.values()
        )

    validations.update(
        {
            "train_rows_4616": len(train_rows) == 4616,
            "dev_rows_1193": len(dev_rows) == 1193,
            "train_scans_24": len(train_scans) == 24,
            "dev_scans_8": len(dev_scans) == 8,
            "train_dev_disjoint": not (train_scans & dev_scans),
            "no_source_or_Z_features": all(
                not any(token in name.lower() for token in ("source", "semantic", "score", "rank", "baseline", "z:"))
                for model in models.values()
                for name in model["feature_names"]
            ),
        }
    )
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "models.json"
    diagnostic_path = out / "dev_diagnostics.json"
    write_json(model_path, {"schema_version": SCHEMA_VERSION, "models": models})
    write_json(
        diagnostic_path,
        {
            "schema_version": SCHEMA_VERSION,
            "role": "calibration_dev_diagnostic_only_no_model_or_condition_selection",
            "diagnostics": diagnostics,
        },
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "factor_models_frozen_pre_fresh_source_inference" if all(validations.values()) else "blocked_factor_model_freeze",
        "counts": {
            "input_rows": len(rows),
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "train_scans": len(train_scans),
            "dev_scans": len(dev_scans),
            "labels_train": dict(sorted(Counter(row["_label"] for row in train_rows).items())),
            "labels_dev": dict(sorted(Counter(row["_label"] for row in dev_rows).items())),
        },
        "warnings": warnings,
        "validations": validations,
        "inputs": {name: {"path": relpath(root, path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "outputs": {
            "models": {"path": relpath(root, model_path), "sha256": sha256_file(model_path)},
            "dev_diagnostics": {"path": relpath(root, diagnostic_path), "sha256": sha256_file(diagnostic_path)},
        },
        "selection_boundary": "No target/source output was read; dev diagnostics did not select features, conditions, or hyperparameters.",
        "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm factor_isolation_model_fit",
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"], "out": relpath(root, out)}))
    return 0 if manifest["status"].startswith("factor_models_frozen") else 2


if __name__ == "__main__":
    raise SystemExit(main())
