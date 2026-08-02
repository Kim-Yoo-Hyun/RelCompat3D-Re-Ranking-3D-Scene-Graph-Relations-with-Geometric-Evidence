#!/usr/bin/env python3
"""Fit/evaluate a small RelCompat3D p_geom_valid calibration smoke model."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from paths import RelCompat3D_HYPOTHESIS_ROOT


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
RelCompat3D_ROOT = RelCompat3D_HYPOTHESIS_ROOT
DEFAULT_INPUT_DIR = RelCompat3D_ROOT / "artifacts" / "calibration" / "train_dev_calib"
DEFAULT_PILOT_ROOT = RelCompat3D_ROOT / "artifacts" / "subset" / "relcompat3d_calib_pilot"
DEFAULT_OUTPUT_DIR = RelCompat3D_ROOT / "artifacts" / "calibration" / "p_geom_valid_smoke"

MODEL_SCHEMA_VERSION = "relcompat3d_p_geom_valid_model_v1"
METRICS_SCHEMA_VERSION = "relcompat3d_p_geom_valid_metrics_v1"
SCORE_SCHEMA_VERSION = "relcompat3d_p_geom_valid_score_v1"

NUMERIC_FEATURES = (
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
DERIVED_FEATURES = (
    "abs_center_delta_z",
    "abs_normalized_center_delta_z",
    "abs_vertical_gap_subject_on_object",
    "predicate_aligned_center_delta_z",
    "predicate_aligned_normalized_center_delta_z",
)
DEFAULT_FAMILIES = ("support_contact", "proximity", "relative_vertical")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit RelCompat3D p_geom_valid calibration smoke model.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_scan_list(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def raw_numeric_features(row: dict[str, Any]) -> dict[str, float]:
    source = row["geometry"]["features"]
    predicate = row["predicate"]["predicate_label"]
    values: dict[str, float] = {}
    for name in NUMERIC_FEATURES:
        value = finite_float(source.get(name))
        if value is not None:
            values[name] = value
    if "center_delta_z" in values:
        values["abs_center_delta_z"] = abs(values["center_delta_z"])
    if "normalized_center_delta_z" in values:
        values["abs_normalized_center_delta_z"] = abs(values["normalized_center_delta_z"])
    if "vertical_gap_subject_on_object" in values:
        values["abs_vertical_gap_subject_on_object"] = abs(
            values["vertical_gap_subject_on_object"]
        )
    direction = 0.0
    if predicate == "higher than":
        direction = 1.0
    elif predicate == "lower than":
        direction = -1.0
    if direction and "center_delta_z" in values:
        values["predicate_aligned_center_delta_z"] = direction * values["center_delta_z"]
    if direction and "normalized_center_delta_z" in values:
        values["predicate_aligned_normalized_center_delta_z"] = (
            direction * values["normalized_center_delta_z"]
        )
    return values


def assign_role(scan_id: str, train_scans: set[str], dev_scans: set[str]) -> str | None:
    if scan_id in train_scans:
        return "train"
    if scan_id in dev_scans:
        return "dev"
    return None


def prepare_rows(
    rows: list[dict[str, Any]],
    train_scans: set[str],
    dev_scans: set[str],
    families: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    prepared: list[dict[str, Any]] = []
    warnings: list[str] = []
    skipped: Counter[str] = Counter()
    for row in rows:
        family = row["predicate"]["predicate_family"]
        if family not in families:
            skipped["family_out_of_scope"] += 1
            continue
        role = assign_role(row["scan_id"], train_scans, dev_scans)
        if role is None:
            skipped["scan_outside_train_dev"] += 1
            continue
        label = row["label"].get("geom_valid")
        if label not in (0, 1):
            skipped["non_binary_label"] += 1
            continue
        copy = dict(row)
        copy["_role"] = role
        copy["_label"] = int(label)
        copy["_raw_numeric"] = raw_numeric_features(row)
        prepared.append(copy)
    if skipped:
        warnings.append(f"skipped_rows:{dict(sorted(skipped.items()))}")
    return prepared, warnings


def build_model_spec(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_names = list(NUMERIC_FEATURES) + list(DERIVED_FEATURES)
    stats: dict[str, dict[str, float]] = {}
    for name in numeric_names:
        values = [row["_raw_numeric"][name] for row in train_rows if name in row["_raw_numeric"]]
        if values:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            std = math.sqrt(variance) if variance > 0 else 1.0
        else:
            mean = 0.0
            std = 1.0
        stats[name] = {"mean": mean, "std": std}

    families = sorted({row["predicate"]["predicate_family"] for row in train_rows})
    predicates = sorted({row["predicate"]["predicate_label"] for row in train_rows})
    feature_names = (
        ["bias"]
        + [f"num:{name}" for name in numeric_names]
        + [f"family:{name}" for name in families]
        + [f"predicate:{name}" for name in predicates]
    )
    return {
        "numeric_features": numeric_names,
        "numeric_stats": stats,
        "families": families,
        "predicates": predicates,
        "feature_names": feature_names,
    }


def vectorize(row: dict[str, Any], spec: dict[str, Any]) -> list[float]:
    vector = [1.0]
    raw = row["_raw_numeric"]
    for name in spec["numeric_features"]:
        value = raw.get(name, spec["numeric_stats"][name]["mean"])
        mean = spec["numeric_stats"][name]["mean"]
        std = spec["numeric_stats"][name]["std"] or 1.0
        vector.append((value - mean) / std)
    family = row["predicate"]["predicate_family"]
    vector.extend(1.0 if family == name else 0.0 for name in spec["families"])
    predicate = row["predicate"]["predicate_label"]
    vector.extend(1.0 if predicate == name else 0.0 for name in spec["predicates"])
    return vector


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def dot(weights: list[float], vector: list[float]) -> float:
    return sum(weight * value for weight, value in zip(weights, vector))


def log_loss(predictions: list[float], labels: list[int]) -> float:
    eps = 1e-12
    total = 0.0
    for prob, label in zip(predictions, labels):
        prob = min(max(prob, eps), 1.0 - eps)
        total -= label * math.log(prob) + (1 - label) * math.log(1.0 - prob)
    return total / len(labels) if labels else 0.0


def fit_logistic(
    vectors: list[list[float]],
    labels: list[int],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[list[float], list[dict[str, float]]]:
    feature_count = len(vectors[0])
    weights = [0.0] * feature_count
    trace: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        gradients = [0.0] * feature_count
        predictions: list[float] = []
        for vector, label in zip(vectors, labels):
            probability = sigmoid(dot(weights, vector))
            predictions.append(probability)
            error = probability - label
            for index, value in enumerate(vector):
                gradients[index] += error * value
        count = float(len(labels))
        for index in range(feature_count):
            gradients[index] /= count
            if index != 0:
                gradients[index] += l2 * weights[index]
            weights[index] -= learning_rate * gradients[index]
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            loss = log_loss(predictions, labels)
            penalty = 0.5 * l2 * sum(weight * weight for weight in weights[1:])
            trace.append({"epoch": epoch, "train_nll": loss + penalty})
    return weights, trace


def predict(vectors: list[list[float]], weights: list[float]) -> list[float]:
    return [sigmoid(dot(weights, vector)) for vector in vectors]


def brier_score(predictions: list[float], labels: list[int]) -> float | None:
    if not labels:
        return None
    return sum((prob - label) ** 2 for prob, label in zip(predictions, labels)) / len(labels)


def calibration_bins(
    predictions: list[float], labels: list[int], bins: int
) -> tuple[list[dict[str, Any]], float | None, float | None]:
    if not labels:
        return [], None, None
    groups = [
        {"bin": index, "lower": index / bins, "upper": (index + 1) / bins, "count": 0}
        for index in range(bins)
    ]
    for prob, label in zip(predictions, labels):
        index = min(int(prob * bins), bins - 1)
        group = groups[index]
        group["count"] += 1
        group["prob_sum"] = group.get("prob_sum", 0.0) + prob
        group["label_sum"] = group.get("label_sum", 0.0) + label
    ece = 0.0
    mce = 0.0
    for group in groups:
        count = group["count"]
        if count:
            avg_conf = group["prob_sum"] / count
            empirical = group["label_sum"] / count
            gap = abs(avg_conf - empirical)
            group["avg_p_geom_valid"] = avg_conf
            group["empirical_geom_valid"] = empirical
            group["gap"] = gap
            ece += (count / len(labels)) * gap
            mce = max(mce, gap)
        else:
            group["avg_p_geom_valid"] = None
            group["empirical_geom_valid"] = None
            group["gap"] = None
        group.pop("prob_sum", None)
        group.pop("label_sum", None)
    return groups, ece, mce


def auroc(predictions: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs = sorted(zip(predictions, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        next_index = index + 1
        while next_index < len(pairs) and pairs[next_index][0] == pairs[index][0]:
            next_index += 1
        avg_rank = (index + 1 + next_index) / 2.0
        positives_in_tie = sum(label for _, label in pairs[index:next_index])
        rank_sum += positives_in_tie * avg_rank
        index = next_index
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(predictions: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    ranked = sorted(zip(predictions, labels), key=lambda item: item[0], reverse=True)
    true_positive = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, 1):
        if label == 1:
            true_positive += 1
            precision_sum += true_positive / rank
    return precision_sum / positives


def invalid_precision(predictions: list[float], labels: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for threshold in (0.1, 0.2, 0.3, 0.5):
        selected = [(prob, label) for prob, label in zip(predictions, labels) if prob <= threshold]
        invalid = sum(1 for _, label in selected if label == 0)
        result[str(threshold)] = {
            "selected": len(selected),
            "invalid": invalid,
            "precision_invalid": invalid / len(selected) if selected else None,
            "coverage": len(selected) / len(labels) if labels else None,
        }
    return result


def summarize_predictions(
    rows: list[dict[str, Any]],
    predictions: list[float],
    bins: int,
) -> dict[str, Any]:
    labels = [row["_label"] for row in rows]
    bin_rows, ece, mce = calibration_bins(predictions, labels, bins)
    by_family: dict[str, Any] = {}
    for family in sorted({row["predicate"]["predicate_family"] for row in rows}):
        indices = [
            index
            for index, row in enumerate(rows)
            if row["predicate"]["predicate_family"] == family
        ]
        family_rows = [rows[index] for index in indices]
        family_predictions = [predictions[index] for index in indices]
        family_labels = [labels[index] for index in indices]
        by_family[family] = {
            "rows": len(family_rows),
            "positives": sum(family_labels),
            "negatives": len(family_labels) - sum(family_labels),
            "positive_rate": sum(family_labels) / len(family_labels) if family_labels else None,
            "mean_p_geom_valid": (
                sum(family_predictions) / len(family_predictions) if family_predictions else None
            ),
            "brier": brier_score(family_predictions, family_labels),
            "nll": log_loss(family_predictions, family_labels) if family_labels else None,
            "auroc_valid": auroc(family_predictions, family_labels),
            "auprc_valid": average_precision(family_predictions, family_labels),
        }
    return {
        "rows": len(rows),
        "positives": sum(labels),
        "negatives": len(labels) - sum(labels),
        "positive_rate": sum(labels) / len(labels) if labels else None,
        "mean_p_geom_valid": sum(predictions) / len(predictions) if predictions else None,
        "brier": brier_score(predictions, labels),
        "nll": log_loss(predictions, labels) if labels else None,
        "ece": ece,
        "mce": mce,
        "auroc_valid": auroc(predictions, labels),
        "auprc_valid": average_precision(predictions, labels),
        "invalid_detection": invalid_precision(predictions, labels),
        "calibration_bins": bin_rows,
        "by_family": by_family,
    }


def prior_predictions(rows: list[dict[str, Any]], prior: float) -> list[float]:
    return [prior] * len(rows)


def family_prior_predictions(
    rows: list[dict[str, Any]], family_prior: dict[str, float], fallback: float
) -> list[float]:
    return [
        family_prior.get(row["predicate"]["predicate_family"], fallback)
        for row in rows
    ]


def count_by_role(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in ("train", "dev"):
        scoped = [row for row in rows if row["_role"] == role]
        result[role] = {
            "rows": len(scoped),
            "positives": sum(row["_label"] for row in scoped),
            "negatives": sum(1 for row in scoped if row["_label"] == 0),
            "by_family": dict(
                sorted(Counter(row["predicate"]["predicate_family"] for row in scoped).items())
            ),
            "by_predicate": dict(
                sorted(Counter(row["predicate"]["predicate_label"] for row in scoped).items())
            ),
        }
    return result


def make_scores(
    rows: list[dict[str, Any]],
    probabilities: list[float],
    model_id: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row, probability in zip(rows, probabilities):
        output.append(
            {
                "schema_version": SCORE_SCHEMA_VERSION,
                "model_id": model_id,
                "candidate_id": row["candidate_id"],
                "role": row["_role"],
                "scan_id": row["scan_id"],
                "subset_split_id": row["subset_split_id"],
                "subgraph_id": row["subgraph_id"],
                "edge": row["edge"],
                "predicate": row["predicate"],
                "label": row["label"],
                "p_geom_valid": probability,
                "p_geom_invalid": 1.0 - probability,
            }
        )
    return output


def make_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# P-Geom Calibration",
        "",
        f"Created at: `{metrics['created_at']}`",
        f"Status: `{metrics['status']}`",
        f"Model id: `{metrics['model_id']}`",
        "",
        "## Inputs",
        "",
        f"- Calibration table: `{metrics['inputs']['table_jsonl']}`",
        f"- Pilot root: `{metrics['inputs']['pilot_root']}`",
        "",
        "## Counts",
        "",
    ]
    for role, counts in metrics["counts"]["by_role"].items():
        lines.append(
            f"- `{role}`: rows `{counts['rows']}`, positives `{counts['positives']}`, "
            f"negatives `{counts['negatives']}`"
        )
    lines.extend(["", "## Dev Metrics", ""])
    dev = metrics["conditions"]["logistic"]["dev"]
    lines.extend(
        [
            f"- Brier: `{dev['brier']}`",
            f"- NLL: `{dev['nll']}`",
            f"- ECE: `{dev['ece']}`",
            f"- MCE: `{dev['mce']}`",
            f"- AUROC(valid): `{dev['auroc_valid']}`",
            f"- AUPRC(valid): `{dev['auprc_valid']}`",
        ]
    )
    lines.extend(["", "## Dev By Family", ""])
    for family, result in dev["by_family"].items():
        lines.append(
            f"- `{family}`: rows `{result['rows']}`, Brier `{result['brier']}`, "
            f"NLL `{result['nll']}`, AUROC `{result['auroc_valid']}`"
        )
    lines.extend(["", "## Baselines", ""])
    for name, result in metrics["conditions"]["baselines"]["dev"].items():
        lines.append(
            f"- `{name}`: Brier `{result['brier']}`, NLL `{result['nll']}`, ECE `{result['ece']}`"
        )
    lines.extend(["", "## Notes", ""])
    for note in metrics["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    table_jsonl = args.input_dir / "table.jsonl"
    manifest_json = args.input_dir / "manifest.json"
    train_scans_path = args.pilot_root / "train_scans.txt"
    dev_scans_path = args.pilot_root / "dev_scans.txt"
    for name, path in {
        "table_jsonl": table_jsonl,
        "manifest_json": manifest_json,
        "train_scans": train_scans_path,
        "dev_scans": dev_scans_path,
    }.items():
        if not path.exists():
            errors.append(f"missing_input:{name}:{relpath(path)}")
    if errors:
        print(json.dumps({"status": "blocked", "errors": errors}, sort_keys=True))
        return 2

    source_manifest = load_json(manifest_json)
    if source_manifest.get("status") != "ready":
        errors.append(f"input_manifest_not_ready:{source_manifest.get('status')}")
    if source_manifest.get("validation", {}).get("errors"):
        errors.append("input_manifest_has_validation_errors")
    if errors:
        print(json.dumps({"status": "blocked", "errors": errors}, sort_keys=True))
        return 2

    rows = load_jsonl(table_jsonl)
    train_scans = read_scan_list(train_scans_path)
    dev_scans = read_scan_list(dev_scans_path)
    train_dev_overlap = sorted(train_scans & dev_scans)
    if train_dev_overlap:
        errors.append(f"train_dev_scan_overlap:{train_dev_overlap[:10]}")
    prepared, prep_warnings = prepare_rows(rows, train_scans, dev_scans, set(args.families))
    warnings.extend(prep_warnings)
    train_rows = [row for row in prepared if row["_role"] == "train"]
    dev_rows = [row for row in prepared if row["_role"] == "dev"]
    if not train_rows:
        errors.append("zero_train_rows")
    if not dev_rows:
        errors.append("zero_dev_rows")
    for role, scoped in (("train", train_rows), ("dev", dev_rows)):
        labels = {row["_label"] for row in scoped}
        if labels != {0, 1}:
            errors.append(f"{role}_missing_binary_labels:{sorted(labels)}")
    if errors:
        print(json.dumps({"status": "blocked", "errors": errors}, sort_keys=True))
        return 2

    spec = build_model_spec(train_rows)
    train_vectors = [vectorize(row, spec) for row in train_rows]
    dev_vectors = [vectorize(row, spec) for row in dev_rows]
    train_labels = [row["_label"] for row in train_rows]
    dev_labels = [row["_label"] for row in dev_rows]
    weights, trace = fit_logistic(
        train_vectors,
        train_labels,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    train_prob = predict(train_vectors, weights)
    dev_prob = predict(dev_vectors, weights)

    train_prior = sum(train_labels) / len(train_labels)
    family_prior = {}
    for family in spec["families"]:
        scoped = [row for row in train_rows if row["predicate"]["predicate_family"] == family]
        family_prior[family] = sum(row["_label"] for row in scoped) / len(scoped)

    model_id = "relcompat3d-p-geom-valid-smoke-v1"
    metrics = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "created_at": date.today().isoformat(),
        "status": "ready",
        "model_id": model_id,
        "inputs": {
            "table_jsonl": relpath(table_jsonl),
            "manifest_json": relpath(manifest_json),
            "pilot_root": relpath(args.pilot_root),
            "train_scans": relpath(train_scans_path),
            "dev_scans": relpath(dev_scans_path),
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "bins": args.bins,
        },
        "counts": {
            "input_rows": len(rows),
            "used_rows": len(prepared),
            "by_role": count_by_role(prepared),
        },
        "conditions": {
            "logistic": {
                "train": summarize_predictions(train_rows, train_prob, args.bins),
                "dev": summarize_predictions(dev_rows, dev_prob, args.bins),
            },
            "baselines": {
                "dev": {
                    "constant_train_prior": summarize_predictions(
                        dev_rows, prior_predictions(dev_rows, train_prior), args.bins
                    ),
                    "family_train_prior": summarize_predictions(
                        dev_rows,
                        family_prior_predictions(dev_rows, family_prior, train_prior),
                        args.bins,
                    ),
                },
            },
        },
        "training_trace": trace,
        "warnings": warnings,
        "notes": [
            "This is a train/dev calibration smoke test, not final held-out hypothesis evidence.",
            "Rows are split by scan id using 25_pilot.md train/dev lists.",
            "The model uses geometry numeric features plus predicate family/label only.",
            "The model does not use candidate_source, label_source, or negative strategy as features.",
            "Semantic scores are not used because p_semantic is null in train_dev_calib.",
        ],
    }
    model = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "created_at": metrics["created_at"],
        "model_id": model_id,
        "source_split": "train_dev_calib",
        "feature_names": spec["feature_names"],
        "numeric_features": spec["numeric_features"],
        "numeric_stats": spec["numeric_stats"],
        "families": spec["families"],
        "predicates": spec["predicates"],
        "weights": weights,
        "hyperparameters": metrics["hyperparameters"],
        "train_prior": train_prior,
        "family_prior": family_prior,
        "notes": metrics["notes"],
    }
    scores = make_scores(train_rows, train_prob, model_id) + make_scores(
        dev_rows, dev_prob, model_id
    )
    manifest = {
        "schema_version": "relcompat3d_p_geom_valid_manifest_v1",
        "created_at": metrics["created_at"],
        "status": "ready",
        "model_id": model_id,
        "source_calibration_split": "train_dev_calib",
        "model_file": "model.json",
        "scores_file": "scores.jsonl",
        "metrics_file": "metrics.json",
        "report_file": "report.md",
        "inputs": metrics["inputs"],
        "counts": metrics["counts"],
        "dev_metrics": metrics["conditions"]["logistic"]["dev"],
        "warnings": warnings,
        "notes": metrics["notes"],
    }

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "manifest.json", manifest)
        write_json(args.output_dir / "model.json", model)
        write_json(args.output_dir / "metrics.json", metrics)
        write_jsonl(args.output_dir / "scores.jsonl", scores)
        (args.output_dir / "report.md").write_text(make_report(metrics), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ready",
                "output_dir": relpath(args.output_dir),
                "train_rows": len(train_rows),
                "dev_rows": len(dev_rows),
                "dev_brier": metrics["conditions"]["logistic"]["dev"]["brier"],
                "dev_ece": metrics["conditions"]["logistic"]["dev"]["ece"],
                "warnings": len(warnings),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
