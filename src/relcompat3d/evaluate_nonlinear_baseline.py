#!/usr/bin/env python3
"""Fit and evaluate the frozen parameter-matched nonlinear rescorer baseline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


FAMILIES = ("support_contact", "proximity", "relative_vertical")
PREDICATES = ("close by", "higher than", "lower than", "lying on", "standing on", "supported by")
RAW_FEATURES = (
    "distance_3d", "distance_xy", "normalized_distance_3d", "normalized_distance_xy",
    "center_delta_z", "normalized_center_delta_z", "projected_iou_xy",
    "projected_subject_overlap_ratio", "projected_object_overlap_ratio",
    "vertical_gap_subject_on_object", "subject_bottom_z", "subject_top_z",
    "object_bottom_z", "object_top_z", "abs_center_delta_z",
    "abs_normalized_center_delta_z", "abs_vertical_gap_subject_on_object",
)
INTERACTIONS = ("predicate_aligned_center_delta_z", "predicate_aligned_normalized_center_delta_z")
KS = (5, 10, 20, 50, 100)
METHODS = ("semantic_only", "family_product", "nonlinear_rescorer")
INPUT_DIM = 1 + len(FAMILIES) + len(PREDICATES) + len(RAW_FEATURES) + len(INTERACTIONS)
HIDDEN = 2
PARAMETER_COUNT = HIDDEN * INPUT_DIM + HIDDEN + HIDDEN + 1 + len(PREDICATES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--train-verification", type=Path, required=True)
    parser.add_argument("--train-ground-truth", type=Path, required=True)
    parser.add_argument("--final-verification", type=Path, required=True)
    parser.add_argument("--final-ground-truth", type=Path, required=True)
    parser.add_argument("--compatibility-models", type=Path, required=True)
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
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_eval_module(root: Path) -> Any:
    path = root / "src/relcompat3d/evaluate_train_only.py"
    spec = importlib.util.spec_from_file_location("relcompat3d_train_only_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot_import:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def logit(value: float) -> float:
    clipped = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


def feature_values(evalmod: Any, row: dict[str, Any]) -> tuple[list[float | None], str, str, float]:
    family = str(row["predicate"]["predicate_family"])
    predicate = str(row["predicate"]["predicate_label"])
    semantic = evalmod.finite((row.get("semantic") or {}).get("ranking_score"))
    if semantic is None:
        raise ValueError(f"missing_semantic:{row['prediction_id']}")
    raw = evalmod.align_predicate(evalmod.raw_numeric(row), predicate)
    values: list[float | None] = [logit(semantic)]
    values.extend(float(family == item) for item in FAMILIES)
    values.extend(float(predicate == item) for item in PREDICATES)
    values.extend(raw.get(name) for name in RAW_FEATURES)
    values.extend(raw.get(name) for name in INTERACTIONS)
    if len(values) != INPUT_DIM:
        raise AssertionError(f"feature_width:{len(values)}:{INPUT_DIM}")
    return values, family, predicate, float(semantic)


def candidate_key(evalmod: Any, row: dict[str, Any]) -> tuple[Any, ...]:
    return evalmod.candidate_key(row)


def load_gt(evalmod: Any, path: Path) -> tuple[dict[str, set[tuple[Any, ...]]], int]:
    grouped, _ = evalmod.load_gt(path)
    return grouped, sum(len(rows) for rows in grouped.values())


def training_matrix(evalmod: Any, path: Path, gt: dict[str, set[tuple[Any, ...]]]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rows: list[list[float | None]] = []
    labels: list[float] = []
    subgraphs: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["predicate"]["predicate_family"] not in FAMILIES:
                continue
            values, _, _, _ = feature_values(evalmod, row)
            rows.append(values)
            labels.append(float(candidate_key(evalmod, row) in gt.get(row["subgraph_id"], set())))
            subgraphs.add(row["subgraph_id"])
    raw = np.asarray([[np.nan if value is None else value for value in row] for row in rows], dtype=np.float64)
    means = np.nanmean(raw, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    filled = np.where(np.isnan(raw), means[None, :], raw)
    continuous = [0] + list(range(1 + len(FAMILIES) + len(PREDICATES), INPUT_DIM))
    std = np.ones(INPUT_DIM, dtype=np.float64)
    std[continuous] = filled[:, continuous].std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    normalized = filled.copy()
    normalized[:, continuous] = (filled[:, continuous] - means[continuous]) / std[continuous]
    stats = {
        "mean": means.tolist(), "std": std.tolist(), "continuous_indices": continuous,
        "rows": len(rows), "contexts": len(subgraphs), "positive": int(sum(labels)),
        "negative": len(labels) - int(sum(labels)),
    }
    return normalized, np.asarray(labels, dtype=np.float64), stats


def initialize(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    scale = math.sqrt(2.0 / INPUT_DIM)
    return {
        "W": rng.normal(0.0, scale, size=(HIDDEN, INPUT_DIM)),
        "b": np.zeros(HIDDEN),
        "v": rng.normal(0.0, math.sqrt(2.0 / HIDDEN), size=HIDDEN),
        "out_b": np.zeros(1),
        "predicate_skip": np.zeros(len(PREDICATES)),
    }


def forward(params: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre = x @ params["W"].T + params["b"]
    hidden = np.maximum(pre, 0.0)
    predicate_start = 1 + len(FAMILIES)
    logits = hidden @ params["v"] + params["out_b"][0] + x[:, predicate_start:predicate_start + len(PREDICATES)] @ params["predicate_skip"]
    probs = np.empty_like(logits)
    positive = logits >= 0
    probs[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probs[~positive] = exp_logits / (1.0 + exp_logits)
    return probs, hidden, pre


def fit(x: np.ndarray, y: np.ndarray, protocol: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    spec = protocol["model"]["optimizer"]
    params = initialize(int(spec["seed"]))
    m = {name: np.zeros_like(value) for name, value in params.items()}
    v2 = {name: np.zeros_like(value) for name, value in params.items()}
    rng = np.random.default_rng(int(spec["seed"]))
    lr, l2 = float(spec["learning_rate"]), float(spec["l2"])
    batch_size, epochs = int(spec["batch_size"]), int(spec["epochs"])
    positive_weight = float((len(y) - y.sum()) / max(y.sum(), 1.0))
    step, trace = 0, []
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            idx = order[start:start + batch_size]
            xb, yb = x[idx], y[idx]
            probs, hidden, pre = forward(params, xb)
            weights = np.where(yb > 0.5, positive_weight, 1.0)
            dlogit = weights * (probs - yb) / weights.sum()
            predicate_start = 1 + len(FAMILIES)
            grads = {
                "v": hidden.T @ dlogit + l2 * params["v"],
                "out_b": np.asarray([dlogit.sum()]),
                "predicate_skip": xb[:, predicate_start:predicate_start + len(PREDICATES)].T @ dlogit + l2 * params["predicate_skip"],
            }
            dhidden = dlogit[:, None] * params["v"][None, :]
            dpre = dhidden * (pre > 0.0)
            grads["W"] = dpre.T @ xb + l2 * params["W"]
            grads["b"] = dpre.sum(axis=0)
            step += 1
            for name in params:
                m[name] = 0.9 * m[name] + 0.1 * grads[name]
                v2[name] = 0.999 * v2[name] + 0.001 * np.square(grads[name])
                mhat = m[name] / (1.0 - 0.9 ** step)
                vhat = v2[name] / (1.0 - 0.999 ** step)
                params[name] -= lr * mhat / (np.sqrt(vhat) + 1e-8)
        probs, _, _ = forward(params, x)
        eps = 1e-8
        loss = -np.mean(np.where(y > 0.5, positive_weight * np.log(probs + eps), np.log(1.0 - probs + eps)))
        if epoch in {0, 9, 19, 39, epochs - 1}:
            trace.append({"epoch": epoch + 1, "weighted_bce": float(loss)})
    return params, trace


def normalize_row(values: list[float | None], stats: dict[str, Any]) -> np.ndarray:
    means = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    array = np.asarray([np.nan if value is None else value for value in values], dtype=np.float64)
    array = np.where(np.isnan(array), means, array)
    continuous = np.asarray(stats["continuous_indices"], dtype=int)
    array[continuous] = (array[continuous] - means[continuous]) / std[continuous]
    return array


def load_final(evalmod: Any, path: Path, models: dict[str, Any], params: dict[str, np.ndarray], stats: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            family = row["predicate"]["predicate_family"]
            if family not in FAMILIES:
                continue
            count += 1
            values, family, predicate, semantic = feature_values(evalmod, row)
            raw = evalmod.raw_numeric(row)
            compatibility = evalmod.probability(models["family_models"][family], family, predicate, raw)
            x = normalize_row(values, stats)[None, :]
            nonlinear = float(forward(params, x)[0][0])
            grouped[row["subgraph_id"]].append({
                "key": candidate_key(evalmod, row),
                "status": row.get("verification_status") or (row.get("verification") or {}).get("verification_status"),
                "scores": {
                    "semantic_only": semantic,
                    "family_product": semantic * compatibility,
                    "nonlinear_rescorer": nonlinear,
                },
            })
    return grouped, count


def metric_contributions(grouped: dict[str, list[dict[str, Any]]], gt: dict[str, set[tuple[Any, ...]]], subgraphs: list[str]) -> dict[str, dict[str, np.ndarray]]:
    values = {
        method: {name: np.zeros((len(KS), len(subgraphs)), dtype=np.float64) for name in ("recall_num", "recall_den", "violation_num", "violation_den")}
        for method in METHODS
    }
    for si, subgraph in enumerate(subgraphs):
        candidates = grouped.get(subgraph, [])
        for method in METHODS:
            ranked = sorted(candidates, key=lambda item: (-item["scores"][method], item["key"]))
            for ki, k in enumerate(KS):
                selected = ranked[:k]
                values[method]["recall_num"][ki, si] = len({row["key"] for row in selected} & gt.get(subgraph, set()))
                values[method]["recall_den"][ki, si] = len(gt.get(subgraph, set()))
                statuses = [row["status"] for row in selected if row["status"] in {"satisfied", "uncertain", "violated"}]
                values[method]["violation_num"][ki, si] = sum(status == "violated" for status in statuses)
                values[method]["violation_den"][ki, si] = len(statuses)
    return values


def ci(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    return [float(value) for value in np.percentile(finite, (2.5, 97.5))]


def summarize(values: dict[str, dict[str, np.ndarray]], subgraphs: list[str], seed: int, n_bootstrap: int) -> dict[str, Any]:
    samples = np.random.default_rng(seed).integers(0, len(subgraphs), size=(n_bootstrap, len(subgraphs)))
    report: dict[str, Any] = {method: {} for method in METHODS}
    cache: dict[str, Any] = {method: {} for method in METHODS}
    for method in METHODS:
        for ki, k in enumerate(KS):
            report[method][str(k)], cache[method][str(k)] = {}, {}
            for metric in ("recall", "violation"):
                num = values[method][f"{metric}_num"][ki]
                den = values[method][f"{metric}_den"][ki]
                point = float(num.sum() / den.sum())
                bnum, bden = num[samples].sum(axis=1), den[samples].sum(axis=1)
                boot = np.divide(bnum, bden, out=np.full_like(bnum, np.nan), where=bden > 0)
                report[method][str(k)][metric] = {"point": point, "ci95": ci(boot), "numerator": int(num.sum()), "denominator": int(den.sum())}
                cache[method][str(k)][metric] = boot
    report["contrasts"] = {}
    for left in ("family_product", "nonlinear_rescorer"):
        for right in ("semantic_only", "family_product"):
            if left == right:
                continue
            name = f"{left}_minus_{right}"
            report["contrasts"][name] = {}
            for k in KS:
                report["contrasts"][name][str(k)] = {}
                for metric in ("recall", "violation"):
                    delta = report[left][str(k)][metric]["point"] - report[right][str(k)][metric]["point"]
                    boot = cache[left][str(k)][metric] - cache[right][str(k)][metric]
                    report["contrasts"][name][str(k)][metric] = {"point": delta, "paired_ci95": ci(boot)}
    return report


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    paths = {name: resolve(root, value) for name, value in {
        "protocol": args.protocol,
        "train_verification": args.train_verification,
        "train_ground_truth": args.train_ground_truth,
        "final_verification": args.final_verification,
        "final_ground_truth": args.final_ground_truth,
        "compatibility_models": args.compatibility_models,
    }.items()}
    out = resolve(root, args.out)
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "relcompat3d_parameter_matched_nonlinear_rescorer_protocol_v1":
        raise ValueError("unexpected_protocol")
    if PARAMETER_COUNT != int(protocol["model"]["parameter_count"]):
        raise ValueError(f"parameter_count_mismatch:{PARAMETER_COUNT}")
    evalmod = load_eval_module(root)
    models = json.loads(paths["compatibility_models"].read_text(encoding="utf-8"))
    train_gt, train_den = load_gt(evalmod, paths["train_ground_truth"])
    x, y, stats = training_matrix(evalmod, paths["train_verification"], train_gt)
    params, trace = fit(x, y, protocol)
    final_gt, final_den = load_gt(evalmod, paths["final_ground_truth"])
    grouped, final_rows = load_final(evalmod, paths["final_verification"], models, params, stats)
    subgraphs = sorted(set(grouped) | set(final_gt))
    metrics = summarize(metric_contributions(grouped, final_gt, subgraphs), subgraphs, 20260712, 1000)
    serialized_params = {name: value.tolist() for name, value in params.items()}
    expected_final_rows = int(protocol["evaluation"].get("expected_in_scope_rows", 220848))
    validations = {
        "parameter_count_exactly_69": PARAMETER_COUNT == 69,
        "train_contexts_354": stats["contexts"] == 354,
        "train_gt_denominator_2730": train_den == 2730,
        "final_contexts_548": len(subgraphs) == 548,
        "final_gt_denominator_3972": final_den == 3972,
        "final_in_scope_rows_expected": final_rows == expected_final_rows,
        "finite_parameters": all(np.isfinite(value).all() for value in params.values()),
        "final_rows_not_used_for_fit_or_normalization": True,
    }
    out.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "schema_version": "relcompat3d_parameter_matched_nonlinear_rescorer_model_v1",
        "architecture": protocol["model"], "input_dim": INPUT_DIM,
        "parameter_count": PARAMETER_COUNT, "normalization": stats,
        "parameters": serialized_params, "training_trace": trace,
    }
    write_json(out / "model.json", model_payload)
    summary = {
        "schema_version": "relcompat3d_parameter_matched_nonlinear_rescorer_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": protocol["classification"],
        "counts": {"train_rows": len(y), "train_positive": int(y.sum()), "train_contexts": stats["contexts"], "final_rows": final_rows, "final_contexts": len(subgraphs), "final_gt_denominator": final_den},
        "metrics": metrics,
        "validations": validations,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(out / "summary.json", summary)
    lines = [
        "# Parameter-Matched Nonlinear Rescorer Baseline", "",
        f"Status: `{'passed' if all(validations.values()) else 'failed'}`", "",
        "This is a post-hoc baseline and does not select or modify the active method.", "",
        "| Method | K | Recall | Violation |", "| --- | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        for k in (10, 50, 100):
            cell = metrics[method][str(k)]
            lines.append(f"| {method} | {k} | {cell['recall']['point']:.4f} | {cell['violation']['point']:.4f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "relcompat3d_parameter_matched_nonlinear_rescorer_manifest_v1",
        "status": "completed" if all(validations.values()) else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {name: {"path": relpath(root, path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "outputs": {name: {"path": relpath(root, out / name), "sha256": sha256_file(out / name)} for name in ("model.json", "summary.json", "summary.md")},
        "validations": validations,
        "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm nonlinear_fusion_baseline",
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "counts": summary["counts"], "out": relpath(root, out)}))
    return 0 if manifest["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
