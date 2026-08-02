#!/usr/bin/env python3
"""Evaluate the frozen train-only RelCompat3D scores on internal-dev or final validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


FAMILIES = ("support_contact", "proximity", "relative_vertical")
KS = (5, 10, 20, 50, 100)
METHODS = (
    "semantic_only",
    "family_product",
    "pooled_product",
    "geometry_only_family",
    "rank_average_family",
    "rrf_c60",
    "product_M_T",
    "product_M_G",
    "product_M_add",
    "product_M_int",
)
COMPAT_MODELS = ("family_specific", "M_T", "M_G", "M_add", "M_int")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--role", choices=("internal_dev", "final_validation"), required=True)
    parser.add_argument("--expected-contexts", type=int, required=True)
    parser.add_argument("--expected-gt-denominator", type=int, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lock-manifest", type=Path)
    parser.add_argument("--docker-service", required=True)
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def raw_numeric(row: dict[str, Any]) -> dict[str, float]:
    source = (row.get("geometry") or {}).get("features") or {}
    names = (
        "distance_3d", "distance_xy", "normalized_distance_3d", "normalized_distance_xy",
        "center_delta_z", "normalized_center_delta_z", "projected_iou_xy",
        "projected_subject_overlap_ratio", "projected_object_overlap_ratio",
        "vertical_gap_subject_on_object", "subject_bottom_z", "subject_top_z",
        "object_bottom_z", "object_top_z",
    )
    values = {name: value for name in names if (value := finite(source.get(name))) is not None}
    for source_name, target_name in (
        ("center_delta_z", "abs_center_delta_z"),
        ("normalized_center_delta_z", "abs_normalized_center_delta_z"),
        ("vertical_gap_subject_on_object", "abs_vertical_gap_subject_on_object"),
    ):
        if source_name in values:
            values[target_name] = abs(values[source_name])
    return values


def align_predicate(raw: dict[str, float], predicate: str) -> dict[str, float]:
    values = dict(raw)
    values.pop("predicate_aligned_center_delta_z", None)
    values.pop("predicate_aligned_normalized_center_delta_z", None)
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


def probability(model: dict[str, Any], family: str, predicate: str, raw: dict[str, float]) -> float:
    values = align_predicate(raw, predicate)
    vector: list[float] = []
    for feature in model["feature_names"]:
        if feature == "bias":
            vector.append(1.0)
        elif feature.startswith("family:"):
            vector.append(float(family == feature.split(":", 1)[1]))
        elif feature.startswith("predicate:"):
            vector.append(float(predicate == feature.split(":", 1)[1]))
        elif feature.startswith("num:"):
            name = feature.split(":", 1)[1]
            stat = model["numeric_stats"][name]
            vector.append((values.get(name, stat["mean"]) - stat["mean"]) / (stat["std"] or 1.0))
        else:
            raise ValueError(f"unsupported_feature:{feature}")
    if len(vector) != len(model["weights"]):
        raise ValueError(f"model_width_mismatch:{model.get('condition', model.get('family'))}")
    return sigmoid(sum(left * right for left, right in zip(model["weights"], vector)))


def candidate_key(row: dict[str, Any]) -> tuple[str, int, int, int, str]:
    return (
        str(row["scan_id"]), int(row["subset_split_id"]), int(row["edge"]["subject_id"]),
        int(row["edge"]["object_id"]), str(row["predicate"]["predicate_label"]),
    )


def gt_key(row: dict[str, Any]) -> tuple[str, int, int, int, str]:
    return (
        str(row["scan_id"]), int(row["subset_split_id"]), int(row["subject_id"]),
        int(row["object_id"]), str(row["predicate_label"]),
    )


def load_gt(path: Path) -> tuple[dict[str, set[tuple[Any, ...]]], dict[str, dict[str, set[tuple[Any, ...]]]]]:
    overall: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    by_family: dict[str, dict[str, set[tuple[Any, ...]]]] = defaultdict(lambda: defaultdict(set))
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            family = row["predicate_family"]
            if family not in FAMILIES:
                continue
            key = gt_key(row)
            overall[row["subgraph_id"]].add(key)
            by_family[row["subgraph_id"]][family].add(key)
    return overall, by_family


def load_candidates(path: Path, payload: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_rows = 0
    family_models, factor_models = payload["family_models"], payload["factor_models"]
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            input_rows += 1
            row = json.loads(line)
            family = row["predicate"]["predicate_family"]
            if family not in FAMILIES:
                continue
            predicate = row["predicate"]["predicate_label"]
            raw = raw_numeric(row)
            compatibility = {
                "family_specific": probability(family_models[family], family, predicate, raw),
                **{name: probability(model, family, predicate, raw) for name, model in factor_models.items()},
            }
            semantic = finite((row.get("semantic") or {}).get("ranking_score"))
            if semantic is None:
                raise ValueError(f"missing_semantic:{row['prediction_id']}")
            grouped[row["subgraph_id"]].append({
                "id": row["prediction_id"], "key": candidate_key(row), "family": family,
                "predicate": predicate, "subject": int(row["edge"]["subject_id"]),
                "object": int(row["edge"]["object_id"]), "semantic": semantic,
                "compat": compatibility, "raw": raw,
                "status": row.get("verification_status") or (row.get("verification") or {}).get("verification_status"),
                "scores": {},
            })
    for candidates in grouped.values():
        count = len(candidates)
        denominator = max(count - 1, 1)
        semantic_order = sorted(candidates, key=lambda item: (-item["semantic"], item["key"]))
        sem_rank = {item["id"]: rank for rank, item in enumerate(semantic_order, 1)}
        geom_order = sorted(candidates, key=lambda item: (-item["compat"]["family_specific"], item["key"]))
        geom_rank = {item["id"]: rank for rank, item in enumerate(geom_order, 1)}
        for item in candidates:
            semantic_pct = 1.0 - (sem_rank[item["id"]] - 1) / denominator
            geometry_pct = 1.0 - (geom_rank[item["id"]] - 1) / denominator
            item["scores"].update({
                "semantic_only": item["semantic"],
                "family_product": item["semantic"] * item["compat"]["family_specific"],
                "pooled_product": item["semantic"] * item["compat"]["M_int"],
                "geometry_only_family": item["compat"]["family_specific"],
                "rank_average_family": 0.5 * (semantic_pct + geometry_pct),
                "rrf_c60": 1.0 / (60 + sem_rank[item["id"]]) + 1.0 / (60 + geom_rank[item["id"]]),
                "product_M_T": item["semantic"] * item["compat"]["M_T"],
                "product_M_G": item["semantic"] * item["compat"]["M_G"],
                "product_M_add": item["semantic"] * item["compat"]["M_add"],
                "product_M_int": item["semantic"] * item["compat"]["M_int"],
            })
    return grouped, input_rows


def empty_arrays(subgraphs: list[str]) -> dict[str, dict[str, np.ndarray]]:
    return {
        method: {name: np.zeros((len(KS), len(subgraphs)), dtype=np.float64) for name in (
            "recall_num", "recall_den", "violation_num", "violation_den",
        )}
        for method in METHODS
    }


def add_cell(target: dict[str, np.ndarray], ki: int, si: int, selected: list[dict[str, Any]], gt: set[tuple[Any, ...]]) -> None:
    target["recall_num"][ki, si] = len({row["key"] for row in selected} & gt)
    target["recall_den"][ki, si] = len(gt)
    statuses = [row["status"] for row in selected if row["status"] in {"satisfied", "uncertain", "violated"}]
    target["violation_num"][ki, si] = sum(status == "violated" for status in statuses)
    target["violation_den"][ki, si] = len(statuses)


def contributions(grouped: dict[str, list[dict[str, Any]]], gt: dict[str, set[tuple[Any, ...]]], gt_family: dict[str, dict[str, set[tuple[Any, ...]]]], subgraphs: list[str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    overall = empty_arrays(subgraphs)
    within = {family: empty_arrays(subgraphs) for family in FAMILIES}
    global_slice = {family: empty_arrays(subgraphs) for family in FAMILIES}
    for si, subgraph in enumerate(subgraphs):
        candidates = grouped.get(subgraph, [])
        for method in METHODS:
            ranked = sorted(candidates, key=lambda item: (-item["scores"][method], item["key"]))
            for ki, k in enumerate(KS):
                selected = ranked[:k]
                add_cell(overall[method], ki, si, selected, gt.get(subgraph, set()))
                for family in FAMILIES:
                    ranked_family = [item for item in ranked if item["family"] == family]
                    add_cell(within[family][method], ki, si, ranked_family[:k], gt_family.get(subgraph, {}).get(family, set()))
                    add_cell(global_slice[family][method], ki, si, [item for item in selected if item["family"] == family], gt_family.get(subgraph, {}).get(family, set()))
    return overall, within, global_slice


def ratio_samples(values: dict[str, np.ndarray], metric: str, ki: int, samples: np.ndarray) -> tuple[float | None, np.ndarray, int, int]:
    numerator, denominator = values[f"{metric}_num"][ki], values[f"{metric}_den"][ki]
    point = float(numerator.sum() / denominator.sum()) if denominator.sum() else None
    boot_num, boot_den = numerator[samples].sum(axis=1), denominator[samples].sum(axis=1)
    boot = np.divide(boot_num, boot_den, out=np.full_like(boot_num, np.nan), where=boot_den > 0)
    return point, boot, int(numerator.sum()), int(denominator.sum())


def ci(values: np.ndarray) -> list[float | None]:
    values = values[np.isfinite(values)]
    return [float(value) for value in np.percentile(values, (2.5, 97.5))] if len(values) else [None, None]


def summarize(values: dict[str, Any], samples: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    report: dict[str, Any] = {}
    cache: dict[str, Any] = {}
    for method in METHODS:
        report[method], cache[method] = {}, {}
        for ki, k in enumerate(KS):
            report[method][str(k)], cache[method][str(k)] = {}, {}
            for metric in ("recall", "violation"):
                point, boot, numerator, denominator = ratio_samples(values[method], metric, ki, samples)
                report[method][str(k)][metric] = {"point": point, "ci95": ci(boot), "numerator": numerator, "denominator": denominator}
                cache[method][str(k)][metric] = boot
    report["deltas_vs_semantic_only"] = {}
    for method in METHODS[1:]:
        report["deltas_vs_semantic_only"][method] = {}
        for k in KS:
            report["deltas_vs_semantic_only"][method][str(k)] = {}
            for metric in ("recall", "violation"):
                left, right = report[method][str(k)][metric]["point"], report["semantic_only"][str(k)][metric]["point"]
                delta_boot = cache[method][str(k)][metric] - cache["semantic_only"][str(k)][metric]
                report["deltas_vs_semantic_only"][method][str(k)][metric] = {
                    "point": left - right if left is not None and right is not None else None,
                    "paired_ci95": ci(delta_boot),
                }
    return report, cache


def add_familywise_ci(report: dict[str, Any], caches: dict[str, Any]) -> None:
    for method in METHODS[1:]:
        for k in KS:
            for metric in ("recall", "violation"):
                deltas, points, active = [], [], []
                for family in FAMILIES:
                    point = report[family]["deltas_vs_semantic_only"][method][str(k)][metric]["point"]
                    boot = caches[family][method][str(k)][metric] - caches[family]["semantic_only"][str(k)][metric]
                    if point is not None and np.any(np.isfinite(boot)):
                        deltas.append(boot)
                        points.append(point)
                        active.append(family)
                radius = None
                if deltas:
                    matrix = np.column_stack(deltas)
                    radius = float(np.nanpercentile(np.nanmax(np.abs(matrix - np.asarray(points)[None, :]), axis=1), 95.0))
                for family in FAMILIES:
                    item = report[family]["deltas_vs_semantic_only"][method][str(k)][metric]
                    item["simultaneous_familywise_ci95"] = [item["point"] - radius, item["point"] + radius] if radius is not None and family in active else [None, None]


def clustered(values: list[tuple[str, float]], subgraphs: list[str], samples: np.ndarray) -> dict[str, Any]:
    sums, counts = np.zeros(len(subgraphs)), np.zeros(len(subgraphs))
    index = {value: offset for offset, value in enumerate(subgraphs)}
    for subgraph, value in values:
        sums[index[subgraph]] += value
        counts[index[subgraph]] += 1
    boot_sum, boot_count = sums[samples].sum(axis=1), counts[samples].sum(axis=1)
    boot = np.divide(boot_sum, boot_count, out=np.full_like(boot_sum, np.nan), where=boot_count > 0)
    raw = np.asarray([value for _, value in values])
    return {
        "rows": len(values), "subgraphs": int(np.sum(counts > 0)),
        "mean": float(raw.mean()) if len(raw) else None,
        "median": float(np.median(raw)) if len(raw) else None,
        "p95": float(np.percentile(raw, 95)) if len(raw) else None,
        "paired_subgraph_ci95": ci(boot),
    }


def controls(grouped: dict[str, list[dict[str, Any]]], gt: dict[str, set[tuple[Any, ...]]], models: dict[str, Any], subgraphs: list[str], samples: np.ndarray) -> dict[str, Any]:
    all_models = {"family_specific": models["family_models"], **models["factor_models"]}
    by_tuple: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for subgraph, candidates in grouped.items():
        for row in candidates:
            by_tuple[(subgraph, row["subject"], row["object"], row["predicate"])] = row
    result: dict[str, Any] = {
        "wrong_T_on_GT_relative_vertical": {}, "close_by_swap": {},
        "vertical_inverse_equivariance": {}, "wrong_pair_geometry_continuity": {},
        "support_contact_endpoint_swap": {"status": "not_run_prohibited_by_frozen_protocol"},
    }
    for name in COMPAT_MODELS:
        correct_wrong, wins = [], []
        close_diffs, inverse_diffs, wrong_pair = [], [], []
        for subgraph, candidates in grouped.items():
            pair_raw: dict[tuple[int, int], dict[str, float]] = {}
            for row in candidates:
                pair_raw.setdefault((row["subject"], row["object"]), row["raw"])
                model = all_models[name][row["family"]] if name == "family_specific" else all_models[name]
                if row["family"] == "relative_vertical" and row["key"] in gt.get(subgraph, set()):
                    wrong_predicate = "lower than" if row["predicate"] == "higher than" else "higher than"
                    wrong = probability(model, row["family"], wrong_predicate, row["raw"])
                    difference = row["compat"][name] - wrong
                    correct_wrong.append((subgraph, difference))
                    wins.append((subgraph, float(difference > 0)))
                if row["predicate"] == "close by":
                    swapped = by_tuple.get((subgraph, row["object"], row["subject"], "close by"))
                    if swapped is not None:
                        close_diffs.append((subgraph, abs(row["compat"][name] - swapped["compat"][name])))
                if row["family"] == "relative_vertical":
                    inverse = "lower than" if row["predicate"] == "higher than" else "higher than"
                    swapped = by_tuple.get((subgraph, row["object"], row["subject"], inverse))
                    if swapped is not None:
                        inverse_diffs.append((subgraph, abs(row["compat"][name] - swapped["compat"][name])))
            ordered_pairs = sorted(pair_raw)
            if len(ordered_pairs) > 1:
                shifted = {pair: ordered_pairs[(offset + 1) % len(ordered_pairs)] for offset, pair in enumerate(ordered_pairs)}
                for row in candidates:
                    if row["key"] not in gt.get(subgraph, set()):
                        continue
                    model = all_models[name][row["family"]] if name == "family_specific" else all_models[name]
                    wrong = probability(model, row["family"], row["predicate"], pair_raw[shifted[(row["subject"], row["object"])]] )
                    wrong_pair.append((subgraph, row["compat"][name] - wrong))
        result["wrong_T_on_GT_relative_vertical"][name] = {
            "correct_minus_wrong": clustered(correct_wrong, subgraphs, samples),
            "correct_above_wrong_rate": clustered(wins, subgraphs, samples),
        }
        result["close_by_swap"][name] = {"absolute_difference": clustered(close_diffs, subgraphs, samples)}
        result["vertical_inverse_equivariance"][name] = {"absolute_difference": clustered(inverse_diffs, subgraphs, samples)}
        result["wrong_pair_geometry_continuity"][name] = {"correct_minus_wrong_pair": clustered(wrong_pair, subgraphs, samples)}
    result["wrong_pair_geometry_continuity"]["transform"] = "within-subgraph lexicographic directed-pair cyclic shift; T fixed; GT rows only"
    return result


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Train-only {report['role']} evaluation", "", f"Status: `{report['status']}`", "",
        "| method | Recall@100 | delta Recall | V@100 | delta V |", "| --- | ---: | ---: | ---: | ---: |",
    ]
    overall = report["overall"]
    for method in METHODS:
        cell = overall[method]["100"]
        delta = overall["deltas_vs_semantic_only"].get(method, {}).get("100", {})
        lines.append(
            f"| {method} | {cell['recall']['point']:.6f} | {delta.get('recall', {}).get('point', 0.0):.6f} | "
            f"{cell['violation']['point']:.6f} | {delta.get('violation', {}).get('point', 0.0):.6f} |"
        )
    lines.extend(["", f"Default product decision: `{report['default_product_decision']['decision']}`", ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    paths = {name: resolve(root, value) for name, value in {
        "verification": args.verification, "ground_truth": args.ground_truth,
        "models": args.models, "protocol": args.protocol,
    }.items()}
    if args.lock_manifest:
        paths["lock_manifest"] = resolve(root, args.lock_manifest)
    out = resolve(root, args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"nonempty_output:{out}")
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    models = json.loads(paths["models"].read_text(encoding="utf-8"))
    if protocol.get("status") != "protocol_frozen_before_strict_calibration_and_internal_dev_inference":
        raise ValueError("protocol_not_frozen")
    if models.get("schema_version") != "relcompat3d_strict_train_only_calibrators_v1":
        raise ValueError("strict_models_not_ready")
    if args.role == "final_validation":
        if "lock_manifest" not in paths:
            raise ValueError("final_validation_requires_lock_manifest")
        lock = json.loads(paths["lock_manifest"].read_text(encoding="utf-8"))
        if lock.get("status") not in {"final_method_locked_after_internal_dev_accept", "final_method_locked_after_internal_dev_reject"}:
            raise ValueError("final_method_not_locked")
        if lock["hashes"]["models_sha256"] != sha256_file(paths["models"]):
            raise ValueError("models_changed_after_lock")
    gt, gt_family = load_gt(paths["ground_truth"])
    grouped, input_rows = load_candidates(paths["verification"], models)
    subgraphs = sorted(set(grouped) | set(gt))
    rng = np.random.default_rng(args.seed)
    sample_indices = rng.integers(0, len(subgraphs), size=(args.n_bootstrap, len(subgraphs)))
    overall_values, within_values, slice_values = contributions(grouped, gt, gt_family, subgraphs)
    overall, overall_cache = summarize(overall_values, sample_indices)
    within, within_cache, global_slice, global_slice_cache = {}, {}, {}, {}
    for family in FAMILIES:
        within[family], within_cache[family] = summarize(within_values[family], sample_indices)
        global_slice[family], global_slice_cache[family] = summarize(slice_values[family], sample_indices)
    add_familywise_ci(within, within_cache)
    add_familywise_ci(global_slice, global_slice_cache)
    gate = overall["deltas_vs_semantic_only"]["family_product"]["100"]
    recall_ci, violation_ci = gate["recall"]["paired_ci95"], gate["violation"]["paired_ci95"]
    recall_pass = recall_ci[0] is not None and recall_ci[0] > -0.01
    violation_pass = violation_ci[1] is not None and violation_ci[1] < 0.0
    decision = "accept" if recall_pass and violation_pass else "reject"
    expected_seed = 20260711 if args.role == "internal_dev" else 20260712
    validations = {
        "contexts_expected": len(subgraphs) == args.expected_contexts,
        "gt_denominator_expected": sum(len(value) for value in gt.values()) == args.expected_gt_denominator,
        "all_methods_reported": set(METHODS) == set(overall) - {"deltas_vs_semantic_only"},
        "all_families_reported": set(within) == set(FAMILIES) == set(global_slice),
        "bootstrap_frozen": args.n_bootstrap == 1000 and args.seed == expected_seed,
        "final_lock_enforced_if_final": args.role != "final_validation" or "lock_manifest" in paths,
    }
    report = {
        "schema_version": "relcompat3d_train_only_source_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": f"{args.role}_evaluation_ready" if all(validations.values()) else f"blocked_{args.role}_evaluation",
        "role": args.role,
        "classification": protocol["classification"],
        "counts": {"input_rows": input_rows, "in_scope_rows": sum(len(value) for value in grouped.values()), "contexts": len(subgraphs), "gt_denominator": sum(len(value) for value in gt.values())},
        "methods": list(METHODS), "ks": list(KS), "overall": overall,
        "within_family": within, "global_topk_family_slice": global_slice,
        "controls": controls(grouped, gt, models, subgraphs, sample_indices),
        "default_product_decision": {
            "decision": decision, "recall_guardrail_pass": recall_pass,
            "violation_gate_pass": violation_pass, "delta_recall_at_100": gate["recall"],
            "delta_violation_at_100": gate["violation"],
            "rule": "accept iff paired delta Recall@100 CI lower > -0.01 and paired delta V@100 CI upper < 0",
        },
        "validations": validations,
        "limitations": [
            "Violation is verifier-derived, not an independent human physical-validity outcome.",
            "The constructed calibration labels are not independent human validity labels.",
            "This run is a leakage-controlled reconstruction, not untouched prospective confirmation.",
        ],
        "inputs": {name: {"path": relpath(root, path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "docker_command": f"env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm {args.docker_service}",
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", report)
    (out / "summary.md").write_text(markdown(report), encoding="utf-8")
    manifest = {
        "schema_version": report["schema_version"], "created_at_utc": report["created_at_utc"],
        "status": report["status"], "validations": validations,
        "outputs": {name: {"path": relpath(root, out / name), "sha256": sha256_file(out / name)} for name in ("summary.json", "summary.md")},
        "docker_command": report["docker_command"],
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"status": report["status"], "counts": report["counts"], "default_product_decision": report["default_product_decision"], "out": relpath(root, out)}))
    return 0 if all(validations.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
