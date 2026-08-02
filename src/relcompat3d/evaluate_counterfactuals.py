#!/usr/bin/env python3
"""Train-only one-factor counterfactual-policy sensitivity for RelCompat3D."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

import build_training_rows as exporter
import compatibility_features as calibration
import evaluate_feature_removal as heldout
import relation_consistency as algebra
import evaluate_main as evaluation
import evaluate_train_only as strict


ACTIVE_FAMILIES = ("proximity", "relative_vertical")
ALL_FAMILIES = ("support_contact", *ACTIVE_FAMILIES)
METRICS = ("recall", "violation_all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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
        if line.strip() and not line.lstrip().startswith("#")
    }


def linked_pairs(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    id_to_index = {row["candidate_id"]: index for index, row in enumerate(rows)}
    pairs: list[tuple[int, int]] = []
    for negative_index, row in enumerate(rows):
        base_id = (row.get("candidate_source") or {}).get("base_candidate_id")
        if row["_label"] == 0 and base_id in id_to_index:
            positive_index = id_to_index[base_id]
            if rows[positive_index]["_label"] == 1:
                pairs.append((positive_index, negative_index))
    return pairs


def fit_family_model(
    family: str,
    rows: list[dict[str, Any]],
    optimizer: dict[str, Any],
    remove_constant_family_indicator: bool = False,
) -> dict[str, Any]:
    train = [
        row
        for row in rows
        if row["_role"] == "train"
        and row["predicate"]["predicate_family"] == family
    ]
    spec = calibration.build_model_spec(train)
    if remove_constant_family_indicator:
        expected = f"family:{family}"
        family_features = [
            feature for feature in spec["feature_names"] if feature.startswith("family:")
        ]
        if family_features != [expected]:
            raise ValueError(
                f"expected_one_constant_family_indicator:{family}:{family_features}"
            )
        spec["feature_names"] = [
            feature for feature in spec["feature_names"] if feature != expected
        ]
        spec["families"] = []
    x = np.asarray(
        [
            algebra.existing_vector(
                spec,
                family,
                row["predicate"]["predicate_label"],
                row["_raw_numeric"],
            )
            for row in train
        ],
        dtype=np.float64,
    )
    y = np.asarray([row["_label"] for row in train], dtype=np.float64)
    pairs = linked_pairs(train)
    pair_diffs = np.asarray([x[pos] - x[neg] for pos, neg in pairs], dtype=np.float64)

    transformed_x = np.asarray(
        [
            algebra.existing_vector(spec, family, predicate, raw)
            for row in train
            for predicate, raw in [
                algebra.transformed_view(
                    family,
                    row["predicate"]["predicate_label"],
                    row["_raw_numeric"],
                )
            ]
        ],
        dtype=np.float64,
    )
    transformed_pair_diffs = np.asarray(
        [transformed_x[pos] - transformed_x[neg] for pos, neg in pairs],
        dtype=np.float64,
    )
    orbit_x = np.concatenate((x, transformed_x), axis=0)
    orbit_y = np.concatenate((y, y), axis=0)
    orbit_pair_diffs = np.concatenate((pair_diffs, transformed_pair_diffs), axis=0)
    weights, trace = algebra.fit_logistic(
        orbit_x,
        orbit_y,
        optimizer,
        pair_diffs=orbit_pair_diffs,
    )
    return {
        "architecture": "family_logistic_orbit_pairwise",
        "family": family,
        "feature_names": spec["feature_names"],
        "numeric_features": spec["numeric_features"],
        "numeric_stats": spec["numeric_stats"],
        "families": spec["families"],
        "predicates": spec["predicates"],
        "weights": weights.tolist(),
        "parameter_count": len(weights),
        "train_rows": len(train),
        "linked_pairs": len(pairs),
        "orbit_rows": len(orbit_y),
        "training_trace": trace,
    }


def projected_scorer(models: dict[str, Any]) -> Callable[[str, str, dict[str, float]], float]:
    return heldout.projected_scorer(models)


def ordering_diagnostic(
    prepared: list[dict[str, Any]],
    scorer: Callable[[str, str, dict[str, float]], float],
) -> dict[str, Any]:
    by_family: dict[str, Any] = {}
    all_margins: list[float] = []
    for family in ACTIVE_FAMILIES:
        rows = [
            row
            for row in prepared
            if row["_role"] == "dev"
            and row["predicate"]["predicate_family"] == family
        ]
        scores = {
            row["candidate_id"]: scorer(
                family,
                row["predicate"]["predicate_label"],
                row["_raw_numeric"],
            )
            for row in rows
        }
        margins: list[float] = []
        transform_errors: list[float] = []
        for row in rows:
            predicate = row["predicate"]["predicate_label"]
            raw = row["_raw_numeric"]
            transformed_predicate, transformed_raw = algebra.transformed_view(
                family, predicate, raw
            )
            transform_errors.append(
                abs(
                    scorer(family, predicate, raw)
                    - scorer(family, transformed_predicate, transformed_raw)
                )
            )
            base_id = (row.get("candidate_source") or {}).get("base_candidate_id")
            if row["_label"] == 0 and base_id in scores:
                margins.append(
                    algebra.logit(scores[base_id])
                    - algebra.logit(scores[row["candidate_id"]])
                )
        all_margins.extend(margins)
        by_family[family] = {
            "pairs": len(margins),
            "positive_win_rate": float(np.mean(np.asarray(margins) > 0.0)),
            "mean_logit_margin": float(np.mean(margins)),
            "max_orbit_error": max(transform_errors) if transform_errors else None,
        }
    return {
        "overall": {
            "pairs": len(all_margins),
            "positive_win_rate": float(np.mean(np.asarray(all_margins) > 0.0)),
            "mean_logit_margin": float(np.mean(all_margins)),
        },
        "by_family": by_family,
    }


def make_target_rows(
    positives: list[exporter.PositiveSpec],
    policy: dict[str, Any],
    relationship_id_map: dict[str, int],
    subset_source: str,
    created_at: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exporter.PROXIMITY_NORM_XY_MIN = float(policy["proximity_threshold"])
    exporter.VERTICAL_ABS_DELTA_Z_MIN = float(policy["vertical_abs_margin_m"])
    negative_rows, negative_records, skipped = exporter.generate_negatives(
        positives=positives,
        split_name="counterfactual_threshold_sensitivity_v1",
        subset_source=subset_source,
        relationship_id_map=relationship_id_map,
        created_at=created_at,
        max_negatives_per_positive=int(policy["negative_cap_per_positive"]),
        max_negatives_per_subgraph_family=int(policy["max_negatives_per_context_family"]),
        max_negative_to_positive_ratio_per_family=float(
            policy["max_negative_to_positive_ratio_per_family"]
        ),
    )
    rows = [spec.row for spec in positives] + negative_rows
    counts = Counter(row["predicate"]["predicate_family"] for row in rows)
    negative_counts = Counter(record["predicate_family"] for record in negative_records)
    return rows, {
        "rows": len(rows),
        "positive_rows": len(positives),
        "negative_rows": len(negative_rows),
        "rows_by_family": dict(sorted(counts.items())),
        "negative_rows_by_family": dict(sorted(negative_counts.items())),
        "skipped_attempts": dict(sorted(skipped.items())),
    }


def model_max_error(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for family in ACTIVE_FAMILIES:
        if left[family]["feature_names"] != right[family]["feature_names"]:
            result[family] = math.inf
            continue
        weight_error = max(
            abs(a - b)
            for a, b in zip(left[family]["weights"], right[family]["weights"])
        )
        stat_error = max(
            abs(left[family]["numeric_stats"][name][field] - right[family]["numeric_stats"][name][field])
            for name in left[family]["numeric_stats"]
            for field in ("mean", "std")
        )
        result[family] = max(weight_error, stat_error)
    return result


def write_metrics(path: Path, sources: dict[str, Any], conditions: tuple[str, ...]) -> None:
    rows: list[dict[str, Any]] = []
    for source, payload in sources.items():
        for condition in conditions:
            for k in evaluation.KS:
                cell = payload["scan_cluster"][condition][str(k)]
                rows.append(
                    {
                        "source": source,
                        "condition": condition,
                        "k": k,
                        "recall": cell["recall"]["point"],
                        "recall_ci_low": cell["recall"]["scan_cluster_ci95"][0],
                        "recall_ci_high": cell["recall"]["scan_cluster_ci95"][1],
                        "violation": cell["violation_all"]["point"],
                        "violation_ci_low": cell["violation_all"]["scan_cluster_ci95"][0],
                        "violation_ci_high": cell["violation_all"]["scan_cluster_ci95"][1],
                    }
                )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dynamic_scan_summary(
    values: dict[str, Any],
    weights: np.ndarray,
    conditions: tuple[str, ...],
) -> dict[str, Any]:
    methods = ("source_score", *conditions)
    report: dict[str, Any] = {method: {} for method in methods}
    cache: dict[str, Any] = {method: {} for method in methods}
    for method in methods:
        for ki, k in enumerate(evaluation.KS):
            report[method][str(k)], cache[method][str(k)] = {}, {}
            for metric in METRICS:
                numerator, denominator = evaluation.ratio_arrays(values[method], metric, ki)
                point = float(numerator.sum() / denominator.sum()) if denominator.sum() else None
                boot = heldout.scan_bootstrap.weighted_ratio(numerator, denominator, weights)
                report[method][str(k)][metric] = {
                    "point": point,
                    "scan_cluster_ci95": evaluation.ci95(boot),
                    "numerator": int(numerator.sum()),
                    "denominator": int(denominator.sum()),
                }
                cache[method][str(k)][metric] = boot
    report["deltas_vs_source_score"] = {}
    report["deltas_vs_default"] = {}
    for method in conditions:
        report["deltas_vs_source_score"][method] = {}
        report["deltas_vs_default"][method] = {}
        for k in evaluation.KS:
            report["deltas_vs_source_score"][method][str(k)] = {}
            report["deltas_vs_default"][method][str(k)] = {}
            for metric in METRICS:
                for reference, target in (
                    ("source_score", "deltas_vs_source_score"),
                    ("default", "deltas_vs_default"),
                ):
                    delta = cache[method][str(k)][metric] - cache[reference][str(k)][metric]
                    report[target][method][str(k)][metric] = {
                        "point": report[method][str(k)][metric]["point"]
                        - report[reference][str(k)][metric]["point"],
                        "paired_scan_cluster_ci95": evaluation.ci95(delta),
                    }
    return report


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Counterfactual-Policy Sensitivity",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Every row is a one-factor-at-a-time train-only target regeneration and refit. Cells are Recall / verifier V.",
        "",
        "| Condition | Dev order | VL-SAT K50/K100 | Open3DSG K50/K100 | SGFN K50/K100 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for condition in summary["condition_order"]:
        order = summary["conditions"][condition]["ordering"]["overall"]["positive_win_rate"]
        cells: list[str] = []
        for source in ("vlsat", "open3dsg", "sgfn"):
            values = summary["sources"][source]["scan_cluster"][condition]
            cells.append(
                f"{values['50']['recall']['point']:.4f}/{values['50']['violation_all']['point']:.4f} ; "
                f"{values['100']['recall']['point']:.4f}/{values['100']['violation_all']['point']:.4f}"
            )
        lines.append(f"| {condition} | {order:.4f} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Support/contact is passed through for every condition. No final-validation row enters target construction, normalization, or fitting.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    protocol_path = resolve(root, args.protocol)
    out = resolve(root, args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"nonempty_output:{out}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_counterfactual_sensitivity_execution":
        raise ValueError("protocol_not_frozen")
    paths = {name: resolve(root, value) for name, value in protocol["inputs"].items()}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing_inputs:{missing}")
    for name, expected in protocol["locked_sha256"].items():
        if sha256(paths[name]) != expected:
            raise ValueError(f"hash_mismatch:{name}")

    train_scans = read_scans(paths["train_scans"])
    dev_scans = read_scans(paths["internal_dev_scans"])
    final_scans = read_scans(paths["final_validation_scans"])
    selected_scans = read_scans(paths["method_development_scans"])
    if train_scans & dev_scans or train_scans & final_scans or dev_scans & final_scans:
        raise ValueError("split_firewall_overlap")
    if selected_scans != train_scans | dev_scans:
        raise ValueError("method_development_scan_union_mismatch")

    subset = json.loads(paths["train_annotations"].read_text(encoding="utf-8"))
    relationship_id_map = exporter.load_relationship_id_map(paths["relationships_file"])
    positives, _, skipped_positive, warnings, errors = exporter.build_contexts_and_positives(
        subset_data=subset,
        selected_scans=selected_scans,
        dataset_root=paths["dataset_root"],
        subset_source=str(paths["train_annotations"].relative_to(root)),
        split_name="counterfactual_threshold_sensitivity_v1",
        relationship_id_map=relationship_id_map,
        created_at=protocol["created_at_kst"],
        allow_selected_scans_without_positive_rows=True,
    )
    if errors:
        raise ValueError(f"positive_export_errors:{errors[:10]}")

    main_models = json.loads(paths["main_models"].read_text(encoding="utf-8"))
    main_orbit_models = main_models["attempts"]["orbit_pairwise"]
    condition_order = tuple(protocol["conditions"])
    fitted_models: dict[str, Any] = {}
    scorers: dict[str, Callable[[str, str, dict[str, float]], float]] = {}
    condition_details: dict[str, Any] = {}
    prepared_cache: dict[str, list[dict[str, Any]]] = {}
    target_cache: dict[str, dict[str, Any]] = {}

    for condition in condition_order:
        spec = protocol["conditions"][condition]
        target_key = spec["target_key"]
        if target_key not in prepared_cache:
            rows, target_counts = make_target_rows(
                positives,
                spec,
                relationship_id_map,
                str(paths["train_annotations"].relative_to(root)),
                protocol["created_at_kst"],
            )
            prepared, target_warnings = calibration.prepare_rows(
                rows, train_scans, dev_scans, set(ALL_FAMILIES)
            )
            prepared_cache[target_key] = prepared
            target_cache[target_key] = {
                **target_counts,
                "warnings": target_warnings,
            }
        prepared = prepared_cache[target_key]
        optimizer = {**protocol["optimizer"], "pairwise_weight": spec["pairwise_weight"]}
        models = {
            "support_contact": {
                **main_orbit_models["support_contact"],
                "role": "unused_support_contact_pass_through",
            }
        }
        for family in ACTIVE_FAMILIES:
            models[family] = fit_family_model(
                family,
                prepared,
                optimizer,
                bool(
                    protocol.get("parameterization", {}).get(
                        "remove_constant_family_indicator", False
                    )
                ),
            )
        scorer = projected_scorer(models)
        fitted_models[condition] = models
        scorers[condition] = scorer
        condition_details[condition] = {
            "policy": spec,
            "target_counts": target_cache[target_key],
            "ordering": ordering_diagnostic(prepared, scorer),
        }
        if target_key != "default":
            del prepared_cache[target_key]

    gt, gt_family = strict.load_gt(paths["ground_truth"])
    official = json.loads(paths["official_context_annotations"].read_text(encoding="utf-8"))
    official_contexts = {f"{row['scan']}_{row['split']}" for row in official["scans"]}
    source_paths = {
        "vlsat": paths["vlsat_verification"],
        "open3dsg": paths["open3dsg_verification"],
        "sgfn": paths["sgfn_verification"],
    }
    original_conditions, original_methods = heldout.CONDITIONS, heldout.METHODS
    original_scan_summary = heldout.scan_summary
    original_eval_methods = evaluation.METHODS
    heldout.CONDITIONS = condition_order
    heldout.METHODS = ("source_score", *condition_order)
    heldout.scan_summary = lambda values, weights: dynamic_scan_summary(
        values, weights, condition_order
    )
    evaluation.METHODS = heldout.METHODS
    try:
        sources = {
            source: heldout.evaluate_source(
                path,
                scorers,
                gt,
                gt_family,
                int(protocol["evaluation"]["bootstrap_seed"]) + index,
                int(protocol["evaluation"]["bootstrap_resamples"]),
                official_contexts if source == "open3dsg" else None,
            )
            for index, (source, path) in enumerate(source_paths.items())
        }
    finally:
        heldout.CONDITIONS, heldout.METHODS = original_conditions, original_methods
        heldout.scan_summary = original_scan_summary
        evaluation.METHODS = original_eval_methods

    default_error = model_max_error(
        fitted_models["default"], main_orbit_models
    )
    routing_reference = json.loads(paths["routing_summary"].read_text(encoding="utf-8"))
    open_reference = json.loads(paths["open3dsg_summary"].read_text(encoding="utf-8"))
    default_points_match = True
    for source, payload in sources.items():
        for k in evaluation.KS:
            for metric in METRICS:
                actual = payload["scan_cluster"]["default"][str(k)][metric]["point"]
                if source == "open3dsg":
                    expected = open_reference["routes"]["official_strict_full_548"]["overall"]["family_slot_rerank"][str(k)][metric]["point"]
                else:
                    expected = routing_reference["sources"][source]["overall"]["family_slot_rerank"][str(k)][metric]["point"]
                default_points_match &= abs(actual - expected) <= 1e-12

    validations = {
        "split_counts_1061_117_157": (len(train_scans), len(dev_scans), len(final_scans)) == (1061, 117, 157),
        "split_sets_pairwise_disjoint": not (train_scans & dev_scans or train_scans & final_scans or dev_scans & final_scans),
        "method_development_is_train_plus_dev": selected_scans == train_scans | dev_scans,
        "zero_final_validation_rows_in_targets": not ({spec.context.scan_id for spec in positives} & final_scans),
        "official_contexts_548": len(official_contexts) == 548,
        "gt_denominator_3972": sum(len(rows) for rows in gt.values()) == 3972,
        "condition_set_exact": set(condition_order) == set(protocol["expected_conditions"]),
        "default_model_matches_main": max(default_error.values()) <= 1e-12,
        "default_points_match_main": default_points_match,
        "all_orbit_errors_zero": all(
            detail["ordering"]["by_family"][family]["max_orbit_error"] <= 1e-12
            for detail in condition_details.values()
            for family in ACTIVE_FAMILIES
        ),
        "all_weights_finite": all(
            math.isfinite(weight)
            for models in fitted_models.values()
            for family in ACTIVE_FAMILIES
            for weight in models[family]["weights"]
        ),
        "no_source_score_or_identity_features": all(
            not any(token in feature.lower() for token in ("source", "semantic", "score", "rank", "identity"))
            for models in fitted_models.values()
            for family in ACTIVE_FAMILIES
            for feature in models[family]["feature_names"]
        ),
        "all_sources_157_scans_548_contexts": all(
            payload["counts"]["scans"] == 157
            and payload["counts"]["evaluation_contexts"] == 548
            for payload in sources.values()
        ),
    }
    status = "completed" if all(validations.values()) else "failed_validation"
    summary = {
        "schema_version": "relcompat3d_counterfactual_threshold_sensitivity_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "classification": "train_only_one_factor_at_a_time_sensitivity",
        "condition_order": list(condition_order),
        "conditions": condition_details,
        "sources": sources,
        "default_model_max_abs_error": default_error,
        "positive_export": {
            "rows": len(positives),
            "skipped": dict(sorted(skipped_positive.items())),
            "warnings": warnings,
        },
        "validations": validations,
        "claim_boundary": protocol["claim_boundary"],
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary)
    write_json(out / "models.json", {"conditions": fitted_models})
    write_metrics(out / "metrics.csv", sources, condition_order)
    (out / "summary.md").write_text(markdown(summary), encoding="utf-8")
    manifest = {
        "schema_version": "relcompat3d_counterfactual_threshold_sensitivity_manifest_v1",
        "created_at_utc": summary["created_at_utc"],
        "status": status,
        "validations": validations,
        "inputs": {
            name: {
                "path": str(path.relative_to(root)) if path.is_file() else str(path.relative_to(root)),
                **({"sha256": sha256(path)} if path.is_file() else {}),
            }
            for name, path in paths.items()
        },
        "outputs": {
            name: {"sha256": sha256(out / name)}
            for name in ("summary.json", "summary.md", "metrics.csv", "models.json")
        },
        "retained_row_level_variant_exports": False,
        "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm counterfactual_threshold_sensitivity",
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"status": status, "validations": validations, "out": str(out.relative_to(root))}))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
