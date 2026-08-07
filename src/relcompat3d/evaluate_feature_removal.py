#!/usr/bin/env python3
"""Refit and evaluate RelCompat3D compatibility with verifier primitives held out."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

import compatibility_features as calibration
import relation_consistency as algebra
import evaluate_all_families as evaluation
import evaluate_support_bootstrap as scan_bootstrap
import evaluate_base_models as model_eval


FAMILIES = ("support_contact", "proximity", "relative_vertical")
CONDITIONS = (
    "main_route",
    "exact_scalar_held_out",
    "primitive_family_held_out",
    "alternative_evidence_only",
)
METHODS = ("source", *CONDITIONS)
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


def relpath(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


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


def numeric_name(feature: str) -> str | None:
    return feature.split(":", 1)[1] if feature.startswith("num:") else None


def select_features(
    model: dict[str, Any],
    family: str,
    condition: str,
    rules: dict[str, Any],
) -> list[str]:
    source = list(model["feature_names"])
    spec = rules[condition][family]
    excluded = set(spec.get("exclude_numeric", []))
    allowed = set(spec.get("allow_numeric", []))
    selected: list[str] = []
    for feature in source:
        name = numeric_name(feature)
        if name is None:
            selected.append(feature)
        elif allowed:
            if name in allowed:
                selected.append(feature)
        elif name not in excluded:
            selected.append(feature)
    if "bias" not in selected:
        raise ValueError(f"missing_bias:{condition}:{family}")
    return selected


def reduced_spec(base_model: dict[str, Any], features: list[str]) -> dict[str, Any]:
    numeric = [name for feature in features if (name := numeric_name(feature))]
    return {
        "feature_names": features,
        "numeric_stats": {name: base_model["numeric_stats"][name] for name in numeric},
        "weights": [0.0] * len(features),
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


def fit_one_model(
    family: str,
    rows: list[dict[str, Any]],
    base_model: dict[str, Any],
    features: list[str],
    optimizer: dict[str, Any],
) -> dict[str, Any]:
    spec = reduced_spec(base_model, features)
    x = np.asarray(
        [
            algebra.existing_vector(
                spec,
                family,
                row["predicate"]["predicate_label"],
                row["_raw_numeric"],
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    y = np.asarray([row["_label"] for row in rows], dtype=np.float64)
    pairs = linked_pairs(rows)
    pair_diffs = np.asarray([x[pos] - x[neg] for pos, neg in pairs], dtype=np.float64)

    transformed_vectors: list[np.ndarray] = []
    for row in rows:
        transformed = algebra.transformed_view(
            family, row["predicate"]["predicate_label"], row["_raw_numeric"]
        )
        if transformed is None:
            raise ValueError(f"missing_declared_transform:{family}")
        predicate, raw = transformed
        transformed_vectors.append(algebra.existing_vector(spec, family, predicate, raw))
    transformed_x = np.asarray(transformed_vectors, dtype=np.float64)
    orbit_x = np.concatenate((x, transformed_x), axis=0)
    orbit_y = np.concatenate((y, y), axis=0)
    transformed_pair_diffs = np.asarray(
        [transformed_x[pos] - transformed_x[neg] for pos, neg in pairs],
        dtype=np.float64,
    )
    orbit_pair_diffs = np.concatenate((pair_diffs, transformed_pair_diffs), axis=0)
    weights, trace = algebra.fit_logistic(
        orbit_x, orbit_y, optimizer, pair_diffs=orbit_pair_diffs
    )
    return {
        "architecture": "feature_removal_family_logistic",
        "family": family,
        "feature_names": features,
        "numeric_stats": spec["numeric_stats"],
        "weights": weights.tolist(),
        "training_trace": trace,
        "parameter_count": len(weights),
        "train_rows": len(rows),
        "linked_pairs": len(pairs),
        "orbit_rows": len(orbit_y),
    }


def projected_scorer(models: dict[str, Any]) -> Callable[[str, str, dict[str, float]], float]:
    def score(family: str, predicate: str, raw: dict[str, float]) -> float:
        model = models[family]
        direct = algebra.existing_probability(model, family, predicate, raw)
        transformed = algebra.transformed_view(family, predicate, raw)
        if transformed is None:
            return direct
        transformed_predicate, transformed_raw = transformed
        inverse = algebra.existing_probability(
            model, family, transformed_predicate, transformed_raw
        )
        return 0.5 * (direct + inverse)

    return score


def fit_conditions(
    prepared: list[dict[str, Any]],
    base_models: dict[str, Any],
    main_models: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Callable[[str, str, dict[str, float]], float]]]:
    train = [row for row in prepared if row["_role"] == "train"]
    main_orbit = main_models["attempts"]["orbit_pairwise"]
    model_sets: dict[str, Any] = {}
    scorers: dict[str, Callable[[str, str, dict[str, float]], float]] = {
        "main_route": projected_scorer(main_orbit)
    }
    for condition in CONDITIONS[1:]:
        models: dict[str, Any] = {
            "support_contact": {
                **main_orbit["support_contact"],
                "role": "unused_by_primary_route_support_contact_passthrough",
            }
        }
        for family in ("proximity", "relative_vertical"):
            rows = [
                row
                for row in train
                if row["predicate"]["predicate_family"] == family
            ]
            base_model = base_models["family_models"][family]
            features = select_features(
                base_model, family, condition, protocol["held_out_conditions"]
            )
            models[family] = fit_one_model(
                family, rows, base_model, features, protocol["optimizer"]
            )
        model_sets[condition] = models
        scorers[condition] = projected_scorer(models)
    return model_sets, scorers


def calibration_diagnostics(
    prepared: list[dict[str, Any]],
    scorers: dict[str, Callable[[str, str, dict[str, float]], float]],
) -> dict[str, Any]:
    dev = [row for row in prepared if row["_role"] == "dev"]
    result: dict[str, Any] = {}
    for condition, scorer in scorers.items():
        result[condition] = {"by_family": {}, "transformation": {}}
        id_scores: dict[str, float] = {}
        for family in ("proximity", "relative_vertical"):
            rows = [row for row in dev if row["predicate"]["predicate_family"] == family]
            probabilities: list[float] = []
            labels: list[int] = []
            errors: list[float] = []
            for row in rows:
                predicate = row["predicate"]["predicate_label"]
                raw = row["_raw_numeric"]
                value = scorer(family, predicate, raw)
                probabilities.append(value)
                labels.append(int(row["_label"]))
                id_scores[f"{condition}:{row['candidate_id']}"] = value
                transformed = algebra.transformed_view(family, predicate, raw)
                if transformed is not None:
                    transformed_predicate, transformed_raw = transformed
                    errors.append(
                        abs(value - scorer(family, transformed_predicate, transformed_raw))
                    )
            result[condition]["by_family"][family] = algebra.calibration_metrics(
                probabilities, labels
            )
            result[condition]["transformation"][family] = {
                "rows": len(errors),
                "max_abs_error": max(errors) if errors else None,
                "mean_abs_error": float(np.mean(errors)) if errors else None,
            }
        margins: list[float] = []
        for row in dev:
            family = row["predicate"]["predicate_family"]
            if family not in {"proximity", "relative_vertical"}:
                continue
            base_id = (row.get("candidate_source") or {}).get("base_candidate_id")
            key = f"{condition}:{base_id}"
            row_key = f"{condition}:{row['candidate_id']}"
            if row["_label"] == 0 and key in id_scores:
                margins.append(algebra.logit(id_scores[key]) - algebra.logit(id_scores[row_key]))
        result[condition]["linked_counterfactual"] = {
            "pairs": len(margins),
            "positive_win_rate": float(np.mean(np.asarray(margins) > 0.0)),
            "mean_logit_margin": float(np.mean(margins)),
        }
    return result


def load_candidates(
    path: Path,
    scorers: dict[str, Callable[[str, str, dict[str, float]], float]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    digest = hashlib.sha256()
    input_rows = 0
    in_scope_rows = 0
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
            predicate = row["predicate"]["predicate_label"]
            raw = model_eval.raw_numeric(row)
            semantic = model_eval.finite((row.get("semantic") or {}).get("ranking_score"))
            if semantic is None:
                raise ValueError(f"missing_semantic:{row['prediction_id']}")
            compatibility = {
                condition: scorer(family, predicate, raw)
                for condition, scorer in scorers.items()
            }
            grouped[row["subgraph_id"]].append(
                {
                    "id": row["prediction_id"],
                    "scan": row["scan_id"],
                    "key": model_eval.candidate_key(row),
                    "family": family,
                    "semantic": float(semantic),
                    "compatibility": compatibility,
                    "status": row.get("verification_status")
                    or (row.get("verification") or {}).get("verification_status"),
                    "scores": {"source": float(semantic)},
                }
            )
    for candidates in grouped.values():
        source_order = sorted(candidates, key=lambda item: (-item["semantic"], item["key"]))
        for condition in CONDITIONS:
            queues: dict[str, list[dict[str, Any]]] = {}
            for family in FAMILIES:
                family_rows = [item for item in candidates if item["family"] == family]
                if family == "support_contact":
                    queues[family] = sorted(
                        family_rows, key=lambda item: (-item["semantic"], item["key"])
                    )
                else:
                    queues[family] = sorted(
                        family_rows,
                        key=lambda item: (
                            -item["semantic"] * item["compatibility"][condition],
                            item["key"],
                        ),
                    )
            offsets = {family: 0 for family in FAMILIES}
            ranked: list[dict[str, Any]] = []
            for source_item in source_order:
                family = source_item["family"]
                ranked.append(queues[family][offsets[family]])
                offsets[family] += 1
            size = len(ranked)
            for rank, item in enumerate(ranked, 1):
                item["scores"][condition] = float(size - rank + 1)
    return grouped, {
        "input_rows": input_rows,
        "in_scope_rows": in_scope_rows,
        "candidate_contexts": len(grouped),
        "input_sha256": digest.hexdigest(),
    }


def scan_summary(values: dict[str, Any], weights: np.ndarray) -> dict[str, Any]:
    report: dict[str, Any] = {method: {} for method in METHODS}
    cache: dict[str, Any] = {method: {} for method in METHODS}
    for method in METHODS:
        for ki, k in enumerate(evaluation.KS):
            report[method][str(k)] = {}
            cache[method][str(k)] = {}
            for metric in METRICS:
                numerator, denominator = evaluation.ratio_arrays(values[method], metric, ki)
                point = float(numerator.sum() / denominator.sum()) if denominator.sum() else None
                boot = scan_bootstrap.weighted_ratio(numerator, denominator, weights)
                report[method][str(k)][metric] = {
                    "point": point,
                    "bootstrap_intervals_ci95": evaluation.ci95(boot),
                    "numerator": int(numerator.sum()),
                    "denominator": int(denominator.sum()),
                }
                cache[method][str(k)][metric] = boot
    report["deltas_vs_source_score"] = {}
    report["deltas_vs_main_route"] = {}
    for method in CONDITIONS:
        report["deltas_vs_source_score"][method] = {}
        report["deltas_vs_main_route"][method] = {}
        for k in evaluation.KS:
            report["deltas_vs_source_score"][method][str(k)] = {}
            report["deltas_vs_main_route"][method][str(k)] = {}
            for metric in METRICS:
                for reference, target in (
                    ("source", "deltas_vs_source_score"),
                    ("main_route", "deltas_vs_main_route"),
                ):
                    point = (
                        report[method][str(k)][metric]["point"]
                        - report[reference][str(k)][metric]["point"]
                    )
                    delta = cache[method][str(k)][metric] - cache[reference][str(k)][metric]
                    report[target][method][str(k)][metric] = {
                        "point": point,
                        "paired_bootstrap_intervals_ci95": evaluation.ci95(delta),
                    }
    return report


def evaluate_source(
    path: Path,
    scorers: dict[str, Callable[[str, str, dict[str, float]], float]],
    gt: dict[str, set[tuple[Any, ...]]],
    gt_family: dict[str, dict[str, set[tuple[Any, ...]]]],
    seed: int,
    resamples: int,
    context_universe: set[str] | None = None,
) -> dict[str, Any]:
    grouped, counts = load_candidates(path, scorers)
    contexts = sorted(context_universe if context_universe is not None else set(grouped) | set(gt))
    context_samples = np.random.default_rng(seed).integers(
        0, len(contexts), size=(resamples, len(contexts))
    )
    overall_values, within_values, global_values = evaluation.contributions(
        grouped, gt, gt_family, contexts
    )
    overall, _ = evaluation.summarize(overall_values, context_samples)
    within: dict[str, Any] = {}
    global_slice: dict[str, Any] = {}
    for family in FAMILIES:
        within[family], _ = evaluation.summarize(within_values[family], context_samples)
        global_slice[family], _ = evaluation.summarize(global_values[family], context_samples)
    weights, cluster_counts = scan_bootstrap.scan_weights(
        grouped, contexts, resamples, seed
    )
    return {
        "counts": {
            **counts,
            "evaluation_contexts": len(contexts),
            "gt_denominator": sum(len(rows) for rows in gt.values()),
            **cluster_counts,
        },
        "overall": overall,
        "bootstrap_intervals": scan_summary(overall_values, weights),
        "within_family": within,
        "global_topk_family_slice": global_slice,
    }


def csv_rows(sources: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, payload in sources.items():
        for method in METHODS:
            for k in evaluation.KS:
                cell = payload["bootstrap_intervals"][method][str(k)]
                rows.append(
                    {
                        "source": source,
                        "method": method,
                        "k": k,
                        "recall": cell["recall"]["point"],
                        "recall_ci_low": cell["recall"]["bootstrap_intervals_ci95"][0],
                        "recall_ci_high": cell["recall"]["bootstrap_intervals_ci95"][1],
                        "violation": cell["violation_all"]["point"],
                        "violation_ci_low": cell["violation_all"]["bootstrap_intervals_ci95"][0],
                        "violation_ci_high": cell["violation_all"]["bootstrap_intervals_ci95"][1],
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(summary: dict[str, Any]) -> str:
    labels = {
        "main_route": "Full RelCompat3D",
        "exact_scalar_held_out": "Exact verifier scalar held out",
        "primitive_family_held_out": "Verifier primitive family held out",
        "alternative_evidence_only": "Alternative evidence only",
    }
    lines = [
        "# Held-out Geometry-Primitive Evaluation",
        "",
        f"Status: `{summary['status']}`",
        "",
        "All variants are refitted on the 1,061-scan training split, use no source score inside compatibility, retain exact orbit projection, and use the same family-slot route. Support/contact is unchanged.",
        "",
        "## K=50 overall",
        "",
        "| Source | Condition | Recall | verifier V | delta R vs source | delta V vs source |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for source in ("vlsat", "open3dsg", "sgfn"):
        payload = summary["sources"][source]["bootstrap_intervals"]
        for method in CONDITIONS:
            cell = payload[method]["50"]
            delta = payload["deltas_vs_source_score"][method]["50"]
            lines.append(
                f"| {source} | {labels[method]} | {cell['recall']['point']:.4f} | "
                f"{cell['violation_all']['point']:.4f} | {delta['recall']['point']:+.4f} | "
                f"{delta['violation_all']['point']:+.4f} |"
            )
    lines.extend(
        [
            "",
            "The exact-scalar condition removes the normalized scalar consumed by the corresponding verifier. The primitive-family condition also removes raw or deterministically related measurements. The alternative-evidence condition retains overlap-only proximity evidence and horizontal-distance/overlap vertical context, so it cannot reconstruct the verifier's directed vertical scalar.",
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
    if protocol.get("status") != "ready":
        raise ValueError("protocol_version_mismatch")
    paths = {name: resolve(root, value) for name, value in protocol["inputs"].items()}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing_inputs:{missing}")
    if sha256(paths["main_models"]) != protocol["expected_hashes"]["main_models_sha256"]:
        raise ValueError("main_model_hash_mismatch")
    if sha256(paths["base_models"]) != protocol["expected_hashes"]["base_models_sha256"]:
        raise ValueError("base_model_hash_mismatch")

    train_scans = read_scans(paths["train_scans"])
    dev_scans = read_scans(paths["development_scans"])
    final_scans = read_scans(paths["final_validation_scans"])
    if train_scans & dev_scans or train_scans & final_scans or dev_scans & final_scans:
        raise ValueError("data_split_overlap")
    table_rows = calibration.load_jsonl(paths["calibration_table"])
    leaked = sorted({row["scan_id"] for row in table_rows} & final_scans)
    if leaked:
        raise ValueError(f"final_validation_rows_in_calibration:{leaked[:10]}")
    prepared, warnings = calibration.prepare_rows(
        table_rows, train_scans, dev_scans, set(FAMILIES)
    )
    base_models = json.loads(paths["base_models"].read_text(encoding="utf-8"))
    main_models = json.loads(paths["main_models"].read_text(encoding="utf-8"))
    held_out_models, scorers = fit_conditions(
        prepared, base_models, main_models, protocol
    )
    diagnostics = calibration_diagnostics(prepared, scorers)

    gt, gt_family = model_eval.load_gt(paths["ground_truth"])
    official_annotations = json.loads(
        paths["official_context_annotations"].read_text(encoding="utf-8")
    )
    official_contexts = {
        f"{row['scan']}_{row['split']}" for row in official_annotations["scans"]
    }
    source_paths = {
        "vlsat": paths["vlsat_verification"],
        "open3dsg": paths["open3dsg_official_verification"],
        "sgfn": paths["sgfn_verification"],
    }
    original_methods = evaluation.METHODS
    evaluation.METHODS = METHODS
    try:
        sources = {
            source: evaluate_source(
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
        evaluation.METHODS = original_methods

    expected = {
        "vlsat": protocol["expected_counts"]["vlsat_in_scope_rows"],
        "open3dsg": protocol["expected_counts"]["open3dsg_in_scope_rows"],
        "sgfn": protocol["expected_counts"]["sgfn_in_scope_rows"],
    }
    routing_reference = json.loads(paths["routing_summary"].read_text(encoding="utf-8"))
    open3dsg_reference = json.loads(
        paths["open3dsg_official_summary"].read_text(encoding="utf-8")
    )
    main_route_matches_reference = True
    for source, payload in sources.items():
        for k in evaluation.KS:
            for metric in METRICS:
                actual = payload["bootstrap_intervals"]["main_route"][str(k)][metric]["point"]
                if source == "open3dsg":
                    expected_point = open3dsg_reference["routes"]["official_full_548"]["overall"]["family_slot_rerank"][str(k)][metric]["point"]
                else:
                    expected_point = routing_reference["sources"][source]["overall"]["family_slot_rerank"][str(k)][metric]["point"]
                main_route_matches_reference &= abs(actual - expected_point) <= 1e-12
    exact_scalar_absent = all(
        not any(
            numeric_name(feature) in set(protocol["held_out_conditions"]["exact_scalar_held_out"][family]["exclude_numeric"])
            for feature in held_out_models["exact_scalar_held_out"][family]["feature_names"]
        )
        for family in ("proximity", "relative_vertical")
    )
    primitive_family_absent = all(
        not any(
            numeric_name(feature) in set(protocol["held_out_conditions"]["primitive_family_held_out"][family]["exclude_numeric"])
            for feature in held_out_models["primitive_family_held_out"][family]["feature_names"]
        )
        for family in ("proximity", "relative_vertical")
    )
    validations = {
        "split_counts_1061_117_157": (len(train_scans), len(dev_scans), len(final_scans)) == (1061, 117, 157),
        "split_sets_pairwise_disjoint": not (train_scans & dev_scans or train_scans & final_scans or dev_scans & final_scans),
        "zero_final_validation_rows_in_fit": not leaked,
        "train_rows_60208": sum(row["_role"] == "train" for row in prepared) == 60208,
        "development_rows_6246": sum(row["_role"] == "dev" for row in prepared) == 6246,
        "exact_verifier_scalars_absent": exact_scalar_absent,
        "primitive_families_absent": primitive_family_absent,
        "no_source_features": all(
            not any(token in feature.lower() for token in ("source", "semantic", "score", "rank"))
            for models in held_out_models.values()
            for family in ("proximity", "relative_vertical")
            for feature in models[family]["feature_names"]
        ),
        "all_weights_finite": all(
            math.isfinite(weight)
            for models in held_out_models.values()
            for family in ("proximity", "relative_vertical")
            for weight in models[family]["weights"]
        ),
        "exact_projection_all_conditions": all(
            diagnostics[condition]["transformation"][family]["max_abs_error"] <= 1e-12
            for condition in CONDITIONS
            for family in ("proximity", "relative_vertical")
        ),
        "all_sources_548_contexts_157_scans": all(
            payload["counts"]["evaluation_contexts"] == 548
            and payload["counts"]["scans"] == 157
            for payload in sources.values()
        ),
        "all_sources_gt_denominator_3972": all(
            payload["counts"]["gt_denominator"] == 3972 for payload in sources.values()
        ),
        "source_row_counts": all(
            sources[source]["counts"]["in_scope_rows"] == count
            for source, count in expected.items()
        ),
        "main_results_match_reported_results": main_route_matches_reference,
        "all_k_and_conditions_reported": all(
            set(payload["bootstrap_intervals"][method]) == {str(k) for k in evaluation.KS}
            for payload in sources.values()
            for method in METHODS
        ),
    }
    status = "completed" if all(validations.values()) else "failed_validation"
    summary = {
        "schema_version": "relcompat3d_relcompat3d_feature_removal_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "classification": protocol["classification"],
        "warnings": warnings,
        "conditions": protocol["held_out_conditions"],
        "diagnostics": diagnostics,
        "sources": sources,
        "validations": validations,
        "evaluation_scope": protocol["evaluation_scope"],
    }
    out.mkdir(parents=True, exist_ok=True)
    models_path = out / "models.json"
    summary_path = out / "summary.json"
    markdown_path = out / "summary.md"
    metrics_path = out / "metrics.csv"
    write_json(models_path, {
        "schema_version": "relcompat3d_feature_removal_models_v1",
        "models": held_out_models,
        "source_score_input": False,
        "source_identity_input": False,
    })
    write_json(summary_path, summary)
    markdown_path.write_text(markdown(summary), encoding="utf-8")
    write_csv(metrics_path, csv_rows(sources))
    outputs = (models_path, summary_path, markdown_path, metrics_path)
    manifest = {
        "schema_version": "relcompat3d_feature_removal_manifest_v1",
        "created_at_utc": summary["created_at_utc"],
        "status": status,
        "protocol": {"path": relpath(root, protocol_path), "sha256": sha256(protocol_path)},
        "inputs": {
            name: {"path": relpath(root, path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for name, path in paths.items()
        },
        "outputs": {
            path.name: {"path": relpath(root, path), "sha256": sha256(path)} for path in outputs
        },
        "validations": validations,
        "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm relcompat3d_feature_removal",
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"status": status, "validations": validations, "out": relpath(root, out)}))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
