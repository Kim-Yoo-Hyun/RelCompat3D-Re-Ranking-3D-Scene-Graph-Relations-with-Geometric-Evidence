#!/usr/bin/env python3
"""Apply the fixed point/mesh audit to RelCompat3D-MLP selections."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import audit_point_mesh as audit
import evaluate_all_families as base
import fit_mlp as nonlinear
import evaluate_base_models as model_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_measurements(path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    result: dict[tuple[str, int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["scan_id"]), int(row["subject_id"]), int(row["object_id"]))
            result[key] = row
    return result


def add_mlp_ranking(grouped: dict[str, list[dict[str, Any]]]) -> None:
    for candidates in grouped.values():
        source_order = sorted(
            candidates, key=lambda row: (-row["scores"]["source"], row["key"])
        )
        queues: dict[str, list[dict[str, Any]]] = {}
        for family in base.FAMILIES:
            rows = [row for row in candidates if row["family"] == family]
            score_name = (
                "source"
                if family == "support_contact"
                else "shared_mlp_pairwise_product"
            )
            queues[family] = sorted(
                rows, key=lambda row: (-row["scores"][score_name], row["key"])
            )
        offsets = {family: 0 for family in base.FAMILIES}
        output: list[dict[str, Any]] = []
        for row in source_order:
            family = row["family"]
            output.append(queues[family][offsets[family]])
            offsets[family] += 1
        for rank, row in enumerate(output, 1):
            row["scores"]["relcompat3d"] = float(len(output) - rank + 1)


def load_rankings(
    path: Path,
    contexts: list[str],
    linear_scorer: Any,
    bce_model: dict[str, Any],
    mlp_model: dict[str, Any],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    grouped, counts = nonlinear.load_candidates(path, linear_scorer, bce_model, mlp_model)
    add_mlp_ranking(grouped)
    rankings: dict[str, dict[str, list[dict[str, Any]]]] = {
        method: {} for method in audit.METHODS
    }
    for context in contexts:
        rows = grouped.get(context, [])
        source = sorted(
            rows, key=lambda row: (-row["scores"]["source"], row["key"])
        )[:100]
        reranked = sorted(
            rows, key=lambda row: (-row["scores"]["relcompat3d"], row["key"])
        )[:100]
        rankings["source"][context] = [audit.lightweight(row, context) for row in source]
        rankings["relcompat3d"][context] = [
            audit.lightweight(row, context) for row in reranked
        ]
    return rankings, {
        **counts,
        "contexts": len(contexts),
        "prediction_contexts": len(grouped),
        "zero_prediction_contexts": len(set(contexts) - set(grouped)),
    }


def read_mechanism_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {
                "source": row["source"],
                "id": row["prediction_id"],
                "family": row["family"],
                "predicate": row["predicate"],
            }
            for row in csv.DictReader(handle)
        ]


def load_case_rows(
    path: Path, target_ids: set[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, float]]]:
    identities: dict[str, dict[str, Any]] = {}
    features: dict[str, dict[str, float]] = {}
    if not target_ids:
        return identities, features
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prediction_id = str(row["prediction_id"])
            if prediction_id not in target_ids:
                continue
            identities[prediction_id] = {
                "scan_id": str(row["scan_id"]),
                "subject_id": int(row["edge"]["subject_id"]),
                "object_id": int(row["edge"]["object_id"]),
            }
            features[prediction_id] = model_eval.raw_numeric(row)
            if len(identities) == len(target_ids):
                break
    return identities, features


def mlp_mechanism_test(
    cases: list[dict[str, Any]],
    measurements: dict[tuple[str, int, int], dict[str, Any]],
    rankings: dict[str, Any],
    source_paths: dict[str, Path],
    model: dict[str, Any],
    levels: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    del rankings
    identities: dict[tuple[str, str], dict[str, Any]] = {}
    raw_features: dict[tuple[str, str], dict[str, float]] = {}
    for source, path in source_paths.items():
        ids = {row["id"] for row in cases if row["source"] == source}
        source_identities, source_features = load_case_rows(path, ids)
        for prediction_id, identity in source_identities.items():
            identities[(source, prediction_id)] = identity
        for prediction_id, raw in source_features.items():
            raw_features[(source, prediction_id)] = raw

    rows: list[dict[str, Any]] = []
    for case in cases:
        ranking_row = identities.get((case["source"], case["id"]))
        raw = raw_features.get((case["source"], case["id"]))
        monotone_flag = None
        endpoint_change = None
        if ranking_row is not None and raw is not None:
            pair_key = (
                ranking_row["scan_id"],
                ranking_row["subject_id"],
                ranking_row["object_id"],
            )
            measurement = measurements.get(pair_key)
            if measurement is not None and measurement["point"].get("available"):
                scale = float(measurement["point"]["pair_scale_m"])
                sequence = [
                    nonlinear.projected_probability(
                        model,
                        case["family"],
                        case["predicate"],
                        audit.intervention_raw(
                            raw, case["family"], case["predicate"], level, scale
                        ),
                    )
                    for level in levels
                ]
                direction = (
                    "nonincreasing" if case["family"] == "proximity" else "nondecreasing"
                )
                monotone_flag = audit.monotone(sequence, direction)
                endpoint_change = float(sequence[-1] - sequence[0])
        rows.append(
            {
                "source": case["source"],
                "prediction_id": case["id"],
                "family": case["family"],
                "predicate": case["predicate"],
                "compatibility_monotone": monotone_flag,
                "compatibility_endpoint_change": endpoint_change,
            }
        )

    summary: dict[str, Any] = {"levels_in_pair_scale_units": levels, "families": {}}
    for family in audit.FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        flags = [
            row["compatibility_monotone"]
            for row in family_rows
            if row["compatibility_monotone"] is not None
        ]
        changes = [
            row["compatibility_endpoint_change"]
            for row in family_rows
            if row["compatibility_endpoint_change"] is not None
        ]
        summary["families"][family] = {
            "selected_cases": len(family_rows),
            "covered_cases": len(flags),
            "monotonicity_rate": float(np.mean(flags)) if flags else None,
            "mean_endpoint_change": float(np.mean(changes)) if changes else None,
            "median_endpoint_change": float(np.median(changes)) if changes else None,
        }
    return summary, rows


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RelCompat3D-MLP Surface Audit",
        "",
        f"Status: `{summary['status']}`",
        "",
        "The fixed point, mesh, and strict-consensus statuses are applied to RelCompat3D-MLP selections. Their absolute values are not directly comparable to the primary OBB-derived Violation metric.",
        "",
        "| Predictor | K | Source consensus V | RelCompat3D-MLP consensus V | Change (95% scan-cluster CI) | MLP coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source, payload in summary["results"].items():
        for k in audit.KS:
            cells = payload["audits"]["consensus"]
            source_cell = cells["source"][str(k)]
            mlp_cell = cells["relcompat3d"][str(k)]
            delta = cells["relcompat3d_minus_source"][str(k)]["violation"]
            ci = delta["paired_bootstrap_intervals_ci95"]
            lines.append(
                f"| {source} | {k} | {source_cell['violation']['point']:.4f} | "
                f"{mlp_cell['violation']['point']:.4f} | {delta['point']:+.4f} "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {mlp_cell['coverage']['point']:.4f} |"
            )
    lines.extend(["", "The audit is an automatic raw-surface construct check, not human physical-validity ground truth.", ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    protocol_path = audit.resolve(root, args.protocol)
    out = audit.resolve(root, args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"nonempty_output:{out}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "ready":
        raise ValueError("protocol_version_mismatch")
    if tuple(protocol["scope"]["ks"]) != audit.KS:
        raise ValueError("rank_cutoffs_mismatch")

    paths = {name: audit.resolve(root, spec["path"]) for name, spec in protocol["inputs"].items()}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing_inputs:{missing}")
    input_checks: dict[str, Any] = {}
    for name, spec in protocol["inputs"].items():
        path = paths[name]
        if path.is_file():
            actual = audit.sha256_file(path)
            if spec.get("sha256") and actual != spec["sha256"]:
                raise ValueError(f"input_hash_mismatch:{name}:{actual}")
            input_checks[name] = {
                "path": audit.relpath(root, path),
                "sha256": actual,
                "size_bytes": path.stat().st_size,
            }
        else:
            input_checks[name] = {"path": audit.relpath(root, path), "type": "directory"}

    linear_models = json.loads(paths["linear_models"].read_text(encoding="utf-8"))
    nonlinear_models = json.loads(paths["nonlinear_models"].read_text(encoding="utf-8"))
    mlp_model = nonlinear_models["shared_mlp_pairwise"]
    bce_model = nonlinear_models["shared_mlp_bce"]
    feature_spec = mlp_model["feature_spec"]
    linear_scorer = base.make_linear_scorer(linear_models)
    annotations = json.loads(paths["official_context_annotations"].read_text(encoding="utf-8"))
    contexts = sorted({f"{row['scan']}_{row['split']}" for row in annotations["scans"]})
    scans = sorted({context.rsplit("_", 1)[0] for context in contexts})
    source_paths = {
        "vlsat": paths["vlsat_verification"],
        "open3dsg": paths["open3dsg_verification"],
        "sgfn": paths["sgfn_verification"],
    }
    rankings: dict[str, Any] = {}
    source_counts: dict[str, Any] = {}
    for source, path in source_paths.items():
        rankings[source], source_counts[source] = load_rankings(
            path, contexts, linear_scorer, bce_model, mlp_model
        )

    measurements = load_measurements(paths["evaluation_measurements"])
    selected = audit.selected_candidates(rankings)
    needed_pairs = {
        (row["scan_id"], row["subject_id"], row["object_id"]) for row in selected
    }
    missing_pairs = needed_pairs - set(measurements)
    additional: dict[tuple[str, int, int], dict[str, Any]] = {}
    additional_inventory: dict[str, Any] = {
        "scans_requested": 0,
        "scans_available": 0,
        "pairs_requested": 0,
        "pairs_measured": 0,
    }
    if missing_pairs:
        grouped_missing: dict[str, set[tuple[int, int]]] = defaultdict(set)
        for scan, subject, object_ in missing_pairs:
            grouped_missing[scan].add((subject, object_))
        additional, _, additional_inventory = audit.measure_pairs(
            paths["raw_scan_root"],
            grouped_missing,
            int(protocol["point_mesh_config"]["maximum_vertices_per_object"]),
            int(protocol["point_mesh_config"]["maximum_triangles_per_object"]),
            float(protocol["point_mesh_config"]["minimum_metric_scale_m"]),
        )
        measurements.update(additional)

    thresholds = json.loads(paths["thresholds"].read_text(encoding="utf-8"))
    all_gt, scope_gt = audit.load_ground_truth_scope(paths["ground_truth"])
    samples = np.random.default_rng(int(protocol["uncertainty"]["seed"])).integers(
        0,
        len(scans),
        size=(int(protocol["uncertainty"]["resamples"]), len(scans)),
    )
    contributions = audit.build_contributions(
        rankings, contexts, scans, all_gt, scope_gt, measurements, thresholds
    )
    results = audit.summarize_contributions(contributions, samples)

    mechanism_cases = read_mechanism_cases(paths["mechanism_cases"])
    mechanism, mechanism_rows = mlp_mechanism_test(
        mechanism_cases,
        measurements,
        rankings,
        source_paths,
        mlp_model,
        [float(value) for value in protocol["synthetic_intervention"]["levels_in_pair_scale_units"]],
    )
    reference = json.loads(paths["routed_comparator_summary"].read_text(encoding="utf-8"))
    recall_equivalence: dict[str, Any] = {}
    for source in source_paths:
        recall_equivalence[source] = {}
        for current_method, reference_method in (
            ("source", "source"),
            ("relcompat3d", "relcompat3d_mlp"),
        ):
            recall_equivalence[source][current_method] = {}
            for k in audit.KS:
                current = results[source]["recall"][current_method][str(k)]["recall_all"]["point"]
                expected = reference["sources"][source]["results"][reference_method][str(k)]["recall"]["point"]
                recall_equivalence[source][current_method][str(k)] = abs(current - expected)

    validations = {
        "all_file_hashes_match": all(
            not spec.get("sha256") or input_checks[name].get("sha256") == spec["sha256"]
            for name, spec in protocol["inputs"].items()
        ),
        "mlp_excludes_source_score_and_identity": not feature_spec["source_score_input"]
        and not feature_spec["source_identity_input"],
        "official_contexts_548": len(contexts) == 548,
        "validation_scans_157": len(scans) == 157,
        "paper_scope_gt_denominator_3972": sum(len(rows) for rows in all_gt.values()) == 3972,
        "audit_scope_gt_denominator_2156": sum(len(rows) for rows in scope_gt.values()) == 2156,
        "all_selected_pairs_measured": needed_pairs <= set(measurements),
        "mlp_recall_matches_routed_comparator": all(
            error <= 1e-15
            for source_payload in recall_equivalence.values()
            for method_payload in source_payload.values()
            for error in method_payload.values()
        ),
        "all_three_audits_reported": all(
            set(payload["audits"]) == {"point", "mesh", "consensus"}
            for payload in results.values()
        ),
        "mechanism_cases_cover_both_families": all(
            mechanism["families"][family]["covered_cases"] > 0
            for family in audit.FAMILIES
        ),
        "all_metric_points_finite": all(
            math.isfinite(payload["audits"][audit_name][method][str(k)]["violation"]["point"])
            for payload in results.values()
            for audit_name in audit.AUDITS
            for method in audit.METHODS
            for k in audit.KS
        ),
    }
    status = "completed" if all(validations.values()) else "failed_validation"
    summary = {
        "schema_version": "relcompat3d_relcompat3d_mlp_point_mesh_audit_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "method": {
            "name": "RelCompat3D-MLP",
            "architecture": mlp_model["architecture"],
            "parameter_count": mlp_model["parameter_count"],
        },
        "scope": protocol["scope"],
        "thresholds": thresholds,
        "coverage": {
            "sources": source_counts,
            "precomputed_measurement_pairs": len(measurements) - len(additional),
            "additional_measurement_pairs": len(additional),
            "additional_inventory": additional_inventory,
        },
        "results": results,
        "mechanism": mechanism,
        "recall_equivalence_to_routed_comparator": recall_equivalence,
        "validations": validations,
        "evaluation_scope": protocol["evaluation_scope"],
        "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm relcompat3d_mlp_point_mesh_audit",
    }

    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    audit.write_json(summary_path, summary)
    summary_md = out / "summary.md"
    summary_md.write_text(markdown(summary), encoding="utf-8")
    metrics_path = out / "metrics.csv"
    audit.write_csv(metrics_path, audit.metrics_csv_rows(results))
    mechanism_path = out / "mechanism_rows.csv"
    audit.write_csv(mechanism_path, mechanism_rows)
    additional_path = out / "additional_evaluation_measurements.jsonl"
    audit.write_jsonl(additional_path, audit.measurement_rows(additional))
    output_paths = (summary_path, summary_md, metrics_path, mechanism_path, additional_path)
    manifest = {
        "schema_version": "relcompat3d_relcompat3d_mlp_point_mesh_audit_manifest_v1",
        "status": status,
        "protocol": {
            "path": audit.relpath(root, protocol_path),
            "sha256": audit.sha256_file(protocol_path),
        },
        "inputs": input_checks,
        "outputs": {
            path.name: {"sha256": audit.sha256_file(path), "size_bytes": path.stat().st_size}
            for path in output_paths
        },
        "validations": validations,
        "docker_command": summary["docker_command"],
    }
    audit.write_json(out / "manifest.json", manifest)
    print(json.dumps({"status": status, "validations": validations}))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
