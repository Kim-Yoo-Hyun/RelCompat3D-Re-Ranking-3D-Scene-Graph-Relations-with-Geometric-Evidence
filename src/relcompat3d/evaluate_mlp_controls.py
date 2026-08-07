#!/usr/bin/env python3
"""Evaluate RelCompat3D-MLP under the fixed family-aware ablation contract."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evaluate_linear_controls as linear_controls
import control_utils as ablation
import fit_mlp as nonlinear
import evaluate_base_models as model_eval


CONDITION_NAMES = {
    "source": "source",
    "all_family_product": "relcompat3d_mlp",
    "wrong_predicate_product": "mlp_wrong_predicate",
    "wrong_pair_product": "mlp_wrong_pair",
    "shuffled_geometry_product": "mlp_shuffled_geometry",
    "endpoint_swap_fixed_label_product": "mlp_fixed_label_endpoint_swap",
    "distance_only": "distance_only",
    "compatibility_only": "mlp_compatibility_only",
}
METHODS = tuple(CONDITION_NAMES.values())
KS = ablation.KS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def rename_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    renamed = {CONDITION_NAMES[name]: raw[name] for name in CONDITION_NAMES}
    renamed["deltas_vs_relcompat3d_mlp"] = {
        CONDITION_NAMES[name]: cells
        for name, cells in raw["deltas_vs_all_family_product"].items()
    }
    return renamed


def make_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, payload in summary["sources"].items():
        for method in METHODS:
            for k in KS:
                cell = payload["metrics"][method][str(k)]
                rows.append(
                    {
                        "source": source,
                        "method": method,
                        "k": k,
                        "recall": cell["recall"]["point"],
                        "recall_scan_ci95_low": cell["recall"]["bootstrap_intervals_ci95"][0],
                        "recall_scan_ci95_high": cell["recall"]["bootstrap_intervals_ci95"][1],
                        "violation": cell["violation"]["point"],
                        "violation_scan_ci95_low": cell["violation"]["bootstrap_intervals_ci95"][0],
                        "violation_scan_ci95_high": cell["violation"]["bootstrap_intervals_ci95"][1],
                        "selected": cell["selected"],
                    }
                )
    return rows


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RelCompat3D-MLP Ablation Evaluation",
        "",
        f"Status: `{summary['status']}`",
        "",
        "All conditions use the fixed nonlinear compatibility head, public/full 548-context target, and family-aware ranking procedure. Support/contact candidates remain in source order.",
        "",
        "| Predictor | Condition | R@50 | V@50 | R@100 | V@100 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    labels = {"vlsat": "VL-SAT", "open3dsg": "Open3DSG", "sgfn": "SGFN"}
    for source, payload in summary["sources"].items():
        for method in METHODS:
            metrics = payload["metrics"][method]
            lines.append(
                f"| {labels[source]} | `{method}` | "
                f"{metrics['50']['recall']['point']:.4f} | "
                f"{metrics['50']['violation']['point']:.4f} | "
                f"{metrics['100']['recall']['point']:.4f} | "
                f"{metrics['100']['violation']['point']:.4f} |"
            )
    lines.extend(
        [
            "",
            "`distance_only` is head-independent and is retained once as a common control. `mlp_compatibility_only` removes the predictor score only from proximity/vertical ordering; it is not a raw-geometry-only model.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    protocol_path = ablation.resolve(root, args.protocol)
    out = ablation.resolve(root, args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"nonempty_output:{out}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "ready":
        raise ValueError("protocol_version_mismatch")
    if tuple(protocol["evaluation"]["ks"]) != KS:
        raise ValueError("rank_cutoffs_mismatch")
    if tuple(protocol["conditions"]) != METHODS:
        raise ValueError("ranking_config_mismatch")

    paths: dict[str, Path] = {}
    input_checks: dict[str, Any] = {}
    for name, spec in protocol["inputs"].items():
        path = ablation.resolve(root, spec["path"])
        if not path.exists():
            raise FileNotFoundError(f"missing_input:{name}:{path}")
        actual = ablation.sha256_file(path)
        if actual != spec["sha256"]:
            raise ValueError(f"input_hash_mismatch:{name}:{actual}")
        paths[name] = path
        input_checks[name] = {
            "path": ablation.relpath(root, path),
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }

    nonlinear_models = json.loads(paths["nonlinear_models"].read_text(encoding="utf-8"))
    model = nonlinear_models["shared_mlp_pairwise"]
    feature_spec = model["feature_spec"]
    if feature_spec["source_score_input"] or feature_spec["source_identity_input"]:
        raise ValueError("mlp_model_uses_source")

    def scorer(family: str, predicate: str, raw: dict[str, float]) -> float:
        return nonlinear.projected_probability(model, family, predicate, raw)

    gt, _ = model_eval.load_gt(paths["ground_truth"])
    context_to_scan = linear_controls.official_context_map(paths["official_context_annotations"])
    contexts = sorted(context_to_scan)
    reference = json.loads(paths["routed_comparator_summary"].read_text(encoding="utf-8"))
    source_paths = {
        "vlsat": paths["vlsat_verification"],
        "open3dsg": paths["open3dsg_verification"],
        "sgfn": paths["sgfn_verification"],
    }
    sources: dict[str, Any] = {}
    equivalence: dict[str, Any] = {}
    for source_index, (source, path) in enumerate(source_paths.items()):
        grouped, counts = ablation.load_rows(path, scorer)
        donor_audit = ablation.add_scores(
            grouped, scorer, protocol["wrong_predicate_mapping"]
        )
        routing_audit = linear_controls.add_ranking_scores(grouped)
        weights, scan_counts = linear_controls.scan_weights(
            contexts,
            context_to_scan,
            int(protocol["evaluation"]["bootstrap_resamples"]),
            int(protocol["evaluation"]["bootstrap_seed"]) + source_index,
        )
        raw_metrics = linear_controls.evaluate(grouped, gt, contexts, weights)
        metrics = rename_metrics(raw_metrics)
        sources[source] = {
            "counts": {
                **counts,
                **scan_counts,
                "candidate_contexts": len(grouped),
                "zero_prediction_contexts": len(set(contexts) - set(grouped)),
                "gt_denominator": sum(len(items) for items in gt.values()),
            },
            "donor_audit": donor_audit,
            "routing_audit": routing_audit,
            "metrics": metrics,
        }
        equivalence[source] = {}
        for current_method, reference_method in (
            ("source", "source"),
            ("relcompat3d_mlp", "relcompat3d_mlp"),
        ):
            equivalence[source][current_method] = {}
            for k in KS:
                current = metrics[current_method][str(k)]
                previous = reference["sources"][source]["results"][reference_method][str(k)]
                equivalence[source][current_method][str(k)] = {
                    "recall_abs_error": abs(
                        current["recall"]["point"] - previous["recall"]["point"]
                    ),
                    "violation_abs_error": abs(
                        current["violation"]["point"]
                        - previous["violation_all"]["point"]
                    ),
                }

    expected_rows = protocol["evaluation"]["expected_in_scope_rows"]
    validations = {
        "all_input_hashes_match": len(input_checks) == len(protocol["inputs"]),
        "mlp_excludes_source_score_and_identity": not feature_spec["source_score_input"]
        and not feature_spec["source_identity_input"],
        "all_sources_have_548_contexts": all(
            payload["counts"]["contexts"] == 548 for payload in sources.values()
        ),
        "all_sources_have_157_scans": all(
            payload["counts"]["scans"] == 157 for payload in sources.values()
        ),
        "gt_denominator_3972": all(
            payload["counts"]["gt_denominator"] == 3972
            for payload in sources.values()
        ),
        "in_scope_row_counts": all(
            payload["counts"]["in_scope_rows"] == expected_rows[source]
            for source, payload in sources.items()
        ),
        "open3dsg_public_route_533_plus_15_zero": (
            sources["open3dsg"]["counts"]["candidate_contexts"] == 533
            and sources["open3dsg"]["counts"]["zero_prediction_contexts"] == 15
        ),
        "support_contact_order_exact": all(
            payload["routing_audit"]["support_contact_order_exact"]
            for payload in sources.values()
        ),
        "family_composition_exact": all(
            payload["routing_audit"]["family_composition_exact"]
            for payload in sources.values()
        ),
        "wrong_pair_full_coverage": all(
            payload["donor_audit"]["wrong_pair_donor_rows"]
            == payload["counts"]["in_scope_rows"]
            for payload in sources.values()
        ),
        "shuffled_geometry_full_coverage": all(
            payload["donor_audit"]["shuffled_donor_rows"]
            == payload["counts"]["in_scope_rows"]
            for payload in sources.values()
        ),
        "wrong_pair_has_no_self_donor": all(
            payload["donor_audit"]["wrong_pair_self_donors"] == 0
            for payload in sources.values()
        ),
        "source_and_mlp_match_routed_comparator": all(
            error <= 1e-15
            for source_payload in equivalence.values()
            for method_payload in source_payload.values()
            for cell in method_payload.values()
            for error in cell.values()
        ),
        "all_metrics_finite": all(
            math.isfinite(payload["metrics"][method][str(k)][metric]["point"])
            for payload in sources.values()
            for method in METHODS
            for k in KS
            for metric in ("recall", "violation")
        ),
    }
    status = "completed" if all(validations.values()) else "failed_validation"
    summary = {
        "schema_version": "relcompat3d_relcompat3d_mlp_ablation_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "classification": protocol["classification"],
        "model": {
            "name": "RelCompat3D-MLP",
            "architecture": model["architecture"],
            "parameter_count": model["parameter_count"],
        },
        "ranking_rule": "family-aware re-ranking",
        "bootstrap_unit": "scan_id cluster",
        "methods": list(METHODS),
        "ks": list(KS),
        "sources": sources,
        "point_equivalence_to_routed_comparator": equivalence,
        "validations": validations,
        "evaluation_scope": protocol["evaluation_scope"],
        "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm relcompat3d_mlp_ablation",
    }

    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    ablation.write_json(summary_path, summary)
    summary_md = out / "summary.md"
    summary_md.write_text(markdown(summary), encoding="utf-8")
    metrics_path = out / "metrics.csv"
    rows = make_rows(summary)
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": "relcompat3d_relcompat3d_mlp_ablation_manifest_v1",
        "status": status,
        "protocol": {
            "path": ablation.relpath(root, protocol_path),
            "sha256": ablation.sha256_file(protocol_path),
        },
        "inputs": input_checks,
        "outputs": {
            path.name: {
                "sha256": ablation.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (summary_path, summary_md, metrics_path)
        },
        "validations": validations,
        "docker_command": summary["docker_command"],
    }
    ablation.write_json(out / "manifest.json", manifest)
    print(json.dumps({"status": status, "validations": validations}))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
