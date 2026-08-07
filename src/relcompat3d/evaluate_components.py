#!/usr/bin/env python3
"""Run matched Linear/MLP diagnostics for two RelCompat3D components."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evaluate_base_models as model_eval
import fit_mlp as nonlinear
import training_control_utils as controls


CONDITIONS = (
    "linear_full",
    "linear_no_pairwise",
    "linear_no_averaging",
    "mlp_full",
    "mlp_no_pairwise",
    "mlp_no_averaging",
)
STREAM_INPUTS = {
    "vlsat_verification",
    "open3dsg_verification",
    "sgfn_verification",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def flatten_values(value: Any) -> list[float]:
    if isinstance(value, list):
        result: list[float] = []
        for item in value:
            result.extend(flatten_values(item))
        return result
    return [float(value)]


def max_parameter_error(left: dict[str, Any], right: dict[str, Any]) -> float:
    errors = []
    for name in left["parameters"]:
        a = flatten_values(left["parameters"][name])
        b = flatten_values(right["parameters"][name])
        if len(a) != len(b):
            raise ValueError(f"parameter_shape_mismatch:{name}")
        errors.extend(abs(x - y) for x, y in zip(a, b))
    return max(errors, default=0.0)


def make_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RelCompat3D Component Diagnostics",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Full, no-pairwise-loss, and no-transformation-averaging are matched "
        "within each estimator. All results use the fixed candidates and "
        "family-aware route.",
        "",
        "## Held-out Linked-Pair Diagnostics",
        "",
        "| Estimator | Condition | Pairs | Positive wins | Mean margin | P05 | Median | P95 | Softplus loss |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for estimator in ("linear", "mlp"):
        for condition in ("full", "no_pairwise"):
            cell = summary["linked_pair_diagnostics"][estimator][condition]
            dist = cell["margin_distribution"]
            lines.append(
                f"| {estimator} | {condition} | {cell['pairs']} | "
                f"{cell['positive_win_rate']:.6f} | {dist['mean']:.6f} | "
                f"{dist['p05']:.6f} | {dist['median']:.6f} | "
                f"{dist['p95']:.6f} | {cell['softplus_margin_loss']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Aggregate Point Estimates",
            "",
            "| Predictor | Condition | R@50 | V@50 | R@100 | V@100 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for source, payload in summary["sources"].items():
        for condition in CONDITIONS:
            result = payload["metrics"][condition]
            lines.append(
                f"| {source} | {condition} | {result['50']['recall']:.4f} | "
                f"{result['50']['violation']:.4f} | "
                f"{result['100']['recall']:.4f} | "
                f"{result['100']['violation']:.4f} |"
            )
    lines.extend(
        [
            "",
            "Transformation-error distributions and transformed-view top-K "
            "membership checks are recorded in `summary.json` and the CSV files.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    protocol_path = controls.resolve(root, args.protocol)
    out = controls.resolve(root, args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"nonempty_output:{out}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "ready":
        raise ValueError("protocol_version_mismatch")
    if tuple(protocol["conditions"]) != CONDITIONS:
        raise ValueError("analysis_conditions_mismatch")
    paths, input_checks = controls.validate_inputs(
        root, protocol, deferred=STREAM_INPUTS
    )

    prepared, training_counts = controls.prepare_rows(paths)
    stats, orbit_x, orbit_y, orbit_pairs, mlp_counts = (
        controls.mlp_training_arrays(prepared)
    )
    mlp_spec = {
        **protocol["mlp_optimizer"],
        "expected_train_rows": mlp_counts["train_rows"],
        "expected_linked_pairs": mlp_counts["linked_pairs"],
    }
    mlp_full = controls.fit_mlp_condition(
        stats, orbit_x, orbit_y, orbit_pairs, mlp_spec, pairwise=True
    )
    mlp_no_pairwise = controls.fit_mlp_condition(
        stats, orbit_x, orbit_y, orbit_pairs, mlp_spec, pairwise=False
    )
    active_mlp_payload = json.loads(
        paths["active_mlp_models"].read_text(encoding="utf-8")
    )
    active_mlp = active_mlp_payload["shared_mlp_pairwise"]
    active_mlp_error = max_parameter_error(mlp_full, active_mlp)

    linear_payload = json.loads(paths["active_linear_models"].read_text(encoding="utf-8"))
    linear_full = controls.CompiledLinear(
        linear_payload["attempts"]["orbit_pairwise"]
    )
    linear_no_pairwise = controls.CompiledLinear(
        linear_payload["attempts"]["orbit_augmented"]
    )
    compiled_mlp_full = controls.CompiledMLP(mlp_full)
    compiled_mlp_no_pairwise = controls.CompiledMLP(mlp_no_pairwise)
    scorers = {
        "linear_full": (linear_full.projected, linear_full.direct),
        "linear_no_pairwise": (
            linear_no_pairwise.projected,
            linear_no_pairwise.direct,
        ),
        "linear_no_averaging": (linear_full.direct, linear_full.direct),
        "mlp_full": (compiled_mlp_full.projected, compiled_mlp_full.direct),
        "mlp_no_pairwise": (
            compiled_mlp_no_pairwise.projected,
            compiled_mlp_no_pairwise.direct,
        ),
        "mlp_no_averaging": (
            compiled_mlp_full.direct,
            compiled_mlp_full.direct,
        ),
    }
    margin = float(protocol["mlp_optimizer"]["pairwise_margin"])
    linked_pair = {
        "linear": {
            "full": controls.linked_pair_diagnostics(
                prepared, linear_full.projected, margin
            ),
            "no_pairwise": controls.linked_pair_diagnostics(
                prepared, linear_no_pairwise.projected, margin
            ),
        },
        "mlp": {
            "full": controls.linked_pair_diagnostics(
                prepared, compiled_mlp_full.projected, margin
            ),
            "no_pairwise": controls.linked_pair_diagnostics(
                prepared, compiled_mlp_no_pairwise.projected, margin
            ),
        },
    }

    context_map = controls.official_context_map(paths["official_context_annotations"])
    contexts = sorted(context_map)
    gt, _ = model_eval.load_gt(paths["ground_truth"])
    source_paths = {
        source: paths[f"{source}_verification"]
        for source in ("vlsat", "open3dsg", "sgfn")
    }
    sources: dict[str, Any] = {}
    for source, path in source_paths.items():
        grouped, counts = controls.load_candidate_rows(
            path,
            protocol["inputs"][f"{source}_verification"]["sha256"],
            scorers,
        )
        metrics, membership, route_checks = controls.evaluate_conditions(
            grouped, gt, contexts, CONDITIONS
        )
        sources[source] = {
            "counts": {
                **counts,
                "candidate_contexts": len(grouped),
                "zero_prediction_contexts": len(set(contexts) - set(grouped)),
                "gt_denominator": sum(len(rows) for rows in gt.values()),
                "candidate_scans": len(
                    {row["scan"] for rows in grouped.values() for row in rows}
                ),
                "official_scans": len(set(context_map.values())),
            },
            "metrics": metrics,
            "transformed_topk_membership": membership,
            "route_checks": route_checks,
        }

    reference = json.loads(paths["main_reference"].read_text(encoding="utf-8"))
    reference_match = True
    for source in sources:
        for k in controls.KS:
            for condition, reference_method in (
                ("linear_full", "relcompat3d_linear"),
                ("mlp_full", "relcompat3d_mlp"),
            ):
                actual = sources[source]["metrics"][condition][str(k)]
                expected = reference["sources"][source]["results"][reference_method][
                    str(k)
                ]
                reference_match &= (
                    abs(actual["recall"] - expected["recall"]["point"]) <= 1e-12
                    and abs(
                        actual["violation"]
                        - expected["violation_all"]["point"]
                    )
                    <= 1e-12
                )
    averaged_errors_zero = all(
        sources[source]["counts"]["transformation_error"][condition][family][
            "max"
        ]
        <= 1e-12
        for source in sources
        for condition in (
            "linear_full",
            "linear_no_pairwise",
            "mlp_full",
            "mlp_no_pairwise",
        )
        for family in controls.RERANKED_FAMILIES
    )
    averaged_membership_exact = all(
        sources[source]["transformed_topk_membership"][condition][str(k)][
            "micro_jaccard"
        ]
        == 1.0
        and sources[source]["transformed_topk_membership"][condition][str(k)][
            "exact_context_fraction"
        ]
        == 1.0
        for source in sources
        for condition in (
            "linear_full",
            "linear_no_pairwise",
            "mlp_full",
            "mlp_no_pairwise",
        )
        for k in controls.KS
    )
    no_average_error_observed = all(
        any(
            sources[source]["counts"]["transformation_error"][condition][family][
                "max"
            ]
            > 1e-12
            for source in sources
            for family in controls.RERANKED_FAMILIES
        )
        for condition in ("linear_no_averaging", "mlp_no_averaging")
    )
    validations = {
        "all_input_hashes_match": True,
        "split_counts_1061_117_157": (
            training_counts["train_scans"],
            training_counts["development_scans"],
            training_counts["final_validation_scans"],
        )
        == (1061, 117, 157),
        "train_rows_60208": training_counts["train_rows"] == 60208,
        "development_rows_6246": training_counts["development_rows"] == 6246,
        "mlp_training_counts_match": mlp_counts
        == {
            "train_rows": 60208,
            "linked_pairs": 33961,
            "orbit_rows": 86032,
            "orbit_pairs": 45167,
        },
        "active_mlp_reproduced": active_mlp_error <= 1e-12,
        "active_metrics_match_main_reference": reference_match,
        "official_context_universe_157_scans": all(
            payload["counts"]["official_scans"] == 157
            for payload in sources.values()
        ),
        "all_gt_denominators_3972": all(
            payload["counts"]["gt_denominator"] == 3972
            for payload in sources.values()
        ),
        "family_sequence_exact": all(
            payload["route_checks"]["family_sequence_exact"]
            for payload in sources.values()
        ),
        "support_subsequence_exact": all(
            payload["route_checks"]["support_subsequence_exact"]
            for payload in sources.values()
        ),
        "averaged_transformation_errors_zero": averaged_errors_zero,
        "averaged_transformed_topk_membership_exact": averaged_membership_exact,
        "no_averaging_exposes_nonzero_error": no_average_error_observed,
        "all_reported_values_finite": all(
            math.isfinite(
                payload["metrics"][condition][str(k)][metric]
            )
            for payload in sources.values()
            for condition in CONDITIONS
            for k in controls.KS
            for metric in ("recall", "violation")
        ),
    }
    status = "completed" if all(validations.values()) else "failed_validation"
    summary = {
        "schema_version": "relcompat3d_component_analysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "conditions": protocol["condition_definitions"],
        "training_counts": training_counts,
        "mlp_training_counts": mlp_counts,
        "active_mlp_max_parameter_abs_error": active_mlp_error,
        "linked_pair_diagnostics": linked_pair,
        "sources": sources,
        "validations": validations,
        "evaluation_scope": protocol["evaluation_scope"],
    }
    out.mkdir(parents=True, exist_ok=True)
    controls.write_json(out / "summary.json", summary)
    controls.write_json(
        out / "models.json",
        {
            "mlp_full": mlp_full,
            "mlp_no_pairwise": mlp_no_pairwise,
        },
    )
    (out / "summary.md").write_text(make_markdown(summary) + "\n", encoding="utf-8")

    metric_rows: list[dict[str, Any]] = []
    transformation_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    for source, payload in sources.items():
        for condition in CONDITIONS:
            for k in controls.KS:
                cell = payload["metrics"][condition][str(k)]
                metric_rows.append(
                    {
                        "source": source,
                        "condition": condition,
                        "k": k,
                        "recall": cell["recall"],
                        "violation": cell["violation"],
                    }
                )
                member = payload["transformed_topk_membership"][condition][str(k)]
                membership_rows.append(
                    {
                        "source": source,
                        "condition": condition,
                        "k": k,
                        "micro_jaccard": member["micro_jaccard"],
                        "exact_context_fraction": member["exact_context_fraction"],
                    }
                )
            for family in sorted(controls.RERANKED_FAMILIES):
                dist = payload["counts"]["transformation_error"][condition][family]
                transformation_rows.append(
                    {
                        "source": source,
                        "condition": condition,
                        "family": family,
                        **dist,
                    }
                )
    for filename, rows in (
        ("metrics.csv", metric_rows),
        ("transformation.csv", transformation_rows),
        ("membership.csv", membership_rows),
    ):
        with (out / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    output_names = (
        "summary.json",
        "summary.md",
        "models.json",
        "metrics.csv",
        "transformation.csv",
        "membership.csv",
    )
    controls.write_json(
        out / "manifest.json",
        {
            "schema_version": "relcompat3d_component_analysis_manifest_v1",
            "status": status,
            "protocol": {
                "path": controls.relpath(root, protocol_path),
                "sha256": controls.sha256_file(protocol_path),
            },
            "inputs": input_checks,
            "outputs": {
                name: {
                    "sha256": controls.sha256_file(out / name),
                    "size_bytes": (out / name).stat().st_size,
                }
                for name in output_names
            },
            "validations": validations,
            "docker_command": (
                "env UID=$(id -u) GID=$(id -g) docker compose -f "
                "configs/relcompat3d/compose.yaml run --rm "
                "relcompat3d_component_analysis"
            ),
        },
    )
    print(json.dumps({"status": status, "validations": validations}))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
