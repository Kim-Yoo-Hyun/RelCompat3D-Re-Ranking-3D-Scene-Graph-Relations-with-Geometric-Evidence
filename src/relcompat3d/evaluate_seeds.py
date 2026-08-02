#!/usr/bin/env python3
"""Evaluate RelCompat3D training robustness over five predeclared seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fit_linear
import fit_mlp as nonlinear
import evaluate_train_only as strict
import relation_consistency as algebra
import training_control_utils as controls


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


def flatten(value: Any) -> list[float]:
    if isinstance(value, list):
        result: list[float] = []
        for item in value:
            result.extend(flatten(item))
        return result
    return [float(value)]


def model_max_error(left: dict[str, Any], right: dict[str, Any]) -> float:
    if "parameters" in left:
        errors = []
        for name in left["parameters"]:
            a = flatten(left["parameters"][name])
            b = flatten(right["parameters"][name])
            errors.extend(abs(x - y) for x, y in zip(a, b))
        return max(errors, default=0.0)
    errors = []
    for family in controls.FAMILIES:
        a = left[family]["weights"]
        b = right[family]["weights"]
        errors.extend(abs(float(x) - float(y)) for x, y in zip(a, b))
    return max(errors, default=0.0)


def model_values(model: dict[str, Any]) -> list[float]:
    if "parameters" in model:
        values: list[float] = []
        for parameter in model["parameters"].values():
            values.extend(flatten(parameter))
        return values
    return [
        float(weight)
        for family in controls.FAMILIES
        for weight in model[family]["weights"]
    ]


def load_seed_candidates(
    path: Path,
    expected_sha256: str,
    scorers: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    digest = hashlib.sha256()
    input_rows = 0
    in_scope_rows = 0
    for raw_line in path.open("rb"):
        digest.update(raw_line)
        if not raw_line.strip():
            continue
        input_rows += 1
        row = json.loads(raw_line)
        family = row["predicate"]["predicate_family"]
        if family not in controls.FAMILIES:
            continue
        in_scope_rows += 1
        predicate = row["predicate"]["predicate_label"]
        raw = strict.raw_numeric(row)
        semantic = strict.finite((row.get("semantic") or {}).get("ranking_score"))
        if semantic is None:
            raise ValueError(f"missing_semantic:{row['prediction_id']}")
        compatibility = {"source": 1.0}
        compatibility.update(
            {
                name: scorer(family, predicate, raw)
                for name, scorer in scorers.items()
            }
        )
        grouped[row["subgraph_id"]].append(
            {
                "id": row["prediction_id"],
                "scan": row["scan_id"],
                "key": strict.candidate_key(row),
                "family": family,
                "semantic": float(semantic),
                "status": row.get("verification_status")
                or (row.get("verification") or {}).get("verification_status"),
                "compatibility": compatibility,
                "transformed_compatibility": compatibility,
            }
        )
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"stream_hash_mismatch:{path}:{actual_sha256}")
    return grouped, {
        "input_rows": input_rows,
        "in_scope_rows": in_scope_rows,
        "input_sha256": actual_sha256,
    }


def make_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RelCompat3D Five-Seed Robustness",
        "",
        f"Status: `{summary['status']}`",
        "",
        "The model seeds were fixed before evaluation. The constructed training "
        "rows and their counterfactual links are held fixed. The active MLP seed "
        "is included but was not reselected from this analysis.",
        "",
        "| Estimator | Predictor | K | Recall mean±std | Violation mean±std | Favorable seeds |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for estimator in ("linear", "mlp"):
        for source, by_k in summary["seed_summary"][estimator].items():
            for k in controls.KS:
                cell = by_k[str(k)]
                lines.append(
                    f"| {estimator} | {source} | {k} | "
                    f"{cell['recall']['mean']:.6f} ± {cell['recall']['std']:.6f} | "
                    f"{cell['violation']['mean']:.6f} ± "
                    f"{cell['violation']['std']:.6f} | "
                    f"{cell['favorable_seed_count']}/{cell['seed_count']} |"
                )
    lines.extend(
        [
            "",
            "Linear uses deterministic zero initialization and full-batch "
            "optimization, so the five declared seed labels reproduce one model "
            "hash. MLP varies only its initialization seed.",
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
    if protocol.get("status") != "frozen_before_five_seed_execution":
        raise ValueError("protocol_not_frozen")
    seeds = [int(seed) for seed in protocol["model_seeds"]]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("five_unique_seeds_required")
    if int(protocol["active_mlp_seed"]) not in seeds:
        raise ValueError("active_seed_not_in_grid")
    paths, input_checks = controls.validate_inputs(
        root, protocol, deferred=STREAM_INPUTS
    )

    prepared, training_counts = controls.prepare_rows(paths)
    current_strict = json.loads(
        paths["current_strict_models"].read_text(encoding="utf-8")
    )
    active_linear_payload = json.loads(
        paths["active_linear_models"].read_text(encoding="utf-8")
    )
    active_linear = active_linear_payload["attempts"]["orbit_pairwise"]
    active_mlp_payload = json.loads(
        paths["active_mlp_models"].read_text(encoding="utf-8")
    )
    active_mlp = active_mlp_payload["shared_nonlinear_structured"]

    linear_models: dict[str, dict[str, Any]] = {}
    linear_model_hashes: dict[str, str] = {}
    for seed in seeds:
        refitted_strict, _ = fit_linear.refit_strict_family_models(
            prepared, current_strict
        )
        attempts, _ = algebra.fit_attempts(
            prepared, refitted_strict, protocol["linear_optimizer"]
        )
        model = attempts["orbit_pairwise"]
        linear_models[str(seed)] = model
        linear_model_hashes[str(seed)] = controls.sha256_json(model)

    stats, orbit_x, orbit_y, orbit_pairs, mlp_counts = (
        controls.mlp_training_arrays(prepared)
    )
    mlp_models: dict[str, dict[str, Any]] = {}
    mlp_model_hashes: dict[str, str] = {}
    for seed in seeds:
        spec = {
            **protocol["mlp_optimizer"],
            "seed": seed,
            "expected_train_rows": mlp_counts["train_rows"],
            "expected_linked_pairs": mlp_counts["linked_pairs"],
        }
        model = controls.fit_mlp_condition(
            stats, orbit_x, orbit_y, orbit_pairs, spec, pairwise=True
        )
        mlp_models[str(seed)] = model
        mlp_model_hashes[str(seed)] = controls.sha256_json(model["parameters"])

    active_seed = str(protocol["active_mlp_seed"])
    active_mlp_error = model_max_error(mlp_models[active_seed], active_mlp)
    active_linear_error = model_max_error(
        linear_models[active_seed], active_linear
    )
    compiled_linear = {
        seed: controls.CompiledLinear(model)
        for seed, model in linear_models.items()
    }
    compiled_mlp = {
        seed: controls.CompiledMLP(model) for seed, model in mlp_models.items()
    }
    scorers = {
        **{
            f"linear_{seed}": model.projected
            for seed, model in compiled_linear.items()
        },
        **{
            f"mlp_{seed}": model.projected
            for seed, model in compiled_mlp.items()
        },
    }
    conditions = ("source",) + tuple(scorers)
    context_map = controls.official_context_map(paths["official_context_annotations"])
    contexts = sorted(context_map)
    gt, _ = strict.load_gt(paths["ground_truth"])
    source_paths = {
        source: paths[f"{source}_verification"]
        for source in ("vlsat", "open3dsg", "sgfn")
    }
    source_payloads: dict[str, Any] = {}
    for source, path in source_paths.items():
        grouped, counts = load_seed_candidates(
            path,
            protocol["inputs"][f"{source}_verification"]["sha256"],
            scorers,
        )
        metrics, _, route_checks = controls.evaluate_conditions(
            grouped, gt, contexts, conditions
        )
        source_payloads[source] = {
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
            "route_checks": route_checks,
        }

    per_seed: dict[str, Any] = {}
    for seed in map(str, seeds):
        per_seed[seed] = {"linear": {}, "mlp": {}}
        for source, payload in source_payloads.items():
            per_seed[seed]["linear"][source] = payload["metrics"][
                f"linear_{seed}"
            ]
            per_seed[seed]["mlp"][source] = payload["metrics"][f"mlp_{seed}"]
    source_metrics = {
        source: payload["metrics"]["source"]
        for source, payload in source_payloads.items()
    }
    seed_summary = controls.summarize_seed_metrics(per_seed, source_metrics)

    reference = json.loads(paths["main_reference"].read_text(encoding="utf-8"))
    active_metrics_match = True
    for source, payload in source_payloads.items():
        for k in controls.KS:
            for condition, method in (
                (f"linear_{active_seed}", "routed_product"),
                (f"mlp_{active_seed}", "routed_matched_mlp"),
            ):
                actual = payload["metrics"][condition][str(k)]
                expected = reference["sources"][source]["results"][method][str(k)]
                active_metrics_match &= (
                    abs(actual["recall"] - expected["recall"]["point"]) <= 1e-12
                    and abs(
                        actual["violation"]
                        - expected["violation_all"]["point"]
                    )
                    <= 1e-12
                )
    direction_reversals = sum(
        cell["source_direction_reversal_count"]
        for estimator in seed_summary.values()
        for source in estimator.values()
        for cell in source.values()
    )
    validations = {
        "all_input_hashes_match": True,
        "five_unique_model_seeds_predeclared": len(seeds) == len(set(seeds)) == 5,
        "active_seed_included_not_selected": (
            int(protocol["active_mlp_seed"]) == 20260714
            and protocol["selection_policy"]
            == "active model fixed before robustness analysis; no reselection"
        ),
        "constructed_rows_fixed_across_seeds": protocol["construction_seed_policy"]
        == "frozen calibration table and linked-pair identities; no resampling",
        "split_counts_1061_117_157": (
            training_counts["train_scans"],
            training_counts["internal_dev_scans"],
            training_counts["final_validation_scans"],
        )
        == (1061, 117, 157),
        "train_rows_60208": training_counts["train_rows"] == 60208,
        "internal_dev_rows_6246": training_counts["internal_dev_rows"] == 6246,
        "mlp_training_counts_match": mlp_counts
        == {
            "train_rows": 60208,
            "linked_pairs": 33961,
            "orbit_rows": 86032,
            "orbit_pairs": 45167,
        },
        "active_linear_reproduced": active_linear_error <= 1e-12,
        "active_mlp_reproduced": active_mlp_error <= 1e-12,
        "linear_is_seed_invariant": len(set(linear_model_hashes.values())) == 1,
        "active_metrics_match_main_reference": active_metrics_match,
        "family_sequence_exact": all(
            payload["route_checks"]["family_sequence_exact"]
            for payload in source_payloads.values()
        ),
        "support_subsequence_exact": all(
            payload["route_checks"]["support_subsequence_exact"]
            for payload in source_payloads.values()
        ),
        "official_context_universe_157_scans": all(
            payload["counts"]["official_scans"] == 157
            for payload in source_payloads.values()
        ),
        "all_gt_denominators_3972": all(
            payload["counts"]["gt_denominator"] == 3972
            for payload in source_payloads.values()
        ),
        "all_model_parameters_finite": all(
            math.isfinite(value)
            for models in (linear_models, mlp_models)
            for model in models.values()
            for value in model_values(model)
        ),
    }
    status = "completed" if all(validations.values()) else "failed_validation"
    summary = {
        "schema_version": "relcompat3d_five_seed_robustness",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "model_seeds": seeds,
        "active_mlp_seed": int(protocol["active_mlp_seed"]),
        "selection_policy": protocol["selection_policy"],
        "construction_seed_policy": protocol["construction_seed_policy"],
        "linear_seed_contract": (
            "deterministic zero initialization and deterministic full-batch "
            "optimization; seed labels are repeated-execution identifiers"
        ),
        "training_counts": training_counts,
        "mlp_training_counts": mlp_counts,
        "model_hashes": {
            "linear": linear_model_hashes,
            "mlp_parameters": mlp_model_hashes,
        },
        "active_model_max_parameter_abs_error": {
            "linear": active_linear_error,
            "mlp": active_mlp_error,
        },
        "per_seed": per_seed,
        "source_metrics": source_metrics,
        "seed_summary": seed_summary,
        "total_source_direction_reversals": direction_reversals,
        "sources": {
            source: {"counts": payload["counts"]}
            for source, payload in source_payloads.items()
        },
        "validations": validations,
        "claim_boundary": protocol["claim_boundary"],
    }
    out.mkdir(parents=True, exist_ok=True)
    controls.write_json(out / "summary.json", summary)
    controls.write_json(
        out / "model_hashes.json",
        {
            "model_seeds": seeds,
            "active_mlp_seed": int(protocol["active_mlp_seed"]),
            "linear": linear_model_hashes,
            "mlp_parameters": mlp_model_hashes,
        },
    )
    (out / "summary.md").write_text(make_markdown(summary) + "\n", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for seed, estimator_payload in per_seed.items():
        for estimator, by_source in estimator_payload.items():
            for source, by_k in by_source.items():
                for k in controls.KS:
                    cell = by_k[str(k)]
                    rows.append(
                        {
                            "seed": seed,
                            "estimator": estimator,
                            "source": source,
                            "k": k,
                            "recall": cell["recall"],
                            "violation": cell["violation"],
                        }
                    )
    with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output_names = (
        "summary.json",
        "summary.md",
        "metrics.csv",
        "model_hashes.json",
    )
    controls.write_json(
        out / "manifest.json",
        {
            "schema_version": "relcompat3d_five_seed_robustness_manifest_v1",
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
            "scientific_result": {
                "total_source_direction_reversals": direction_reversals
            },
            "docker_command": (
                "env UID=$(id -u) GID=$(id -g) docker compose -f "
                "configs/relcompat3d/compose.yaml run --rm "
                "relcompat3d_seed_robustness"
            ),
        },
    )
    print(
        json.dumps(
            {
                "status": status,
                "total_source_direction_reversals": direction_reversals,
                "validations": validations,
            }
        )
    )
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
