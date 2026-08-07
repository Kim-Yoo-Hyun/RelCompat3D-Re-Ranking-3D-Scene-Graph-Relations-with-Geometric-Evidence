#!/usr/bin/env python3
"""Evaluate matched family-aware routing constraints on the active candidate pool."""

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

import evaluate_all_families as base
import evaluate_score_robustness as robust
import evaluate_support_bootstrap as scan_bootstrap
import evaluate_base_models as model_eval
import fit_mlp as nonlinear


KS = base.KS
FAMILIES = base.FAMILIES
RERANKED_FAMILIES = ("proximity", "relative_vertical")
ESTIMATORS = ("linear", "mlp")
ROUTES = ("family_slots", "pv_global", "support_order_only", "all_families")
METHODS = (
    "source",
    "identity_family_slots",
    *(f"{estimator}_{route}" for estimator in ESTIMATORS for route in ROUTES),
)


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty_csv:{path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_order(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda row: (-row["semantic"], row["key"]))


def load_candidates(
    path: Path,
    linear_score: Callable[[str, str, dict[str, float]], float],
    mlp_model: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    digest = hashlib.sha256()
    seen: set[str] = set()
    input_rows = in_scope_rows = duplicates = 0
    score_min, score_max = math.inf, -math.inf
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
            prediction_id = row["prediction_id"]
            duplicates += int(prediction_id in seen)
            seen.add(prediction_id)
            predicate = row["predicate"]["predicate_label"]
            raw = model_eval.raw_numeric(row)
            semantic = model_eval.finite((row.get("semantic") or {}).get("ranking_score"))
            if semantic is None:
                raise ValueError(f"missing_semantic:{prediction_id}")
            score_min = min(score_min, semantic)
            score_max = max(score_max, semantic)
            grouped[row["subgraph_id"]].append(
                {
                    "id": prediction_id,
                    "scan": row["scan_id"],
                    "key": model_eval.candidate_key(row),
                    "family": family,
                    "predicate": predicate,
                    "semantic": float(semantic),
                    "linear": float(linear_score(family, predicate, raw)),
                    "mlp": float(
                        nonlinear.projected_probability(
                            mlp_model, family, predicate, raw
                        )
                    ),
                    "status": row.get("verification_status")
                    or (row.get("verification") or {}).get("verification_status"),
                }
            )
    return grouped, {
        "input_rows": input_rows,
        "in_scope_rows": in_scope_rows,
        "input_sha256": digest.hexdigest(),
        "unique_prediction_ids": len(seen),
        "duplicate_prediction_ids": duplicates,
        "observed_score_min": score_min,
        "observed_score_max": score_max,
        "all_scores_nonnegative": score_min >= 0.0,
    }


def family_slots_order(
    candidates: list[dict[str, Any]],
    utility: Callable[[dict[str, Any]], float],
) -> list[dict[str, Any]]:
    original = source_order(candidates)
    queues: dict[str, list[dict[str, Any]]] = {}
    for family in FAMILIES:
        rows = [row for row in candidates if row["family"] == family]
        queues[family] = (
            source_order(rows)
            if family == "support_contact"
            else sorted(rows, key=lambda row: (-utility(row), row["key"]))
        )
    offsets = {family: 0 for family in FAMILIES}
    result: list[dict[str, Any]] = []
    for slot in original:
        family = slot["family"]
        result.append(queues[family][offsets[family]])
        offsets[family] += 1
    return result


def identity_family_slots_order(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    original = source_order(candidates)
    queues = {
        family: source_order(
            [row for row in candidates if row["family"] == family]
        )
        for family in FAMILIES
    }
    offsets = {family: 0 for family in FAMILIES}
    result: list[dict[str, Any]] = []
    for slot in original:
        family = slot["family"]
        result.append(queues[family][offsets[family]])
        offsets[family] += 1
    return result


def pv_global_order(
    candidates: list[dict[str, Any]],
    utility: Callable[[dict[str, Any]], float],
) -> list[dict[str, Any]]:
    """Merge proximity and vertical rows while fixing every support slot."""
    original = source_order(candidates)
    support = source_order(
        [row for row in candidates if row["family"] == "support_contact"]
    )
    reranked = sorted(
        [row for row in candidates if row["family"] in RERANKED_FAMILIES],
        key=lambda row: (-utility(row), row["key"]),
    )
    support_offset = reranked_offset = 0
    result: list[dict[str, Any]] = []
    for slot in original:
        if slot["family"] == "support_contact":
            result.append(support[support_offset])
            support_offset += 1
        else:
            result.append(reranked[reranked_offset])
            reranked_offset += 1
    return result


def support_order_only_order(
    candidates: list[dict[str, Any]],
    utility: Callable[[dict[str, Any]], float],
) -> list[dict[str, Any]]:
    """Allow global competition but retain source-score utility for support rows."""
    return sorted(
        candidates,
        key=lambda row: (
            -(
                row["semantic"]
                if row["family"] == "support_contact"
                else utility(row)
            ),
            row["key"],
        ),
    )


def all_families_order(
    candidates: list[dict[str, Any]],
    estimator: str,
) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda row: (
            -(row["semantic"] * row[estimator]),
            row["key"],
        ),
    )


def build_orders(
    candidates: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    orders = {
        "source": source_order(candidates),
        "identity_family_slots": identity_family_slots_order(candidates),
    }
    for estimator in ESTIMATORS:
        utility = lambda row, estimator=estimator: (
            row["semantic"] * row[estimator]
        )
        orders[f"{estimator}_family_slots"] = family_slots_order(
            candidates, utility
        )
        orders[f"{estimator}_pv_global"] = pv_global_order(
            candidates, utility
        )
        orders[f"{estimator}_support_order_only"] = support_order_only_order(
            candidates, utility
        )
        orders[f"{estimator}_all_families"] = all_families_order(
            candidates, estimator
        )
    return orders


def method_rows(
    sources: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, payload in sources.items():
        for method in METHODS:
            for k in KS:
                cell = payload["overall"][method][str(k)]
                delta = payload["overall"]["deltas_vs_source"].get(
                    method, {}
                ).get(str(k), {})
                rows.append(
                    {
                        "source": source,
                        "method": method,
                        "k": k,
                        "recall": cell["recall"]["point"],
                        "recall_delta": (delta.get("recall") or {}).get("point"),
                        "recall_delta_ci_low": (
                            (delta.get("recall") or {}).get(
                                "paired_bootstrap_intervals_ci95", [None, None]
                            )[0]
                        ),
                        "recall_delta_ci_high": (
                            (delta.get("recall") or {}).get(
                                "paired_bootstrap_intervals_ci95", [None, None]
                            )[1]
                        ),
                        "violation": cell["violation_all"]["point"],
                        "violation_delta": (
                            delta.get("violation_all") or {}
                        ).get("point"),
                        "violation_delta_ci_low": (
                            (delta.get("violation_all") or {}).get(
                                "paired_bootstrap_intervals_ci95", [None, None]
                            )[0]
                        ),
                        "violation_delta_ci_high": (
                            (delta.get("violation_all") or {}).get(
                                "paired_bootstrap_intervals_ci95", [None, None]
                            )[1]
                        ),
                    }
                )
    return rows


def family_rows(
    sources: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, payload in sources.items():
        for family in FAMILIES:
            report = payload["global_topk_family_slice"][family]
            for method in METHODS:
                for k in KS:
                    cell = report[method][str(k)]
                    rows.append(
                        {
                            "source": source,
                            "family": family,
                            "method": method,
                            "k": k,
                            "selected": cell["counts"]["selected"],
                            "recall": cell["recall"]["point"],
                            "recall_numerator": cell["recall"]["numerator"],
                            "recall_denominator": cell["recall"]["denominator"],
                            "violation": cell["violation_all"]["point"],
                            "violation_numerator": cell["violation_all"][
                                "numerator"
                            ],
                            "violation_denominator": cell["violation_all"][
                                "denominator"
                            ],
                        }
                    )
    return rows


def evaluate_source(
    grouped: dict[str, list[dict[str, Any]]],
    gt: dict[str, set[tuple[Any, ...]]],
    gt_family: dict[str, dict[str, set[tuple[Any, ...]]]],
    contexts: list[str],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    overall_values = robust.empty_values(list(METHODS), contexts)
    family_values = {
        family: robust.empty_values(list(METHODS), contexts)
        for family in FAMILIES
    }
    membership: dict[str, dict[str, dict[str, int]]] = {
        method: {
            str(k): {"promoted": 0, "demoted": 0}
            for k in KS
        }
        for method in METHODS
    }
    checks = {
        "identity_full_order_exact": True,
        "family_slots_family_sequence_exact": True,
        "family_slots_support_ids_exact": True,
        "pv_global_support_slots_exact": True,
        "pv_global_support_ids_exact": True,
        "support_order_only_support_subsequence_exact": True,
    }
    for ci, context in enumerate(contexts):
        candidates = grouped.get(context, [])
        orders = build_orders(candidates)
        source = orders["source"]
        source_support = [
            row["id"] for row in source
            if row["family"] == "support_contact"
        ]
        checks["identity_full_order_exact"] &= (
            [row["id"] for row in source]
            == [row["id"] for row in orders["identity_family_slots"]]
        )
        for estimator in ESTIMATORS:
            family_slots = orders[f"{estimator}_family_slots"]
            checks["family_slots_family_sequence_exact"] &= (
                [row["family"] for row in source]
                == [row["family"] for row in family_slots]
            )
            checks["family_slots_support_ids_exact"] &= (
                source_support
                == [
                    row["id"] for row in family_slots
                    if row["family"] == "support_contact"
                ]
            )
            pv_global = orders[f"{estimator}_pv_global"]
            checks["pv_global_support_slots_exact"] &= (
                [
                    row["family"] == "support_contact"
                    for row in source
                ]
                == [
                    row["family"] == "support_contact"
                    for row in pv_global
                ]
            )
            checks["pv_global_support_ids_exact"] &= (
                source_support
                == [
                    row["id"] for row in pv_global
                    if row["family"] == "support_contact"
                ]
            )
            support_only = orders[f"{estimator}_support_order_only"]
            checks["support_order_only_support_subsequence_exact"] &= (
                source_support
                == [
                    row["id"] for row in support_only
                    if row["family"] == "support_contact"
                ]
            )
        for method, order in orders.items():
            for ki, k in enumerate(KS):
                chosen = order[:k]
                robust.add_cell(
                    overall_values[method],
                    ki,
                    ci,
                    chosen,
                    gt.get(context, set()),
                )
                for family in FAMILIES:
                    robust.add_cell(
                        family_values[family][method],
                        ki,
                        ci,
                        [
                            row for row in chosen
                            if row["family"] == family
                        ],
                        gt_family.get(context, {}).get(family, set()),
                    )
                source_ids = {row["id"] for row in source[:k]}
                method_ids = {row["id"] for row in chosen}
                membership[method][str(k)]["promoted"] += len(
                    method_ids - source_ids
                )
                membership[method][str(k)]["demoted"] += len(
                    source_ids - method_ids
                )
    weights, cluster_counts = scan_bootstrap.scan_weights(
        grouped, contexts, resamples, seed
    )
    route_contrasts: dict[str, Any] = {}
    for estimator in ESTIMATORS:
        reference = f"{estimator}_family_slots"
        route_contrasts[estimator] = {}
        for route in ("pv_global", "support_order_only", "all_families"):
            method = f"{estimator}_{route}"
            route_contrasts[estimator][route] = {}
            for ki, k in enumerate(KS):
                route_contrasts[estimator][route][str(k)] = {}
                for metric in ("recall", "violation_all"):
                    left_num, left_den = robust.ratio_arrays(
                        overall_values[method], metric, ki
                    )
                    right_num, right_den = robust.ratio_arrays(
                        overall_values[reference], metric, ki
                    )
                    left_point = (
                        float(left_num.sum() / left_den.sum())
                        if left_den.sum()
                        else None
                    )
                    right_point = (
                        float(right_num.sum() / right_den.sum())
                        if right_den.sum()
                        else None
                    )
                    delta = (
                        robust.weighted_ratio(left_num, left_den, weights)
                        - robust.weighted_ratio(right_num, right_den, weights)
                    )
                    route_contrasts[estimator][route][str(k)][metric] = {
                        "point": (
                            left_point - right_point
                            if left_point is not None
                            and right_point is not None
                            else None
                        ),
                        "paired_bootstrap_intervals_ci95": base.ci95(delta),
                    }
    return {
        "counts": {
            **cluster_counts,
            "evaluation_contexts": len(contexts),
            "zero_prediction_contexts": len(set(contexts) - set(grouped)),
            "gt_denominator": sum(len(rows) for rows in gt.values()),
        },
        "overall": robust.summarize(
            overall_values, list(METHODS), weights
        ),
        "global_topk_family_slice": {
            family: robust.summarize(
                family_values[family], list(METHODS), weights
            )
            for family in FAMILIES
        },
        "membership_vs_source": membership,
        "route_checks": checks,
        "route_minus_family_slots": route_contrasts,
    }


def reported_match(
    sources: dict[str, Any],
    reported: dict[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    mapping = {
        "source": "source",
        "identity_family_slots": "source",
        "linear_family_slots": "relcompat3d_linear",
        "mlp_family_slots": "relcompat3d_mlp",
    }
    rows: list[dict[str, Any]] = []
    exact = True
    for source, payload in sources.items():
        for local, reference_method in mapping.items():
            for k in KS:
                for local_metric, reference_metric in (
                    ("recall", "recall"),
                    ("violation_all", "violation_all"),
                ):
                    actual = payload["overall"][local][str(k)][
                        local_metric
                    ]["point"]
                    expected = reported["sources"][source]["results"][
                        reference_method
                    ][str(k)][reference_metric]["point"]
                    error = abs(actual - expected)
                    exact &= error <= 1e-12
                    rows.append(
                        {
                            "source": source,
                            "method": local,
                            "reference_method": reference_method,
                            "k": k,
                            "metric": local_metric,
                            "actual": actual,
                            "expected": expected,
                            "absolute_error": error,
                        }
                    )
    return exact, rows


def summary_markdown(sources: dict[str, Any], status: str) -> str:
    lines = [
        "# Family-Aware Routing Constraint Controls",
        "",
        f"Status: `{status}`",
        "",
        "The direct matched control is `pv_global`: it uses the same candidates, "
        "compatibility estimator, product utility, support/contact slots, and "
        "support/contact order as `family_slots`, but merges proximity and "
        "vertical-order candidates into one queue.",
        "",
        "| Source | Estimator | Route | R@50 | V@50 | R@100 | V@100 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for source, payload in sources.items():
        for estimator in ESTIMATORS:
            for route in ROUTES:
                method = f"{estimator}_{route}"
                k50 = payload["overall"][method]["50"]
                k100 = payload["overall"][method]["100"]
                lines.append(
                    f"| {source} | {estimator} | {route} | "
                    f"{100 * k50['recall']['point']:.2f} | "
                    f"{100 * k50['violation_all']['point']:.2f} | "
                    f"{100 * k100['recall']['point']:.2f} | "
                    f"{100 * k100['violation_all']['point']:.2f} |"
                )
    lines.extend(
        [
            "",
            "`support_order_only` additionally removes fixed support positions "
            "while preserving the relative source order of support/contact rows. "
            "`all_families` is a scope comparison that also applies compatibility "
            "to support/contact and is not a matched test of the family-slot "
            "constraint.",
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
    paths = {
        name: resolve(root, value)
        for name, value in protocol["inputs"].items()
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing_inputs:{missing}")
    for name, expected in protocol["expected_sha256"].items():
        actual = sha256_file(paths[name])
        if actual != expected:
            raise ValueError(
                f"hash_mismatch:{name}:expected={expected}:actual={actual}"
            )

    linear_models = json.loads(
        paths["linear_models"].read_text(encoding="utf-8")
    )
    nonlinear_models = json.loads(
        paths["nonlinear_models"].read_text(encoding="utf-8")
    )
    linear_score = base.make_linear_scorer(linear_models)
    mlp_model = nonlinear_models["shared_mlp_pairwise"]
    gt, gt_family = model_eval.load_gt(paths["ground_truth"])
    annotations = json.loads(
        paths["official_context_annotations"].read_text(encoding="utf-8")
    )
    contexts = sorted(
        f"{row['scan']}_{row['split']}" for row in annotations["scans"]
    )
    source_paths = {
        "vlsat": paths["vlsat_verification"],
        "open3dsg": paths["open3dsg_verification"],
        "sgfn": paths["sgfn_verification"],
    }
    resamples = int(protocol["evaluation"]["bootstrap_resamples"])
    seed = int(protocol["evaluation"]["bootstrap_seed"])
    sources: dict[str, Any] = {}
    input_counts: dict[str, Any] = {}
    for index, (source, path) in enumerate(source_paths.items()):
        grouped, input_counts[source] = load_candidates(
            path, linear_score, mlp_model
        )
        sources[source] = evaluate_source(
            grouped,
            gt,
            gt_family,
            contexts,
            resamples,
            seed + index,
        )

    reference = json.loads(
        paths["reported_summary"].read_text(encoding="utf-8")
    )
    reference_exact, reference_rows = reported_match(sources, reference)
    route_checks = {
        source: payload["route_checks"]
        for source, payload in sources.items()
    }
    validations = {
        "all_input_hashes_match": True,
        "all_sources_have_548_contexts": all(
            payload["counts"]["evaluation_contexts"] == 548
            for payload in sources.values()
        ),
        "all_sources_have_157_scans": all(
            payload["counts"]["scans"] == 157
            for payload in sources.values()
        ),
        "all_prediction_ids_unique": all(
            cell["duplicate_prediction_ids"] == 0
            for cell in input_counts.values()
        ),
        "all_source_scores_nonnegative": all(
            cell["all_scores_nonnegative"]
            for cell in input_counts.values()
        ),
        "reported_points_exact": reference_exact,
        "identity_route_exact": all(
            checks["identity_full_order_exact"]
            for checks in route_checks.values()
        ),
        "family_slots_preserve_family_sequence": all(
            checks["family_slots_family_sequence_exact"]
            for checks in route_checks.values()
        ),
        "family_slots_preserve_support_ids": all(
            checks["family_slots_support_ids_exact"]
            for checks in route_checks.values()
        ),
        "pv_global_preserves_support_slots": all(
            checks["pv_global_support_slots_exact"]
            for checks in route_checks.values()
        ),
        "pv_global_preserves_support_ids": all(
            checks["pv_global_support_ids_exact"]
            for checks in route_checks.values()
        ),
        "support_order_only_preserves_support_subsequence": all(
            checks["support_order_only_support_subsequence_exact"]
            for checks in route_checks.values()
        ),
    }
    status = "completed" if all(validations.values()) else "failed_validation"
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "relcompat3d_routing_constraint_controls_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "methods": list(METHODS),
        "route_definitions": protocol["routes"],
        "bootstrap_unit": protocol["evaluation"]["bootstrap_unit"],
        "bootstrap_resamples": resamples,
        "input_counts": input_counts,
        "sources": sources,
        "validations": validations,
        "evaluation_scope": protocol["evaluation_scope"],
    }
    write_json(out / "summary.json", summary)
    write_csv(out / "metrics.csv", method_rows(sources))
    write_csv(out / "family_metrics.csv", family_rows(sources))
    write_csv(out / "result_check.csv", reference_rows)
    membership_rows = [
        {
            "source": source,
            "method": method,
            "k": k,
            **payload["membership_vs_source"][method][str(k)],
        }
        for source, payload in sources.items()
        for method in METHODS
        for k in KS
    ]
    write_csv(out / "membership.csv", membership_rows)
    (out / "summary.md").write_text(
        summary_markdown(sources, status), encoding="utf-8"
    )
    outputs = (
        "summary.json",
        "summary.md",
        "metrics.csv",
        "family_metrics.csv",
        "result_check.csv",
        "membership.csv",
    )
    manifest = {
        "schema_version": "relcompat3d_routing_constraint_manifest_v1",
        "status": status,
        "protocol": {
            "path": relpath(root, protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "inputs": {
            name: {
                "path": relpath(root, path),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
        "outputs": {
            name: {
                "path": relpath(root, out / name),
                "sha256": sha256_file(out / name),
            }
            for name in outputs
        },
        "validations": validations,
        "docker_command": (
            "env UID=$(id -u) GID=$(id -g) docker compose "
            "-f configs/relcompat3d/compose.yaml run --rm "
            "relcompat3d_routing_constraints"
        ),
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"status": status, "validations": validations}))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
