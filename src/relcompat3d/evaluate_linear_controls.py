#!/usr/bin/env python3
"""Evaluate fixed RelCompat3D ablations through the primary family-slot route."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import control_utils as ablation
import evaluate_main as structured
import evaluate_train_only as strict


METHODS = ablation.METHODS
KS = ablation.KS
FAMILIES = ablation.FAMILIES
SOURCE_LABELS = ablation.SOURCE_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def add_routed_scores(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    support_exact = True
    composition_exact = True
    for rows in grouped.values():
        source_order = sorted(rows, key=lambda row: (-row["semantic"], row["key"]))
        source_support_order = [row["id"] for row in source_order if row["family"] == "support_contact"]
        source_family_sequence = [row["family"] for row in source_order]
        for method in METHODS:
            queues: dict[str, list[dict[str, Any]]] = {}
            for family in FAMILIES:
                family_rows = [row for row in rows if row["family"] == family]
                if family == "support_contact":
                    queues[family] = sorted(
                        family_rows, key=lambda row: (-row["semantic"], row["key"])
                    )
                else:
                    queues[family] = sorted(
                        family_rows,
                        key=lambda row: (-row["scores"][method], row["key"]),
                    )
            offsets = {family: 0 for family in FAMILIES}
            routed: list[dict[str, Any]] = []
            for family in source_family_sequence:
                routed.append(queues[family][offsets[family]])
                offsets[family] += 1
            support_exact &= (
                [row["id"] for row in routed if row["family"] == "support_contact"]
                == source_support_order
            )
            composition_exact &= [row["family"] for row in routed] == source_family_sequence
            for rank, row in enumerate(routed, 1):
                row.setdefault("routed_scores", {})[method] = float(len(routed) - rank + 1)
    return {
        "support_contact_order_exact": support_exact,
        "family_composition_exact": composition_exact,
    }


def official_context_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {
        f"{row['scan']}_{row['split']}": str(row["scan"])
        for row in payload["scans"]
    }
    if len(result) != 548:
        raise ValueError(f"official_context_count:{len(result)}")
    return result


def scan_weights(
    contexts: list[str],
    context_to_scan: dict[str, str],
    resamples: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    scans = sorted(set(context_to_scan.values()))
    scan_index = {scan: index for index, scan in enumerate(scans)}
    context_scan = np.asarray([scan_index[context_to_scan[context]] for context in contexts])
    sampled = np.random.default_rng(seed).integers(
        0, len(scans), size=(resamples, len(scans))
    )
    counts = np.zeros((resamples, len(scans)), dtype=np.float64)
    for index in range(resamples):
        counts[index] = np.bincount(sampled[index], minlength=len(scans))
    per_scan = Counter(context_to_scan.values())
    return counts[:, context_scan], {
        "scans": len(scans),
        "contexts": len(contexts),
        "min_contexts_per_scan": min(per_scan.values()),
        "max_contexts_per_scan": max(per_scan.values()),
    }


def weighted_ratio(
    numerator: np.ndarray, denominator: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    boot_num = weights @ numerator
    boot_den = weights @ denominator
    return np.divide(
        boot_num,
        boot_den,
        out=np.full(boot_num.shape, np.nan, dtype=np.float64),
        where=boot_den > 0,
    )


def evaluate(
    grouped: dict[str, list[dict[str, Any]]],
    gt: dict[str, set[tuple[Any, ...]]],
    contexts: list[str],
    weights: np.ndarray,
) -> dict[str, Any]:
    arrays = {
        method: {
            name: np.zeros((len(KS), len(contexts)), dtype=np.float64)
            for name in (
                "recall_num",
                "recall_den",
                "violation_num",
                "status_den",
                "selected",
            )
        }
        for method in METHODS
    }
    for context_index, context in enumerate(contexts):
        rows = grouped.get(context, [])
        gt_rows = gt.get(context, set())
        for method in METHODS:
            ranked = sorted(
                rows,
                key=lambda row: (-row["routed_scores"][method], row["key"]),
            )
            for k_index, k in enumerate(KS):
                selected = ranked[:k]
                cell = arrays[method]
                cell["recall_num"][k_index, context_index] = len(
                    {row["key"] for row in selected} & gt_rows
                )
                cell["recall_den"][k_index, context_index] = len(gt_rows)
                cell["selected"][k_index, context_index] = len(selected)
                statuses = [row["status"] for row in selected]
                cell["violation_num"][k_index, context_index] = statuses.count(
                    "violated"
                )
                cell["status_den"][k_index, context_index] = sum(
                    status in {"satisfied", "uncertain", "violated"}
                    for status in statuses
                )

    report: dict[str, Any] = {}
    cache: dict[str, dict[str, dict[str, np.ndarray]]] = {
        method: {} for method in METHODS
    }
    for method in METHODS:
        report[method] = {}
        for k_index, k in enumerate(KS):
            report[method][str(k)] = {}
            cache[method][str(k)] = {}
            for metric, numerator_name, denominator_name in (
                ("recall", "recall_num", "recall_den"),
                ("violation", "violation_num", "status_den"),
            ):
                numerator = arrays[method][numerator_name][k_index]
                denominator = arrays[method][denominator_name][k_index]
                point = float(numerator.sum() / denominator.sum())
                boot = weighted_ratio(numerator, denominator, weights)
                report[method][str(k)][metric] = {
                    "point": point,
                    "scan_cluster_ci95": ablation.ci95(boot),
                    "numerator": int(numerator.sum()),
                    "denominator": int(denominator.sum()),
                }
                cache[method][str(k)][metric] = boot
            report[method][str(k)]["selected"] = int(
                arrays[method]["selected"][k_index].sum()
            )

    report["deltas_vs_structured_product"] = {}
    for method in METHODS:
        if method == "structured_product":
            continue
        report["deltas_vs_structured_product"][method] = {}
        for k in KS:
            report["deltas_vs_structured_product"][method][str(k)] = {}
            for metric in ("recall", "violation"):
                left = report[method][str(k)][metric]["point"]
                right = report["structured_product"][str(k)][metric]["point"]
                delta = (
                    cache[method][str(k)][metric]
                    - cache["structured_product"][str(k)][metric]
                )
                report["deltas_vs_structured_product"][method][str(k)][metric] = {
                    "point": left - right,
                    "paired_scan_cluster_ci95": ablation.ci95(delta),
                }
    return report


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
                        "recall_scan_ci95_low": cell["recall"]["scan_cluster_ci95"][0],
                        "recall_scan_ci95_high": cell["recall"]["scan_cluster_ci95"][1],
                        "violation": cell["violation"]["point"],
                        "violation_scan_ci95_low": cell["violation"]["scan_cluster_ci95"][0],
                        "violation_scan_ci95_high": cell["violation"]["scan_cluster_ci95"][1],
                        "selected": cell["selected"],
                    }
                )
    return rows


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Public/Full Routed Ablation Evaluation",
        "",
        f"Status: `{summary['status']}`",
        "",
        "All conditions preserve the source family-slot sequence and support/contact order; only proximity and vertical candidates are reordered.",
        "",
        "| Source | Condition | R@50 | V@50 | R@100 | V@100 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for source, payload in summary["sources"].items():
        for method in METHODS:
            metrics = payload["metrics"][method]
            lines.append(
                f"| {SOURCE_LABELS[source]} | `{method}` | "
                f"{metrics['50']['recall']['point']:.4f} | "
                f"{metrics['50']['violation']['point']:.4f} | "
                f"{metrics['100']['recall']['point']:.4f} | "
                f"{metrics['100']['violation']['point']:.4f} |"
            )
    lines.extend(
        [
            "",
            "`compatibility_only` removes the source score only inside the routed proximity/vertical families; support/contact remains a source-order pass-through.",
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
    if protocol.get("status") != "frozen_before_routed_public_ablation_execution":
        raise ValueError("protocol_not_frozen")
    if tuple(protocol["evaluation"]["ks"]) != KS:
        raise ValueError("k_contract_mismatch")
    # JSON object order is not part of the protocol semantics.  The evaluator
    # uses the fixed METHODS tuple below, so validate exact membership rather
    # than the serialization order of the condition descriptions.
    if len(protocol["conditions"]) != len(METHODS) or set(protocol["conditions"]) != set(METHODS):
        raise ValueError("method_contract_mismatch")

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

    models = json.loads(paths["structured_models"].read_text(encoding="utf-8"))
    if models.get("source_score_used") is not False or models.get("source_identity_used") is not False:
        raise ValueError("structured_model_uses_source")
    scorer = structured.make_structured_scorer(models)
    gt, _ = strict.load_gt(paths["ground_truth"])
    context_to_scan = official_context_map(paths["official_context_annotations"])
    contexts = sorted(context_to_scan)
    routing_summary = json.loads(paths["routing_summary"].read_text(encoding="utf-8"))
    official_summary = json.loads(paths["open3dsg_official_summary"].read_text(encoding="utf-8"))
    source_paths = {
        "vlsat": paths["vlsat_verification"],
        "open3dsg": paths["open3dsg_verification"],
        "sgfn": paths["sgfn_verification"],
    }

    expected_rows = protocol["evaluation"]["expected_in_scope_rows"]
    sources: dict[str, Any] = {}
    equivalence: dict[str, Any] = {}
    for source_index, (source, path) in enumerate(source_paths.items()):
        grouped, counts = ablation.load_rows(path, scorer)
        donor_audit = ablation.add_scores(
            grouped, scorer, protocol["wrong_predicate_mapping"]
        )
        routing_audit = add_routed_scores(grouped)
        weights, scan_counts = scan_weights(
            contexts,
            context_to_scan,
            int(protocol["evaluation"]["bootstrap_resamples"]),
            int(protocol["evaluation"]["bootstrap_seed"]) + source_index,
        )
        metrics = evaluate(grouped, gt, contexts, weights)
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
        if source == "open3dsg":
            reference = official_summary["routes"]["official_strict_full_548"]["overall"]
        else:
            reference = routing_summary["sources"][source]["overall"]
        equivalence[source] = {}
        for current_method, reference_method in (
            ("source_score", "source_score"),
            ("structured_product", "family_slot_rerank"),
        ):
            equivalence[source][current_method] = {}
            for k in KS:
                current = metrics[current_method][str(k)]
                previous = reference[reference_method][str(k)]
                equivalence[source][current_method][str(k)] = {
                    "recall_abs_error": abs(
                        current["recall"]["point"] - previous["recall"]["point"]
                    ),
                    "violation_abs_error": abs(
                        current["violation"]["point"]
                        - previous["violation_all"]["point"]
                    ),
                }

    validations = {
        "all_input_hashes_match": len(input_checks) == len(protocol["inputs"]),
        "model_excludes_source": models.get("source_score_used") is False
        and models.get("source_identity_used") is False,
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
        "primary_point_estimates_match_main_route": all(
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
        "schema_version": "relcompat3d_routed_public_ablation_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "classification": protocol["classification"],
        "ranking_rule": "family-slot applicability route",
        "bootstrap_unit": "scan_id cluster",
        "methods": list(METHODS),
        "ks": list(KS),
        "sources": sources,
        "point_equivalence_to_main_route": equivalence,
        "validations": validations,
        "claim_boundary": protocol["claim_boundary"],
        "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm routed_public_ablation_evaluation",
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
        "schema_version": "relcompat3d_routed_public_ablation_manifest_v1",
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
