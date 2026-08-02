#!/usr/bin/env python3
"""Run fixed-model RelCompat3D corruption and information ablations."""

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

import relation_consistency as algebra
import evaluate_main as structured
import evaluate_train_only as strict


FAMILIES = ("support_contact", "proximity", "relative_vertical")
METHODS = (
    "source_score",
    "structured_product",
    "wrong_predicate_product",
    "wrong_pair_product",
    "shuffled_geometry_product",
    "endpoint_swap_fixed_label_product",
    "distance_only",
    "compatibility_only",
)
KS = (50, 100)
EXPECTED_ROWS = {"vlsat": 220848, "open3dsg": 160596, "sgfn": 220848}
SOURCE_LABELS = {"vlsat": "VL-SAT", "open3dsg": "Open3DSG", "sgfn": "SGFN"}


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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def ci95(samples: np.ndarray) -> list[float | None]:
    finite = samples[np.isfinite(samples)]
    if not len(finite):
        return [None, None]
    return [float(value) for value in np.percentile(finite, (2.5, 97.5))]


def ratio_samples(
    numerator: np.ndarray, denominator: np.ndarray, samples: np.ndarray
) -> np.ndarray:
    boot_num = numerator[samples].sum(axis=1)
    boot_den = denominator[samples].sum(axis=1)
    return np.divide(
        boot_num,
        boot_den,
        out=np.full_like(boot_num, np.nan, dtype=np.float64),
        where=boot_den > 0,
    )


def endpoint_pair(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["edge"]["subject_id"]), int(row["edge"]["object_id"])


def load_rows(
    path: Path,
    score_compatibility: Callable[[str, str, dict[str, float]], float],
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
            source_row = json.loads(raw_line)
            family = source_row["predicate"]["predicate_family"]
            if family not in FAMILIES:
                continue
            in_scope_rows += 1
            predicate = source_row["predicate"]["predicate_label"]
            raw = strict.raw_numeric(source_row)
            semantic = strict.finite(
                (source_row.get("semantic") or {}).get("ranking_score")
            )
            if semantic is None:
                raise ValueError(f"missing_semantic:{source_row['prediction_id']}")
            grouped[source_row["subgraph_id"]].append(
                {
                    "id": source_row["prediction_id"],
                    "scan": source_row["scan_id"],
                    "key": strict.candidate_key(source_row),
                    "pair": endpoint_pair(source_row),
                    "family": family,
                    "predicate": predicate,
                    "raw": raw,
                    "semantic": float(semantic),
                    "compatibility": float(score_compatibility(family, predicate, raw)),
                    "status": source_row.get("verification_status")
                    or (source_row.get("verification") or {}).get("verification_status"),
                    "scores": {},
                }
            )
    return grouped, {
        "input_rows": input_rows,
        "in_scope_rows": in_scope_rows,
        "input_sha256": digest.hexdigest(),
    }


def rotate(items: list[dict[str, Any]], offset: int) -> dict[str, dict[str, float]]:
    if len(items) < 2:
        raise ValueError("donor_group_too_small")
    ordered = sorted(items, key=lambda row: row["key"])
    shift = offset % len(ordered)
    if shift == 0:
        shift = 1
    return {
        row["id"]: ordered[(index + shift) % len(ordered)]["raw"]
        for index, row in enumerate(ordered)
    }


def add_scores(
    grouped: dict[str, list[dict[str, Any]]],
    score_compatibility: Callable[[str, str, dict[str, float]], float],
    wrong_predicate: dict[str, str],
) -> dict[str, Any]:
    wrong_pair_donors: dict[str, dict[str, float]] = {}
    stream_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    wrong_pair_groups = 0
    for rows in grouped.values():
        local: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            local[(row["family"], row["predicate"])].append(row)
            stream_groups[(row["family"], row["predicate"])].append(row)
        for items in local.values():
            wrong_pair_donors.update(rotate(items, 1))
            wrong_pair_groups += 1

    shuffled_donors: dict[str, dict[str, float]] = {}
    for items in stream_groups.values():
        shuffled_donors.update(rotate(items, max(1, len(items) // 2)))

    wrong_pair_self = 0
    endpoint_applicable = 0
    wrong_predicate_rows = 0
    for rows in grouped.values():
        for row in rows:
            family = row["family"]
            predicate = row["predicate"]
            raw = row["raw"]
            semantic = row["semantic"]
            wrong_name = wrong_predicate.get(predicate)
            if wrong_name is None:
                raise ValueError(f"missing_wrong_predicate_mapping:{predicate}")
            wrong_predicate_rows += 1
            wrong_pair_raw = wrong_pair_donors[row["id"]]
            shuffled_raw = shuffled_donors[row["id"]]
            if wrong_pair_raw is raw:
                wrong_pair_self += 1

            endpoint = algebra.transformed_view(family, predicate, raw)
            endpoint_raw = raw
            if endpoint is not None:
                _, endpoint_raw = endpoint
                endpoint_applicable += 1

            distance = raw.get("distance_3d")
            distance_score = (
                1.0 / (1.0 + max(distance, 0.0))
                if distance is not None and math.isfinite(distance)
                else 0.0
            )
            row["scores"] = {
                "source_score": semantic,
                "structured_product": semantic * row["compatibility"],
                "wrong_predicate_product": semantic
                * score_compatibility(family, wrong_name, raw),
                "wrong_pair_product": semantic
                * score_compatibility(family, predicate, wrong_pair_raw),
                "shuffled_geometry_product": semantic
                * score_compatibility(family, predicate, shuffled_raw),
                "endpoint_swap_fixed_label_product": semantic
                * score_compatibility(family, predicate, endpoint_raw),
                "distance_only": distance_score,
                "compatibility_only": row["compatibility"],
            }
    return {
        "wrong_pair_donor_rows": len(wrong_pair_donors),
        "shuffled_donor_rows": len(shuffled_donors),
        "wrong_pair_groups": wrong_pair_groups,
        "wrong_pair_self_donors": wrong_pair_self,
        "wrong_predicate_rows": wrong_predicate_rows,
        "endpoint_applicable_rows": endpoint_applicable,
    }


def evaluate(
    grouped: dict[str, list[dict[str, Any]]],
    gt: dict[str, set[tuple[Any, ...]]],
    contexts: list[str],
    samples: np.ndarray,
) -> dict[str, Any]:
    arrays: dict[str, dict[str, np.ndarray]] = {
        method: {
            name: np.zeros((len(KS), len(contexts)), dtype=np.float64)
            for name in ("recall_num", "recall_den", "violation_num", "status_den", "selected")
        }
        for method in METHODS
    }
    for ci, context in enumerate(contexts):
        rows = grouped.get(context, [])
        gt_rows = gt.get(context, set())
        for method in METHODS:
            ranked = sorted(rows, key=lambda row: (-row["scores"][method], row["key"]))
            for ki, k in enumerate(KS):
                selected = ranked[:k]
                cell = arrays[method]
                cell["recall_num"][ki, ci] = len(
                    {row["key"] for row in selected} & gt_rows
                )
                cell["recall_den"][ki, ci] = len(gt_rows)
                cell["selected"][ki, ci] = len(selected)
                statuses = [row["status"] for row in selected]
                cell["violation_num"][ki, ci] = statuses.count("violated")
                cell["status_den"][ki, ci] = sum(
                    status in {"satisfied", "uncertain", "violated"}
                    for status in statuses
                )

    report: dict[str, Any] = {}
    cache: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for method in METHODS:
        report[method] = {}
        for ki, k in enumerate(KS):
            report[method][str(k)] = {}
            for metric, numerator_name, denominator_name in (
                ("recall", "recall_num", "recall_den"),
                ("violation", "violation_num", "status_den"),
            ):
                numerator = arrays[method][numerator_name][ki]
                denominator = arrays[method][denominator_name][ki]
                boot = ratio_samples(numerator, denominator, samples)
                point = float(numerator.sum() / denominator.sum())
                report[method][str(k)][metric] = {
                    "point": point,
                    "ci95": ci95(boot),
                    "numerator": int(numerator.sum()),
                    "denominator": int(denominator.sum()),
                }
                cache[method][str(k)][metric] = boot
            report[method][str(k)]["selected"] = int(
                arrays[method]["selected"][ki].sum()
            )

    report["deltas_vs_structured_product"] = {}
    for method in METHODS:
        if method == "structured_product":
            continue
        report["deltas_vs_structured_product"][method] = {}
        for k in KS:
            report["deltas_vs_structured_product"][method][str(k)] = {}
            for metric in ("recall", "violation"):
                point = (
                    report[method][str(k)][metric]["point"]
                    - report["structured_product"][str(k)][metric]["point"]
                )
                delta = (
                    cache[method][str(k)][metric]
                    - cache["structured_product"][str(k)][metric]
                )
                report["deltas_vs_structured_product"][method][str(k)][metric] = {
                    "point": point,
                    "paired_ci95": ci95(delta),
                }
    return report


def make_csv(summary: dict[str, Any]) -> list[dict[str, Any]]:
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
                        "recall_ci95_low": cell["recall"]["ci95"][0],
                        "recall_ci95_high": cell["recall"]["ci95"][1],
                        "violation": cell["violation"]["point"],
                        "violation_ci95_low": cell["violation"]["ci95"][0],
                        "violation_ci95_high": cell["violation"]["ci95"][1],
                        "selected": cell["selected"],
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Structured Ablation Evaluation",
        "",
        f"Status: `{summary['status']}`",
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
            "`compatibility_only` excludes the predictor score but remains predicate-conditioned; it is not true raw-G-only. The endpoint row corrupts the ordered-pair geometry while retaining the original label and leaves support/contact untouched.",
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
    if protocol.get("status") != "frozen_before_structured_ablation_execution":
        raise ValueError("protocol_not_frozen")
    if tuple(protocol["evaluation"]["ks"]) != KS:
        raise ValueError("k_contract_mismatch")
    if tuple(protocol["conditions"]) != METHODS:
        raise ValueError("method_contract_mismatch")

    paths: dict[str, Path] = {}
    input_checks: dict[str, Any] = {}
    for name, spec in protocol["inputs"].items():
        path = resolve(root, spec["path"])
        if not path.exists():
            raise FileNotFoundError(f"missing_input:{name}:{path}")
        actual = sha256_file(path)
        if actual != spec["sha256"]:
            raise ValueError(f"input_hash_mismatch:{name}:{actual}")
        paths[name] = path
        input_checks[name] = {
            "path": relpath(root, path),
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }

    models = json.loads(paths["structured_models"].read_text(encoding="utf-8"))
    if models.get("source_score_used") is not False or models.get("source_identity_used") is not False:
        raise ValueError("structured_model_uses_source")
    score_compatibility = structured.make_structured_scorer(models)
    gt, _ = strict.load_gt(paths["ground_truth"])
    main_summary = json.loads(paths["structured_main_summary"].read_text(encoding="utf-8"))
    source_paths = {
        "vlsat": paths["vlsat_verification"],
        "open3dsg": paths["open3dsg_verification"],
        "sgfn": paths["sgfn_verification"],
    }

    sources: dict[str, Any] = {}
    equivalence: dict[str, Any] = {}
    for source_index, (source, path) in enumerate(source_paths.items()):
        grouped, counts = load_rows(path, score_compatibility)
        donor_audit = add_scores(
            grouped, score_compatibility, protocol["wrong_predicate_mapping"]
        )
        contexts = sorted(set(grouped) | set(gt))
        samples = np.random.default_rng(
            int(protocol["evaluation"]["bootstrap_seed"]) + source_index
        ).integers(
            0,
            len(contexts),
            size=(
                int(protocol["evaluation"]["bootstrap_resamples"]),
                len(contexts),
            ),
        )
        metrics = evaluate(grouped, gt, contexts, samples)
        sources[source] = {
            "counts": {
                **counts,
                "contexts": len(contexts),
                "gt_denominator": sum(len(items) for items in gt.values()),
            },
            "donor_audit": donor_audit,
            "metrics": metrics,
        }
        source_equivalence: dict[str, Any] = {}
        for method in ("source_score", "structured_product"):
            source_equivalence[method] = {}
            for k in KS:
                current = metrics[method][str(k)]
                previous = main_summary["sources"][source]["overall"][method][str(k)]
                source_equivalence[method][str(k)] = {
                    "recall_abs_error": abs(
                        current["recall"]["point"] - previous["recall"]["point"]
                    ),
                    "violation_abs_error": abs(
                        current["violation"]["point"]
                        - previous["violation_all"]["point"]
                    ),
                }
        equivalence[source] = source_equivalence

    validations = {
        "all_input_hashes_match": len(input_checks) == len(protocol["inputs"]),
        "model_excludes_source": models.get("source_score_used") is False
        and models.get("source_identity_used") is False,
        "contexts_548": all(payload["counts"]["contexts"] == 548 for payload in sources.values()),
        "gt_denominator_3972": all(
            payload["counts"]["gt_denominator"] == 3972 for payload in sources.values()
        ),
        "in_scope_row_counts": all(
            sources[source]["counts"]["in_scope_rows"] == expected
            for source, expected in EXPECTED_ROWS.items()
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
        "wrong_predicate_full_coverage": all(
            payload["donor_audit"]["wrong_predicate_rows"]
            == payload["counts"]["in_scope_rows"]
            for payload in sources.values()
        ),
        "source_and_main_point_equivalence": all(
            cell[metric] <= 1e-15
            for source_payload in equivalence.values()
            for method_payload in source_payload.values()
            for cell in method_payload.values()
            for metric in ("recall_abs_error", "violation_abs_error")
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
        "schema_version": "relcompat3d_structured_ablation_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "classification": protocol["classification"],
        "methods": list(METHODS),
        "ks": list(KS),
        "sources": sources,
        "point_equivalence_to_structured_main": equivalence,
        "validations": validations,
        "claim_boundary": protocol["claim_boundary"],
        "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm structured_ablation_evaluation",
    }

    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    write_json(summary_path, summary)
    summary_md = out / "summary.md"
    summary_md.write_text(markdown(summary), encoding="utf-8")
    metrics_path = out / "metrics.csv"
    write_csv(metrics_path, make_csv(summary))
    manifest = {
        "schema_version": "relcompat3d_structured_ablation_manifest_v1",
        "status": status,
        "protocol": {
            "path": relpath(root, protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "inputs": input_checks,
        "outputs": {
            path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in (summary_path, summary_md, metrics_path)
        },
        "validations": validations,
    }
    write_json(out / "manifest.json", manifest)
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
