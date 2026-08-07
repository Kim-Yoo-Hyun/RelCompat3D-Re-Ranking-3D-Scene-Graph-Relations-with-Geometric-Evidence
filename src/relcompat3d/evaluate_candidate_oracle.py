#!/usr/bin/env python3
"""Compute fixed-candidate exact-match Recall upper bounds from derived rows."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SOURCES = ("vlsat", "open3dsg", "sgfn")
SOURCE_LABELS = {
    "vlsat": "VL-SAT",
    "open3dsg": "Open3DSG",
    "sgfn": "SGFN",
}
FAMILIES = ("support_contact", "proximity", "relative_vertical")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
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


def read_csv_gz(path: Path) -> Iterable[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_ground_truth(
    path: Path,
) -> tuple[
    dict[str, set[str]],
    dict[str, dict[str, set[str]]],
    int,
]:
    overall: dict[str, set[str]] = defaultdict(set)
    by_family: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    count = 0
    for row in read_csv_gz(path):
        context = row["context_uid"]
        family = row["family"]
        candidate = row["candidate_uid"]
        overall[context].add(candidate)
        by_family[context][family].add(candidate)
        count += 1
    return overall, by_family, count


def evaluate_source(
    path: Path,
    ground_truth: dict[str, set[str]],
    ground_truth_by_family: dict[str, dict[str, set[str]]],
    denominator: int,
    ks: tuple[int, ...],
) -> dict[str, Any]:
    pool: dict[str, set[str]] = defaultdict(set)
    pool_by_family: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    source_slots: dict[int, dict[str, dict[str, int]]] = {
        k: defaultdict(lambda: defaultdict(int)) for k in ks
    }
    source_support_hits: dict[int, dict[str, int]] = {
        k: defaultdict(int) for k in ks
    }
    observed: dict[str, dict[int, int]] = {
        "Source": {k: 0 for k in ks},
        "RelCompat3D-Linear": {k: 0 for k in ks},
        "RelCompat3D-MLP": {k: 0 for k in ks},
    }
    row_count = 0
    seen_candidates: set[tuple[str, str]] = set()
    for row in read_csv_gz(path):
        row_count += 1
        context = row["context_uid"]
        family = row["family"]
        candidate = row["candidate_uid"]
        key = (context, candidate)
        if key in seen_candidates:
            raise ValueError(f"duplicate_relation_candidate:{path}:{key}")
        seen_candidates.add(key)
        exact = bool(int(row["exact_match"]))
        if exact:
            pool[context].add(candidate)
            pool_by_family[context][family].add(candidate)
        source_rank = int(row["rank_source"])
        for k in ks:
            if source_rank <= k:
                source_slots[k][context][family] += 1
                if family == "support_contact" and exact:
                    source_support_hits[k][context] += 1
            for method, rank_column in (
                ("Source", "rank_source"),
                ("RelCompat3D-Linear", "rank_linear"),
                ("RelCompat3D-MLP", "rank_mlp"),
            ):
                if exact and int(row[rank_column]) <= k:
                    observed[method][k] += 1

    contexts = sorted(ground_truth)
    pool_hits = sum(len(pool.get(context, set())) for context in contexts)
    oracle: dict[str, dict[int, int]] = {
        "Unconstrained oracle": {k: 0 for k in ks},
        "Family-slot oracle": {k: 0 for k in ks},
        "Active-route oracle": {k: 0 for k in ks},
    }
    for context in contexts:
        pool_total = len(pool.get(context, set()))
        for k in ks:
            oracle["Unconstrained oracle"][k] += min(k, pool_total)
            family_total = 0
            for family in FAMILIES:
                slots = source_slots[k][context].get(family, 0)
                available = len(pool_by_family.get(context, {}).get(family, set()))
                family_total += min(slots, available)
            oracle["Family-slot oracle"][k] += family_total
            active_total = source_support_hits[k][context]
            for family in ("proximity", "relative_vertical"):
                slots = source_slots[k][context].get(family, 0)
                available = len(pool_by_family.get(context, {}).get(family, set()))
                active_total += min(slots, available)
            oracle["Active-route oracle"][k] += active_total

    rows: list[dict[str, Any]] = []
    for method, values in {**observed, **oracle}.items():
        for k, numerator in values.items():
            rows.append(
                {
                    "method": method,
                    "k": k,
                    "recall_numerator": numerator,
                    "recall_denominator": denominator,
                    "recall": numerator / denominator,
                }
            )
    return {
        "candidate_rows": row_count,
        "candidate_relations": len(seen_candidates),
        "candidate_pool_gt_relations": pool_hits,
        "candidate_pool_coverage": pool_hits / denominator,
        "rows": rows,
        "observed": observed,
        "oracle": oracle,
    }


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    protocol_path = resolve(root, args.protocol)
    rows_dir = resolve(root, args.rows)
    out = resolve(root, args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"nonempty_output:{out}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "ready":
        raise ValueError("protocol_version_mismatch")
    manifest_path = rows_dir / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != protocol["input_rows"]["manifest_sha256"]:
        raise ValueError(f"table_rows_manifest_hash_mismatch:{manifest_sha}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("table_rows_incomplete")
    for name, spec in manifest["files"].items():
        path = rows_dir / name
        if sha256_file(path) != spec["sha256"]:
            raise ValueError(f"table_rows_file_hash_mismatch:{name}")

    ground_truth, ground_truth_by_family, denominator = load_ground_truth(
        rows_dir / "ground_truth.csv.gz"
    )
    ks = tuple(int(value) for value in protocol["scope"]["ks"])
    sources: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for source in SOURCES:
        payload = evaluate_source(
            rows_dir / f"{source}_candidates.csv.gz",
            ground_truth,
            ground_truth_by_family,
            denominator,
            ks,
        )
        sources[source] = payload
        for row in payload["rows"]:
            csv_rows.append(
                {
                    "predictor": SOURCE_LABELS[source],
                    **row,
                    "recall_percent": 100.0 * float(row["recall"]),
                }
            )

    validations = {
        "table_rows_hashes_match": True,
        "ground_truth_denominator": (
            denominator == protocol["scope"]["ground_truth_denominator"]
        ),
        "candidate_identity_unique": all(
            payload["candidate_rows"] == payload["candidate_relations"]
            for payload in sources.values()
        ),
        "oracle_ordering": all(
            payload["observed"]["Source"][k]
            <= payload["oracle"]["Active-route oracle"][k]
            <= payload["oracle"]["Family-slot oracle"][k]
            <= payload["oracle"]["Unconstrained oracle"][k]
            <= payload["candidate_pool_gt_relations"]
            for payload in sources.values()
            for k in ks
        ),
        "evaluated_methods_within_oracle": all(
            payload["observed"][method][k]
            <= payload["oracle"]["Active-route oracle"][k]
            for payload in sources.values()
            for method in ("RelCompat3D-Linear", "RelCompat3D-MLP")
            for k in ks
        ),
        "oracle_recall_monotone_in_k": all(
            all(
                values[right] >= values[left]
                for left, right in zip(ks, ks[1:])
            )
            for payload in sources.values()
            for values in payload["oracle"].values()
        ),
        "all_points_finite": all(
            math.isfinite(float(row["recall"]))
            for row in csv_rows
        ),
    }
    status = "completed" if all(validations.values()) else "failed_validation"
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "oracle_recall.csv"
    write_csv(metrics_path, csv_rows)
    coverage_rows = [
        {
            "predictor": SOURCE_LABELS[source],
            "candidate_rows": payload["candidate_rows"],
            "ground_truth_in_candidate_pool": payload[
                "candidate_pool_gt_relations"
            ],
            "ground_truth_denominator": denominator,
            "candidate_pool_coverage": payload["candidate_pool_coverage"],
            "candidate_pool_coverage_percent": 100.0
            * payload["candidate_pool_coverage"],
        }
        for source, payload in sources.items()
    ]
    coverage_path = out / "candidate_pool_coverage.csv"
    write_csv(coverage_path, coverage_rows)
    summary = {
        "schema_version": "relcompat3d_candidate_oracle_summary_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "definitions": protocol["definitions"],
        "ground_truth_denominator": denominator,
        "sources": sources,
        "validations": validations,
        "evaluation_scope": protocol["evaluation_scope"],
        "docker_command": (
            "env UID=$(id -u) GID=$(id -g) docker compose "
            "-f configs/relcompat3d/compose.yaml run --rm "
            "relcompat3d_candidate_oracle"
        ),
    }
    summary_path = out / "summary.json"
    write_json(summary_path, summary)
    lines = [
        "# Candidate-Pool Oracle Recall",
        "",
        f"Status: `{status}`",
        "",
        "| Predictor | Pool coverage | K | Source | Linear | MLP | Active-route oracle | Family-slot oracle | Unconstrained oracle |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source, payload in sources.items():
        for k in ks:
            lines.append(
                f"| {SOURCE_LABELS[source]} | {100 * payload['candidate_pool_coverage']:.2f} | {k} | "
                f"{100 * payload['observed']['Source'][k] / denominator:.2f} | "
                f"{100 * payload['observed']['RelCompat3D-Linear'][k] / denominator:.2f} | "
                f"{100 * payload['observed']['RelCompat3D-MLP'][k] / denominator:.2f} | "
                f"{100 * payload['oracle']['Active-route oracle'][k] / denominator:.2f} | "
                f"{100 * payload['oracle']['Family-slot oracle'][k] / denominator:.2f} | "
                f"{100 * payload['oracle']['Unconstrained oracle'][k] / denominator:.2f} |"
            )
    lines.extend(
        (
            "",
            "The oracles are diagnostic upper bounds for fixed candidates. They do not represent attainable model performance.",
            "",
        )
    )
    summary_md = out / "summary.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    outputs = (metrics_path, coverage_path, summary_path, summary_md)
    manifest_out = out / "manifest.json"
    write_json(
        manifest_out,
        {
            "schema_version": "relcompat3d_candidate_oracle_manifest_v1",
            "status": status,
            "protocol": {
                "path": relpath(root, protocol_path),
                "sha256": sha256_file(protocol_path),
            },
            "table_rows": {
                "path": relpath(root, rows_dir),
                "manifest_sha256": manifest_sha,
            },
            "outputs": {
                path.name: {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in outputs
            },
            "validations": validations,
            "docker_command": summary["docker_command"],
        },
    )
    print(
        json.dumps(
            {
                "status": status,
                "candidate_pool_coverage": {
                    source: payload["candidate_pool_coverage"]
                    for source, payload in sources.items()
                },
                "validations": validations,
            },
            sort_keys=True,
        )
    )
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
