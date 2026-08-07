#!/usr/bin/env python3
"""Benchmark RelCompat3D compatibility scoring and family-aware re-ranking."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import resource
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_all_families as linear_eval
import evaluate_base_models as model_eval


FAMILIES = linear_eval.FAMILIES
RERANKED_FAMILIES = {"proximity", "relative_vertical"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


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


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def load_rows(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_rows = 0
    family_counts = {family: 0 for family in FAMILIES}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            input_rows += 1
            row = json.loads(line)
            family = row["predicate"]["predicate_family"]
            if family not in FAMILIES:
                continue
            family_counts[family] += 1
            semantic = model_eval.finite((row.get("semantic") or {}).get("ranking_score"))
            if semantic is None:
                raise ValueError(f"missing_semantic:{row['prediction_id']}")
            grouped[row["subgraph_id"]].append(
                {
                    "id": row["prediction_id"],
                    "key": model_eval.candidate_key(row),
                    "family": family,
                    "predicate": row["predicate"]["predicate_label"],
                    "semantic": float(semantic),
                    "raw": model_eval.raw_numeric(row) if family in RERANKED_FAMILIES else None,
                }
            )
    return grouped, {
        "input_rows": input_rows,
        "in_scope_rows": sum(family_counts.values()),
        "scored_rows": sum(family_counts[family] for family in RERANKED_FAMILIES),
        "contexts": len(grouped),
        **{f"{family}_rows": count for family, count in family_counts.items()},
    }


def rank_rows(
    grouped: dict[str, list[dict[str, Any]]],
    scorer: Any,
) -> dict[str, list[tuple[str, str]]]:
    outputs: dict[str, list[tuple[str, str]]] = {}
    for context, rows in grouped.items():
        scored: list[dict[str, Any]] = []
        for row in rows:
            compatibility = (
                scorer(row["family"], row["predicate"], row["raw"])
                if row["family"] in RERANKED_FAMILIES
                else None
            )
            scored.append(
                {
                    "id": row["id"],
                    "key": row["key"],
                    "family": row["family"],
                    "semantic": row["semantic"],
                    "utility": (
                        row["semantic"] * float(compatibility)
                        if compatibility is not None
                        else row["semantic"]
                    ),
                }
            )
        source_order = sorted(scored, key=lambda item: (-item["semantic"], item["key"]))
        queues: dict[str, list[dict[str, Any]]] = {}
        for family in FAMILIES:
            family_rows = [item for item in scored if item["family"] == family]
            score_name = "semantic" if family == "support_contact" else "utility"
            queues[family] = sorted(
                family_rows, key=lambda item: (-item[score_name], item["key"])
            )
        offsets = {family: 0 for family in FAMILIES}
        ordered: list[tuple[str, str]] = []
        for source_item in source_order:
            family = source_item["family"]
            item = queues[family][offsets[family]]
            offsets[family] += 1
            ordered.append((str(item["id"]), family))
        outputs[context] = ordered
    return outputs


def validate_ordering(
    grouped: dict[str, list[dict[str, Any]]],
    outputs: dict[str, list[tuple[str, str]]],
) -> dict[str, bool]:
    family_exact = True
    support_exact = True
    for context, rows in grouped.items():
        source_order = sorted(rows, key=lambda item: (-item["semantic"], item["key"]))
        source_families = [item["family"] for item in source_order]
        output_families = [family for _, family in outputs[context]]
        family_exact = family_exact and source_families == output_families
        source_support = [str(item["id"]) for item in source_order if item["family"] == "support_contact"]
        output_support = [item_id for item_id, family in outputs[context] if family == "support_contact"]
        support_exact = support_exact and source_support == output_support
    return {
        "family_sequence_exact": family_exact,
        "support_contact_order_exact": support_exact,
    }


def output_fingerprint(outputs: dict[str, list[tuple[str, str]]]) -> str:
    digest = hashlib.sha256()
    for context in sorted(outputs):
        digest.update(context.encode("utf-8"))
        digest.update(b"\0")
        for item_id, family in outputs[context]:
            digest.update(item_id.encode("utf-8"))
            digest.update(b"\t")
            digest.update(family.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


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
    model_payload = json.loads(paths["linear_models"].read_text(encoding="utf-8"))
    scorer = linear_eval.make_linear_scorer(model_payload)
    family_models = model_payload["attempts"]["orbit_pairwise"]
    parameter_counts = {
        family: int(family_models[family]["parameter_count"])
        for family in FAMILIES
    }
    warmups = int(protocol["measurement"]["warmup_runs"])
    repeats = int(protocol["measurement"]["timed_runs"])

    source_results: dict[str, Any] = {}
    validations: dict[str, bool] = {"inputs_exist": not missing}
    expected = protocol["expected_counts"]
    for source in protocol["sources"]:
        grouped, counts = load_rows(paths[source])
        validations[f"{source}_counts_match"] = all(
            int(counts[key]) == int(value)
            for key, value in expected[source].items()
        )
        warm_fingerprint = None
        for _ in range(warmups):
            warm_output = rank_rows(grouped, scorer)
            warm_fingerprint = output_fingerprint(warm_output)
        times: list[float] = []
        fingerprints: list[str] = []
        last_output: dict[str, list[tuple[str, str]]] | None = None
        for _ in range(repeats):
            gc.collect()
            start = time.perf_counter()
            last_output = rank_rows(grouped, scorer)
            times.append(time.perf_counter() - start)
            fingerprints.append(output_fingerprint(last_output))
        if last_output is None:
            raise RuntimeError("no_timed_output")
        order_checks = validate_ordering(grouped, last_output)
        validations[f"{source}_deterministic"] = (
            len(set(fingerprints)) == 1
            and (warm_fingerprint is None or warm_fingerprint == fingerprints[0])
        )
        validations[f"{source}_family_sequence_exact"] = order_checks["family_sequence_exact"]
        validations[f"{source}_support_contact_order_exact"] = order_checks["support_contact_order_exact"]
        median_seconds = statistics.median(times)
        source_results[source] = {
            "counts": counts,
            "runs_seconds": times,
            "median_seconds": median_seconds,
            "min_seconds": min(times),
            "max_seconds": max(times),
            "p95_seconds": percentile(times, 95),
            "median_ms_per_context": 1000.0 * median_seconds / counts["contexts"],
            "scored_rows_per_second": counts["scored_rows"] / median_seconds,
            "all_in_scope_rows_per_second": counts["in_scope_rows"] / median_seconds,
            "ranking_fingerprint_sha256": fingerprints[0],
            "validations": order_checks,
        }
        del grouped, last_output
        gc.collect()

    status = "completed" if all(validations.values()) else "completed_with_validation_errors"
    summary = {
        "schema_version": "relcompat3d_relcompat3d_runtime_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "measurement_config": protocol["measurement"],
        "environment": {
            "cpu": cpu_model(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        },
        "parameters": {
            "by_family": parameter_counts,
            "stored_total": sum(parameter_counts.values()),
            "active_primary_total": sum(parameter_counts[f] for f in RERANKED_FAMILIES),
            "fitted_fusion_parameters": 0,
        },
        "sources": source_results,
        "validations": validations,
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary)
    lines = [
        "# RelCompat3D Runtime Benchmark",
        "",
        f"Status: `{status}`",
        "",
        "The timing starts from preloaded verification rows and includes compatibility scoring, transformation averaging, and family-aware sorting. It excludes source prediction, geometry reconstruction/join, file parsing, metrics, and bootstrap.",
        "",
        "| Predictor | Contexts | In-scope rows | Scored rows | Median total (s) | Median ms/context | Scored rows/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {"vlsat": "VL-SAT", "open3dsg": "Open3DSG", "sgfn": "SGFN"}
    for source in protocol["sources"]:
        item = source_results[source]
        counts = item["counts"]
        lines.append(
            f"| {labels[source]} | {counts['contexts']:,} | {counts['in_scope_rows']:,} | "
            f"{counts['scored_rows']:,} | {item['median_seconds']:.4f} | "
            f"{item['median_ms_per_context']:.3f} | {item['scored_rows_per_second']:,.0f} |"
        )
    lines.extend([
        "",
        f"Stored parameters: `{sum(parameter_counts.values())}`; active proximity/vertical parameters: `{sum(parameter_counts[f] for f in RERANKED_FAMILIES)}`; fitted fusion parameters: `0`.",
        f"Peak process RSS: `{summary['environment']['peak_rss_mib']:.1f} MiB`.",
        "",
    ])
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    output_paths = (out / "summary.json", out / "summary.md")
    write_json(out / "manifest.json", {
        "schema_version": "relcompat3d_relcompat3d_runtime_manifest_v1",
        "status": status,
        "protocol": {"path": str(protocol_path.relative_to(root)), "sha256": sha256_file(protocol_path)},
        "inputs": {name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for name, path in paths.items()},
        "outputs": {path.name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in output_paths},
        "validations": validations,
        "docker_command": "env UID=$(id -u) GID=$(id -g) docker compose -f configs/relcompat3d/compose.yaml run --rm relcompat3d_runtime",
    })
    print(json.dumps({"status": status, "validations": validations}, sort_keys=True))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
