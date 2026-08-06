#!/usr/bin/env python3
"""Export deterministic local table rows for paper reproduction."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import hmac
import io
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import audit_point_mesh as surface
import control_utils as controls
import evaluate_main as linear
import evaluate_train_only as strict
import fit_mlp as nonlinear


SOURCE_PATH_KEYS = {
    "vlsat": "vlsat_verification",
    "open3dsg": "open3dsg_verification",
    "sgfn": "sgfn_verification",
}
FAMILIES = ("support_contact", "proximity", "relative_vertical")

CANDIDATE_FIELDS = (
    "predictor",
    "scan_uid",
    "scan_index",
    "context_uid",
    "context_index",
    "pair_uid",
    "candidate_uid",
    "row_uid",
    "predicate",
    "family",
    "source_score",
    "linear_compatibility",
    "mlp_compatibility",
    "linear_utility",
    "mlp_utility",
    "exact_match",
    "verifier_status",
    "surface_point_status",
    "surface_mesh_status",
    "surface_agreement_status",
    "rank_source",
    "rank_linear",
    "rank_mlp",
    "rank_rankavg",
    "rank_rrf",
    "rank_product_all_families",
    "rank_wrong_predicate",
    "rank_wrong_pair",
    "rank_shuffled_geometry",
    "rank_fixed_predicate_swap",
    "rank_distance_only",
    "rank_compatibility_only",
)

GROUND_TRUTH_FIELDS = (
    "scan_uid",
    "scan_index",
    "context_uid",
    "context_index",
    "pair_uid",
    "candidate_uid",
    "predicate",
    "family",
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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def deterministic_csv_gz(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: Iterable[dict[str, Any]],
) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text,
                    fieldnames=fieldnames,
                    lineterminator="\n",
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
                    count += 1
    return count


class Pseudonyms:
    def __init__(self, key: bytes) -> None:
        self.key = key

    def make(self, namespace: str, *values: Any) -> str:
        message = "\x1f".join((namespace, *(str(value) for value in values)))
        return hmac.new(self.key, message.encode("utf-8"), hashlib.sha256).hexdigest()[:24]


def official_maps(path: Path) -> tuple[list[str], dict[str, int], list[str], dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    contexts = sorted({f"{row['scan']}_{row['split']}" for row in payload["scans"]})
    scans = sorted({str(row["scan"]) for row in payload["scans"]})
    return (
        contexts,
        {context: index for index, context in enumerate(contexts)},
        scans,
        {scan: index for index, scan in enumerate(scans)},
    )


def load_measurements(paths: Iterable[Path]) -> dict[tuple[str, int, int], dict[str, Any]]:
    result: dict[tuple[str, int, int], dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (
                    str(row["scan_id"]),
                    int(row["subject_id"]),
                    int(row["object_id"]),
                )
                result[key] = row
    return result


def direct_rank(rows: list[dict[str, Any]], score_name: str) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: (-float(row["scores"][score_name]), row["key"]))
    return {str(row["id"]): rank for rank, row in enumerate(ordered, 1)}


def routed_rank(rows: list[dict[str, Any]], score_name: str) -> dict[str, int]:
    source_order = sorted(rows, key=lambda row: (-float(row["scores"]["source_score"]), row["key"]))
    queues: dict[str, list[dict[str, Any]]] = {}
    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        family_score = "source_score" if family == "support_contact" else score_name
        queues[family] = sorted(
            family_rows,
            key=lambda row: (-float(row["scores"][family_score]), row["key"]),
        )
    offsets = {family: 0 for family in FAMILIES}
    output: list[dict[str, Any]] = []
    for source_row in source_order:
        family = source_row["family"]
        output.append(queues[family][offsets[family]])
        offsets[family] += 1
    return {str(row["id"]): rank for rank, row in enumerate(output, 1)}


def add_rank_fusion_scores(rows: list[dict[str, Any]]) -> None:
    denominator = max(len(rows) - 1, 1)
    source_rank = direct_rank(rows, "source_score")
    compatibility_order = sorted(
        rows,
        key=lambda row: (-float(row["compatibility"]), row["key"]),
    )
    compatibility_rank = {
        str(row["id"]): rank for rank, row in enumerate(compatibility_order, 1)
    }
    for row in rows:
        row_id = str(row["id"])
        rank_z = source_rank[row_id]
        rank_c = compatibility_rank[row_id]
        q_z = 1.0 - (rank_z - 1) / denominator
        q_c = 1.0 - (rank_c - 1) / denominator
        row["scores"]["rankavg"] = 0.5 * (q_z + q_c)
        row["scores"]["rrf"] = 1.0 / (60 + rank_z) + 1.0 / (60 + rank_c)


def assign_ranks(grouped: dict[str, list[dict[str, Any]]]) -> None:
    route_columns = {
        "rank_linear": "structured_product",
        "rank_mlp": "mlp_product",
        "rank_rankavg": "rankavg",
        "rank_rrf": "rrf",
        "rank_wrong_predicate": "wrong_predicate_product",
        "rank_wrong_pair": "wrong_pair_product",
        "rank_shuffled_geometry": "shuffled_geometry_product",
        "rank_fixed_predicate_swap": "endpoint_swap_fixed_label_product",
        "rank_distance_only": "distance_only",
        "rank_compatibility_only": "compatibility_only",
    }
    for rows in grouped.values():
        add_rank_fusion_scores(rows)
        rank_maps = {
            "rank_source": direct_rank(rows, "source_score"),
            "rank_product_all_families": direct_rank(rows, "structured_product"),
            **{
                column: routed_rank(rows, score_name)
                for column, score_name in route_columns.items()
            },
        }
        for row in rows:
            row_id = str(row["id"])
            row["paper_ranks"] = {
                column: ranks[row_id] for column, ranks in rank_maps.items()
            }


def relation_identity(row: dict[str, Any]) -> tuple[str, int, int, int, str]:
    return tuple(row["key"])  # type: ignore[return-value]


def candidate_export_rows(
    source: str,
    grouped: dict[str, list[dict[str, Any]]],
    gt: dict[str, set[tuple[Any, ...]]],
    contexts: list[str],
    context_index: dict[str, int],
    scan_index: dict[str, int],
    aliases: Pseudonyms,
    measurements: dict[tuple[str, int, int], dict[str, Any]],
    thresholds: dict[str, Any],
) -> Iterable[dict[str, Any]]:
    for context in contexts:
        rows = sorted(grouped.get(context, []), key=lambda row: row["key"])
        for row in rows:
            scan, split, subject, object_, predicate = relation_identity(row)
            family = str(row["family"])
            pair_uid = aliases.make("pair", scan, split, subject, object_)
            candidate_uid = aliases.make(
                "relation", scan, split, subject, predicate, object_
            )
            point_status = mesh_status = agreement_status = ""
            if family in {"proximity", "relative_vertical"}:
                measurement = measurements.get((scan, int(subject), int(object_)))
                if measurement is not None:
                    statuses = surface.all_audit_statuses(
                        measurement,
                        family,
                        str(predicate),
                        thresholds,
                    )
                    point_status = statuses["point"]
                    mesh_status = statuses["mesh"]
                    agreement_status = statuses["consensus"]
            scores = row["scores"]
            ranks = row["paper_ranks"]
            yield {
                "predictor": source,
                "scan_uid": aliases.make("scan", scan),
                "scan_index": scan_index[scan],
                "context_uid": aliases.make("context", context),
                "context_index": context_index[context],
                "pair_uid": pair_uid,
                "candidate_uid": candidate_uid,
                "row_uid": aliases.make("row", source, candidate_uid),
                "predicate": predicate,
                "family": family,
                "source_score": format(float(row["semantic"]), ".17g"),
                "linear_compatibility": format(float(row["compatibility"]), ".17g"),
                "mlp_compatibility": format(float(row["mlp_compatibility"]), ".17g"),
                "linear_utility": format(float(scores["structured_product"]), ".17g"),
                "mlp_utility": format(float(scores["mlp_product"]), ".17g"),
                "exact_match": int(row["key"] in gt.get(context, set())),
                "verifier_status": row["status"] or "",
                "surface_point_status": point_status,
                "surface_mesh_status": mesh_status,
                "surface_agreement_status": agreement_status,
                **ranks,
            }


def ground_truth_export_rows(
    path: Path,
    context_index: dict[str, int],
    scan_index: dict[str, int],
    aliases: Pseudonyms,
) -> Iterable[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            family = str(row["predicate_family"])
            if family not in FAMILIES:
                continue
            scan = str(row["scan_id"])
            split = int(row["subset_split_id"])
            subject = int(row["subject_id"])
            object_ = int(row["object_id"])
            predicate = str(row["predicate_label"])
            context = str(row["subgraph_id"])
            rows.append(
                {
                    "scan_uid": aliases.make("scan", scan),
                    "scan_index": scan_index[scan],
                    "context_uid": aliases.make("context", context),
                    "context_index": context_index[context],
                    "pair_uid": aliases.make("pair", scan, split, subject, object_),
                    "candidate_uid": aliases.make(
                        "relation", scan, split, subject, predicate, object_
                    ),
                    "predicate": predicate,
                    "family": family,
                }
            )
    rows.sort(key=lambda row: (int(row["context_index"]), str(row["candidate_uid"])))
    return rows


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    protocol_path = resolve(root, args.protocol)
    out = resolve(root, args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"nonempty_output:{out}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") not in {
        "frozen_before_table_row_export",
        "public_execution_protocol",
    }:
        raise ValueError("unsupported_protocol_status")
    public_mode = protocol.get("status") == "public_execution_protocol"

    paths: dict[str, Path] = {}
    input_checks: dict[str, Any] = {}
    for name, spec in protocol["inputs"].items():
        path = resolve(root, spec["path"])
        if not path.exists():
            raise FileNotFoundError(f"missing_input:{name}:{path}")
        paths[name] = path
        if spec.get("sha256"):
            actual = sha256_file(path)
            if actual != spec["sha256"]:
                raise ValueError(f"input_hash_mismatch:{name}:{actual}")
            input_checks[name] = {
                "path": relpath(root, path),
                "sha256": actual,
                "size_bytes": path.stat().st_size,
            }
        else:
            input_checks[name] = {
                "path": relpath(root, path),
                "distribution": spec.get("distribution", "local input"),
            }

    key_text = paths["id_key"].read_text(encoding="utf-8").strip()
    if len(key_text) < 32:
        raise ValueError("id_key_too_short")
    aliases = Pseudonyms(bytes.fromhex(key_text))
    contexts, context_index, scans, scan_index = official_maps(
        paths["official_context_annotations"]
    )
    gt, _ = strict.load_gt(paths["ground_truth"])
    structured_models = json.loads(
        paths["structured_models"].read_text(encoding="utf-8")
    )
    nonlinear_models = json.loads(
        paths["nonlinear_models"].read_text(encoding="utf-8")
    )
    linear_scorer = linear.make_structured_scorer(structured_models)
    mlp_model = nonlinear_models["shared_nonlinear_structured"]
    thresholds = json.loads(paths["surface_thresholds"].read_text(encoding="utf-8"))
    measurement_paths = [paths["surface_measurements"]]
    if "additional_surface_measurements" in paths:
        measurement_paths.append(paths["additional_surface_measurements"])
    measurements = load_measurements(measurement_paths)

    out.mkdir(parents=True, exist_ok=True)
    gt_path = out / "ground_truth.csv.gz"
    gt_rows = deterministic_csv_gz(
        gt_path,
        GROUND_TRUTH_FIELDS,
        ground_truth_export_rows(
            paths["ground_truth"],
            context_index,
            scan_index,
            aliases,
        ),
    )

    candidate_counts: dict[str, int] = {}
    candidate_paths: dict[str, Path] = {}
    route_checks: dict[str, Any] = {}
    for source, input_name in SOURCE_PATH_KEYS.items():
        grouped, load_counts = controls.load_rows(
            paths[input_name],
            linear_scorer,
        )
        donor = controls.add_scores(
            grouped,
            linear_scorer,
            protocol["wrong_predicate_mapping"],
        )
        for rows in grouped.values():
            for row in rows:
                mlp_compatibility = nonlinear.projected_probability(
                    mlp_model,
                    row["family"],
                    row["predicate"],
                    row["raw"],
                )
                row["mlp_compatibility"] = mlp_compatibility
                row["scores"]["mlp_product"] = (
                    float(row["semantic"]) * mlp_compatibility
                )
        assign_ranks(grouped)
        candidate_path = out / f"{source}_candidates.csv.gz"
        candidate_count = deterministic_csv_gz(
            candidate_path,
            CANDIDATE_FIELDS,
            candidate_export_rows(
                source,
                grouped,
                gt,
                contexts,
                context_index,
                scan_index,
                aliases,
                measurements,
                thresholds,
            ),
        )
        candidate_paths[source] = candidate_path
        candidate_counts[source] = candidate_count
        route_checks[source] = {
            "load_counts": load_counts,
            "donor_audit": donor,
            "contexts_with_candidates": len(grouped),
            "zero_candidate_contexts": len(set(contexts) - set(grouped)),
        }

    expected = protocol["scope"]
    validations = {
        "input_hashes_match": all(
            "sha256" not in spec
            or input_checks[name]["sha256"] == spec["sha256"]
            for name, spec in protocol["inputs"].items()
        ),
        "official_contexts": len(contexts) == expected["expected_contexts"],
        "official_scans": len(scans) == expected["expected_scans"],
        "ground_truth_rows": gt_rows == expected["expected_ground_truth_rows"],
        "candidate_rows_present": all(count > 0 for count in candidate_counts.values()),
        "candidate_headers_exclude_raw_geometry": not any(
            token in CANDIDATE_FIELDS
            for token in (
                "scan_id",
                "subject_id",
                "object_id",
                "distance_3d",
                "distance_xy",
                "center_delta_z",
            )
        ),
        "all_wrong_pair_donors_nonself": all(
            payload["donor_audit"]["wrong_pair_self_donors"] == 0
            for payload in route_checks.values()
        ),
        "surface_measurements_available": len(measurements) > 0,
    }
    if "expected_candidate_rows" in expected:
        validations["candidate_rows_match_reference"] = (
            candidate_counts == expected["expected_candidate_rows"]
        )
    status = "completed" if all(validations.values()) else "failed_validation"
    schema_path = out / "schema.json"
    write_json(
        schema_path,
        {
            "schema_version": "relcompat3d_pseudonymized_rows_v1",
            "status": status,
            "candidate_fields": list(CANDIDATE_FIELDS),
            "ground_truth_fields": list(GROUND_TRUTH_FIELDS),
            "identifier_contract": protocol["row_contract"]["identifiers"],
            "included": protocol["row_contract"]["included"],
            "excluded": protocol["row_contract"]["excluded"],
            "redistribution_status": protocol["row_contract"]["redistribution_status"],
            "hmac_key_fingerprint": hashlib.sha256(
                paths["id_key"].read_bytes()
            ).hexdigest(),
        },
    )
    manifest_path = out / "manifest.json"
    files = {
        "ground_truth.csv.gz": gt_path,
        "schema.json": schema_path,
        **{
            f"{source}_candidates.csv.gz": path
            for source, path in candidate_paths.items()
        },
    }
    write_json(
        manifest_path,
        {
            "schema_version": "relcompat3d_table_rows_manifest_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "protocol": {
                "path": relpath(root, protocol_path),
                "sha256": sha256_file(protocol_path),
            },
            "counts": {
                "ground_truth": gt_rows,
                "candidates": candidate_counts,
                "contexts": len(contexts),
                "scans": len(scans),
            },
            "files": {
                name: {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for name, path in files.items()
            },
            "input_checks": input_checks,
            "route_checks": route_checks,
            "validations": validations,
            "redistribution_status": protocol["row_contract"][
                "redistribution_status"
            ],
            "docker_export_command": (
                "env UID=$(id -u) GID=$(id -g) docker compose "
                "-f configs/relcompat3d/compose.yaml run --rm "
                + (
                    "relcompat3d_export_trained_rows"
                    if public_mode
                    else "relcompat3d_export_rows"
                )
            ),
        },
    )
    print(
        json.dumps(
            {
                "status": status,
                "candidate_counts": candidate_counts,
                "ground_truth_rows": gt_rows,
                "manifest_sha256": sha256_file(manifest_path),
                "validations": validations,
            },
            sort_keys=True,
        )
    )
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
