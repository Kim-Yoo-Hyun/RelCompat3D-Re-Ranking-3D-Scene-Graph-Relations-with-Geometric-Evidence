#!/usr/bin/env python3
"""Convert source-predictor score dumps to RelCompat3D prediction rows.

The adapter preserves scan, context, ordered-pair, predicate, and score
identity. Geometry is joined in the next preprocessing stage; this command
does not infer or synthesize missing source predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "relcompat3d_prediction_v1"
TARGET_FAMILIES = {
    "standing on": "support_contact",
    "lying on": "support_contact",
    "supported by": "support_contact",
    "close by": "proximity",
    "higher than": "relative_vertical",
    "lower than": "relative_vertical",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("vlsat", "sgfn", "open3dsg"), required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--relationships", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--baseline-run-id", required=True)
    parser.add_argument("--split-name", default="official_validation")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def family(label: str) -> str:
    if label in TARGET_FAMILIES:
        return TARGET_FAMILIES[label]
    if label in {"left", "right", "front", "behind", "in front of"}:
        return "relative_horizontal"
    if label in {"attached to", "hanging on", "mounted on", "connected to"}:
        return "attachment_deferred"
    return "unsupported"


def finite_score(value: Any, location: str) -> float:
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"non-finite score at {location}")
    return score


def load_contexts(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in payload["scans"]:
        scan = str(entry["scan"])
        split = int(entry["split"])
        contexts[scan].append(
            {
                "scan_id": scan,
                "subset_split_id": split,
                "subgraph_id": f"{scan}_{split}",
                "objects": {int(key): str(value) for key, value in entry["objects"].items()},
            }
        )
    return contexts


def parse_split(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        if len(text) == 1 and text in "abcdef":
            return int(text, 16)
    return None


def open3dsg_identity(raw: dict[str, Any]) -> tuple[str, int, str] | None:
    scan_value = raw.get("scan_id")
    if scan_value is None:
        return None
    scan_id = str(scan_value)
    split = parse_split(raw.get("subset_split_id"))
    if split is None:
        base, separator, suffix = scan_id.rpartition("-")
        parsed = parse_split(suffix) if separator else None
        if parsed is not None and base:
            scan_id, split = base, parsed
    if split is None:
        return None
    return scan_id, split, f"{scan_id}_{split}"


def make_row(
    *,
    source: str,
    run_id: str,
    split_name: str,
    scan_id: str,
    subset_split_id: int,
    subgraph_id: str,
    edge_index: int,
    subject_node_index: int,
    object_node_index: int,
    subject_id: int,
    object_id: int,
    subject_label: str | None,
    object_label: str | None,
    predicate_label: str,
    predicate_index: int,
    raw_predicate_id: int,
    score: float,
) -> dict[str, Any]:
    prediction_id = (
        f"{source}:{split_name}:{scan_id}:{subset_split_id}:"
        f"{subject_id}:{object_id}:{predicate_label}"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "prediction",
        "prediction_id": prediction_id,
        "baseline_name": source,
        "baseline_run_id": run_id,
        "split_name": split_name,
        "scan_id": scan_id,
        "subset_split_id": subset_split_id,
        "subgraph_id": subgraph_id,
        "task_mode": "predcls_relation",
        "edge": {
            "edge_index": edge_index,
            "edge_source": f"{source}_raw_scores",
            "subject_id": subject_id,
            "object_id": object_id,
            "subject_node_index": subject_node_index,
            "object_node_index": object_node_index,
            "subject_label": subject_label,
            "object_label": object_label,
            "subject_label_source": "3DSSG_subset",
            "object_label_source": "3DSSG_subset",
        },
        "predicate": {
            "predicate_label": predicate_label,
            "predicate_family": family(predicate_label),
            "predicate_index": predicate_index,
            "raw_3dssg_predicate_id": raw_predicate_id,
            "predicate_vocab": "3DSSG_relationships",
        },
        "scores": {
            "predicate_score": score,
            "predicate_score_type": (
                "cosine_similarity" if source == "open3dsg" else "sigmoid_probability"
            ),
            "ranking_score": score,
            "ranking_score_type": "predicate_score",
        },
        "ranks": {
            "predicate_rank_for_pair": None,
            "semantic_rank_in_subgraph": None,
        },
        "adapter": {
            "name": f"{source}_to_relcompat3d",
            "version": "v1",
            "identity_preserving": True,
        },
    }


def adapt_vlsat(
    raw_path: Path,
    contexts: dict[str, list[dict[str, Any]]],
    relationship_ids: dict[str, int],
    run_id: str,
    split_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    context_lookup = {
        context["subgraph_id"]: context
        for scan_contexts in contexts.values()
        for context in scan_contexts
    }
    for raw_offset, raw in enumerate(iter_jsonl(raw_path)):
        identity = open3dsg_identity(raw)
        if identity is None:
            continue
        scan_id, subset_split_id, subgraph_id = identity
        context = context_lookup.get(subgraph_id)
        if context is None:
            raise ValueError(f"VL-SAT context is outside the official subset: {subgraph_id}")
        node_ids = [int(value) for value in raw["node_instance_ids"]]
        labels = [str(value) for value in raw["relation_names"]]
        scores = raw["rel_scores_3d"]
        edges = raw["edge_indices"]
        if len(edges) != len(scores):
            raise ValueError(f"VL-SAT edge/score mismatch: {subgraph_id}")
        for edge_offset, (edge, score_row) in enumerate(zip(edges, scores)):
            if len(score_row) != len(labels):
                raise ValueError(f"VL-SAT score width mismatch: {subgraph_id}:{edge_offset}")
            s_node, o_node = int(edge[0]), int(edge[1])
            subject_id, object_id = node_ids[s_node], node_ids[o_node]
            for predicate_index, (label, value) in enumerate(zip(labels, score_row)):
                if label == "none":
                    continue
                rows.append(
                    make_row(
                        source="vlsat",
                        run_id=run_id,
                        split_name=split_name,
                        scan_id=context["scan_id"],
                        subset_split_id=context["subset_split_id"],
                        subgraph_id=subgraph_id,
                        edge_index=edge_offset,
                        subject_node_index=s_node,
                        object_node_index=o_node,
                        subject_id=subject_id,
                        object_id=object_id,
                        subject_label=context["objects"].get(subject_id),
                        object_label=context["objects"].get(object_id),
                        predicate_label=label,
                        predicate_index=predicate_index,
                        raw_predicate_id=relationship_ids[label],
                        score=finite_score(value, f"vlsat:{raw_offset}:{edge_offset}"),
                    )
                )
    return rows


def adapt_sgfn(
    raw_path: Path,
    contexts: dict[str, list[dict[str, Any]]],
    relationship_ids: dict[str, int],
    run_id: str,
    split_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_offset, raw in enumerate(iter_jsonl(raw_path)):
        scan_id = str(raw["scan_id"])
        labels = [str(value) for value in raw["relation_names"]]
        node_ids = [int(value) for value in raw["node_instance_ids"]]
        edges = raw["edge_indices"]
        scores = raw.get("rel_scores", raw.get("rel_scores_3d"))
        if scores is None or len(edges) != len(scores):
            raise ValueError(f"SGFN edge/score mismatch: {scan_id}")
        for context in contexts.get(scan_id, []):
            objects = context["objects"]
            for edge_offset, (edge, score_row) in enumerate(zip(edges, scores)):
                if len(score_row) != len(labels):
                    raise ValueError(f"SGFN score width mismatch: {scan_id}:{edge_offset}")
                s_node, o_node = int(edge[0]), int(edge[1])
                subject_id, object_id = node_ids[s_node], node_ids[o_node]
                if subject_id not in objects or object_id not in objects:
                    continue
                for predicate_index, (label, value) in enumerate(zip(labels, score_row)):
                    if label == "none":
                        continue
                    rows.append(
                        make_row(
                            source="sgfn",
                            run_id=run_id,
                            split_name=split_name,
                            scan_id=scan_id,
                            subset_split_id=context["subset_split_id"],
                            subgraph_id=context["subgraph_id"],
                            edge_index=edge_offset,
                            subject_node_index=s_node,
                            object_node_index=o_node,
                            subject_id=subject_id,
                            object_id=object_id,
                            subject_label=objects[subject_id],
                            object_label=objects[object_id],
                            predicate_label=label,
                            predicate_index=predicate_index,
                            raw_predicate_id=relationship_ids[label],
                            score=finite_score(value, f"sgfn:{raw_offset}:{edge_offset}"),
                        )
                    )
    return rows


def adapt_open3dsg(
    raw_path: Path,
    contexts: dict[str, list[dict[str, Any]]],
    relationship_ids: dict[str, int],
    run_id: str,
    split_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    context_lookup = {
        context["subgraph_id"]: context
        for scan_contexts in contexts.values()
        for context in scan_contexts
    }
    for raw_offset, raw in enumerate(iter_jsonl(raw_path)):
        identity = open3dsg_identity(raw)
        if identity is None:
            continue
        scan_id, subset_split_id, subgraph_id = identity
        context = context_lookup.get(subgraph_id)
        if context is None:
            # Official Open3DSG dumps may also contain non-evaluation contexts.
            # The frozen evaluation universe is defined by the 3DSSG subset.
            continue
        edge = raw["edge"]
        subject_id, object_id = int(edge["subject_id"]), int(edge["object_id"])
        if subject_id == object_id:
            continue
        if subject_id not in context["objects"] or object_id not in context["objects"]:
            continue
        for default_index, predicate in enumerate(raw["predicate_scores"]):
            label = str(predicate["predicate_label"])
            if label == "none":
                continue
            predicate_index = int(predicate.get("open3dsg_predicate_index", default_index))
            rows.append(
                make_row(
                    source="open3dsg",
                    run_id=run_id,
                    split_name=split_name,
                    scan_id=scan_id,
                    subset_split_id=subset_split_id,
                    subgraph_id=subgraph_id,
                    edge_index=int(raw.get("edge_index", raw_offset)),
                    subject_node_index=int(edge.get("subject_node_index", -1)),
                    object_node_index=int(edge.get("object_node_index", -1)),
                    subject_id=subject_id,
                    object_id=object_id,
                    subject_label=context["objects"].get(subject_id, edge.get("subject_label")),
                    object_label=context["objects"].get(object_id, edge.get("object_label")),
                    predicate_label=label,
                    predicate_index=predicate_index,
                    raw_predicate_id=int(predicate.get("raw_3dssg_predicate_id", relationship_ids[label])),
                    score=finite_score(predicate["score"], f"open3dsg:{raw_offset}:{predicate_index}"),
                )
            )
    return rows


def assign_ranks(rows: list[dict[str, Any]]) -> None:
    by_pair: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        edge = row["edge"]
        by_pair[(row["subgraph_id"], edge["subject_id"], edge["object_id"])].append(row)
        by_context[row["subgraph_id"]].append(row)
    for group in by_pair.values():
        group.sort(key=lambda row: (-row["scores"]["ranking_score"], row["prediction_id"]))
        for rank, row in enumerate(group, 1):
            row["ranks"]["predicate_rank_for_pair"] = rank
    for group in by_context.values():
        group.sort(key=lambda row: (-row["scores"]["ranking_score"], row["prediction_id"]))
        for rank, row in enumerate(group, 1):
            row["ranks"]["semantic_rank_in_subgraph"] = rank


def main() -> int:
    args = parse_args()
    contexts = load_contexts(args.subset)
    relationship_names = [
        line.strip()
        for line in args.relationships.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    relationship_ids = {label: index for index, label in enumerate(relationship_names)}
    adapter = {
        "vlsat": adapt_vlsat,
        "sgfn": adapt_sgfn,
        "open3dsg": adapt_open3dsg,
    }[args.source]
    rows = adapter(
        args.raw,
        contexts,
        relationship_ids,
        args.baseline_run_id,
        args.split_name,
    )
    assign_ranks(rows)
    rows.sort(key=lambda row: row["prediction_id"])
    duplicate_ids = [
        prediction_id
        for prediction_id, count in Counter(row["prediction_id"] for row in rows).items()
        if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"duplicate prediction identities: {duplicate_ids[:3]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest_path = args.manifest or args.out.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "relcompat3d_source_adapter_manifest_v1",
        "status": "completed",
        "source": args.source,
        "baseline_run_id": args.baseline_run_id,
        "input": {"path": str(args.raw), "sha256": sha256_file(args.raw)},
        "output": {"path": str(args.out), "sha256": sha256_file(args.out)},
        "counts": {
            "rows": len(rows),
            "contexts": len({row["subgraph_id"] for row in rows}),
            "ordered_pairs": len(
                {
                    (row["subgraph_id"], row["edge"]["subject_id"], row["edge"]["object_id"])
                    for row in rows
                }
            ),
        },
        "validations": {
            "prediction_ids_unique": True,
            "ordered_pair_identity_preserved": True,
            "scores_finite": True,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
