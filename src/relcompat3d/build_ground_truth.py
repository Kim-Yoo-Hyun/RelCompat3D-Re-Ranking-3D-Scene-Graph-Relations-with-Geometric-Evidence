#!/usr/bin/env python3
"""Export exact-relation ground truth from official 3DSSG annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


FAMILIES = {
    "standing on": "support_contact",
    "lying on": "support_contact",
    "supported by": "support_contact",
    "close by": "proximity",
    "higher than": "relative_vertical",
    "lower than": "relative_vertical",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--relationships", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-name", default="official_validation")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def family(label: str) -> str:
    if label in FAMILIES:
        return FAMILIES[label]
    if label in {"attached to", "hanging on", "mounted on", "connected to"}:
        return "attachment_deferred"
    if label in {"left", "right", "front", "behind", "in front of"}:
        return "relative_horizontal"
    return "unsupported"


def relation_parts(relation: list[Any]) -> tuple[int, int, int, str]:
    if len(relation) < 4:
        raise ValueError(f"invalid 3DSSG relation row: {relation!r}")
    return int(relation[0]), int(relation[1]), int(relation[2]), str(relation[3])


def main() -> int:
    args = parse_args()
    payload = json.loads(args.subset.read_text(encoding="utf-8"))
    relationship_names = [
        line.strip()
        for line in args.relationships.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    relationship_ids = {label: index for index, label in enumerate(relationship_names)}
    rows: list[dict[str, Any]] = []
    for context in payload["scans"]:
        scan_id = str(context["scan"])
        split_id = int(context["split"])
        subgraph_id = f"{scan_id}_{split_id}"
        objects = {int(key): str(value) for key, value in context["objects"].items()}
        for relation_index, relation in enumerate(context.get("relationships", [])):
            subject_id, object_id, raw_id, label = relation_parts(relation)
            expected = relationship_ids.get(label)
            if expected is not None and expected != raw_id:
                raise ValueError(
                    f"predicate id mismatch in {subgraph_id}:{relation_index}: "
                    f"annotation={raw_id}, vocabulary={expected}"
                )
            rows.append(
                {
                    "schema_version": "relcompat3d_ground_truth_v1",
                    "record_type": "ground_truth",
                    "gt_id": (
                        f"gt:{args.split_name}:{scan_id}:{split_id}:"
                        f"{subject_id}:{object_id}:{label}"
                    ),
                    "split_name": args.split_name,
                    "scan_id": scan_id,
                    "subset_split_id": split_id,
                    "subgraph_id": subgraph_id,
                    "subject_id": subject_id,
                    "object_id": object_id,
                    "subject_label": objects.get(subject_id),
                    "object_label": objects.get(object_id),
                    "predicate_label": label,
                    "predicate_family": family(label),
                    "raw_3dssg_predicate_id": raw_id,
                    "source_relation_index": relation_index,
                    "subset_source": str(args.subset),
                }
            )
    rows.sort(key=lambda row: row["gt_id"])
    duplicate_ids = [
        key for key, count in Counter(row["gt_id"] for row in rows).items() if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"duplicate ground-truth identities: {duplicate_ids[:3]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "relcompat3d_ground_truth_manifest_v1",
        "status": "completed",
        "inputs": {
            "subset": {"path": str(args.subset), "sha256": sha256_file(args.subset)},
            "relationships": {
                "path": str(args.relationships),
                "sha256": sha256_file(args.relationships),
            },
        },
        "output": {"path": str(args.out), "sha256": sha256_file(args.out)},
        "counts": {
            "rows": len(rows),
            "contexts": len({row["subgraph_id"] for row in rows}),
            "scans": len({row["scan_id"] for row in rows}),
            "evaluated_family_rows": sum(row["predicate_family"] in FAMILIES.values() for row in rows),
        },
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
