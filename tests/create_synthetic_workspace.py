#!/usr/bin/env python3
"""Create a minimal licensed-data-free workspace for integration checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def create_workspace(root: Path) -> dict[str, Path]:
    data_root = root / "local_dataset"
    relcompat_root = data_root / "RelCompat3D"
    subset_root = relcompat_root / "3DSSG_subset"
    scan_root = data_root / "3RScan" / "scans" / "scan-1"
    source_root = relcompat_root / "source_outputs"

    relationships = subset_root / "relationships.txt"
    subset = subset_root / "relationships_validation.json"
    relationships.parent.mkdir(parents=True, exist_ok=True)
    relationships.write_text("close by\nsupported by\n", encoding="utf-8")
    write_json(
        subset,
        {
            "scans": [
                {
                    "scan": "scan-1",
                    "split": 1,
                    "objects": {"1": "desk", "2": "floor"},
                    "relationships": [
                        [1, 2, 0, "close by"],
                        [1, 2, 1, "supported by"],
                    ],
                }
            ]
        },
    )

    shared_scores = {
        "scan_id": "scan-1",
        "node_instance_ids": [1, 2],
        "edge_indices": [[0, 1]],
        "relation_names": ["close by", "supported by"],
    }
    raw_paths = {
        "vlsat": source_root / "vlsat" / "raw.jsonl",
        "sgfn": source_root / "sgfn" / "raw.jsonl",
        "open3dsg": source_root / "open3dsg" / "raw.jsonl",
    }
    write_jsonl(
        raw_paths["vlsat"],
        [
            {
                **shared_scores,
                "subset_split_id": 1,
                "subgraph_id": "scan-1_1",
                "rel_scores_3d": [[0.8, 0.7]],
            }
        ],
    )
    write_jsonl(
        raw_paths["sgfn"],
        [{**shared_scores, "rel_scores": [[0.8, 0.7]]}],
    )
    write_jsonl(
        raw_paths["open3dsg"],
        [
            {
                "scan_id": "scan-1",
                "subset_split_id": 1,
                "subgraph_id": "scan-1_1",
                "edge_index": 0,
                "edge": {
                    "subject_id": 1,
                    "object_id": 2,
                    "subject_node_index": 0,
                    "object_node_index": 1,
                },
                "predicate_scores": [
                    {
                        "predicate_label": "close by",
                        "score": 0.8,
                        "open3dsg_predicate_index": 0,
                        "raw_3dssg_predicate_id": 0,
                    },
                    {
                        "predicate_label": "supported by",
                        "score": 0.7,
                        "open3dsg_predicate_index": 1,
                        "raw_3dssg_predicate_id": 1,
                    },
                ],
            }
        ],
    )

    def obb(center: list[float]) -> dict[str, Any]:
        return {
            "centroid": center,
            "axesLengths": [1.0, 1.0, 1.0],
            "normalizedAxes": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }

    write_json(
        scan_root / "semseg.v2.json",
        {
            "segGroups": [
                {"objectId": 1, "obb": obb([0.0, 0.0, 1.0])},
                {"objectId": 2, "obb": obb([0.2, 0.0, 0.0])},
            ]
        },
    )
    vertices: list[str] = []
    for index in range(60):
        offset = (index % 10) * 0.01
        vertices.append(f"{offset} {offset} 1.0 1")
        vertices.append(f"{offset} {offset} 0.95 2")
    scan_root.mkdir(parents=True, exist_ok=True)
    (scan_root / "labels.instances.annotated.v2.ply").write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                f"element vertex {len(vertices)}",
                "property float x",
                "property float y",
                "property float z",
                "property int objectId",
                "element face 0",
                "property list uchar int vertex_indices",
                "end_header",
                *vertices,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "data_root": data_root,
        "subset": subset,
        "relationships": relationships,
        **{f"raw_{source}": path for source, path in raw_paths.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    paths = create_workspace(args.root.resolve())
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
