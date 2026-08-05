from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "relcompat3d"
sys.path.insert(0, str(SRC))

import evaluate_comparators as routing  # noqa: E402
import relation_consistency as consistency  # noqa: E402


def run_script(name: str, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, str(SRC / name), *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class SyntheticPipelineTest(unittest.TestCase):
    def test_adapter_geometry_join_and_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subset = root / "relationships_validation.json"
            relationships = root / "relationships.txt"
            raw = root / "raw.jsonl"
            predictions = root / "canonical" / "vlsat" / "predictions.jsonl"
            ground_truth = root / "canonical" / "ground_truth.jsonl"
            scan_dir = root / "dataset" / "3RScan" / "scans" / "scan-1"
            scan_dir.mkdir(parents=True)

            relationships.write_text("close by\nsupported by\n", encoding="utf-8")
            subset.write_text(
                json.dumps(
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
                    }
                ),
                encoding="utf-8",
            )
            raw.write_text(
                json.dumps(
                    {
                        "scan_id": "scan-1",
                        "subset_split_id": 1,
                        "subgraph_id": "scan-1_1",
                        "node_instance_ids": [1, 2],
                        "edge_indices": [[0, 1]],
                        "relation_names": ["close by", "supported by"],
                        "rel_scores_3d": [[0.8, 0.7]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            obb = lambda center: {
                "centroid": center,
                "axesLengths": [1.0, 1.0, 1.0],
                "normalizedAxes": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            }
            (scan_dir / "semseg.v2.json").write_text(
                json.dumps(
                    {
                        "segGroups": [
                            {"objectId": 1, "obb": obb([0.0, 0.0, 1.0])},
                            {"objectId": 2, "obb": obb([0.2, 0.0, 0.0])},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            vertices = []
            for index in range(60):
                offset = (index % 10) * 0.01
                vertices.append(f"{offset} {offset} 1.0 1")
                vertices.append(f"{offset} {offset} 0.95 2")
            (scan_dir / "labels.instances.annotated.v2.ply").write_text(
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

            run_script(
                "adapt_source_predictions.py",
                "--source", "vlsat",
                "--raw", str(raw),
                "--subset", str(subset),
                "--relationships", str(relationships),
                "--out", str(predictions),
                "--baseline-run-id", "synthetic-vlsat",
            )
            run_script(
                "build_ground_truth.py",
                "--subset", str(subset),
                "--relationships", str(relationships),
                "--out", str(ground_truth),
            )
            run_script(
                "build_verification_rows.py",
                "--predictions-jsonl", str(predictions),
                "--dataset-root", str(root / "dataset"),
                "--output-dir", str(predictions.parent),
                "--verification-policy", "point_subtype",
            )

            prediction_rows = read_jsonl(predictions)
            verification_rows = read_jsonl(predictions.parent / "verification.jsonl")
            ground_truth_rows = read_jsonl(ground_truth)
            self.assertEqual(len(prediction_rows), 2)
            self.assertEqual(len(verification_rows), 2)
            self.assertEqual(len(ground_truth_rows), 2)
            self.assertTrue(all(row["quality"]["row_preserved"] for row in verification_rows))
            by_predicate = {row["predicate"]["predicate_label"]: row for row in verification_rows}
            self.assertEqual(by_predicate["close by"]["verification_status"], "satisfied")
            self.assertEqual(by_predicate["supported by"]["verification_status"], "satisfied")
            self.assertTrue(by_predicate["supported by"]["verification"]["point_evidence_available"])

    def test_transform_and_family_route_invariants(self) -> None:
        raw = {
            "center_delta_z": 1.0,
            "normalized_center_delta_z": 0.5,
            "projected_subject_overlap_ratio": 0.2,
            "projected_object_overlap_ratio": 0.4,
            "subject_bottom_z": 1.0,
            "subject_top_z": 2.0,
            "object_bottom_z": 0.0,
            "object_top_z": 1.0,
        }
        predicate, transformed = consistency.transformed_view(
            "relative_vertical", "higher than", raw
        )
        self.assertEqual(predicate, "lower than")
        self.assertEqual(transformed["center_delta_z"], -1.0)
        self.assertEqual(transformed["projected_subject_overlap_ratio"], 0.4)

        rows = [
            {
                "id": "support-a", "family": "support_contact", "semantic": 0.9,
                "key": ("scan", 1, 1, 2, "supported by"),
                "scores": {"source_score": 0.9, "structured_product": 0.1,
                           "structured_rank_average": 0.1, "structured_rrf_c60": 0.1,
                           "shared_nonlinear_structured_product": 0.1},
            },
            {
                "id": "proximity-a", "family": "proximity", "semantic": 0.8,
                "key": ("scan", 1, 2, 3, "close by"),
                "scores": {"source_score": 0.8, "structured_product": 0.2,
                           "structured_rank_average": 0.2, "structured_rrf_c60": 0.2,
                           "shared_nonlinear_structured_product": 0.2},
            },
            {
                "id": "proximity-b", "family": "proximity", "semantic": 0.7,
                "key": ("scan", 1, 3, 4, "close by"),
                "scores": {"source_score": 0.7, "structured_product": 0.9,
                           "structured_rank_average": 0.9, "structured_rrf_c60": 0.9,
                           "shared_nonlinear_structured_product": 0.9},
            },
        ]
        checks = routing.add_family_slot_routes({"scan_1": rows})
        self.assertTrue(checks["family_composition_exact"])
        self.assertTrue(checks["support_selection_exact"])


if __name__ == "__main__":
    unittest.main()
