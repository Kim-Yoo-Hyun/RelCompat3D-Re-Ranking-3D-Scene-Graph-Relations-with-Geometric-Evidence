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
from create_synthetic_workspace import create_workspace  # noqa: E402


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
            workspace = create_workspace(root)
            subset = workspace["subset"]
            relationships = workspace["relationships"]
            canonical = root / "canonical"
            ground_truth = root / "canonical" / "ground_truth.jsonl"
            run_script(
                "build_ground_truth.py",
                "--subset", str(subset),
                "--relationships", str(relationships),
                "--out", str(ground_truth),
            )
            ground_truth_rows = read_jsonl(ground_truth)
            self.assertEqual(len(ground_truth_rows), 2)

            for source in ("vlsat", "sgfn", "open3dsg"):
                predictions = canonical / source / "predictions.jsonl"
                run_script(
                    "adapt_source_predictions.py",
                    "--source", source,
                    "--raw", str(workspace[f"raw_{source}"]),
                    "--subset", str(subset),
                    "--relationships", str(relationships),
                    "--out", str(predictions),
                    "--baseline-run-id", f"synthetic-{source}",
                )
                run_script(
                    "build_verification_rows.py",
                    "--predictions-jsonl", str(predictions),
                    "--dataset-root", str(workspace["data_root"]),
                    "--output-dir", str(predictions.parent),
                    "--verification-policy", "point_subtype",
                )

                prediction_rows = read_jsonl(predictions)
                verification_rows = read_jsonl(predictions.parent / "verification.jsonl")
                self.assertEqual(len(prediction_rows), 2, source)
                self.assertEqual(len(verification_rows), 2, source)
                self.assertTrue(
                    all(row["baseline_name"] == source for row in prediction_rows),
                    source,
                )
                self.assertTrue(
                    all(row["quality"]["row_preserved"] for row in verification_rows),
                    source,
                )
                by_predicate = {
                    row["predicate"]["predicate_label"]: row
                    for row in verification_rows
                }
                self.assertEqual(
                    by_predicate["close by"]["verification_status"],
                    "satisfied",
                    source,
                )
                self.assertEqual(
                    by_predicate["supported by"]["verification_status"],
                    "satisfied",
                    source,
                )
                self.assertTrue(
                    by_predicate["supported by"]["verification"]["point_evidence_available"],
                    source,
                )

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
