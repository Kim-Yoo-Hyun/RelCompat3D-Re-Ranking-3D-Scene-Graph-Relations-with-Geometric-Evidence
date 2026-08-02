"""Shared repository paths for RelCompat3D command-line entry points.

Scripts in this folder are executed directly by Docker compose, so this module
uses only file-relative paths and does not require package installation.
"""

from __future__ import annotations

from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SRC_ROOT.parents[1]

HYPOTHESIS_RECORDS_REL = Path("archive/hypothesis_records/hypothesis")
RelCompat3D_HYPOTHESIS_REL = HYPOTHESIS_RECORDS_REL / "CAND-001" / "RelCompat3D_geometry-grounded-verification"
RelCompat3D_ARTIFACTS_REL = RelCompat3D_HYPOTHESIS_REL / "artifacts"

RelCompat3D_HYPOTHESIS_ROOT = REPO_ROOT / RelCompat3D_HYPOTHESIS_REL
RelCompat3D_ARTIFACTS_ROOT = REPO_ROOT / RelCompat3D_ARTIFACTS_REL

EXPERIMENT_ROOT_REL = Path("experiments/RelCompat3D_geom_reliability")
RESULT_ROOT_REL = Path("results/relcompat3d_geom_reliability")
ARCHIVE_EXPERIMENT_ROOT_REL = Path("archive/experiments/RelCompat3D_geom_reliability")

EXPERIMENT_ROOT = REPO_ROOT / EXPERIMENT_ROOT_REL
RESULT_ROOT = REPO_ROOT / RESULT_ROOT_REL
ARCHIVE_EXPERIMENT_ROOT = REPO_ROOT / ARCHIVE_EXPERIMENT_ROOT_REL


def repo_rel(path: Path, repo_root: Path = REPO_ROOT) -> str:
    """Return a repository-relative path string when possible."""

    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)
