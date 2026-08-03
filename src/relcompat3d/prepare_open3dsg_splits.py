#!/usr/bin/env python3
"""Prepare the fixed Open3DSG training and development relationship files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED = {
    "stage": {"train": 3852, "development": 160},
    "filter": {"train": 3744, "development": 156},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("stage", "filter"), required=True)
    parser.add_argument("--official-train", type=Path, required=True)
    parser.add_argument("--official-validation", type=Path, required=True)
    parser.add_argument("--development-scans", type=Path, required=True)
    parser.add_argument("--preprocessed-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("scans")
    if not isinstance(rows, list):
        raise SystemExit(f"missing scans list: {path}")
    return rows


def read_scan_ids(path: Path) -> set[str]:
    values = {line.strip() for line in path.read_text(encoding="utf-8").splitlines()}
    values.discard("")
    return values


def preprocessed_path(root: Path, row: dict[str, Any]) -> Path:
    scan = str(row["scan"])
    split = int(row["split"])
    suffix = str(hex(split))[-1]
    return root / scan / f"data_dict_{suffix}.pkl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps({"scans": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def row_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "subgraphs": len(rows),
        "scans": len({str(row["scan"]) for row in rows}),
        "relations": sum(len(row.get("relationships", [])) for row in rows),
    }


def main() -> None:
    args = parse_args()
    train = read_rows(args.official_train)
    official_validation = read_rows(args.official_validation)
    development_scans = read_scan_ids(args.development_scans)
    development = [
        row for row in official_validation if str(row.get("scan")) in development_scans
    ]

    missing_development_scans = development_scans - {
        str(row["scan"]) for row in development
    }
    if missing_development_scans:
        raise SystemExit(
            "development scans absent from official validation annotations: "
            + ", ".join(sorted(missing_development_scans))
        )

    if args.phase == "filter":
        if args.preprocessed_root is None:
            raise SystemExit("--preprocessed-root is required for the filter phase")
        train = [
            row for row in train if preprocessed_path(args.preprocessed_root, row).is_file()
        ]
        development = [
            row
            for row in development
            if preprocessed_path(args.preprocessed_root, row).is_file()
        ]

    observed = {"train": len(train), "development": len(development)}
    if args.strict and observed != EXPECTED[args.phase]:
        raise SystemExit(
            f"Open3DSG split coverage mismatch for {args.phase}: "
            f"expected {EXPECTED[args.phase]}, observed {observed}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "relationships_train.json"
    development_path = args.output_dir / "relationships_validation.json"
    write_rows(train_path, train)
    write_rows(development_path, development)
    (args.output_dir / "train_scans.txt").write_text(
        "\n".join(sorted({str(row["scan"]) for row in train})) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "validation_scans.txt").write_text(
        "\n".join(sorted(development_scans)) + "\n",
        encoding="utf-8",
    )

    payload = {
        "status": "ready",
        "phase": args.phase,
        "expected_subgraphs": EXPECTED[args.phase],
        "train": row_summary(train),
        "development": row_summary(development),
        "outputs": {
            "train": str(train_path),
            "development": str(development_path),
        },
        "sha256": {
            "train": sha256(train_path),
            "development": sha256(development_path),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
