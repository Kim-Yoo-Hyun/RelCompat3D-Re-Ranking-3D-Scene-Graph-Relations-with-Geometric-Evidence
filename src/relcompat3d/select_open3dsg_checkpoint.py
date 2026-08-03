#!/usr/bin/env python3
"""Select an Open3DSG checkpoint using development loss only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlflow-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--link", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_run_dirs(root: Path) -> list[Path]:
    candidates = []
    for metric in root.rglob("metrics/val/loss"):
        run_dir = metric.parents[2]
        if (run_dir / "checkpoints").is_dir():
            candidates.append(run_dir)
    return sorted(set(candidates), key=lambda path: path.stat().st_mtime)


def select_run(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return args.run_dir.resolve()
    candidates = candidate_run_dirs(args.mlflow_root)
    if not candidates:
        raise SystemExit(f"no Open3DSG MLflow run found under {args.mlflow_root}")
    return candidates[-1]


def read_best_metric(path: Path) -> tuple[int, float, int]:
    records: list[tuple[int, float, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        timestamp, value, step = fields
        records.append((int(timestamp), float(value), int(step)))
    if not records:
        raise SystemExit(f"development loss history is empty: {path}")
    return min(records, key=lambda record: (record[1], record[2]))


def main() -> None:
    args = parse_args()
    run_dir = select_run(args)
    metric_path = run_dir / "metrics/val/loss"
    timestamp, value, metric_step = read_best_metric(metric_path)
    checkpoint_step = metric_step + 1
    matches = sorted((run_dir / "checkpoints").glob(f"*step={checkpoint_step}.ckpt"))
    if len(matches) != 1:
        raise SystemExit(
            f"expected one checkpoint for development step {metric_step}, found {len(matches)}"
        )
    checkpoint = matches[0].resolve()

    payload = {
        "status": "selected",
        "selection_metric": "val/loss",
        "selection_mode": "min",
        "relation_evaluation_metrics_used": False,
        "development_loss": value,
        "development_metric_step": metric_step,
        "development_metric_timestamp": timestamp,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint),
        "run_directory": str(run_dir),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.link:
        args.link.parent.mkdir(parents=True, exist_ok=True)
        args.link.unlink(missing_ok=True)
        os.symlink(checkpoint, args.link)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
