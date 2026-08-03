#!/usr/bin/env python3
"""Configure a pinned Open3DSG checkout for the containerized protocol."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


ASSIGNMENTS = {
    "CONF.PATH.HOME": 'os.environ.get("OPEN3DSG_HOME", "/workspace/local_dataset/Open3DSG")',
    "CONF.PATH.BASE": 'os.environ.get("OPEN3DSG_BASE", "/workspace/local_dataset/Open3DSG")',
    "CONF.PATH.DATA": 'os.environ.get("OPEN3DSG_DATA", "/workspace/local_dataset/Open3DSG/data")',
    "CONF.PATH.DATA_OUT": 'os.environ.get("OPEN3DSG_DATA_OUT", "/workspace/local_dataset/Open3DSG/output")',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def source_commit(source_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def configure(source_root: Path) -> Path:
    config_path = source_root / "open3dsg/config/config.py"
    if not config_path.is_file():
        raise SystemExit(f"Open3DSG configuration file not found: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    for variable, expression in ASSIGNMENTS.items():
        pattern = rf"^{re.escape(variable)}\s*=.*$"
        replacement = f"{variable} = {expression}"
        text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if count != 1:
            raise SystemExit(f"unable to configure {variable} in {config_path}")
    config_path.write_text(text, encoding="utf-8")
    return config_path


def prepare_runtime(source_root: Path) -> dict[str, str]:
    base = Path(os.environ.get("OPEN3DSG_BASE", "/workspace/local_dataset/Open3DSG"))
    data = Path(os.environ.get("OPEN3DSG_DATA", str(base / "data")))
    data_out = Path(os.environ.get("OPEN3DSG_DATA_OUT", str(base / "output")))
    directories = [
        base,
        data / "3RScan/3DSSG_subset",
        data / "SCANNET/scannet_3d/data",
        data / "SCANNET/scannet_2d",
        data_out / "datasets/OpenSG_3RScan",
        data_out / "datasets/OpenSG_ScanNet/subgraphs",
        data_out / "checkpoints",
        data_out / "features",
        base / "mlops/opensg/mlflow",
        base / "mlops/opensg/tensorboards",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    source_package = source_root / "open3dsg"
    runtime_package = base / "open3dsg"
    if runtime_package.is_symlink() or runtime_package.exists():
        if runtime_package.resolve() != source_package.resolve():
            raise SystemExit(
                f"runtime Open3DSG package points to a different path: {runtime_package}"
            )
    else:
        runtime_package.symlink_to(source_package, target_is_directory=True)

    empty_scans = json.dumps({"scans": []}, indent=2) + "\n"
    scannet_subgraphs = data_out / "datasets/OpenSG_ScanNet/subgraphs"
    for name in ("relationships_train.json", "relationships_validation.json"):
        path = scannet_subgraphs / name
        if not path.exists():
            path.write_text(empty_scans, encoding="utf-8")

    return {
        "base": str(base),
        "data": str(data),
        "data_out": str(data_out),
        "runtime_package": str(runtime_package),
    }


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    observed_commit = source_commit(source_root)
    if observed_commit != args.expected_commit:
        raise SystemExit(
            "Open3DSG source revision mismatch: "
            f"expected {args.expected_commit}, observed {observed_commit}"
        )

    runtime = prepare_runtime(source_root)
    config_path = configure(source_root)
    payload = {
        "status": "configured",
        "source_root": str(source_root),
        "source_commit": observed_commit,
        "configuration": str(config_path),
        "environment_variables": sorted(ASSIGNMENTS),
        "runtime": runtime,
    }
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
