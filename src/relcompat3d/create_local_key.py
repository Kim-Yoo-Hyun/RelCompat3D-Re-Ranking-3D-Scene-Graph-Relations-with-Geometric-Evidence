#!/usr/bin/env python3
"""Create a private local key for deterministic table-row identifiers."""

from __future__ import annotations

import argparse
import secrets
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(secrets.token_hex(32) + "\n", encoding="utf-8")
    args.out.chmod(0o600)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
