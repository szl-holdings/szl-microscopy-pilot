"""Command-line entry point for the bounded microscopy example."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import PilotError, run


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline microscopy image-quality report")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        if name == "run":
            command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args.manifest, args.out if args.command == "run" else None)
    except PilotError as exc:
        parser.exit(2, f"pilot: {exc}\n")
    print(json.dumps({"status": "verified", "images": len(report["images"]),
                      "comparison": report["comparison"]["status"],
                      "report": str(args.out) if args.command == "run" else None}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
