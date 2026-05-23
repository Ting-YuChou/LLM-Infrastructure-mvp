#!/usr/bin/env python3
"""Validate and optionally execute the AWQ conversion handoff contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from training.pipeline import validate_checkpoint_artifact, write_json  # noqa: E402


def build_handoff(
    source_checkpoint: str,
    target_path: str,
    conversion_command: str = "",
    dry_run: bool = False,
) -> dict:
    source = Path(source_checkpoint)
    target = Path(target_path)
    source_validation = validate_checkpoint_artifact(source)
    target_validation = validate_checkpoint_artifact(target)
    return {
        "schema_version": "1.0",
        "source_checkpoint": str(source),
        "target_path": str(target),
        "source_validation": source_validation,
        "target_validation": target_validation,
        "quantization": "awq",
        "conversion_command": conversion_command,
        "dry_run": dry_run,
        "status": "ready_for_conversion"
        if source_validation["ready_for_awq_conversion"]
        else "blocked_source_checkpoint_not_ready",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AWQ conversion handoff validator")
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--target-path", required=True)
    parser.add_argument("--output", required=True, help="Path to write awq_handoff.json")
    parser.add_argument(
        "--conversion-command",
        default="",
        help="Optional command template using {source} and {target}",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    handoff = build_handoff(
        source_checkpoint=args.source_checkpoint,
        target_path=args.target_path,
        conversion_command=args.conversion_command,
        dry_run=args.dry_run,
    )
    write_json(Path(args.output), handoff)
    print(json.dumps(handoff, indent=2, sort_keys=True))

    if not args.conversion_command:
        return 0 if args.dry_run else 2
    if args.dry_run:
        return 0
    if handoff["status"] != "ready_for_conversion":
        return 1

    command = args.conversion_command.format(
        source=args.source_checkpoint,
        target=args.target_path,
    )
    subprocess.run(command, shell=True, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
