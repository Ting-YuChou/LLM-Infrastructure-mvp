#!/usr/bin/env python3
"""Fail CI when the serving runtime dependency matrix drifts."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REQUIRED_PACKAGES = (
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "tokenizers",
    "accelerate",
    "sentencepiece",
    "protobuf",
    "fastapi",
    "uvicorn",
    "pydantic",
    "httpx",
    "vllm",
)

EXPECTED_CUDA = "12.9.1"
EXPECTED_PYTHON = "python3.10"
FORBIDDEN_DOCKERFILE_PATTERNS = (
    "torch==2.1.2",
    "vllm==0.2.7",
    "pydantic==1.10.13",
    "cu118",
    "cuda:11.8",
)


def normalize_package_name(raw: str) -> str:
    """Normalize requirement names, stripping extras such as uvicorn[standard]."""
    name = raw.strip().lower()
    if "[" in name:
        name = name.split("[", 1)[0]
    return name.replace("_", "-")


def parse_pinned_requirements(path: Path) -> Dict[str, str]:
    """Parse simple pinned requirements from a requirements file."""
    pins: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        line = line.split("#", 1)[0].strip()
        match = re.match(r"^([A-Za-z0-9_.\-\[\]]+)==([^;\s]+)", line)
        if not match:
            continue
        pins[normalize_package_name(match.group(1))] = match.group(2)
    return pins


def validate_runtime_matrix(
    root_requirements: Path,
    serving_requirements: Path,
    serving_dockerfile: Path,
) -> List[str]:
    """Return validation error strings for runtime matrix drift."""
    errors: List[str] = []
    root_pins = parse_pinned_requirements(root_requirements)
    serving_pins = parse_pinned_requirements(serving_requirements)
    dockerfile = serving_dockerfile.read_text(encoding="utf-8")

    for package in REQUIRED_PACKAGES:
        root_version = root_pins.get(package)
        serving_version = serving_pins.get(package)
        if root_version is None:
            errors.append(f"{root_requirements}: missing required package {package}")
            continue
        if serving_version is None:
            errors.append(f"{serving_requirements}: missing required package {package}")
            continue
        if root_version != serving_version:
            errors.append(
                f"{package} mismatch: {root_requirements} has {root_version}, "
                f"{serving_requirements} has {serving_version}"
            )

    from_line = next(
        (line.strip() for line in dockerfile.splitlines() if line.strip().startswith("FROM ")),
        "",
    )
    if EXPECTED_CUDA not in from_line:
        errors.append(
            f"{serving_dockerfile}: FROM must use CUDA {EXPECTED_CUDA}, got: {from_line}"
        )

    if EXPECTED_PYTHON not in dockerfile:
        errors.append(f"{serving_dockerfile}: must install {EXPECTED_PYTHON}")

    for pattern in FORBIDDEN_DOCKERFILE_PATTERNS:
        if pattern in dockerfile:
            errors.append(f"{serving_dockerfile}: forbidden stale runtime pattern {pattern!r}")

    if "-r /tmp/requirements.serving.txt" not in dockerfile:
        errors.append(f"{serving_dockerfile}: must install docker/requirements.serving.txt")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check serving runtime dependency matrix")
    parser.add_argument("--root-requirements", default="requirements.txt")
    parser.add_argument("--serving-requirements", default="docker/requirements.serving.txt")
    parser.add_argument("--serving-dockerfile", default="docker/Dockerfile.serving")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = validate_runtime_matrix(
        Path(args.root_requirements),
        Path(args.serving_requirements),
        Path(args.serving_dockerfile),
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("serving-runtime-matrix-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
