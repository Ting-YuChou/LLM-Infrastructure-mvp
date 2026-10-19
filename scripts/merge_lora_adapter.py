#!/usr/bin/env python3
"""Merge a LoRA adapter (e.g. a LoRA SFT run's final/) into a full bf16 checkpoint."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from training.export import merge_adapter  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base model the adapter was trained on")
    parser.add_argument("--adapter", required=True, help="Adapter directory (adapter_config.json)")
    parser.add_argument("--output", required=True, help="Output directory for the merged model")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    args = parser.parse_args()
    merge_adapter(args.base, args.adapter, args.output, dtype=DTYPES[args.dtype])


if __name__ == "__main__":
    main()
