#!/usr/bin/env python3
"""Bit-for-bit guard for gemm_w4a16_tiled changes that keep the summation order.

Computes the tiled kernel's output for fixed-seed weights and inputs (the
Llama shapes at 1, 8, 16 and 64 rows) and either saves them or compares them
with a saved file. Run --save once before a kernel change and --check after
it, on the same GPU and software (a later run on the same Colab VM).

  python golden.py --save /workspace/p0/w4a16_golden.pt
  python golden.py --check /workspace/p0/w4a16_golden.pt   # exit 1 on any difference
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "phase0"))

from bandwidth import SHAPES  # noqa: E402
from src.serving.nano_vllm.kernels.w4a16 import quantize_w4a16, w4a16_linear  # noqa: E402

ROWS = (1, 8, 16, 64)


def outputs() -> dict[str, torch.Tensor]:
    out = {}
    for layer, (n, k) in SHAPES.items():
        if layer == "lm_head":
            continue
        torch.manual_seed(0)
        weight = quantize_w4a16(torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * k**-0.5, layout="tiled")
        for rows in ROWS:
            x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
            out[f"{layer} M={rows}"] = w4a16_linear(x, weight).cpu()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--save", type=Path)
    group.add_argument("--check", type=Path)
    args = parser.parse_args()
    current = outputs()
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(current, args.save)
        print(f"GOLDEN saved {len(current)} outputs to {args.save}")
        return
    saved = torch.load(args.check)
    differ = [name for name in saved if not torch.equal(saved[name], current[name])]
    for name in differ:
        diff = (saved[name].float() - current[name].float()).abs()
        print(f"GOLDEN differs: {name} max |diff| {diff.max().item():.3g}, {int((diff > 0).sum())} elements")
    print(f"GOLDEN {'FAIL' if differ else 'ok'}: {len(saved) - len(differ)}/{len(saved)} bit-identical")
    sys.exit(1 if differ else 0)


if __name__ == "__main__":
    main()
