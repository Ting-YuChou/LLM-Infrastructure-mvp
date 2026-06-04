#!/usr/bin/env python3
"""Microbenchmark nano-vLLM PyTorch/Triton kernel helpers."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch

from serving.nano_vllm.kernels import rms_norm, silu_mul


def bench(label: str, fn, repeats: int) -> float:
    for _ in range(5):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    ms = elapsed * 1000.0 / repeats
    print(f"{label}: {ms:.3f} ms")
    return ms


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark nano-vLLM kernels")
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available", file=sys.stderr)
        return 2
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    x = torch.randn(args.rows, args.hidden_size, device=device, dtype=dtype)
    w = torch.randn(args.hidden_size, device=device, dtype=dtype)
    up = torch.randn_like(x)

    bench("rms_norm[pytorch]", lambda: rms_norm(x, w, use_triton=False), args.repeats)
    bench("silu_mul[pytorch]", lambda: silu_mul(x, up, use_triton=False), args.repeats)
    if device.type == "cuda":
        bench(
            "rms_norm[triton-if-available]",
            lambda: rms_norm(x, w, use_triton=True),
            args.repeats,
        )
        bench(
            "silu_mul[triton-if-available]",
            lambda: silu_mul(x, up, use_triton=True),
            args.repeats,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
