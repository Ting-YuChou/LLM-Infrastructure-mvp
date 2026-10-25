#!/usr/bin/env python3
"""Achievable HBM bandwidth and cuBLAS linear-layer times on this GPU (Phase 0).

Measures, with CUDA events:
  * device-to-device copy of 1 GiB (read + write bytes);
  * F.linear on Llama 3.1 8B's fused decode shapes (qkv, o, gate_up, down,
    lm_head) for a range of token counts M, so the linear layers of one decode
    step have a measured floor next to the datasheet roofline.

Weights are rotated through enough copies that the working set is far bigger
than the 40 MB L2; otherwise o_proj (32 MiB) would be timed from cache.

  python bandwidth.py --out bandwidth.json
"""

from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

HIDDEN, INTER, KV_DIM, VOCAB, LAYERS = 4096, 14336, 1024, 128256, 32
SHAPES = {  # name: (out_features, in_features), the fused layouts vLLM uses
    "qkv_proj": (HIDDEN + 2 * KV_DIM, HIDDEN),
    "o_proj": (HIDDEN, HIDDEN),
    "gate_up_proj": (2 * INTER, HIDDEN),
    "down_proj": (HIDDEN, INTER),
    "lm_head": (VOCAB, HIDDEN),
}
TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
ROTATE_BYTES = 512 * 1024**2


def time_ms(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def copy_bandwidth() -> float:
    src = torch.empty(256 * 1024**2, dtype=torch.float32, device="cuda")  # 1 GiB
    dst = torch.empty_like(src)
    ms = time_ms(lambda: dst.copy_(src), iters=20)
    return 2 * src.numel() * src.element_size() / (ms * 1e-3) / 1e9


def linear_times() -> list[dict]:
    rows = []
    for name, (n, k) in SHAPES.items():
        weight_bytes = n * k * 2
        copies = max(1, math.ceil(ROTATE_BYTES / weight_bytes))
        weights = [torch.randn(n, k, dtype=torch.bfloat16, device="cuda") for _ in range(copies)]
        for m in TOKENS:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            state = {"i": 0}

            def step():
                state["i"] = (state["i"] + 1) % copies
                F.linear(x, weights[state["i"]])

            ms = time_ms(step, iters=max(20, 2 * copies))
            rows.append(
                {
                    "layer": name,
                    "M": m,
                    "N": n,
                    "K": k,
                    "ms": ms,
                    "weight_GBps": weight_bytes / (ms * 1e-3) / 1e9,
                    "tflops": 2 * m * n * k / (ms * 1e-3) / 1e12,
                }
            )
        del weights
        torch.cuda.empty_cache()
    return rows


def decode_linear_floor(rows: list[dict]) -> list[dict]:
    """Per decode step: 32 x (qkv + o + gate_up + down) + lm_head, at batch M."""
    by = {(r["layer"], r["M"]): r["ms"] for r in rows}
    out = []
    for m in TOKENS:
        layer_ms = sum(by[(name, m)] for name in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"))
        out.append({"M": m, "ms_per_step": LAYERS * layer_ms + by[("lm_head", m)]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "copy_GBps": copy_bandwidth(),
    }
    rows = linear_times()
    result["linear"] = rows
    result["best_gemm_tflops"] = max(r["tflops"] for r in rows)
    result["best_gemv_GBps"] = max(r["weight_GBps"] for r in rows if r["M"] == 1)
    result["decode_linear_floor"] = decode_linear_floor(rows)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"copy: {result['copy_GBps']:.0f} GB/s; best GEMM {result['best_gemm_tflops']:.0f} TFLOPs")
    for r in rows:
        if r["M"] in (1, 64, 4096):
            print(
                f"  {r['layer']:>12} M={r['M']:<5} {r['ms'] * 1e3:8.1f} us "
                f"{r['weight_GBps']:7.0f} GB/s {r['tflops']:6.1f} TFLOPs"
            )
    for r in result["decode_linear_floor"]:
        print(f"  decode linear floor M={r['M']:<5} {r['ms_per_step']:.2f} ms/step")


if __name__ == "__main__":
    main()
