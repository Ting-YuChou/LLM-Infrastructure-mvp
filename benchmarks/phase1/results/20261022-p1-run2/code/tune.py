#!/usr/bin/env python3
"""Sweep launch settings of the Phase 1 Triton kernels on the GPU.

For each kernel and shape, times every config with kbench's CUDA-graph
timer (inputs rotated past L2) and reports the fastest next to the current
default (DECODE_CONFIG, PREFILL_CONFIG, GEMV_CONFIG). Configs that fail to
compile or run (e.g. too much shared memory) are recorded and skipped.

  python tune.py --out results/tune                  # all kernels
  python tune.py --kernels gemv --out results/tune   # one of decode, prefill, gemv
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from kbench import SCALE, _paged_inputs, copies_for, randn, time_us  # noqa: E402  (puts the repo on sys.path)
from src.serving.nano_vllm.kernels import linear as linear_mod  # noqa: E402
from src.serving.nano_vllm.kernels import paged_attention as attention  # noqa: E402

L2_BYTES = 40 * 1024**2


def timed(make_call, input_bytes: int) -> float:
    copies = copies_for(input_bytes, L2_BYTES)
    fns = [make_call() for _ in range(copies)]
    us, mode, _ = time_us(fns)
    if mode != "graph":
        raise RuntimeError(f"not capturable: {mode}")
    return us


def sweep(name, shapes, configs, make_call, input_bytes, default):
    rows = []
    for shape in shapes:
        results = []
        for config in configs:
            try:
                us = timed(lambda: make_call(shape, config), input_bytes(shape))
                results.append({"config": config, "us": us})
            except Exception as e:  # compile or launch failure for this config
                results.append({"config": config, "error": f"{type(e).__name__}: {str(e)[:120]}"})
            torch.cuda.empty_cache()
        ok = [r for r in results if "us" in r]
        best = min(ok, key=lambda r: r["us"]) if ok else None
        base = next((r for r in ok if r["config"] == default), None)
        row = {
            "kernel": name,
            "shape": shape,
            "best": best,
            "default": base,
            "speedup_vs_default": base["us"] / best["us"] if best and base else None,
            "all": results,
        }
        print("TUNE " + json.dumps({k: v for k, v in row.items() if k != "all"}), flush=True)
        rows.append(row)
    return rows


def grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, values)) for values in itertools.product(*axes.values())]


def tune_decode():
    shapes = [{"B": b, "ctx": c} for b, c in ((1, 1024), (1, 4096), (16, 1024), (64, 1024), (64, 4096), (128, 1024))]
    configs = grid(block_n=[32, 64, 128], num_warps=[2, 4, 8], num_stages=[1, 2, 3])

    def make_call(shape, config):
        i = _paged_inputs(shape["B"], shape["ctx"], [1] * shape["B"], "cuda")
        return lambda: attention.paged_decode_attention(
            i["q"], i["k_cache"], i["v_cache"], i["tables"], i["context_lens"], SCALE,
            use_triton=True, config=config,
        )

    def input_bytes(shape):
        return 2 * 2 * shape["B"] * shape["ctx"] * 8 * 128

    return sweep("decode", shapes, configs, make_call, input_bytes, attention.DECODE_CONFIG)


def tune_prefill():
    shapes = [{"seqs": n, "qlen": q} for n, q in ((1, 1024), (1, 4096), (4, 1024))]
    configs = grid(block_m=[64, 128], block_n=[32, 64, 128], num_warps=[4, 8], num_stages=[1, 2, 3])

    def make_call(shape, config):
        lens = [shape["qlen"]] * shape["seqs"]
        i = _paged_inputs(shape["seqs"], shape["qlen"], lens, "cuda")
        return lambda: attention.paged_prefill_attention(
            i["q"], i["k_cache"], i["v_cache"], i["tables"], i["cu_seqlens"], i["context_lens"],
            lens, lens, SCALE, use_triton=True, config=config,
        )

    def input_bytes(shape):
        return 3 * 2 * shape["seqs"] * shape["qlen"] * 32 * 128

    return sweep("prefill", shapes, configs, make_call, input_bytes, attention.PREFILL_CONFIG)


def tune_gemv():
    shapes = [
        {"layer": layer, "N": n, "K": k}
        for layer, (n, k) in {
            "qkv": (6144, 4096), "o_proj": (4096, 4096), "gate_up": (28672, 4096),
            "down": (4096, 14336), "lm_head": (128256, 4096),
        }.items()
    ]
    configs = grid(block_n=[32, 64, 128], block_k=[128, 256, 512], num_warps=[4, 8], num_stages=[2, 3, 4])
    if linear_mod.GEMV_CONFIG not in configs:
        configs.append(dict(linear_mod.GEMV_CONFIG))

    def make_call(shape, config):
        x = randn(1, shape["K"], device="cuda")
        w = randn(shape["N"], shape["K"], device="cuda", scale=shape["K"] ** -0.5)
        return lambda: linear_mod.linear(x, w, backend="triton", config=config)

    def input_bytes(shape):
        return 2 * shape["N"] * shape["K"]

    return sweep("gemv", shapes, configs, make_call, input_bytes, linear_mod.GEMV_CONFIG)


def markdown(rows) -> str:
    lines = ["# Triton launch-setting sweep", "",
             "| kernel | shape | best config | best us | default us | speedup |",
             "|---|---|---|---:|---:|---:|"]
    for r in rows:
        shape = " ".join(f"{k}={v}" for k, v in r["shape"].items())
        best, base = r["best"], r["default"]
        lines.append(
            f"| {r['kernel']} | {shape} | {best['config'] if best else '-'} "
            f"| {best['us']:.1f} | {base['us']:.1f} | {r['speedup_vs_default']:.2f}x |"
            if best and base else f"| {r['kernel']} | {shape} | {best} | - | - | - |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kernels", default="decode,prefill,gemv")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "tune.py needs a GPU"
    torch.manual_seed(0)
    rows = []
    for name in args.kernels.split(","):
        rows += {"decode": tune_decode, "prefill": tune_prefill, "gemv": tune_gemv}[name]()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tune.json").write_text(json.dumps(rows, indent=2))
    text = markdown(rows)
    (out / "tune.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
