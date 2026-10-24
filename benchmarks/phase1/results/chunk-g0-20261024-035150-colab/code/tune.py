#!/usr/bin/env python3
"""Sweep launch settings of the Phase 1 Triton kernels on the GPU.

For each kernel and shape, times every config with kbench's CUDA-graph
timer (inputs rotated past L2) and reports the fastest next to the config
the code currently picks for that shape (decode_config(B), PREFILL_CONFIG,
GEMV_CONFIG, CUDA_GEMV_CONFIG, W4A16_CONFIG). Configs that fail to compile or run (e.g.
too much shared memory) are recorded and skipped.

  python tune.py --out results/tune                  # all kernels
  python tune.py --kernels gemv --out results/tune   # decode, decode_fp8, prefill, gemv, cuda_gemv, w4a16, w4a16_tiled
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
from pathlib import Path

import torch

from kbench import SCALE, _paged_inputs, copies_for, randn, time_us  # noqa: E402  (puts the repo on sys.path)
# import_module, not `from ...kernels import linear`: the package re-exports
# the linear() function under the submodule's name.
linear_mod = importlib.import_module("src.serving.nano_vllm.kernels.linear")
attention = importlib.import_module("src.serving.nano_vllm.kernels.paged_attention")
fp8 = importlib.import_module("src.serving.nano_vllm.kernels.fp8")

L2_BYTES = 40 * 1024**2


def timed(make_call, input_bytes: int) -> float:
    copies = copies_for(input_bytes, L2_BYTES)
    fns = [make_call() for _ in range(copies)]
    us, mode, _ = time_us(fns)
    if mode != "graph":
        raise RuntimeError(f"not capturable: {mode}")
    return us


def sweep(name, shapes, configs, make_call, input_bytes, default):
    """``default`` is the current config, or a function of the shape returning it."""
    rows = []
    for shape in shapes:
        current = default(shape) if callable(default) else default
        results = []
        for config in configs + ([current] if current not in configs else []):
            try:
                us = timed(lambda: make_call(shape, config), input_bytes(shape))
                results.append({"config": config, "us": us})
            except Exception as e:  # compile or launch failure for this config
                results.append({"config": config, "error": f"{type(e).__name__}: {str(e)[:120]}"})
            torch.cuda.empty_cache()
        ok = [r for r in results if "us" in r]
        best = min(ok, key=lambda r: r["us"]) if ok else None
        base = next((r for r in ok if r["config"] == current), None)
        row = {
            "kernel": name,
            "shape": shape,
            "best": best,
            "default": base,
            "speedup_vs_default": base["us"] / best["us"] if best and base else None,
            "all": results,
        }
        # Every config's time goes into the log too, so a later crash loses nothing.
        compact = {**{k: v for k, v in row.items() if k != "all"},
                   "all": [[r["config"], round(r["us"], 2)] for r in ok]}
        print("TUNE " + json.dumps(compact), flush=True)
        rows.append(row)
    return rows


def grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, values)) for values in itertools.product(*axes.values())]


def tune_decode():
    shapes = [
        {"B": b, "ctx": c}
        for b, c in ((1, 1024), (1, 4096), (16, 1024), (32, 1024), (64, 1024), (64, 4096), (128, 1024))
    ]
    configs = grid(block_n=[32, 64, 128], num_warps=[4, 8], num_stages=[1, 2, 3])

    def make_call(shape, config):
        i = _paged_inputs(shape["B"], shape["ctx"], [1] * shape["B"], "cuda")
        return lambda: attention.paged_decode_attention(
            i["q"], i["k_cache"], i["v_cache"], i["tables"], i["context_lens"], SCALE,
            use_triton=True, config=config,
        )

    def input_bytes(shape):
        return 2 * 2 * shape["B"] * shape["ctx"] * 8 * 128

    return sweep(
        "decode", shapes, configs, make_call, input_bytes, lambda s: attention.decode_config(s["B"])
    )


def tune_decode_fp8():
    """The decode kernel on an fp8 cache: half the bytes per tile, plus the
    integer decode, so the bf16 tiers need not be the best here."""
    shapes = [
        {"B": b, "ctx": c}
        for b, c in ((1, 1024), (1, 4096), (16, 1024), (16, 4096), (64, 1024), (64, 4096), (128, 1024))
    ]
    configs = grid(block_n=[32, 64, 128, 256], num_warps=[4, 8], num_stages=[1, 2, 3, 4])

    def make_call(shape, config):
        i = _paged_inputs(shape["B"], shape["ctx"], [1] * shape["B"], "cuda")
        k, k_scales = fp8.fp8_quantize_reference(i["k_cache"])
        v, v_scales = fp8.fp8_quantize_reference(i["v_cache"])
        del i["k_cache"], i["v_cache"]
        return lambda: attention.paged_decode_attention(
            i["q"], k, v, i["tables"], i["context_lens"], SCALE, use_triton=True, config=config,
            key_scales=k_scales, value_scales=v_scales,
        )

    def input_bytes(shape):
        return 2 * shape["B"] * shape["ctx"] * 8 * (128 + 4)

    return sweep(
        "decode_fp8", shapes, configs, make_call, input_bytes,
        lambda s: attention.decode_config(s["B"], kv_fp8=True),
    )


def tune_prefill():
    shapes = [{"seqs": n, "qlen": q} for n, q in ((1, 1024), (1, 4096), (4, 1024))]
    configs = grid(block_m=[64, 128], block_n=[32, 64, 128], num_warps=[4, 8], num_stages=[2, 3])

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

    def make_call(shape, config):
        x = randn(1, shape["K"], device="cuda")
        w = randn(shape["N"], shape["K"], device="cuda", scale=shape["K"] ** -0.5)
        return lambda: linear_mod.linear(x, w, backend="triton", config=config)

    def input_bytes(shape):
        return 2 * shape["N"] * shape["K"]

    return sweep("gemv", shapes, configs, make_call, input_bytes, linear_mod.GEMV_CONFIG)


def tune_cuda_gemv():
    if linear_mod.load_cuda_extension() is None:
        raise RuntimeError("the CUDA GEMV did not build")
    shapes = [
        {"layer": layer, "M": m, "N": n, "K": k}
        for layer, (n, k) in {
            "qkv": (6144, 4096), "o_proj": (4096, 4096), "gate_up": (28672, 4096),
            "down": (4096, 14336), "lm_head": (128256, 4096),
        }.items()
        for m in (1, 4, 8)
    ]
    configs = grid(rows_per_warp=[1, 2, 4])

    def make_call(shape, config):
        x = randn(shape["M"], shape["K"], device="cuda")
        w = randn(shape["N"], shape["K"], device="cuda", scale=shape["K"] ** -0.5)
        return lambda: linear_mod.linear(x, w, backend="cuda", config=config)

    def input_bytes(shape):
        return 2 * shape["N"] * shape["K"]

    return sweep("cuda_gemv", shapes, configs, make_call, input_bytes, linear_mod.CUDA_GEMV_CONFIG)


def tune_w4a16():
    w4a16 = importlib.import_module("src.serving.nano_vllm.kernels.w4a16")
    if linear_mod.load_cuda_extension("w4a16") is None:
        raise RuntimeError("the W4A16 kernels did not build")
    shapes = [
        {"layer": layer, "M": m, "N": n, "K": k}
        for layer, (n, k) in {
            "qkv": (6144, 4096), "o_proj": (4096, 4096), "gate_up": (28672, 4096),
            "down": (4096, 14336),
        }.items()
        for m in (1, 2, 4, 8, 16, 32, 64)
    ]
    configs = [{"kernel": "gemv", "rows_per_warp": r} for r in (1, 2, 4)] + [
        {"kernel": kernel, "splits": s} for kernel in ("mma", "wide") for s in (0, 1, 2, 4, 8)
    ]

    def make_call(shape, config):
        if config.get("kernel") == "gemv" and shape["M"] > w4a16.W4A16_GEMV_MAX_ROWS:
            raise ValueError("outside the GEMV's row limit")
        x = randn(shape["M"], shape["K"], device="cuda")
        w = w4a16.quantize_w4a16(randn(shape["N"], shape["K"], device="cuda", scale=shape["K"] ** -0.5))
        return lambda: w4a16.w4a16_linear(x, w, config=config)

    def input_bytes(shape):
        return shape["N"] * shape["K"] // 2

    return sweep("w4a16", shapes, configs, make_call, input_bytes, w4a16.W4A16_CONFIG)


def tune_w4a16_tiled():
    """gemm_w4a16_tiled: tile shape and blocks per SM of the persistent grid,
    and, above 64 rows, several 64-row launches vs dequantize + F.linear."""
    w4a16 = importlib.import_module("src.serving.nano_vllm.kernels.w4a16")
    if linear_mod.load_cuda_extension("w4a16") is None:
        raise RuntimeError("the W4A16 kernels did not build")
    shapes = [
        {"layer": layer, "M": m, "N": n, "K": k}
        for layer, (n, k) in {
            "qkv": (6144, 4096), "o_proj": (4096, 4096), "gate_up": (28672, 4096),
            "down": (4096, 14336),
        }.items()
        for m in (1, 4, 16, 32, 48, 64, 128, 256)
    ]
    keys = ("tiled_tile", "tiled_blocks_per_sm", "tiled_max_rows")
    configs = grid(tiled_tile=list(w4a16.W4A16_TILED_TILES), tiled_blocks_per_sm=[1, 2], tiled_max_rows=[256])
    configs.append({"tiled_tile": "auto", "tiled_blocks_per_sm": 1, "tiled_max_rows": 64})

    def make_call(shape, config):
        mt = (min(shape["M"], w4a16.W4A16_TILED_MAX_LAUNCH_ROWS) + 15) // 16
        if config["tiled_tile"] != "auto" and config["tiled_tile"] not in w4a16._TILED_TILES_BY_MT[mt]:
            raise ValueError("tile not built for this many rows")
        if config["tiled_max_rows"] < 256 and shape["M"] <= config["tiled_max_rows"]:
            raise ValueError("the same launch as tiled_tile auto")
        x = randn(shape["M"], shape["K"], device="cuda")
        w = w4a16.quantize_w4a16(randn(shape["N"], shape["K"], device="cuda", scale=shape["K"] ** -0.5),
                                 layout="tiled")
        return lambda: w4a16.w4a16_linear(x, w, config=config)

    def input_bytes(shape):
        return shape["N"] * shape["K"] // 2

    def default(shape):
        """The launch W4A16_CONFIG gives this shape, as one of the configs."""
        current = {k: w4a16.W4A16_CONFIG[k] for k in keys}
        if shape["M"] > current["tiled_max_rows"]:
            return current
        tile_k, tile_n, _ = w4a16.tiled_launch_config(  # each launch takes up to 64 rows
            min(shape["M"], w4a16.W4A16_TILED_MAX_LAUNCH_ROWS), shape["N"], shape["K"],
            {**w4a16.W4A16_CONFIG, **current}, 1,
        )
        name = next(name for name, tile in w4a16.W4A16_TILED_TILES.items() if tile == (tile_k, tile_n))
        return {**current, "tiled_tile": name, "tiled_max_rows": 256}

    return sweep("w4a16_tiled", shapes, configs, make_call, input_bytes, default)


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
    parser.add_argument("--kernels", default="decode,prefill,gemv,cuda_gemv,w4a16")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "tune.py needs a GPU"
    torch.manual_seed(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.kernels.split(","):
        rows += {
            "decode": tune_decode,
            "decode_fp8": tune_decode_fp8,
            "prefill": tune_prefill,
            "gemv": tune_gemv,
            "cuda_gemv": tune_cuda_gemv,
            "w4a16": tune_w4a16,
            "w4a16_tiled": tune_w4a16_tiled,
        }[name]()
        # Written after each kernel so a failure in a later one keeps these.
        (out / "tune.json").write_text(json.dumps(rows, indent=2))
        (out / "tune.md").write_text(markdown(rows) + "\n")
    print(markdown(rows))


if __name__ == "__main__":
    main()
