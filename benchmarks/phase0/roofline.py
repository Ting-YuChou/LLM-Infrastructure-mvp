#!/usr/bin/env python3
"""Roofline bounds for Llama 3.1 8B decode and serving on A100 (Phase 0).

Bytes and FLOPs come from the model shape, so the bound counts the weights a
decode step actually reads: the embedding table contributes one row per token,
not the whole 1 GB matrix (lm_head is read in full).

  python benchmarks/phase0/roofline.py                      # peak numbers
  python benchmarks/phase0/roofline.py --bandwidth-json results/<run>/bandwidth.json

With a bandwidth.json from bandwidth.py the "achievable" columns use the
measured copy bandwidth and the best measured GEMM TFLOPs instead of the
datasheet peaks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LLAMA31_8B = {
    "vocab": 128256,
    "hidden": 4096,
    "intermediate": 14336,
    "layers": 32,
    "heads": 32,
    "kv_heads": 8,
    "head_dim": 128,
    "tied_embeddings": False,
}

GPUS = {
    "A100-SXM4-80GB": {"hbm_GBps": 2039.0, "bf16_tflops": 312.0},
    "A100-SXM4-40GB": {"hbm_GBps": 1555.0, "bf16_tflops": 312.0},
}

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def model_stats(cfg: dict, dtype_bytes: int = 2) -> dict:
    h, inter, layers = cfg["hidden"], cfg["intermediate"], cfg["layers"]
    kv_dim = cfg["kv_heads"] * cfg["head_dim"]
    per_layer_matmul = h * h + 2 * h * kv_dim + h * h + 3 * h * inter  # q, k+v, o, gate/up/down
    per_layer = per_layer_matmul + 2 * h  # + two RMSNorm weights
    embed = cfg["vocab"] * h
    lm_head = 0 if cfg["tied_embeddings"] else embed
    total_params = layers * per_layer + embed + lm_head + h
    # Weights one decode step streams from HBM: every layer, the final norm and
    # lm_head; the embedding lookup touches one row per token (added per token).
    step_params = layers * per_layer + h + (embed if cfg["tied_embeddings"] else lm_head)
    matmul_params = layers * per_layer_matmul + embed  # lm_head (or tied embed) is a matmul
    return {
        "total_params": total_params,
        "total_weight_bytes": total_params * dtype_bytes,
        "decode_step_weight_bytes": step_params * dtype_bytes,
        "matmul_params": matmul_params,
        "kv_bytes_per_token": 2 * layers * kv_dim * dtype_bytes,
        # QK^T and PV: 2 FLOPs per MAC, two matmuls, per head and layer.
        "attn_flops_per_query_key": 4 * layers * cfg["heads"] * cfg["head_dim"],
        "embed_row_bytes": h * dtype_bytes,
    }


def decode_step_bound(m: dict, batch: int, ctx: int, gbps: float, tflops: float) -> dict:
    """One decode step for `batch` sequences that each attend to `ctx` tokens."""
    bytes_ = (
        m["decode_step_weight_bytes"]
        + batch * ctx * m["kv_bytes_per_token"]
        + batch * m["embed_row_bytes"]
    )
    flops = batch * (2 * m["matmul_params"] + m["attn_flops_per_query_key"] * ctx)
    t_mem = bytes_ / (gbps * 1e9)
    t_compute = flops / (tflops * 1e12)
    t = max(t_mem, t_compute)
    return {
        "batch": batch,
        "ctx": ctx,
        "ms_per_step": t * 1e3,
        "tok_per_s": batch / t,
        "bound": "memory" if t_mem >= t_compute else "compute",
        "mem_ms": t_mem * 1e3,
        "compute_ms": t_compute * 1e3,
    }


def serve_round_bound(
    m: dict, concurrency: int, input_len: int, output_len: int, gbps: float, tflops: float
) -> dict:
    """Closed-loop serving: `concurrency` requests, each prefill `input_len`
    then decode `output_len` tokens. Lower bound on the time for one round, with
    perfect overlap of prefill and decode and every weight read shared by the
    whole batch."""
    c, i, o = concurrency, input_len, output_len
    kv = m["kv_bytes_per_token"]
    ctx_sum = o * i + o * (o - 1) // 2  # sum of attended lengths over the decode steps
    bytes_ = o * m["decode_step_weight_bytes"] + c * kv * (ctx_sum + i)  # + prefill KV writes
    attn_prefill = c * m["attn_flops_per_query_key"] * i * (i + 1) // 2
    attn_decode = c * m["attn_flops_per_query_key"] * ctx_sum
    flops = c * (i + o) * 2 * m["matmul_params"] + attn_prefill + attn_decode
    t_mem = bytes_ / (gbps * 1e9)
    t_compute = flops / (tflops * 1e12)
    t = max(t_mem, t_compute)
    return {
        "concurrency": c,
        "output_tok_per_s": c * o / t,
        "total_tok_per_s": c * (i + o) / t,
        "bound": "memory" if t_mem >= t_compute else "compute",
    }


def build(gpu: str, gbps: float, tflops: float, cfg: dict = LLAMA31_8B) -> dict:
    m = model_stats(cfg)
    latency_ctx = 128 + 256 // 2  # mid-point of the latency workload (in 128, out 256)
    sweep_ctx = 1024 + 256 // 2  # mid-point of the sweep workload (in 1024, out 256)
    return {
        "gpu": gpu,
        "hbm_GBps": gbps,
        "bf16_tflops": tflops,
        "model": m,
        "batch1_ms_per_token": decode_step_bound(m, 1, latency_ctx, gbps, tflops)["ms_per_step"],
        "decode_latency_ctx": [decode_step_bound(m, b, latency_ctx, gbps, tflops) for b in BATCHES],
        "decode_sweep_ctx": [decode_step_bound(m, b, sweep_ctx, gbps, tflops) for b in BATCHES],
        "serve_sweep": [serve_round_bound(m, c, 1024, 256, gbps, tflops) for c in BATCHES],
    }


def markdown(report: dict) -> str:
    m = report["model"]
    lines = [
        f"### {report['gpu']}: {report['hbm_GBps']:.0f} GB/s, {report['bf16_tflops']:.0f} TFLOPs",
        "",
        f"- params {m['total_params'] / 1e9:.3f} B, all weights {m['total_weight_bytes'] / 1e9:.2f} GB; "
        f"one decode step reads {m['decode_step_weight_bytes'] / 1e9:.2f} GB "
        "(embedding table excluded: one row per token)",
        f"- KV cache {m['kv_bytes_per_token'] / 1024:.0f} KiB per token",
        f"- batch=1 decode bound: **{report['batch1_ms_per_token']:.2f} ms/token** "
        f"({1e3 / report['batch1_ms_per_token']:.0f} tok/s)",
        "",
        "| batch | ctx 256: ms/step | tok/s | ctx 1152: ms/step | tok/s | bound | serve 1024/256: out tok/s | bound |",
        "|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for short, long_, serve in zip(
        report["decode_latency_ctx"], report["decode_sweep_ctx"], report["serve_sweep"]
    ):
        lines.append(
            f"| {short['batch']} | {short['ms_per_step']:.2f} | {short['tok_per_s']:.0f} "
            f"| {long_['ms_per_step']:.2f} | {long_['tok_per_s']:.0f} | {long_['bound']} "
            f"| {serve['output_tok_per_s']:.0f} | {serve['bound']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu", default="A100-SXM4-80GB", choices=sorted(GPUS))
    parser.add_argument(
        "--bandwidth-json",
        type=Path,
        help="bandwidth.py output; adds bounds at the measured bandwidth and TFLOPs",
    )
    parser.add_argument("--out", type=Path, help="write the reports as JSON here")
    args = parser.parse_args()

    spec = GPUS[args.gpu]
    reports = [build(f"{args.gpu} (datasheet)", spec["hbm_GBps"], spec["bf16_tflops"])]
    if args.bandwidth_json:
        bw = json.loads(args.bandwidth_json.read_text())
        reports.append(
            build(
                f"{args.gpu} (measured on this pod)",
                bw["copy_GBps"],
                bw["best_gemm_tflops"],
            )
        )
    for report in reports:
        print(markdown(report))
        print()
    if args.out:
        args.out.write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
