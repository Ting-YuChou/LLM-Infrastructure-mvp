#!/usr/bin/env python3
"""Each decoder projection's GEMM time inside an end-to-end decode run (nsys).

kbench times one GEMM in a loop; this reads the GEMMs as decode runs them,
every weight cold (31 other layers pass through L2 in between). The GEMM
kernels of the capture (a regex on the kernel name) are labelled by position:
the first one after the activation kernel (SiLU-and-mul) is down_proj, and
the three GEMMs before it are gate_up, o_proj and qkv. Works for nano_vllm's
gemm_w4a16_tiled and vLLM's Marlin alike.

One projection can take several kernels: GEMM launches with no other kernel
between them are one call (nano's tiled kernel takes 64 rows per launch), and
an optional prelude regex adds kernels to the next GEMM call (past 128 rows
nano dequantizes the weight, then calls cuBLAS).

  python gemm_breakdown.py --out d/gemm_breakdown.md \\
      nano_b1=results/<run>/nsys/nano_w4a16_tiled_b1.sqlite:gemm_w4a16 \\
      nano_b256=results/<run>/nsys/nano_w4a16_tiled_b256.sqlite:gemm:dequantize \\
      vllm_awq_b1=results/<run>/nsys/vllmw4_b1.sqlite:Marlin
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase0"))

LAYERS = ("qkv", "o_proj", "gate_up", "down")
ACTIVATION = r"silu|act_and_mul"


def label_layers(
    names: list[str], durations: list[float], gemm: str, activation: str = ACTIVATION, prelude: str | None = None
) -> dict:
    """Kernel names and durations in launch order -> {layer: [durations]}."""
    gemm_re, act_re = re.compile(gemm), re.compile(activation, re.I)
    prelude_re = re.compile(prelude) if prelude else None
    calls = []  # [duration, follows the activation]
    pending, after_activation, previous_gemm = 0.0, False, False
    for name, duration in zip(names, durations):
        if prelude_re and prelude_re.search(name):
            pending += duration
            previous_gemm = False
        elif gemm_re.search(name):
            if previous_gemm and not pending:
                calls[-1][0] += duration
            else:
                calls.append([pending + duration, after_activation])
            pending, after_activation, previous_gemm = 0.0, False, True
        else:
            after_activation = after_activation or bool(act_re.search(name))
            previous_gemm = False
    out = {layer: [] for layer in LAYERS}
    for i, (_, follows_activation) in enumerate(calls):
        if follows_activation and i >= 3:
            for offset, layer in enumerate(LAYERS):
                out[layer].append(calls[i - 3 + offset][0])
    return out


def from_sqlite(path: Path, gemm: str, prelude: str | None = None) -> dict:
    from analyze_nsys import load

    data = load(sqlite3.connect(path))
    names = [k[3] if k[3] != "?" else k[2] for k in data["kernels"]]
    durations = [(k[1] - k[0]) / 1e3 for k in data["kernels"]]
    return label_layers(names, durations, gemm, prelude=prelude)


def summary(layers: dict) -> dict:
    out = {layer: {"n": len(v), "median_us": statistics.median(v) if v else None} for layer, v in layers.items()}
    if all(out[layer]["median_us"] for layer in LAYERS):
        out["sum_us"] = sum(out[layer]["median_us"] for layer in LAYERS)
    return out


def markdown(results: dict) -> str:
    lines = [
        "# GEMM time per projection inside decode (nsys, median µs per call)",
        "",
        "| run | qkv | o_proj | gate_up | down | per layer | calls per projection |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, s in results.items():
        cells = [f"{s[layer]['median_us']:.1f}" if s[layer]["median_us"] else "-" for layer in LAYERS]
        total = f"{s['sum_us']:.1f}" if "sum_us" in s else "-"
        lines.append(f"| {label} | {' | '.join(cells)} | {total} | {s['down']['n']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="label=path.sqlite:gemm_regex[:prelude_regex]")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    results = {}
    for spec in args.runs:
        label, rest = spec.split("=", 1)
        path, gemm, *prelude = rest.split(":")
        if not Path(path).exists():
            print(f"{label}: {path} missing, skipped")
            continue
        results[label] = summary(from_sqlite(Path(path), gemm, *prelude))
    out = Path(args.out)
    out.write_text(markdown(results))
    out.with_suffix(".json").write_text(json.dumps(results, indent=1))
    print(markdown(results))


if __name__ == "__main__":
    main()
