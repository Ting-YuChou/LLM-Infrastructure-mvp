#!/usr/bin/env python3
"""Where gemm_w4a16_tiled spends its time, block by block (the kTrace build).

For every Llama shape and row count the weight is rotated through copies
adding up to ~512 MiB (kbench's rule), so each traced call reads it from HBM
as decode does, and the trace build runs --runs times. Each run is split into:

  launch skew   spread of the blocks' entry times (globaltimer)
  prologue      entry -> first tile in shared memory
  per tile      the stretch from the first tile to the last slice end minus the
                slice ends, per tile: streaming plus math, against the DRAM time of
                one tile when the blocks share the copy bandwidth evenly
  finishes      each slice end: warps summed / counter bumped or y written /
                done (the last block to arrive sums the slots in that step)
  end skew      spread of exit times, and the block that exits last
  critical path the last block's time from the first entry, by segment

Clocks are converted with the median cycles-per-ns over all blocks, since
globaltimer ticks every ~1 µs on an A100. The launches go back to back after
a warm-up, with one sync at the end: a host round trip after each launch let
the SM clock drop to ~1.08 GHz (run down-d1). Writes trace.json, trace.md and
a Gantt chart of one run per shape (gantt_<layer>_M<m>.png).

  python tiled_trace.py --out results/trace
  python tiled_trace.py --out d --shapes down --rows 1,16 --runs 20
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "phase0"))

from src.serving.nano_vllm.kernels.w4a16 import (  # noqa: E402
    TILED_TRACE_CLK_ENTRY as CLK_ENTRY,
    TILED_TRACE_CLK_EXIT as CLK_EXIT,
    TILED_TRACE_CLK_FIRST as CLK_FIRST,
    TILED_TRACE_FINISH0 as FINISH0,
    TILED_TRACE_FINISHES as FINISHES,
    TILED_TRACE_FLAGS as FLAGS,
    TILED_TRACE_GT_ENTRY as GT_ENTRY,
    TILED_TRACE_GT_EXIT as GT_EXIT,
    TILED_TRACE_TICKET as TICKET,
    TILED_TRACE_TILES as TILES,
)

ROTATE_BYTES = 512 * 1024**2
ROWS = (1, 8, 16, 64)


def clock_rate(runs: list[list[list[int]]]) -> float:
    """Median SM cycles per ns over every block with a positive globaltimer span."""
    rates = [
        (row[CLK_EXIT] - row[CLK_ENTRY]) / (row[GT_EXIT] - row[GT_ENTRY])
        for rows in runs
        for row in rows
        if row[GT_EXIT] > row[GT_ENTRY]
    ]
    if not rates:
        raise ValueError("no block spans a globaltimer tick")
    return statistics.median(rates)


def block_segments(row: list[int], ranges: tuple[int, int], k_tiles: int, rate: float) -> dict:
    """One block's time in ns, from its own clock."""
    begin, end = ranges
    ns = lambda cycles: cycles / rate  # noqa: E731
    finishes = []
    for f in range(min(FINISHES, (end - 1) // k_tiles - begin // k_tiles + 1)):
        start, summed, handed, done = row[FINISH0 + 4 * f : FINISH0 + 4 * f + 4]
        finishes.append({
            "final": f == (end - 1) // k_tiles - begin // k_tiles,
            "last": bool(row[FLAGS] >> f & 1),
            "sole": bool(row[FLAGS] >> (8 + f) & 1),
            "start": ns(start - row[CLK_ENTRY]),
            "sum_warps": ns(summed - start),
            "handoff": ns(handed - summed),
            "tail": ns(done - handed),
            "total": ns(done - start),
        })
    finish_ns = sum(f["total"] for f in finishes)
    last_done = row[FINISH0 + 4 * (len(finishes) - 1) + 3] if finishes else row[CLK_FIRST]
    tiles = row[TILES]
    return {
        "tiles": tiles,
        "prologue": ns(row[CLK_FIRST] - row[CLK_ENTRY]),
        "per_tile": (ns(last_done - row[CLK_FIRST]) - finish_ns) / tiles,
        "finishes": finishes,
        "exit": ns(row[CLK_EXIT] - last_done),
        "span": ns(row[CLK_EXIT] - row[CLK_ENTRY]),
    }


def analyze_run(rows: list[list[int]], schedule: dict, k_tiles: int, rate: float) -> dict:
    """One traced launch: kernel-wide spreads, block medians and the critical path."""
    blocks = [block_segments(row, r, k_tiles, rate) for row, r in zip(rows, schedule["ranges"])]
    t0 = min(row[GT_ENTRY] for row in rows)
    last = max(range(len(rows)), key=lambda b: rows[b][TICKET])  # exit order, exact
    crit, crit_row = blocks[last], rows[last]
    median = lambda xs: statistics.median(xs) if xs else 0.0  # noqa: E731
    mid = [f for b in blocks for f in b["finishes"] if not f["final"]]
    final = [f for b in blocks for f in b["finishes"] if f["final"]]
    full = [row for row, (begin, end) in zip(rows, schedule["ranges"]) if end - begin == schedule["iters"]]
    return {
        "kernel": max(row[GT_EXIT] for row in rows) - t0,
        "launch_skew": max(row[GT_ENTRY] for row in rows) - t0,
        "end_skew": max(row[GT_EXIT] for row in full) - min(row[GT_EXIT] for row in full),  # full stripes
        "prologue": median([b["prologue"] for b in blocks]),
        "per_tile": median([b["per_tile"] for b in blocks]),
        "finish_mid": median([f["total"] for f in mid]),
        "finish_final": median([f["total"] for f in final]),
        "finish_final_last": median([f["total"] for f in final if f["last"]]),
        "finish_final_other": median([f["total"] for f in final if not f["last"] and not f["sole"]]),
        "sum_warps": median([f["sum_warps"] for f in mid + final]),
        "handoff": median([f["handoff"] for f in mid + final if not f["sole"]]),
        "last_tail": median([f["tail"] for f in mid + final if f["last"]]),
        "critical": {
            "block": last,
            "tiles": crit["tiles"],
            "finishes": len(crit["finishes"]),
            "last_arriver": any(f["last"] for f in crit["finishes"] if f["final"]),
            "start": crit_row[GT_ENTRY] - t0,
            "prologue": crit["prologue"],
            "streaming": crit["per_tile"] * crit["tiles"],
            "finishes_ns": sum(f["total"] for f in crit["finishes"]),
            "exit": crit["exit"],
        },
    }


def summarize(runs: list[dict]) -> dict:
    """Median over runs of every number (the critical block's id from the median run)."""
    out = {}
    for key, value in runs[0].items():
        if isinstance(value, dict):
            out[key] = {k: statistics.median(r[key][k] for r in runs) for k in value}
        else:
            out[key] = statistics.median(r[key] for r in runs)
    return out


def globaltimer_tick(runs: list[list[list[int]]]) -> int:
    """The smallest step between distinct globaltimer stamps of one launch."""
    steps = []
    for rows in runs:
        stamps = sorted({row[GT_ENTRY] for row in rows} | {row[GT_EXIT] for row in rows})
        steps += [b - a for a, b in zip(stamps, stamps[1:])]
    return min(steps, default=0)


def tile_dram_ns(rows: int, tile_k: int, tile_n: int, blocks: int, copy_gbps: float) -> float:
    """One tile's bytes (4-bit weight, scales, x) at the block's even share of the copy bandwidth."""
    nbytes = tile_k * tile_n // 2 + tile_k // 128 * tile_n * 2 + rows * tile_k * 2
    return nbytes / (copy_gbps / blocks)  # GB/s = bytes/ns


def gantt(rows: list[list[int]], schedule: dict, k_tiles: int, rate: float, title: str, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    t0 = min(row[GT_ENTRY] for row in rows)
    fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.06)))
    for b, (row, r) in enumerate(zip(rows, schedule["ranges"])):
        seg = block_segments(row, r, k_tiles, rate)
        start = (row[GT_ENTRY] - t0) / 1e3
        ax.barh(b, seg["prologue"] / 1e3, left=start, color="#9aa0a6", height=0.8)
        cursor = start + seg["prologue"] / 1e3
        for f in seg["finishes"]:
            f_start = start + f["start"] / 1e3
            ax.barh(b, f_start - cursor, left=cursor, color="#4c78a8", height=0.8)
            ax.barh(b, f["total"] / 1e3, left=f_start, color="#e45756" if f["last"] else "#f2a541", height=0.8)
            cursor = f_start + f["total"] / 1e3
        ax.barh(b, seg["exit"] / 1e3, left=cursor, color="#9aa0a6", height=0.8)
    ax.set_xlabel("µs from the first block's entry")
    ax.set_ylabel("block")
    ax.set_title(f"{title}: prologue (grey), tiles (blue), slice ends (orange; red = last to arrive)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def trace_shape(layer: str, n: int, k: int, rows: int, runs: int, copy_gbps: float, out: Path) -> dict:
    import torch

    from src.serving.nano_vllm.kernels.w4a16 import (
        W4A16_CONFIG,
        _sm_count,
        quantize_w4a16,
        tiled_launch_config,
        tiled_schedule,
        w4a16_linear,
        w4a16_tiled_trace,
    )

    torch.manual_seed(0)
    first = quantize_w4a16(torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * k**-0.5, layout="tiled")
    copies = max(2, -(-ROTATE_BYTES // first.nbytes()))
    weights = [first] + [
        type(first)(first.qweight.clone(), first.scales.clone(), first.group_size, first.layout)
        for _ in range(copies - 1)
    ]
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    tile_k, tile_n, grid = tiled_launch_config(rows, n, k, W4A16_CONFIG, _sm_count(x.device))
    schedule = tiled_schedule(n, k, tile_k, tile_n, grid)
    for _ in range(20):  # warm-up at full load, so the SM clock is up
        for weight in weights:
            w4a16_linear(x, weight)
    pending = []
    for run in range(runs):
        target = run % copies
        for c in range(copies):  # every other copy, so the target's weight has left L2
            if c != target:
                w4a16_linear(x, weights[c])
        pending.append(w4a16_tiled_trace(x, weights[target])[1])
    traces = [trace.cpu().tolist() for trace in pending]  # the one sync
    k_tiles = k // tile_k
    rate = clock_rate(traces)
    rates = sorted(
        statistics.median(
            (row[CLK_EXIT] - row[CLK_ENTRY]) / (row[GT_EXIT] - row[GT_ENTRY]) for row in t if row[GT_EXIT] > row[GT_ENTRY]
        )
        for t in traces
    )
    analyzed = [analyze_run(t, schedule, k_tiles, rate) for t in traces]
    result = {
        "layer": layer, "N": n, "K": k, "M": rows, "tile": f"{tile_k}x{tile_n}",
        "blocks": len(schedule["ranges"]), "iters": schedule["iters"], "copies": copies, "runs": runs,
        "cycles_per_ns": rate,
        "cycles_per_ns_runs": [rates[0], rates[-1]],  # slowest and fastest run
        "globaltimer_tick_ns": globaltimer_tick(traces),
        "tile_dram_ns": tile_dram_ns(rows, tile_k, tile_n, len(schedule["ranges"]), copy_gbps),
        "median": summarize(analyzed),
    }
    median_run = sorted(range(runs), key=lambda r: analyzed[r]["kernel"])[runs // 2]
    gantt(traces[median_run], schedule, k_tiles, rate, f"{layer} M={rows} ({result['tile']})",
          out / f"gantt_{layer}_M{rows}.png")
    return result


def markdown(results: list[dict]) -> str:
    us = lambda ns: f"{ns / 1e3:.2f}"  # noqa: E731
    lines = [
        "# gemm_w4a16_tiled trace (cold weights, medians over runs, µs)",
        "",
        "| shape | tile, blocks × tiles | kernel | launch skew | prologue | per tile / DRAM share | "
        "slice end mid / final (last block) | end skew (full stripes) | critical block: start + prologue + tiles + ends + exit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        m, c = r["median"], r["median"]["critical"]
        lines.append(
            f"| {r['layer']} M={r['M']} | {r['tile']}, {r['blocks']} × {r['iters']} | {us(m['kernel'])} | "
            f"{us(m['launch_skew'])} | {us(m['prologue'])} | {us(m['per_tile'])} / {us(r['tile_dram_ns'])} | "
            f"{us(m['finish_mid'])} / {us(m['finish_final'])} ({us(m['finish_final_last'])}) | {us(m['end_skew'])} | "
            f"{us(c['start'])} + {us(c['prologue'])} + {us(c['streaming'])} + {us(c['finishes_ns'])} + {us(c['exit'])} |"
        )
    lines += [
        "",
        "Slice ends split (median µs): warps summed / counter bumped (slot stored, fenced) / last block's slot sum.",
        "",
        "| shape | sum warps | handoff | last block's sum | SM GHz (runs) | globaltimer tick |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        m, (slow, fast) = r["median"], r["cycles_per_ns_runs"]
        lines.append(
            f"| {r['layer']} M={r['M']} | {us(m['sum_warps'])} | {us(m['handoff'])} | {us(m['last_tail'])} | "
            f"{slow:.2f}–{fast:.2f} | {r['globaltimer_tick_ns']} ns |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--shapes", default="", help="regex on the layer name (default: all but lm_head)")
    parser.add_argument("--rows", default=",".join(map(str, ROWS)))
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--copy-gbps", type=float, default=0.0, help="default: measured (phase0 bandwidth.py)")
    args = parser.parse_args()

    from bandwidth import SHAPES, copy_bandwidth

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    copy_gbps = args.copy_gbps or copy_bandwidth()
    results = []
    for layer, (n, k) in SHAPES.items():
        if layer == "lm_head" or (args.shapes and not re.search(args.shapes, layer)):
            continue
        for rows in map(int, args.rows.split(",")):
            result = trace_shape(layer, n, k, rows, args.runs, copy_gbps, out)
            print("TRACE " + json.dumps(result), flush=True)
            results.append(result)
    (out / "trace.json").write_text(json.dumps({"copy_GBps": copy_gbps, "rows": results}, indent=1))
    (out / "trace.md").write_text(markdown(results))
    print(markdown(results))


if __name__ == "__main__":
    main()
