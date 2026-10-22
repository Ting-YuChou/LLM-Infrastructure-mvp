#!/usr/bin/env python3
"""Where a decode step's time goes, from an nsys SQLite export (Phase 0).

  nsys export --type sqlite -o run.sqlite run.nsys-rep
  python analyze_nsys.py run.sqlite --label vllm_graph_b1 --steps 256 --batch 1 --out-dir out/

The window is the NVTX range "measured" when present (vllm_offline.py), else
the span of all captured kernels. Per decode step it reports wall time, GPU
busy time (union of kernels, memcpys and memsets on the device, clipped to the
step), the idle remainder, kernels and launch API calls, and kernel time by
category. With NVTX step ranges (engine_step_*, decode_step_*) the per-step
numbers are medians over the ranges (steps over 3x the median, i.e. prefill,
dropped); without them they are window / --steps.

Writes <label>.json, <label>_timeline.csv (a few steps from the middle of the
window) and, when matplotlib is installed, <label>_timeline.png.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

CATEGORIES = [  # first match wins; attention before gemm (flash/fmha kernels mention cutlass, splitk)
    # softmax counts as attention: these runs decode greedily, so only an unfused
    # attention (nano's PyTorch path) launches softmax kernels.
    ("attention", r"flash|attn|attention|fmha|paged|reshape_and_cache|kv_cache|decode_kernel|prefill_kernel|softmax"),
    ("norm", r"rms|norm|rsqrt"),
    ("rope", r"rope|rotary"),
    ("activation", r"silu|act_and_mul|gelu|swiglu"),
    ("sampling", r"argmax|topk|top_k|sampl|sort|gumbel|exponential"),
    ("gemm", r"gemm|gemv|cutlass|cublas|xmma|matmul|nvjet|splitk|s16816|wgrad|dgrad"),
    ("elementwise/copy", r"elementwise|copy|cat_|index|fill|unrolled|vectorized|scatter|gather|embedding|triton_poi"),
]
LAUNCH_APIS = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx", "cudaGraphLaunch")


def classify(name: str) -> str:
    lowered = name.lower()
    for category, pattern in CATEGORIES:
        if re.search(pattern, lowered):
            return category
    return "other"


def columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def has_table(db: sqlite3.Connection, table: str) -> bool:
    return db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def load(db: sqlite3.Connection) -> dict:
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    kernels = [
        (start, end, strings.get(short, "?"), strings.get(demangled, "?"))
        for start, end, short, demangled in db.execute(
            "SELECT start, end, shortName, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"
        )
    ]
    other_gpu = []
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if has_table(db, table):
            other_gpu += list(db.execute(f"SELECT start, end FROM {table}"))
    apis = []
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if has_table(db, table):
            apis += [
                (start, end, re.sub(r"_v\d+$", "", strings.get(name_id, "?")))
                for start, end, name_id in db.execute(f"SELECT start, end, nameId FROM {table}")
            ]
    ranges = []
    if has_table(db, "NVTX_EVENTS"):
        cols = columns(db, "NVTX_EVENTS")
        text_id = "textId" if "textId" in cols else "NULL"
        for start, end, text, tid in db.execute(
            f"SELECT start, end, text, {text_id} FROM NVTX_EVENTS WHERE end IS NOT NULL"
        ):
            ranges.append((start, end, text if text is not None else strings.get(tid, "")))
    kernels.sort()
    apis.sort()
    gpu = sorted([(k[0], k[1]) for k in kernels] + other_gpu)
    return {
        "kernels": kernels,
        "kernel_starts": [k[0] for k in kernels],
        "gpu": gpu,
        "gpu_starts": [g[0] for g in gpu],
        "max_gpu_ns": max((e - s for s, e in gpu), default=0),
        "apis": apis,
        "api_starts": [a[0] for a in apis],
        "ranges": ranges,
    }


def starting_in(items: list, starts: list[int], lo: int, hi: int) -> list:
    return items[bisect.bisect_left(starts, lo) : bisect.bisect_left(starts, hi)]


def union_length(intervals: list[tuple[int, int]], lo: int, hi: int) -> int:
    clipped = sorted((max(s, lo), min(e, hi)) for s, e in intervals if e > lo and s < hi)
    total, cur_s, cur_e = 0, None, None
    for s, e in clipped:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def window_stats(data: dict, lo: int, hi: int) -> dict:
    """Kernels and API calls that start inside [lo, hi); busy time clipped to it."""
    kernels = starting_in(data["kernels"], data["kernel_starts"], lo, hi)
    gpu = starting_in(data["gpu"], data["gpu_starts"], lo - data["max_gpu_ns"], hi)
    apis = starting_in(data["apis"], data["api_starts"], lo, hi)
    launches = [a for a in apis if a[2] in LAUNCH_APIS]
    return {
        "wall_ns": hi - lo,
        "busy_ns": union_length(gpu, lo, hi),
        "kernel_ns": sum(e - s for s, e, _, _ in kernels),
        "kernels": len(kernels),
        "launch_calls": len(launches),
        "launch_api_ns": sum(e - s for s, e, _ in launches),
        "kernel_list": kernels,
        "api_list": apis,
    }


def analyze(data: dict, label: str, steps: int | None, batch: int) -> dict:
    measured = [r for r in data["ranges"] if r[2] == "measured"]
    if measured:
        lo, hi = measured[0][0], measured[0][1]
    else:
        lo = min(k[0] for k in data["kernels"])
        hi = max(k[1] for k in data["kernels"])
    whole = window_stats(data, lo, hi)

    step_ranges = sorted(
        (r for r in data["ranges"] if re.match(r"(decode|engine)_step_\d+$", r[2])),
        key=lambda r: r[0],
    )
    if step_ranges:
        per = [window_stats(data, s, e) for s, e, _ in step_ranges]
        # nano's engine steps include the prefill step(s); keep the typical decode steps.
        median_wall = statistics.median(p["wall_ns"] for p in per)
        per = [p for p in per if p["wall_ns"] < 3 * median_wall]

        def med(key):
            return statistics.median(p[key] for p in per)

        step = {k: med(k) for k in ("wall_ns", "busy_ns", "kernels", "launch_calls", "launch_api_ns")}
        step_source = f"median of {len(per)} NVTX step ranges"
        n_steps = len(step_ranges)
    else:
        n_steps = steps or 1
        step = {
            k: whole[k] / n_steps
            for k in ("wall_ns", "busy_ns", "kernels", "launch_calls", "launch_api_ns")
        }
        step_source = f"capture window / {n_steps} steps"

    by_category: dict[str, int] = defaultdict(int)
    by_kernel: dict[str, list] = defaultdict(lambda: [0, 0, ""])
    for s, e, short, demangled in whole["kernel_list"]:
        category = classify(f"{short} {demangled}")
        by_category[category] += e - s
        entry = by_kernel[short]
        entry[0] += 1
        entry[1] += e - s
        entry[2] = category
    total_kernel = max(1, whole["kernel_ns"])
    api_counts: dict[str, list] = defaultdict(lambda: [0, 0])
    for s, e, name in whole["api_list"]:
        api_counts[name][0] += 1
        api_counts[name][1] += e - s

    report = {
        "label": label,
        "window_ms": whole["wall_ns"] / 1e6,
        "gpu_busy_frac": whole["busy_ns"] / max(1, whole["wall_ns"]),
        "per_step_source": step_source,
        "steps": n_steps,
        "batch": batch,
        "per_step": {
            "wall_ms": step["wall_ns"] / 1e6,
            "gpu_busy_ms": step["busy_ns"] / 1e6,
            "gpu_idle_ms": (step["wall_ns"] - step["busy_ns"]) / 1e6,
            "kernels": step["kernels"],
            "launch_calls": step["launch_calls"],
            "launch_api_ms": step["launch_api_ns"] / 1e6,
        },
        "kernel_time_by_category": {
            c: {"ms": ns / 1e6, "frac": ns / total_kernel}
            for c, ns in sorted(by_category.items(), key=lambda kv: -kv[1])
        },
        "top_kernels": [
            {
                "name": name[:120],
                "category": cat,
                "count": count,
                "total_ms": ns / 1e6,
                "avg_us": ns / count / 1e3,
                "frac": ns / total_kernel,
            }
            for name, (count, ns, cat) in sorted(by_kernel.items(), key=lambda kv: -kv[1][1])[:20]
        ],
        "cuda_api": {
            name: {"calls": calls, "total_ms": ns / 1e6}
            for name, (calls, ns) in sorted(api_counts.items(), key=lambda kv: -kv[1][1])[:15]
        },
    }
    # Three steps from the middle of the window when a step is short; otherwise
    # 5 ms from the middle of one step, where launch gaps are still visible.
    if step_ranges:
        middle = step_ranges[len(step_ranges) // 2]
        step_lo, step_hi = middle[0], middle[1]
    else:
        step_lo = (lo + hi) // 2
        step_hi = step_lo + int(step["wall_ns"])
    if step["wall_ns"] <= 15e6:
        slice_lo, slice_hi = step_lo, step_lo + int(3 * step["wall_ns"])
    else:
        slice_lo = (step_lo + step_hi) // 2
        slice_hi = slice_lo + 5_000_000
    report["timeline_slice_ns"] = [slice_lo, slice_hi]
    return report


def write_timeline(data: dict, report: dict, out_dir: Path) -> None:
    lo, hi = report["timeline_slice_ns"]
    rows = []
    for s, e, short, demangled in data["kernels"]:
        if e > lo and s < hi:
            rows.append(("GPU", (s - lo) / 1e3, (e - s) / 1e3, short[:80], classify(f"{short} {demangled}")))
    for s, e, name in data["apis"]:
        if e > lo and s < hi:
            rows.append(("CPU (CUDA API)", (s - lo) / 1e3, (e - s) / 1e3, name, "api"))
    for s, e, text in data["ranges"]:
        if e > lo and s < hi and text != "measured":
            rows.append(("NVTX", (s - lo) / 1e3, (e - s) / 1e3, text, "nvtx"))
    label = report["label"]
    with open(out_dir / f"{label}_timeline.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track", "start_us", "dur_us", "name", "category"])
        writer.writerows(sorted(rows, key=lambda r: (r[0], r[1])))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colors = {
        "gemm": "#4C72B0", "attention": "#DD8452", "norm": "#55A868", "rope": "#C44E52",
        "activation": "#8172B3", "sampling": "#937860", "elementwise/copy": "#DA8BC3",
        "other": "#8C8C8C", "api": "#BBBBBB", "nvtx": "#CCB974",
    }
    tracks = ["NVTX", "CPU (CUDA API)", "GPU"]
    fig, ax = plt.subplots(figsize=(14, 3.2))
    bars: dict[tuple[str, str], list] = defaultdict(list)  # one broken_barh call per group
    for track, start, dur, _, category in rows:
        bars[(track, category)].append((start, max(dur, 0.3)))
    for (track, category), spans in bars.items():
        y = tracks.index(track)
        ax.broken_barh(spans, (y - 0.35, 0.7), color=colors.get(category, "#8C8C8C"))
    ax.set_yticks(range(len(tracks)), tracks)
    ax.set_xlim(0, (hi - lo) / 1e3)
    ax.set_xlabel("time (us)")
    per = report["per_step"]
    ax.set_title(
        f"{label}: {per['wall_ms']:.2f} ms/step, GPU busy {per['gpu_busy_ms']:.2f} ms, "
        f"{per['kernels']:.0f} kernels/step"
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[c])
        for c in colors
        if any(r[4] == c for r in rows if r[0] == "GPU")
    ]
    names = [c for c in colors if any(r[4] == c for r in rows if r[0] == "GPU")]
    ax.legend(handles, names, loc="upper right", fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_timeline.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--steps", type=int, help="decode steps in the window (no NVTX steps)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.sqlite) as db:
        data = load(db)
    if not data["kernels"]:
        raise SystemExit(f"{args.sqlite}: no kernels captured")
    report = analyze(data, args.label, args.steps, args.batch)
    (args.out_dir / f"{args.label}.json").write_text(json.dumps(report, indent=2))
    write_timeline(data, report, args.out_dir)
    per = report["per_step"]
    print(
        "NSYS " + json.dumps(
            {
                "label": args.label,
                **{k: round(v, 3) for k, v in per.items()},
                "categories": {
                    c: round(v["frac"], 3) for c, v in report["kernel_time_by_category"].items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
