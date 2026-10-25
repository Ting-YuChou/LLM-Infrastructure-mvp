#!/usr/bin/env python3
"""Per-kernel hardware counters from an ncu raw CSV export (Phase 0).

  ncu --import run.ncu-rep --csv --page raw > run_raw.csv
  python analyze_ncu.py run_raw.csv --label vllm_graph_b1 --out-dir out/

Groups kernels by name and reports, per group: launches, time, DRAM
throughput (% of peak and GB/s from bytes moved), SM throughput, achieved vs
theoretical occupancy, registers per thread, L2 hit rate and the top warp
stall reason. Writes <label>.json and prints a markdown table.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from analyze_nsys import classify

UNIT_SCALE = {  # to microseconds and bytes
    "nsecond": 1e-3, "usecond": 1.0, "msecond": 1e3, "second": 1e6,
    "byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12,
}
METRICS = {
    "time_us": "gpu__time_duration.sum",
    "dram_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_read": "dram__bytes_read.sum",
    "dram_write": "dram__bytes_write.sum",
    "sm_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "achieved_occ": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "theoretical_occ": "sm__maximum_warps_per_active_cycle_pct",
    "regs": "launch__registers_per_thread",
    "l2_hit": "lts__t_sector_hit_rate.pct",
}
STALL = re.compile(r"smsp__average_warps_issue_stalled_(\w+?)_per_issue_active\.ratio$")


def number(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except (AttributeError, ValueError):
        return None


def read_rows(path: Path) -> list[dict]:
    lines = path.read_text(errors="replace").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    reader = csv.reader(lines[start:])
    header, units = next(reader), next(reader)
    rows = []
    for values in reader:
        if len(values) != len(header):
            continue
        row = {"name": values[header.index("Kernel Name")]}
        for key, metric in METRICS.items():
            if metric in header:
                i = header.index(metric)
                value = number(values[i])
                if value is not None:
                    value *= UNIT_SCALE.get(units[i], 1)
                row[key] = value
        stalls = {}
        for i, column in enumerate(header):
            match = STALL.match(column)
            if match and number(values[i]) is not None:
                stalls[match.group(1)] = number(values[i])
        row["stalls"] = stalls
        rows.append(row)
    return rows


def short_name(name: str) -> str:
    name = re.sub(r"\(.*$", "", name)  # drop the parameter list
    return name[:100]


def mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def analyze(rows: list[dict], label: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[short_name(row["name"])].append(row)
    total_us = sum(r.get("time_us") or 0 for r in rows)
    total_bytes = sum((r.get("dram_read") or 0) + (r.get("dram_write") or 0) for r in rows)
    kernels = []
    for name, members in groups.items():
        time_us = sum(r.get("time_us") or 0 for r in members)
        moved = sum((r.get("dram_read") or 0) + (r.get("dram_write") or 0) for r in members)
        stalls: dict[str, float] = defaultdict(float)
        for r in members:
            for reason, value in r["stalls"].items():
                stalls[reason] += value * (r.get("time_us") or 0)
        top = sorted(stalls.items(), key=lambda kv: -kv[1])[:2]
        kernels.append(
            {
                "name": name,
                "category": classify(name),
                "launches": len(members),
                "time_us": time_us,
                "time_frac": time_us / total_us if total_us else None,
                "avg_us": time_us / len(members),
                "dram_pct": mean([r.get("dram_pct") for r in members]),
                "dram_GBps": moved / (time_us * 1e-6) / 1e9 if time_us else None,
                "sm_pct": mean([r.get("sm_pct") for r in members]),
                "achieved_occ": mean([r.get("achieved_occ") for r in members]),
                "theoretical_occ": mean([r.get("theoretical_occ") for r in members]),
                "regs": mean([r.get("regs") for r in members]),
                "l2_hit": mean([r.get("l2_hit") for r in members]),
                "top_stalls": [reason for reason, _ in top],
            }
        )
    kernels.sort(key=lambda k: -k["time_us"])
    by_category: dict[str, float] = defaultdict(float)
    for k in kernels:
        by_category[k["category"]] += k["time_us"]
    return {
        "label": label,
        "kernels_profiled": len(rows),
        "total_kernel_us": total_us,
        "dram_GBps_over_kernel_time": total_bytes / (total_us * 1e-6) / 1e9 if total_us else None,
        "time_by_category": {
            c: {"us": us, "frac": us / total_us if total_us else None}
            for c, us in sorted(by_category.items(), key=lambda kv: -kv[1])
        },
        "kernels": kernels,
    }


def fmt(value, digits=0) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def markdown(report: dict, top: int = 15) -> str:
    lines = [
        f"### ncu {report['label']}: {report['kernels_profiled']} kernels, "
        f"{report['total_kernel_us']:.0f} us, "
        f"{fmt(report['dram_GBps_over_kernel_time'])} GB/s DRAM over kernel time",
        "",
        "| kernel | cat | n | time % | avg us | DRAM % | GB/s | SM % | occ ach/theo | regs | L2 hit | top stalls |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for k in report["kernels"][:top]:
        lines.append(
            f"| `{k['name'][:60]}` | {k['category']} | {k['launches']} | {fmt(100 * (k['time_frac'] or 0))} "
            f"| {fmt(k['avg_us'], 1)} | {fmt(k['dram_pct'])} | {fmt(k['dram_GBps'])} | {fmt(k['sm_pct'])} "
            f"| {fmt(k['achieved_occ'])}/{fmt(k['theoretical_occ'])} | {fmt(k['regs'])} | {fmt(k['l2_hit'])} "
            f"| {', '.join(k['top_stalls'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw_csv", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    report = analyze(read_rows(args.raw_csv), args.label)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.label}.json").write_text(json.dumps(report, indent=2))
    print(markdown(report), flush=True)


if __name__ == "__main__":
    main()
