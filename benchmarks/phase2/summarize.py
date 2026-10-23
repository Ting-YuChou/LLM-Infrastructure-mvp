#!/usr/bin/env python3
"""summary.md for one Phase 2 run directory (run_phase2.sh's summarize step).

Reads the P2_SERVE lines from the run's console (one per `vllm bench serve` run),
the nsys/*.json reports and status.tsv, and writes tables per workload:
  w1  throughput, TTFT and ITL by concurrency, per server config
  w1u the same at c=1 and 16 with prompts no earlier run sent
  w2  shared-prefix runs
  w3  the background decodes' ITL tail and the long prompts' TTFT
plus per-step GPU idle and the engine's host phases from nsys.

  python summarize.py <run dir>
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def serve_rows(out: Path) -> list[dict]:
    """The P2_SERVE lines (bench_serve prints them to the run's console).

    On the VM the console is <root>/console.log next to results/<run>; the
    report tarball puts a copy in the run directory.
    """
    rows, seen = [], set()
    consoles = [out / "console.log", out.resolve().parents[1] / "console.log"]
    for path in [*consoles, *sorted((out / "logs").glob("*.log"))]:
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("P2_SERVE {") and line not in seen:
                seen.add(line)
                rows.append(json.loads(line[len("P2_SERVE ") :]))
    return rows


def fmt(value, digits=1) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def w1_tables(rows: list[dict], workload: str = "w1", title: str = "W1: random 1024/256") -> list[str]:
    by_label: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        m = re.fullmatch(rf"{workload}_c(\d+)", r["run"])
        if m:
            by_label[r["engine"]][int(m.group(1))] = r
    if not by_label:
        return []
    levels = sorted({c for runs in by_label.values() for c in runs})
    lines = [f"## {title}", ""]
    for title, key, digits in (
        ("output tok/s", "out_tok_s", 0),
        ("TTFT p50 (ms)", "ttft_p50_ms", 1),
        ("ITL p50 (ms)", "itl_p50_ms", 2),
        ("ITL p99 (ms)", "itl_p99_ms", 2),
    ):
        lines += [f"**{title}**", "", "| config | " + " | ".join(f"c={c}" for c in levels) + " |",
                  "|---|" + "---:|" * len(levels)]
        for label, runs in by_label.items():
            cells = []
            for c in levels:
                r = runs.get(c)
                if r is None:
                    cells.append("-")
                else:
                    cells.append(fmt(r.get(key), digits) + ("" if r.get("ok") else " ✗"))
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def _prom_total(text: str, names: tuple[str, ...]) -> float | None:
    """Sum of the samples of the first metric name present in a /metrics dump."""
    for name in names:
        values = [
            float(line.rsplit(" ", 1)[1])
            for line in text.splitlines()
            if line.startswith(name) and (line[len(name)] in " {")
        ]
        if values:
            return sum(values)
    return None


def hit_rate(out: Path, label: str, workload: str) -> float | None:
    """Prefix-cache token hit rate during one workload, from the snapshots around it."""
    serve = out / "serve"
    before, after = serve / f"{label}_{workload}_before", serve / f"{label}_{workload}_after"
    stats = [p.with_suffix(".stats.json") for p in (before, after)]
    if all(p.exists() for p in stats):  # nano
        b, a = (json.loads(p.read_text()) for p in stats)
        queries = a.get("prefix_query_tokens", 0) - b.get("prefix_query_tokens", 0)
        hits = a.get("prefix_hit_tokens", 0) - b.get("prefix_hit_tokens", 0)
        return hits / queries if queries else None
    dumps = [p.with_suffix(".metrics.txt") for p in (before, after)]
    if all(p.exists() for p in dumps):  # vLLM: prefix_cache_{queries,hits} counters
        b, a = (p.read_text() for p in dumps)
        names_q = ("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total")
        names_h = ("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total")
        qb, qa = _prom_total(b, names_q), _prom_total(a, names_q)
        hb, ha = _prom_total(b, names_h), _prom_total(a, names_h)
        if None not in (qb, qa, hb, ha) and qa > qb:
            return (ha - hb) / (qa - qb)
    return None


def w2_table(rows: list[dict], out: Path) -> list[str]:
    picked = [r for r in rows if r["run"] in ("w2a", "w2b")]
    if not picked:
        return []
    lines = [
        "## W2: shared prefixes", "",
        "w2a: one 2048-token prefix + 128 tokens, output 128, c=16. "
        "w2b: 48 prefixes of 4096 tokens + 256, output 128, c=16. "
        "Hit rate: prompt tokens served from the prefix cache during the run.", "",
        "| config | run | ok | req/s | output tok/s | TTFT p50 | TTFT p99 | ITL p50 | hit rate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(picked, key=lambda r: (r["run"], r["engine"])):
        rate = hit_rate(out, r["engine"], r["run"])
        lines.append(
            f"| {r['engine']} | {r['run']} | {'yes' if r.get('ok') else 'no'} | {fmt(r.get('req_s'), 2)} "
            f"| {fmt(r.get('out_tok_s'), 0)} | {fmt(r.get('ttft_p50_ms'))} | {fmt(r.get('ttft_p99_ms'))} "
            f"| {fmt(r.get('itl_p50_ms'), 2)} | {'-' if rate is None else f'{100 * rate:.1f}%'} |"
        )
    return lines + [""]


def w3_table(rows: list[dict]) -> list[str]:
    bg = {r["engine"]: r for r in rows if r["run"] == "w3_bg"}
    fg = {r["engine"]: r for r in rows if r["run"] == "w3_fg"}
    if not bg and not fg:
        return []
    lines = [
        "## W3: prefill interference", "",
        "Background 256/1024 at c=32; 4096-token prompts (output 8) at 0.25 req/s on top. "
        "ITL columns are the background's.", "",
        "| config | ITL p50 | ITL p99 | ITL p99.9 | ITL max | bg tok/s | long TTFT p50 | long TTFT max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in sorted(set(bg) | set(fg)):
        b, f = bg.get(label, {}), fg.get(label, {})
        lines.append(
            f"| {label} | {fmt(b.get('itl_p50_ms'), 2)} | {fmt(b.get('itl_p99_ms'))} "
            f"| {fmt(b.get('itl_p99.9_ms'))} | {fmt(b.get('itl_p100_ms'))} | {fmt(b.get('out_tok_s'), 0)} "
            f"| {fmt(f.get('ttft_p50_ms'))} | {fmt(f.get('ttft_p100_ms'))} |"
        )
    return lines + [""]


def nsys_table(out: Path) -> list[str]:
    reports = [json.loads(p.read_text()) for p in sorted((out / "nsys").glob("*.json"))]
    reports = [r for r in reports if "per_step" in r]
    if not reports:
        return []
    lines = [
        "## nsys: nano server per engine step", "",
        "| capture | wall ms | GPU busy ms | GPU idle ms | kernels | host phases (median µs) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in reports:
        per = r["per_step"]
        phases = ", ".join(
            f"{name} {v['median_us']:.0f}" for name, v in list(r.get("nvtx_phases", {}).items())[:7]
        )
        lines.append(
            f"| {r['label']} | {per['wall_ms']:.2f} | {per['gpu_busy_ms']:.2f} | {per['gpu_idle_ms']:.2f} "
            f"| {per['kernels']:.0f} | {phases or '-'} |"
        )
    return lines + [""]


def offline_table(out: Path) -> list[str]:
    rows = []
    for path in sorted((out / "offline").glob("nano_*.json")):
        if path.stem.endswith("_nsys"):
            continue
        rows += json.loads(path.read_text())
    if not rows:
        return []
    lines = [
        "## Offline decode (nano_bench, prefill-first, CUDA graphs)", "",
        "| async | batch | in/out | decode ms/step | p50 | tok/s |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {'yes' if r.get('async_scheduling') else 'no'} | {r['batch']} | {r['input_len']}/{r['output_len']} "
            f"| {r['decode_ms_per_step']:.2f} | {r['decode_ms_p50']:.2f} | {r['decode_tok_per_s']:.0f} |"
        )
    return lines + [""]


def server_settings(out: Path) -> list[str]:
    lines = []
    for path in sorted((out / "logs").glob("*_server.log")):
        text = path.read_text(errors="replace")
        found = re.findall(
            r"(GPU KV cache size: [0-9,]+ tokens|max_num_seqs=\d+|max_num_batched_tokens=\d+"
            r"|chunked_prefill_size=\d+|max_total_num_tokens=\d+)",
            text,
        )
        if found:
            lines.append(f"- {path.stem}: " + ", ".join(dict.fromkeys(found)))
    return ["## Server settings from the logs", "", *lines, ""] if lines else []


def main() -> None:
    out = Path(sys.argv[1])
    rows = serve_rows(out)
    lines = [f"# Phase 2 run {out.name}", ""]
    status = out / "status.tsv"
    if status.exists():
        lines += ["| step | rc | seconds |", "|---|---:|---:|"]
        lines += [f"| {' | '.join(line.split(chr(9)))} |" for line in status.read_text().splitlines()]
        lines.append("")
    bad = [f"{r['engine']}/{r['run']}" for r in rows if not r.get("ok")]
    if bad:
        lines += [f"Runs that did not complete every request with the full output: {', '.join(bad)}", ""]
    bench = out / "prefix_bench" / "prefix_bench.md"
    if bench.exists():
        lines += ["## " + bench.read_text().lstrip("# "), ""]
    lines += w1_tables(rows)
    lines += w1_tables(rows, "w1u", "W1u: random 1024/128, prompts no earlier run sent (uncached)")
    lines += w2_table(rows, out) + w3_table(rows) + offline_table(out)
    lines += nsys_table(out) + server_settings(out)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "serve_rows.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
