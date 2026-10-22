#!/usr/bin/env python3
"""Tables and plots for one Phase 0 run directory.

  python summarize.py results/<run_id>

Reads what run_phase0.sh wrote (serve/, offline/, nsys/, bandwidth.json,
roofline.json) and writes summary.md plus plots/*.png (when matplotlib is
installed). Prints summary.md.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def fmt(value, digits=1) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def serve_runs(run: Path) -> dict:
    runs: dict = {}
    for path in sorted((run / "serve").glob("*_*.json")):
        match = re.match(r"(\w+?)_(latency_c1|sweep_c(\d+))$", path.stem)
        data = load_json(path)
        if not match or not data:
            continue
        engine, label, conc = match.group(1), match.group(2), match.group(3)
        runs.setdefault(engine, {})[label if conc is None else int(conc)] = data
    return runs


def main() -> None:
    run = Path(sys.argv[1])
    roofline = load_json(run / "roofline.json") or []
    sheet = roofline[0] if roofline else None
    measured = roofline[1] if len(roofline) > 1 else None
    bandwidth = load_json(run / "bandwidth.json")
    serve = serve_runs(run)
    offline = {  # the *_ncu runs time steps with ncu attached; leave them out
        p.stem: load_json(p)
        for p in sorted((run / "offline").glob("vllm_*.json"))
        if not p.stem.endswith("_ncu")
    }
    nano = load_json(run / "offline" / "nano.json") or []
    nsys = {
        p.stem: load_json(p)
        for p in sorted((run / "nsys").glob("*.json"))
        if not p.stem.endswith("_nsys")
    }
    lines = [f"# Phase 0 results: run {run.name}", ""]

    env = run / "environment.txt"
    if env.exists():
        lines += ["## Environment", "", "```", env.read_text().strip(), "```", ""]

    # --- batch=1 decode against the bound -----------------------------------
    bound = sheet["batch1_ms_per_token"] if sheet else None
    rows = []
    if sheet:
        rows.append((f"roofline, datasheet {sheet['hbm_GBps']:,.0f} GB/s", bound))
    if measured:
        rows.append(
            (f"roofline, measured copy {measured['hbm_GBps']:,.0f} GB/s", measured["batch1_ms_per_token"])
        )
    if bandwidth:
        floor = {r["M"]: r["ms_per_step"] for r in bandwidth["decode_linear_floor"]}
        rows.append(("cuBLAS linear layers only (M=1, no attention)", floor.get(1)))
    for engine in ("vllm", "sglang"):
        lat = serve.get(engine, {}).get("latency_c1")
        if lat:
            rows.append((f"{engine} serve, TPOT p50 (in 128 / out 256)", lat.get("median_tpot_ms")))
    for label, data in offline.items():
        if data and data.get("batch") == 1:
            rows.append((f"{label} offline engine step", data["decode_ms_per_step"]))
    for data in nano:
        if data["batch"] == 1:
            rows.append(
                (f"nano_vllm {data['path']} (in {data['input_len']})", data["decode_ms_per_step"])
            )
    lines += [
        "## Batch=1 decode",
        "",
        "| what | ms/token | tok/s | % of roofline |",
        "|---|---:|---:|---:|",
    ]
    for name, ms in rows:
        if ms:
            pct = f"{100 * bound / ms:.0f}%" if bound else "-"
            lines.append(f"| {name} | {ms:.2f} | {1e3 / ms:.0f} | {pct} |")
    lines.append("")

    # --- serving sweep -------------------------------------------------------
    serve_bound = {r["concurrency"]: r for r in sheet["serve_sweep"]} if sheet else {}
    lines += [
        "## Serving sweep (random, in 1024 / out 256, ignore_eos)",
        "",
        "| engine | conc | out tok/s | roofline % | total tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for engine, runs in serve.items():
        for conc in sorted(k for k in runs if isinstance(k, int)):
            r = runs[conc]
            ub = serve_bound.get(conc, {}).get("output_tok_per_s")
            pct = f"{100 * r['output_throughput'] / ub:.0f}%" if ub else "-"
            lines.append(
                f"| {engine} | {conc} | {fmt(r.get('output_throughput'), 0)} | {pct} "
                f"| {fmt(r.get('total_token_throughput'), 0)} "
                f"| {fmt(r.get('median_ttft_ms'))} | {fmt(r.get('p99_ttft_ms'))} "
                f"| {fmt(r.get('median_tpot_ms'), 2)} | {fmt(r.get('p99_tpot_ms'), 2)} "
                f"| {fmt(r.get('median_itl_ms'), 2)} | {fmt(r.get('p99_itl_ms'), 2)} |"
            )
    for engine in serve:
        startup = load_json(run / "serve" / f"{engine}_startup.json")
        if startup:
            lines.append(f"\n{engine} server ready after {startup['startup_s']} s.")
    lines.append("")

    # --- nsys ----------------------------------------------------------------
    if nsys:
        lines += [
            "## Where a decode step goes (nsys)",
            "",
            "| capture | batch | ms/step | GPU busy ms | GPU idle ms | kernels/step | launch calls/step | top kernel categories |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for label, rep in nsys.items():
            if not rep:
                continue
            per = rep["per_step"]
            cats = ", ".join(
                f"{c} {100 * v['frac']:.0f}%"
                for c, v in list(rep["kernel_time_by_category"].items())[:4]
            )
            lines.append(
                f"| {label} | {rep.get('batch', '-')} | {per['wall_ms']:.2f} | {per['gpu_busy_ms']:.2f} "
                f"| {per['gpu_idle_ms']:.2f} | {per['kernels']:.0f} | {per['launch_calls']:.0f} | {cats} |"
            )
        lines.append("")

    # --- ncu -----------------------------------------------------------------
    ncu = [load_json(p) for p in sorted((run / "ncu").glob("*.json"))]
    if any(ncu):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from analyze_ncu import markdown as ncu_markdown

        lines += ["## Decode kernels under ncu (one decode step)", ""]
        for rep in filter(None, ncu):
            lines += [ncu_markdown(rep), ""]

    # --- nano ----------------------------------------------------------------
    if nano:
        lines += [
            "## nano_vllm today",
            "",
            "| path | batch | in | ms/step | tok/s | model ms/step | engine overhead ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for d in nano:
            lines.append(
                f"| {d['path']} | {d['batch']} | {d['input_len']} | {d['decode_ms_per_step']:.2f} "
                f"| {d['decode_tok_per_s']:.0f} | {fmt(d.get('decode_model_ms_per_step'), 2)} "
                f"| {fmt(d.get('decode_engine_overhead_ms'), 2)} |"
            )
        lines.append("")

    text = "\n".join(lines)
    (run / "summary.md").write_text(text + "\n")
    print(text)
    plot(run, serve, sheet, measured)


def plot(run: Path, serve: dict, sheet, measured) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    out = run / "plots"
    out.mkdir(exist_ok=True)
    metrics = [
        ("output_throughput", "output tokens/s", "throughput_vs_concurrency.png"),
        ("median_tpot_ms", "TPOT p50 (ms)", "tpot_vs_concurrency.png"),
        ("median_ttft_ms", "TTFT p50 (ms)", "ttft_vs_concurrency.png"),
    ]
    for key, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for engine, runs in serve.items():
            concs = sorted(k for k in runs if isinstance(k, int))
            ax.plot(concs, [runs[c].get(key) for c in concs], marker="o", label=engine)
        if key == "output_throughput":
            for report, style, name in ((sheet, "--", "roofline (datasheet)"), (measured, ":", "roofline (measured)")):
                if report:
                    rows = report["serve_sweep"]
                    ax.plot(
                        [r["concurrency"] for r in rows],
                        [r["output_tok_per_s"] for r in rows],
                        style, color="gray", label=name,
                    )
        if key == "median_tpot_ms" and sheet:
            rows = sheet["decode_sweep_ctx"]
            ax.plot([r["batch"] for r in rows], [r["ms_per_step"] for r in rows], "--", color="gray",
                    label="decode-step bound (ctx 1152)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("max concurrency")
        ax.set_ylabel(ylabel)
        gpu = sheet["gpu"].split(" ")[0] if sheet else "GPU"
        ax.set_title(f"Llama 3.1 8B BF16, 1x {gpu}: {ylabel}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
