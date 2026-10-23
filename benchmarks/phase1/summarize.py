#!/usr/bin/env python3
"""summary.md for one Phase 1 run directory.

  python summarize.py results/<run_id>

Collects what run_phase1.sh wrote: status.tsv, the pytest summary, kbench.md,
the gemm_w4a16_tiled trace (trace.md), the ncu tables of the kbench kernels, nano_vllm against vLLM at the same decode
points (offline/) and the nsys step breakdowns. Writes and prints summary.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase0"))
from analyze_ncu import markdown as ncu_markdown  # noqa: E402


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def fmt(value, digits=2) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def main() -> None:
    run = Path(sys.argv[1])
    lines = [f"# Phase 1 results: run {run.name}", ""]

    status = run / "status.tsv"
    if status.exists():
        lines += ["## Steps", "", "| step | rc | seconds |", "|---|---:|---:|"]
        for row in status.read_text().splitlines():
            name, rc, seconds = (row.split("\t") + ["", ""])[:3]
            lines.append(f"| {name} | {'ok' if rc == '0' else f'**{rc}**'} | {seconds} |")
        lines.append("")

    env = run / "environment.txt"
    if env.exists():
        lines += ["## Environment", "", "```", env.read_text().strip(), "```", ""]

    pytest_log = run / "tests" / "pytest.log"
    if pytest_log.exists():
        text = pytest_log.read_text().splitlines()
        notable = [line for line in text if line.startswith(("SKIPPED", "FAILED", "ERROR"))]
        lines += ["## Tests", "", "```", *(text[-1:] + notable), "```", ""]

    kbench = run / "kbench" / "kbench.md"
    if kbench.exists():  # its own title becomes this section's
        body = kbench.read_text().splitlines()[1:]
        lines += ["## Kernels (kbench)", *[f"#{line}" if line.startswith("## ") else line for line in body], ""]

    trace = run / "trace" / "trace.md"
    if trace.exists():  # tiled_trace.py; its title becomes this section's
        lines += ["## gemm_w4a16_tiled trace", *trace.read_text().splitlines()[1:], ""]

    ncu = [load_json(p) for p in sorted((run / "ncu").glob("kbench_*.json"))]
    if any(ncu):
        lines += ["## Kernels under ncu", ""]
        for report in filter(None, ncu):
            lines += [ncu_markdown(report), ""]

    # One nano_<config>.json per nano_bench configuration (run_phase1.sh e2e).
    nano = [
        (path.stem.removeprefix("nano_"), d)
        for path in sorted((run / "offline").glob("nano_*.json"))
        if not path.stem.endswith("_nsys")
        for d in (load_json(path) or [])
    ]
    def point(d: dict) -> tuple:  # sampled runs with sampled runs, fp8 KV with fp8 KV
        return (d["batch"], d["input_len"], d["output_len"], d.get("temperature", 0.0) > 0,
                d.get("kv_cache_dtype", "auto"))

    def vllm_runs(pattern: str) -> dict:
        return {
            point(d): d
            for d in filter(None, (load_json(p) for p in sorted((run / "offline").glob(pattern))))
        }

    vllm = vllm_runs("vllm_*.json")
    # w4a16 nano configs are compared with vLLM on the AWQ int4 checkpoint.
    vllm_w4 = vllm_runs("vllmw4_*.json")
    if nano or vllm or vllm_w4:
        lines += [
            "## End to end decode (nano_vllm vs vLLM CUDA graph)",
            "",
            "| nano config | path | batch | in | out | nano ms/step | nano tok/s | vLLM | vLLM ms/step "
            "| vLLM tok/s | nano / vLLM time |",
            "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
        for config, d in nano:
            quantized = d.get("weight_quant", "none") != "none"
            v = (vllm_w4 if quantized else vllm).get(point(d), {})
            baseline = f"{v.get('quantization') or 'bf16'}" if v else "-"
            ratio = d["decode_ms_per_step"] / v["decode_ms_per_step"] if v else None
            lines.append(
                f"| {config} | {d['path']} | {d['batch']} | {d['input_len']} | {d['output_len']} "
                f"| {fmt(d['decode_ms_per_step'])} | {fmt(d['decode_tok_per_s'], 0)} | {baseline} "
                f"| {fmt(v.get('decode_ms_per_step'))} | {fmt(v.get('decode_tok_per_s'), 0)} "
                f"| {fmt(ratio)}{'x' if ratio else ''} |"
            )
        lines.append("")

    nsys = {p.stem: load_json(p) for p in sorted((run / "nsys").glob("*.json"))}
    if any(nsys.values()):
        lines += [
            "## Where a nano decode step goes (nsys)",
            "",
            "| capture | ms/step | GPU busy ms | GPU idle ms | kernels/step | launch calls/step | top kernel categories |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for label, report in nsys.items():
            if not report:
                continue
            per = report["per_step"]
            cats = ", ".join(
                f"{c} {100 * v['frac']:.0f}%" for c, v in list(report["kernel_time_by_category"].items())[:4]
            )
            lines.append(
                f"| {label} | {per['wall_ms']:.2f} | {per['gpu_busy_ms']:.2f} | {per['gpu_idle_ms']:.2f} "
                f"| {per['kernels']:.0f} | {per['launch_calls']:.0f} | {cats} |"
            )
        lines.append("")

    quality = run / "quality" / "quality.md"
    if quality.exists():
        body = quality.read_text().splitlines()
        lines += [f"#{body[0]}", *body[1:], ""]

    text = "\n".join(lines)
    (run / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
