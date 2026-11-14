#!/usr/bin/env python3
"""Summarize one `vllm bench serve` result and slim its saved JSON.

Prints one line, "<tag> {json}", with throughput and the TTFT/TPOT/ITL
percentiles, and rewrites the result file without its per-request arrays
(--keep-raw first copies the full file to <name>.raw.json). ok means every
request completed with exactly output_len tokens (the runs use --ignore-eos);
with --variable-output (requests may stop at EOS) only that all completed.

  serve_summary.py result.json vllm sweep_c16 64 256 --tag P0_SERVE
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("result", type=Path)
    parser.add_argument("engine")
    parser.add_argument("label")
    parser.add_argument("num_prompts", type=int)
    parser.add_argument("output_len", type=int)
    parser.add_argument("--tag", default="P0_SERVE")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument(
        "--variable-output",
        action="store_true",
        help="requests may stop at EOS: ok only needs every request completed, "
        "and the line reports the token totals",
    )
    args = parser.parse_args()

    try:
        result = json.loads(args.result.read_text())
    except (OSError, ValueError) as e:
        print(f"{args.tag} {args.engine} {args.label} missing result: {e}")
        return
    if args.keep_raw:
        shutil.copyfile(args.result, args.result.with_suffix(".raw.json"))
    slim = {k: v for k, v in result.items() if not isinstance(v, (list, dict))}
    args.result.write_text(json.dumps(slim, indent=2))

    n = args.num_prompts
    full = slim.get("total_output_tokens") == n * args.output_len
    summary = {
        "engine": args.engine,
        "run": args.label,
        "ok": slim.get("completed") == n and (full or args.variable_output),
        "completed": slim.get("completed"),
        "req_s": round(slim.get("request_throughput", 0), 3),
        "out_tok_s": round(slim.get("output_throughput", 0), 1),
    }
    if args.variable_output:
        summary["in_tok"] = slim.get("total_input_tokens")
        summary["out_tok"] = slim.get("total_output_tokens")
    for metric in ("ttft", "tpot", "itl", "e2el"):
        for stat in ("median", "p90", "p99", "p99.9", "p100"):
            value = slim.get(f"{stat}_{metric}_ms")
            if value is not None:
                summary[f"{metric}_{'p50' if stat == 'median' else stat}_ms"] = round(value, 2)
    print(f"{args.tag} " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
