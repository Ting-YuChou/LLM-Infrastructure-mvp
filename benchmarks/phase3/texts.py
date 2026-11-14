#!/usr/bin/env python3
"""Completions of the same prompts, kept as text (run_phase3.sh's texts workload).

Speculative decoding must not change what a server answers. This sends the
first prompts of each dataset's c=1 file to /v1/completions at temperature 0,
once one at a time and once 16 at a time, and saves the texts; summarize.py
compares a drafter server's against the same server without one, and the
server's own two passes against each other (logits depend slightly on the
batch size, so a few greedy streams diverge without any speculation).

  python texts.py --base-url http://127.0.0.1:8100 --model <name> \
      --workloads <dir> --out <file.json> [--prompts 8] [--max-tokens 256]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATASETS = ("alpaca", "humaneval", "gsm8k", "long")


def complete(base_url: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    ).encode()
    request = urllib.request.Request(
        f"{base_url}/v1/completions", body, {"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    reply = json.load(urllib.request.urlopen(request, timeout=600))
    choice = reply["choices"][0]
    return {
        "text": choice["text"],
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": reply["usage"]["completion_tokens"],
        "seconds": time.perf_counter() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=8, help="per dataset")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    prompts = []
    for dataset in DATASETS:
        lines = (args.workloads / f"{dataset}_c1.jsonl").read_text().splitlines()[: args.prompts]
        prompts += [(dataset, json.loads(line)["prompt"]) for line in lines]

    def one(item):
        return {"dataset": item[0], **complete(args.base_url, args.model, item[1], args.max_tokens)}

    result = {"c1": [one(item) for item in prompts]}
    with ThreadPoolExecutor(max_workers=16) as pool:
        result["c16"] = list(pool.map(one, prompts))
    args.out.write_text(json.dumps(result))
    for name, rows in result.items():
        tokens = sum(row["completion_tokens"] for row in rows)
        print(f"P3_TEXTS {args.out.stem} {name}: {len(rows)} completions, {tokens} tokens")


if __name__ == "__main__":
    main()
