#!/usr/bin/env python3
"""Compare request latency between two OpenAI-like completion endpoints."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def post_completion(
    base_url: str, model: str, prompt: str, max_tokens: int, token: str | None
) -> float:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=payload,
        headers=headers,
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        response.read()
    return (time.perf_counter() - start) * 1000.0


def run(label: str, base_url: str, model: str, args: argparse.Namespace) -> None:
    samples = [
        post_completion(base_url, model, args.prompt, args.max_tokens, args.token)
        for _ in range(args.requests)
    ]
    print(
        f"{label}: mean={statistics.mean(samples):.1f}ms "
        f"p50={statistics.median(samples):.1f}ms "
        f"max={max(samples):.1f}ms"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare nano-vLLM and vLLM endpoint latency"
    )
    parser.add_argument("--nano-url", required=True)
    parser.add_argument("--nano-model", default="nano-llm-local")
    parser.add_argument("--vllm-url")
    parser.add_argument("--vllm-model")
    parser.add_argument("--token")
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt", default="Explain LLM serving in one sentence.")
    args = parser.parse_args()

    run("nano-vllm", args.nano_url, args.nano_model, args)
    if args.vllm_url and args.vllm_model:
        run("vllm", args.vllm_url, args.vllm_model, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
