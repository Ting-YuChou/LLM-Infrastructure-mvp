#!/usr/bin/env python3
"""Authenticated OpenAI-compatible completion smoke test."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a completion smoke test")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Say ok.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.dumps({
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": 8,
        "temperature": 0.0,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {args.token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        return 1

    if not data.get("choices"):
        print("completion response missing choices", file=sys.stderr)
        return 1
    print("completion-smoke-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
