#!/usr/bin/env python3
"""GPU-side AWQ serving smoke test for an already-running vLLM/gateway stack."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test AWQ model serving")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--token")
    parser.add_argument("--require-local-path", action="store_true")
    return parser.parse_args()


def request_json(url: str, payload: dict | None = None, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        raise


def main() -> int:
    args = parse_args()
    if args.require_local_path and args.model.startswith("/") and not Path(args.model).is_dir():
        print(f"AWQ model path does not exist: {args.model}", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    request_json(f"{base_url}/ready")
    payload = {
        "model": args.model,
        "prompt": "Reply with ok.",
        "max_tokens": 8,
        "temperature": 0.0,
    }
    result = request_json(f"{base_url}/v1/completions", payload=payload, token=args.token)
    if not result.get("choices"):
        print("completion response missing choices", file=sys.stderr)
        return 1
    print("vllm-awq-smoke-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
