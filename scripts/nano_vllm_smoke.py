#!/usr/bin/env python3
"""Smoke test an already-running nano-vLLM backend or gateway route."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


def request_json(
    url: str, payload: dict | None = None, token: Optional[str] = None
) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test nano-vLLM serving")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--model", default="nano-llm-local")
    parser.add_argument("--token")
    parser.add_argument("--require-local-path", action="store_true")
    args = parser.parse_args()

    if (
        args.require_local_path
        and args.model.startswith("/")
        and not Path(args.model).is_dir()
    ):
        print(f"nano-vLLM model path does not exist: {args.model}", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    request_json(f"{base_url}/ready")
    completion = request_json(
        f"{base_url}/v1/completions",
        payload={
            "model": args.model,
            "prompt": "Explain nano-vLLM in one short sentence.",
            "max_tokens": 16,
            "temperature": 0.0,
        },
        token=args.token,
    )
    if completion.get("object") != "text_completion" or not completion.get("choices"):
        print(f"invalid completion response: {completion}", file=sys.stderr)
        return 1

    chat = request_json(
        f"{base_url}/v1/chat/completions",
        payload={
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with ready."}],
            "max_tokens": 16,
            "temperature": 0.0,
        },
        token=args.token,
    )
    if chat.get("object") != "chat.completion" or not chat.get("choices"):
        print(f"invalid chat response: {chat}", file=sys.stderr)
        return 1

    print("nano-vllm-smoke-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
