#!/usr/bin/env python3
"""
Smoke test for the local Docker Compose serving stack.

Expected stack:
  docker compose -f docker-compose.local.yml up -d --build
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


def request(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
    timeout: float = 10.0,
) -> tuple[int, str]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def get_json(base_url: str, path: str, token: Optional[str] = None) -> Dict[str, Any]:
    status, body = request("GET", f"{base_url}{path}", token=token)
    assert_status(status, 200, path, body)
    return json.loads(body)


def post_json(
    base_url: str,
    path: str,
    payload: Dict[str, Any],
    token: Optional[str] = None,
) -> Dict[str, Any]:
    status, body = request("POST", f"{base_url}{path}", payload=payload, token=token)
    assert_status(status, 200, path, body)
    return json.loads(body)


def assert_status(status: int, expected: int, path: str, body: str) -> None:
    if status != expected:
        raise AssertionError(f"{path} returned HTTP {status}, expected {expected}: {body[:300]}")


def wait_for_health(base_url: str, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            health = get_json(base_url, "/health")
            if health.get("status") == "healthy":
                return
        except Exception as exc:  # pragma: no cover - exercised by real compose only
            last_error = str(exc)
        time.sleep(1.0)
    raise TimeoutError(f"Gateway did not become healthy: {last_error}")


def run_smoke(base_url: str, username: str, password: str, timeout_seconds: float) -> None:
    base_url = base_url.rstrip("/")
    wait_for_health(base_url, timeout_seconds)

    token_response = post_json(
        base_url,
        "/auth/token",
        {"username": username, "password": password},
    )
    token = token_response["access_token"]

    completion = post_json(
        base_url,
        "/v1/completions",
        {
            "prompt": "Explain reliable LLM serving in one sentence.",
            "max_tokens": 32,
            "temperature": 0.2,
        },
        token=token,
    )
    if completion.get("object") != "text_completion" or not completion.get("choices"):
        raise AssertionError(f"Invalid completion contract: {completion}")
    if "usage" not in completion:
        raise AssertionError(f"Completion response missing usage: {completion}")

    chat = post_json(
        base_url,
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "Say ready."}],
            "max_tokens": 16,
            "temperature": 0.2,
        },
        token=token,
    )
    if chat.get("object") != "chat.completion" or not chat.get("choices"):
        raise AssertionError(f"Invalid chat contract: {chat}")
    message = chat["choices"][0].get("message") or {}
    if message.get("role") != "assistant" or not message.get("content"):
        raise AssertionError(f"Invalid chat message: {chat}")

    status, stream_body = request(
        "POST",
        f"{base_url}/v1/completions",
        {
            "prompt": "Stream a short readiness response.",
            "max_tokens": 24,
            "stream": True,
        },
        token=token,
        timeout=timeout_seconds,
    )
    assert_status(status, 200, "/v1/completions stream", stream_body)
    if "data: [DONE]" not in stream_body:
        raise AssertionError("Streaming response did not include data: [DONE]")

    usage = get_json(base_url, "/usage", token=token)
    if usage.get("total_requests", 0) < 3 or usage.get("total_tokens", 0) <= 0:
        raise AssertionError(f"Usage was not recorded: {usage}")

    status, metrics = request("GET", f"{base_url}/metrics", timeout=timeout_seconds)
    assert_status(status, 200, "/metrics", metrics)
    for metric_name in (
        "llm_requests_total",
        "llm_tokens_processed_total",
        "llm_time_to_first_token_seconds",
    ):
        if metric_name not in metrics:
            raise AssertionError(f"Missing metric {metric_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test local LLM serving compose stack")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--username", default="local")
    parser.add_argument("--password", default="local")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    run_smoke(
        base_url=args.base_url,
        username=args.username,
        password=args.password,
        timeout_seconds=args.timeout,
    )
    print("compose-smoke-ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
