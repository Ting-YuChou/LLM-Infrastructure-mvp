#!/usr/bin/env python3
"""Return a gateway JWT for deployment smoke tests."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Authenticate against the gateway")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.username or not args.password:
        print("smoke credentials are required", file=sys.stderr)
        return 2

    payload = json.dumps({
        "username": args.username,
        "password": args.password,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/auth/token",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        return 1

    print(data["access_token"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
