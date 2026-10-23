#!/usr/bin/env python3
"""Host cost of the prefix cache: the radix tree against Phase 1's hash chain.

Replays the W2 workloads against a BlockManager at serving scale (9000 blocks
of 16 tokens, about vLLM's KV capacity on the 40 GB A100) without a model:
each request is admitted (can_allocate + allocate), its prefill committed in
one step, then decoded for a few tokens with a commit per token, and freed.

  shared  one 2048-token prefix + 128 random tokens per request (W2a)
  many    48 prefixes of 4096 tokens + 256, shuffled; more than the cache holds,
          so the cache evicts (W2b)

Reports microseconds per admission, per prefill commit and per decode-step
commit, and the token hit rate, for Phase 1's hash chain, the Python radix
tree and the C++ one (when nanobind is installed).

  PYTHONPATH=. python benchmarks/phase2/prefix_bench.py --out results/prefix_bench
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hash_block_manager import BlockManager as HashBlockManager  # noqa: E402
from src.serving.nano_vllm.block_manager import BlockManager  # noqa: E402
from src.serving.nano_vllm.sampling import SamplingOptions  # noqa: E402
from src.serving.nano_vllm.sequence import Sequence  # noqa: E402

BLOCK_SIZE = 16
DECODE_TOKENS = 32
IMPLS = {HashBlockManager: "hash (Phase 1)", BlockManager: "radix (Python)"}


def prompts(kind: str, n: int, rng: random.Random) -> list[list[int]]:
    vocab = 128_000
    if kind == "shared":
        prefix = [rng.randrange(vocab) for _ in range(2048)]
        return [prefix + [rng.randrange(vocab) for _ in range(128)] for _ in range(n)]
    stems = [[rng.randrange(vocab) for _ in range(4096)] for _ in range(48)]
    out = [stems[i % 48] + [rng.randrange(vocab) for _ in range(256)] for i in range(n)]
    rng.shuffle(out)
    return out


def run(manager_cls, kind: str, n: int, num_blocks: int, seed: int) -> dict:
    rng = random.Random(seed)
    Sequence.block_size = BLOCK_SIZE
    manager = manager_cls(num_blocks=num_blocks, block_size=BLOCK_SIZE, enable_prefix_cache=True)
    commit = getattr(manager, "commit", None) or manager.hash_blocks
    admit_us, prefill_us, decode_us = [], [], []
    queries = hits = 0
    for tokens in prompts(kind, n, rng):
        seq = Sequence(tokens, SamplingOptions(max_tokens=DECODE_TOKENS + 1))
        t0 = time.perf_counter_ns()
        cached = manager.can_allocate(seq)
        manager.allocate(seq, cached)
        admit_us.append((time.perf_counter_ns() - t0) / 1e3)
        queries += seq.num_tokens
        hits += cached * BLOCK_SIZE

        seq.num_scheduled_tokens = seq.num_tokens - seq.num_cached_tokens
        t0 = time.perf_counter_ns()
        commit(seq)
        prefill_us.append((time.perf_counter_ns() - t0) / 1e3)
        seq.num_cached_tokens += seq.num_scheduled_tokens

        for _ in range(DECODE_TOKENS):
            seq.append_token(rng.randrange(128_000))
            manager.may_append(seq)
            seq.num_scheduled_tokens = 1
            t0 = time.perf_counter_ns()
            commit(seq)
            decode_us.append((time.perf_counter_ns() - t0) / 1e3)
            seq.num_cached_tokens += 1
        manager.deallocate(seq)

    def summary(values: list[float]) -> dict:
        ordered = sorted(values)
        return {
            "mean_us": statistics.fmean(values),
            "p50_us": ordered[len(ordered) // 2],
            "p99_us": ordered[int(0.99 * (len(ordered) - 1))],
        }

    return {
        "impl": IMPLS[manager_cls],
        "workload": kind,
        "requests": n,
        "hit_rate": hits / queries,
        "admit": summary(admit_us),
        "prefill_commit": summary(prefill_us),
        "decode_commit": summary(decode_us),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=192)
    parser.add_argument("--num-blocks", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    impls = [HashBlockManager, BlockManager]
    try:
        from src.serving.nano_vllm.radix_cpp import CppBlockManager

        IMPLS[CppBlockManager] = "radix (C++)"
        impls.append(CppBlockManager)
    except ImportError as e:  # nanobind missing
        print(f"skipping the C++ radix tree: {e}")
    rows = [
        run(cls, kind, args.requests, args.num_blocks, args.seed)
        for kind in ("shared", "many")
        for cls in impls
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "prefix_bench.json").write_text(json.dumps(rows, indent=2))
    lines = [
        f"# Prefix cache host cost ({args.num_blocks} blocks of {BLOCK_SIZE}, {args.requests} requests)",
        "",
        "| workload | impl | hit rate | admit p50 µs | admit p99 µs | prefill commit p50 µs "
        "| decode commit mean µs |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['workload']} | {r['impl']} | {100 * r['hit_rate']:.1f}% | {r['admit']['p50_us']:.0f} "
            f"| {r['admit']['p99_us']:.0f} | {r['prefill_commit']['p50_us']:.0f} "
            f"| {r['decode_commit']['mean_us']:.1f} |"
        )
    (args.out / "prefix_bench.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
