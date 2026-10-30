#!/usr/bin/env python3
"""Teacher-forced quality of vLLM on quality.py's chunks.

Runs after quality.py with the same --out: reads chunks.pt, scores every
position of each chunk through vLLM's prompt logprobs (the raw model
output), and appends a row to quality.json / quality.md with the
perplexity and the top-1 agreement against quality.py's first config.
quality.json also gets the agreement against each of quality.py's configs
(top1/<label>.pt), e.g. nano and vLLM on the same AWQ checkpoint.

  python quality_vllm.py --model <dir> --label vllm_bf16 --out results/quality
  python quality_vllm.py --model <awq dir> --label vllm_awq --out results/quality
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quality import write_tables  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True, help="quality.py's --out")
    # One chunk per step: prompt logprobs materialize fp32 logits for every
    # scheduled token (2,048 x 128,256 x 4 B = 1 GB).
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    out = Path(args.out)
    data = torch.load(out / "chunks.pt").tolist()
    length = len(data[0])
    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        max_model_len=length + 16,
        max_num_batched_tokens=max(args.max_num_batched_tokens, length + 16),
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        seed=0,
    )
    load_s = time.perf_counter() - t0
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1, detokenize=False)
    t0 = time.perf_counter()
    outputs = llm.generate([{"prompt_token_ids": chunk} for chunk in data], params)
    nll, count, top1 = 0.0, 0, []
    for chunk, output in zip(data, outputs):
        # prompt_logprobs[i] scores token i given tokens < i: the target's
        # logprob and the rank-1 token (quality.py's argmax of row i - 1).
        for pos in range(1, length):
            entries = output.prompt_logprobs[pos]
            nll -= entries[chunk[pos]].logprob
            count += 1
            top1.append(next(token for token, lp in entries.items() if lp.rank == 1))
    top1 = torch.tensor(top1, dtype=torch.int32)
    config = llm.llm_engine.model_config
    row = {
        "label": args.label,
        "engine": "vllm",
        "attention_backend": "vllm",
        "linear_backend": "vllm",
        "weight_quant": str(config.quantization or "none"),
        "dtype": str(config.dtype).replace("torch.", ""),
        "kv_cache_dtype": "auto",
        "ppl": math.exp(nll / count),
        "tokens": count,
        "weights_gib": None,
        "load_s": load_s,
        "eval_s": time.perf_counter() - t0,
    }
    results = json.loads((out / "quality.json").read_text())
    agreement = {}
    for other in results:
        path = out / "top1" / f"{other['label']}.pt"
        if path.exists() and other.get("engine", "nano") == "nano":
            agreement[other["label"]] = float((top1 == torch.load(path)).float().mean())
    if results:
        row["top1_agreement_vs"] = results[0]["label"]
        row["top1_agreement"] = agreement.get(results[0]["label"])
    row["top1_agreement_by_config"] = agreement
    torch.save(top1, out / "top1" / f"{args.label}.pt")
    print("QUALITY " + json.dumps(row), flush=True)
    results = [r for r in results if r["label"] != args.label] + [row]
    print(write_tables(out, results, length))


if __name__ == "__main__":
    main()
