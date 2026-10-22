#!/usr/bin/env python3
"""Offline vLLM decode steps for nsys (Phase 0).

Runs vLLM in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0), so one nsys
capture sees the scheduler, the model runner and every kernel, and drives
LLMEngine.step() itself to time each engine step. Two passes over the same
batch of requests:
  1. timing: every step timed, no profiler range;
  2. profile: once every request has its first token (prefill done), the next
     --profile-steps decode steps run between cudaProfilerStart/Stop, each in an
     NVTX range engine_step_<n>, inside an outer NVTX range "measured".
Run under  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop
so model loading, torch.compile and CUDA graph capture stay out of the report.

  python vllm_offline.py --model /models/llama --batch 1 --input-len 128 --output-len 256 --out r.json
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.sampling_params import RequestOutputKind  # noqa: E402

_request_ids = itertools.count()


def sampling(max_tokens: int) -> SamplingParams:
    # A fresh object every time: LLM.generate() switches the params it gets to
    # FINAL_ONLY in place, and then engine.step() reports tokens only at the end.
    return SamplingParams(
        max_tokens=max_tokens, ignore_eos=True, temperature=0.0, output_kind=RequestOutputKind.DELTA
    )


def run_steps(llm: LLM, prompts: list, max_tokens: int, profile_steps: int = 0) -> list[dict]:
    engine = llm.llm_engine
    params = sampling(max_tokens)
    ids = []
    for prompt in prompts:
        request_id = f"req-{next(_request_ids)}"
        engine.add_request(request_id, prompt, params)
        ids.append(request_id)
    produced = dict.fromkeys(ids, 0)
    records, step = [], 0
    profile_from, profiling = None, False
    while engine.has_unfinished_requests():
        prefill_done = all(n > 0 for n in produced.values())
        if profile_steps and prefill_done and profile_from is None:
            profile_from = step + 4  # a few steps past the last prefill
        if profile_from is not None and step == profile_from:
            torch.cuda.profiler.start()
            torch.cuda.nvtx.range_push("measured")
            profiling = True
        if profiling:
            torch.cuda.nvtx.range_push(f"engine_step_{step}")
        start = time.perf_counter()
        outputs = engine.step()
        elapsed = time.perf_counter() - start
        if profiling:
            torch.cuda.nvtx.range_pop()
            if step == profile_from + profile_steps - 1:
                torch.cuda.nvtx.range_pop()
                torch.cuda.profiler.stop()
                profiling = False
        for out in outputs:
            rid = out.request_id
            if rid not in produced:  # some versions add a suffix to the id internally
                rid = next(k for k in produced if rid.startswith(k))
            produced[rid] += len(out.outputs[0].token_ids)  # DELTA: new tokens only
        records.append({"step": step, "s": elapsed, "decode": prefill_done})
        step += 1
    if profiling:  # the requests finished before profile_steps ran out
        torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()
    assert set(produced.values()) == {max_tokens}, f"tokens per request: {set(produced.values())}"
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--eager", action="store_true", help="disable CUDA graphs")
    parser.add_argument("--profile-steps", type=int, default=64)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="lower it under ncu, so its per-pass memory backup fits on the GPU",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        enforce_eager=args.eager,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
    )
    rng = random.Random(0)
    prompts = [
        {"prompt_token_ids": [rng.randrange(1000, 100000) for _ in range(args.input_len)]}
        for _ in range(args.batch)
    ]
    llm.generate(prompts, sampling(args.output_len), use_tqdm=False)  # warm-up
    timing = run_steps(llm, prompts, args.output_len)
    profiled = run_steps(llm, prompts, args.output_len, profile_steps=args.profile_steps)

    # Pure decode steps: after every request has its first token, excluding the
    # final step (requests finishing, outputs flushed).
    decode = [r["s"] for r in timing if r["decode"]][:-1]
    prefill_steps = [r for r in timing if not r["decode"]]
    mean = statistics.fmean(decode)
    result = {
        "engine": "vllm",
        "quantization": llm.llm_engine.model_config.quantization,
        "cuda_graphs": not args.eager,
        "batch": args.batch,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "prefill_steps": len(prefill_steps),
        "prefill_s": sum(r["s"] for r in prefill_steps),
        "decode_steps": len(decode),
        "decode_ms_per_step": mean * 1e3,
        "decode_ms_p50": statistics.median(decode) * 1e3,
        "decode_tok_per_s": args.batch / mean,
        "profiled_decode_ms_per_step": statistics.fmean(
            r["s"] for r in profiled if r["decode"]
        ) * 1e3,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("VLLM_OFFLINE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
