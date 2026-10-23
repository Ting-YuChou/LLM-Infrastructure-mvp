#!/usr/bin/env python3
"""Decode timing of this repo's nano_vllm engine on the same workloads (Phase 0).

Two paths, both as they are today:
  static  NanoLlamaForCausalLM.forward on a StaticKVCache, batch 1, one
          argmax + .item() per token (what NanoLlamaEngine.generate does);
  paged   the continuous-batching engine core: scheduler -> ModelRunner
          (forward over the batch, batched sampling) -> postprocess, driven
          step by step (EngineCore.step). --attention-backend and --enforce-eager pick the Phase 1
          paths (Triton attention, decode CUDA graphs).

EOS is disabled so every sequence decodes exactly --output-len tokens. The
decode steps of each measured run (after the prefill) sit between
cudaProfilerStart/Stop with one NVTX range per step, so nsys
--capture-range=cudaProfilerApi or ncu --profile-from-start off records just
those steps.

  python nano_bench.py --model /models/llama --runs static:1:128:256,paged:16:1024:256 --out nano.json
Needs this repo on PYTHONPATH (src.serving.nano_vllm).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import time

import torch

from src.serving.nano_vllm import NanoLlamaEngine, NanoVLLMConfig, SamplingOptions, Sequence
from src.serving.nano_vllm.core import EngineRequest
from src.serving.nano_vllm import nvtx as engine_nvtx


CUDA = torch.cuda.is_available()


def nvtx(name: str):
    return torch.cuda.nvtx.range(name) if CUDA else contextlib.nullcontext()


def sync() -> None:
    if CUDA:
        torch.cuda.synchronize()


def profiler(on: bool) -> None:
    if CUDA:
        (torch.cuda.profiler.start if on else torch.cuda.profiler.stop)()


def random_prompt(rng: random.Random, length: int, vocab: int) -> list[int]:
    return [rng.randrange(min(1000, vocab // 2), min(100000, vocab)) for _ in range(length)]


def run_static(engine: NanoLlamaEngine, prompt: list[int], output_len: int, profile: bool) -> dict:
    model = engine.model
    cache = model.new_cache(batch_size=1, max_seq_len=len(prompt) + output_len)
    ids = torch.tensor([prompt], dtype=torch.long, device=model.device)
    sync()
    start = time.perf_counter()
    logits = model.forward(ids, cache=cache, start_pos=0)
    token = int(torch.argmax(logits[0, -1]).item())
    prefill_s = time.perf_counter() - start
    step_s = []
    pos = len(prompt)
    if profile:
        profiler(True)
    for step in range(output_len - 1):
        t0 = time.perf_counter()
        with nvtx(f"decode_step_{step}"):
            logits = model.forward(
                torch.tensor([[token]], dtype=torch.long, device=model.device),
                cache=cache,
                start_pos=pos,
            )
            token = int(torch.argmax(logits[0, -1]).item())
        step_s.append(time.perf_counter() - t0)
        pos += 1
    if profile:
        profiler(False)
    return {"prefill_s": prefill_s, "decode_step_s": step_s}


def _drive(core, num_requests: int, records: list, profile: bool) -> dict:
    """Step the engine core until every request finishes; tokens per request."""
    produced: dict[int, int] = {}
    step = 0
    profiling = False
    while core.has_work():
        if profile and not profiling and len(produced) == num_requests:
            profiler(True)  # every sequence is past its prefill
            profiling = True
        before = len(records)
        t0 = time.perf_counter()
        # With NANO_NVTX=1 the core marks its own steps and phases.
        with contextlib.nullcontext() if engine_nvtx.ENABLED else nvtx(f"engine_step_{step}"):
            outputs = core.step()
        if len(records) > before:  # a step that scheduled nothing ran no model
            records[-1]["step_s"] = time.perf_counter() - t0
        for output in outputs:
            if output.error is not None:
                raise output.error
            if output.token_id is not None:
                produced[output.req_id] = produced.get(output.req_id, 0) + 1
        step += 1
    if profiling:
        profiler(False)
    return produced


def run_paged(engine: NanoLlamaEngine, prompts: list[list[int]], output_len: int, profile: bool) -> dict:
    core = engine.core
    runner = core.model_runner
    records: list[dict] = []
    launch, collect = runner.launch, runner.collect
    by_handle: dict[int, dict] = {}

    # model_s: launching a step plus waiting for it (with async scheduling the
    # wait overlaps the next step's launch, so it is no longer a model time).
    def timed_launch(seqs, is_prefill, gather_rows=None):
        t0 = time.perf_counter()
        handle = launch(seqs, is_prefill, gather_rows)
        record = {"prefill": is_prefill, "seqs": len(seqs), "model_s": time.perf_counter() - t0}
        records.append(record)
        by_handle[id(handle)] = record
        return handle

    def timed_collect(handle):
        t0 = time.perf_counter()
        token_ids = collect(handle)
        by_handle.pop(id(handle))["model_s"] += time.perf_counter() - t0
        return token_ids

    runner.launch, runner.collect = timed_launch, timed_collect
    sampling = SamplingOptions(max_tokens=output_len, temperature=0.0, ignore_eos=True)
    for prompt in prompts:
        core.add_request(
            EngineRequest(req_id=next(Sequence._counter), prompt_token_ids=prompt, sampling=sampling)
        )
    try:
        produced = _drive(core, len(prompts), records, profile)
    finally:
        runner.launch, runner.collect = launch, collect
    counts = set(produced.values())
    assert len(produced) == len(prompts) and counts == {output_len}, (
        f"expected {output_len} tokens per sequence, got {counts}"
    )
    return {"steps": records}


def summarize(kind: str, batch: int, input_len: int, output_len: int, raw: dict) -> dict:
    if kind == "static":
        steps = raw["decode_step_s"]
        prefill_s = raw["prefill_s"]
    else:
        full = [r for r in raw["steps"] if not r["prefill"] and r["seqs"] == batch]
        steps = [r["step_s"] for r in full]
        prefill_s = sum(r["step_s"] for r in raw["steps"] if r["prefill"])
    steps = sorted(steps)
    mean = sum(steps) / len(steps)
    result = {
        "engine": "nano_vllm",
        "path": kind,
        "batch": batch,
        "input_len": input_len,
        "output_len": output_len,
        "prefill_s": prefill_s,
        "decode_steps_measured": len(steps),
        "decode_ms_per_step": mean * 1e3,
        "decode_ms_p50": steps[len(steps) // 2] * 1e3,
        "decode_tok_per_s": batch / mean,
    }
    if kind == "paged":
        decode = [r for r in raw["steps"] if not r["prefill"] and r["seqs"] == batch]
        model_mean = sum(r["model_s"] for r in decode) / len(decode)
        # Step time outside run_batch: scheduling, postprocess.
        result["decode_model_ms_per_step"] = model_mean * 1e3
        result["decode_engine_overhead_ms"] = (mean - model_mean) * 1e3
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--runs",
        default="static:1:128:256,paged:1:128:256,paged:1:1024:256,paged:4:1024:256,paged:16:1024:256",
        help="comma-separated kind:batch:input_len:output_len",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda", help="cpu only for testing the harness")
    parser.add_argument("--attention-backend", default="torch", choices=["torch", "triton"])
    parser.add_argument("--linear-backend", default="torch", choices=["torch", "triton", "cuda"])
    parser.add_argument("--weight-quant", default="none", choices=["none", "w4a16"])
    parser.add_argument(
        "--enforce-eager", action="store_true", help="no decode CUDA graphs (paged path)"
    )
    parser.add_argument(
        "--async-scheduling", action="store_true",
        help="launch step N+1 before collecting step N (paged path)",
    )
    args = parser.parse_args()

    runs = [tuple(r.split(":")) for r in args.runs.split(",")]
    max_batch = max(int(b) for _, b, _, _ in runs)
    max_len = max(int(i) + int(o) for _, _, i, o in runs)
    config = NanoVLLMConfig(
        model_name=args.model,
        dtype="bfloat16" if args.device == "cuda" else "float32",
        device=args.device,
        max_model_len=max_len + 16,
        max_new_tokens=max_len,
        num_kvcache_blocks=max_batch * (max_len // 16 + 2) + 64,
        max_num_seqs=max_batch,
        max_num_batched_tokens=max(4096, max_len),
        # Phase 0/1 methodology: all prompts prefill, then the batch decodes.
        scheduling="prefill_first",
        async_scheduling=args.async_scheduling,
        attention_backend=args.attention_backend,
        linear_backend=args.linear_backend,
        weight_quant=args.weight_quant,
        enforce_eager=args.enforce_eager,
    )
    engine = NanoLlamaEngine(config)
    t0 = time.perf_counter()
    engine.initialize()
    load_s = time.perf_counter() - t0
    rng = random.Random(0)
    vocab = engine.model.config.vocab_size

    # Warm-up: Triton JIT and cuBLAS heuristics, outside any measured run.
    run_static(engine, random_prompt(rng, 32, vocab), 8, profile=False)
    run_paged(engine, [random_prompt(rng, 32, vocab)], 8, profile=False)

    results = []
    for kind, batch, input_len, output_len in runs:
        batch, input_len, output_len = int(batch), int(input_len), int(output_len)
        prompts = [random_prompt(rng, input_len, vocab) for _ in range(batch)]
        if kind == "static":
            assert batch == 1, "the static path is single-request"
            raw = run_static(engine, prompts[0], output_len, profile=True)
        else:
            raw = run_paged(engine, prompts, output_len, profile=True)
        result = summarize(kind, batch, input_len, output_len, raw)
        result["load_s"] = load_s
        result["attention_backend"] = args.attention_backend
        result["linear_backend"] = args.linear_backend
        result["weight_quant"] = args.weight_quant
        result["async_scheduling"] = args.async_scheduling
        result["cuda_graphs"] = (
            kind == "paged" and not args.enforce_eager and args.device == "cuda"
        )
        results.append(result)
        print("NANO_BENCH " + json.dumps(result), flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
