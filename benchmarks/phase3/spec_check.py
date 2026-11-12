#!/usr/bin/env python3
"""Is every token nano emits the target's own choice? (Phase 3, M2 on the GPU.)

On the GPU a speculative run and a plain run of the same prompt need not
give the same text: logits depend slightly on how many rows a kernel
computes, so greedy streams split at near-ties (nano's own one-at-a-time and
16-at-a-time passes already differ). What must hold is that each emitted
token is the argmax of the target's logits given the tokens before it, up to
that noise. This checks it directly:

  1. generate with the engine core (plain and speculative, one request at a
     time and all at once), temperature 0;
  2. run every finished sequence through the target in one prefill pass
     (teacher forcing) and, at each generated position, compare the emitted
     token with that pass's argmax. Where they differ, the margin (top logit
     minus the emitted token's logit) says whether it was a near-tie.

A wrong position, a stale KV entry or an accepted wrong draft would show as
mismatches with large margins in the speculative runs only. The lm_head runs
as int8 without the dense copy, so one-row and many-row passes read the same
weights.

It also times the engine's steps: the wall time of each core.step() once
every request of the group has its first token, so a step is a decode step
(plain) or a speculative cycle with no prompt rows next to it. The core
collects a step before it returns, so that is the whole cycle: host work,
draft, verify and sampling.

  PYTHONPATH=<repo> python spec_check.py --model <dir> [--weight-quant awq] \
      --drafter <dir> --workloads <dir> --out <file.json> [--prompts 4] \
      [--drafter-quants none,w4a16]
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DATASETS = ("alpaca", "humaneval", "gsm8k", "long")
BLOCK = 16


def generate(core, requests: list[list[int]], eos, max_tokens: int, together: bool):
    """Token ids per request, wall seconds, the core's stats, and per group
    of requests run together the seconds of each step after the group's
    prompts (see the module docstring)."""
    from src.serving.nano_vllm.core import EngineRequest
    from src.serving.nano_vllm.sampling import SamplingOptions

    outputs: dict[int, list[int]] = {index: [] for index in range(len(requests))}
    offset = generate.next_id
    generate.next_id += len(requests)
    start = time.perf_counter()
    groups = [range(len(requests))] if together else [[index] for index in range(len(requests))]
    step_seconds = []
    for group in groups:
        for index in group:
            core.add_request(EngineRequest(
                offset + index, requests[index],
                SamplingOptions(max_tokens=max_tokens, temperature=0.0), eos_token_ids=eos,
            ))
        no_token_yet = set(group)
        steps = []
        while core.has_work():
            timed = not no_token_yet
            step_start = time.perf_counter()
            step_outputs = core.step()
            if timed:
                steps.append(time.perf_counter() - step_start)
            for output in step_outputs:
                if output.error is not None:
                    raise output.error
                if output.token_id is not None:
                    outputs[output.req_id - offset].append(output.token_id)
                    no_token_yet.discard(output.req_id - offset)
        step_seconds.append(steps)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return [outputs[index] for index in range(len(requests))], seconds, core.stats(), step_seconds


generate.next_id = 0


@torch.inference_mode()
def teacher_margins(model, prompt: list[int], output: list[int]) -> list[float]:
    """Per generated token: 0 if it is the argmax given its prefix in one
    prefill pass, else top logit - its logit."""
    from src.serving.nano_vllm.metadata import build_batch

    tokens = prompt + output
    total = len(tokens) - 1  # the last token's own logits are not needed
    blocks = -(-total // BLOCK)
    cache = model.new_paged_cache(num_blocks=blocks, block_size=BLOCK)
    device = model.device
    input_ids, positions, metadata = build_batch(
        input_ids=tokens[:total], positions=list(range(total)), slots=list(range(total)),
        block_tables=[list(range(blocks))], context_lens=[total], query_lens=[total],
        is_prefill=True, null_block=cache.null_block, num_slots=cache.total_slots, device=device,
    )
    # Logits for the rows that predict the generated tokens only.
    metadata.logits_indices = torch.arange(len(prompt) - 1, total, device=device)
    logits = model.forward_batch(input_ids, positions, cache, metadata).float()
    emitted = torch.tensor(output, device=device)
    margins = logits.max(dim=-1).values - logits.gather(1, emitted.view(-1, 1)).view(-1)
    return margins.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--weight-quant", default="none")
    parser.add_argument("--drafter", required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=4, help="per dataset")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--drafter-quants", default="none",
        help="comma-separated speculative_weight_quant settings, a speculative run for each",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from src.serving.nano_vllm.config import NanoVLLMConfig
    from src.serving.nano_vllm.core import EngineCore, load_model

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    eos = tuple(sorted({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")}))
    datasets, requests = [], []
    for dataset in DATASETS:
        lines = (args.workloads / f"{dataset}_c1.jsonl").read_text().splitlines()[: args.prompts]
        for line in lines:
            datasets.append(dataset)
            requests.append(tokenizer(json.loads(line)["prompt"], add_special_tokens=True).input_ids)

    def config(drafter, drafter_quant="none"):
        return NanoVLLMConfig(
            model_name=args.model, dtype="bfloat16", device="cuda", max_model_len=4096,
            max_new_tokens=args.max_tokens, num_kvcache_blocks=4096, max_num_seqs=16,
            max_num_batched_tokens=2048, weight_quant=args.weight_quant,
            lm_head_quant="int8" if args.weight_quant != "none" else "none",
            async_scheduling=False, speculative_model=drafter,
            speculative_weight_quant=drafter_quant,
        )

    model = load_model(config(None))
    rows = []
    modes = [("plain", None, "none")]
    for quant in args.drafter_quants.split(","):
        modes.append(("spec" if quant == "none" else f"spec, {quant} drafter", args.drafter, quant))
    for mode, drafter, drafter_quant in modes:
        core = EngineCore(config(drafter, drafter_quant), model)
        core.capture_graphs()
        generate(core, requests[:2], eos, 16, together=False)  # first-use compiles
        generate(core, requests[:2], eos, 16, together=True)
        for together in (False, True):
            before = core.stats()
            outputs, seconds, stats, step_seconds = generate(
                core, requests, eos, args.max_tokens, together
            )
            # Per dataset when the requests ran one at a time (a group each).
            by_dataset: dict[str, list[float]] = {}
            if not together:
                for dataset, steps in zip(datasets, step_seconds):
                    by_dataset.setdefault(dataset, []).extend(steps)
            all_steps = [seconds_ for steps in step_seconds for seconds_ in steps]
            margins = [m for prompt, out in zip(requests, outputs) for m in teacher_margins(model, prompt, out)]
            wrong = sorted(m for m in margins if m > 0)
            cycles = stats.get("spec_num_drafts", 0) - before.get("spec_num_drafts", 0)
            accepted = stats.get("spec_num_accepted_tokens", 0) - before.get("spec_num_accepted_tokens", 0)
            row = {
                "label": args.label,
                "mode": mode,
                "batch": len(requests) if together else 1,
                "tokens": len(margins),
                "not_argmax": len(wrong),
                "max_margin": wrong[-1] if wrong else 0.0,
                "margins_over_0.5": sum(m > 0.5 for m in wrong),
                "margins_over_1": sum(m > 1.0 for m in wrong),
                "tokens_per_cycle": (accepted + cycles) / cycles if cycles else None,
                "out_tok_s": len(margins) / seconds,
                "steps": len(all_steps),
                "step_ms_median": 1000 * statistics.median(all_steps) if all_steps else None,
                "step_ms_mean": 1000 * statistics.fmean(all_steps) if all_steps else None,
                "step_ms_median_by_dataset": {
                    dataset: 1000 * statistics.median(steps) for dataset, steps in by_dataset.items() if steps
                },
            }
            rows.append(row)
            print("P3_SPEC_CHECK " + json.dumps(row), flush=True)
        del core
        gc.collect()
        torch.cuda.empty_cache()
    args.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
