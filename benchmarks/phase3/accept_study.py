#!/usr/bin/env python3
"""How many tokens a DFlash draft gets accepted, before any engine work
(Phase 3, M0; run_phase3.sh's accept step). Plain PyTorch and transformers,
nothing of nano_vllm's serving path.

Lossless speculative decoding emits exactly the target's own tokens, so a
target's output stream does not depend on the drafter. The study uses that:

1. For each prompt, the reference DFlash loop (the checkpoint's own drafter
   code, greedy drafts, a block verified per target forward) generates the
   target's stream and keeps the target's hidden-state features of every
   position. Greedy streams take the argmax; sampled streams (temperature
   0.7, top-p 0.9, nano's keep rule) draw with Gumbel noise keyed by
   (seed, stream, position), the way nano's sampler keys its Philox noise.
2. Each arm then replays the drafter alone along that fixed stream: at an
   anchor it drafts a block from the stored features, counts the leading
   tokens that equal the stream, and moves the anchor past them and the
   bonus token. No target forward is needed, so an arm costs only drafter
   passes. The arm (bf16 drafter, greedy drafts, block 10) must reproduce the
   live loop's accept lengths; the summary says how often it does.

Arms: drafter weights bf16 or rounded to 4 bits (the engine's RTN W4A16);
block 10, 6 or 4; and for sampled streams how the drafter draws: greedy (the
reference), coupled (the target's noise for that position) or independent
(its own noise). Targets: the BF16 model, and the same model with every
projection replaced by the AWQ checkpoint's dequantized weights.

  python accept_study.py --target <dir> --drafter <dir> --workloads <dir> \
      --out <dir> [--awq <AWQ checkpoint dir>] [--num-prompts 16]
  python accept_study.py --self-test    # bookkeeping check on stub models, CPU
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

TEMPERATURE = 0.7
TOP_P = 0.9
INDEPENDENT_STREAM = 1 << 20  # stream id offset of a drafter drawing with its own noise
DATASETS = ("alpaca", "humaneval", "gsm8k", "long")
# (drafter weights, drafter draw, block size)
GREEDY_ARMS = (("bf16", "greedy", 10), ("bf16", "greedy", 6), ("bf16", "greedy", 4), ("rtn4", "greedy", 10))
SAMPLED_ARMS = (
    ("bf16", "greedy", 10),
    ("bf16", "coupled", 10),
    ("bf16", "independent", 10),
    ("bf16", "coupled", 6),
    ("bf16", "coupled", 4),
    ("rtn4", "coupled", 10),
)
REFERENCE_ARM = ("bf16", "greedy", 10)  # what the live loop runs


# --- draws ---------------------------------------------------------------------


def keyed_noise(seed: int, stream: int, position: int, vocab: int, device) -> torch.Tensor:
    """Gumbel noise of one position of one stream, the same on every call."""
    generator = torch.Generator(device=device)
    generator.manual_seed((seed * 1_000_003 + stream) * 1_000_003 + position)
    uniform = torch.rand(vocab, generator=generator, device=device, dtype=torch.float32)
    return -torch.log(-torch.log(uniform.clamp(1e-10, 1 - 1e-7)))


def keep(logits: torch.Tensor) -> torch.Tensor:
    """nano's top-p rule at TEMPERATURE / TOP_P: [rows, vocab] bool."""
    from src.serving.nano_vllm.sampling import keep_mask

    rows = logits.shape[0]
    return keep_mask(
        logits,
        logits.new_full((rows,), TEMPERATURE),
        logits.new_full((rows,), TOP_P),
        torch.zeros(rows, dtype=torch.long, device=logits.device),
    )


def draw(logits: torch.Tensor, positions, sampled: bool, seed: int, stream: int) -> torch.Tensor:
    """One token per row of [rows, vocab]; row i is the token at positions[i]."""
    logits = logits.float()
    if not sampled:
        return logits.argmax(dim=-1)
    scores = (logits / TEMPERATURE).masked_fill(~keep(logits), float("-inf"))
    noise = torch.stack(
        [keyed_noise(seed, stream, int(p), logits.shape[-1], logits.device) for p in positions]
    )
    return (scores + noise).argmax(dim=-1)


# --- the models ------------------------------------------------------------------


class HFTarget:
    """A transformers causal LM, one sequence, with its KV cache."""

    def __init__(self, model, layer_ids: list[int]):
        self.model = model
        self.layer_ids = layer_ids
        self.cache = None

    def start(self) -> None:
        from transformers import DynamicCache

        self.cache = DynamicCache()

    def forward(self, ids: list[int], start: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokens at positions start.. on top of the cache -> (logits [n, vocab],
        features [n, layers * hidden]); logits row i predicts position start + i + 1."""
        device = self.model.device
        out = self.model(
            input_ids=torch.tensor([ids], device=device),
            position_ids=torch.arange(start, start + len(ids), device=device)[None],
            past_key_values=self.cache,
            use_cache=True,
            output_hidden_states=True,
        )
        self.cache = out.past_key_values
        # hidden_states[0] is the embedding: layer L's output is index L + 1.
        features = torch.cat([out.hidden_states[layer + 1] for layer in self.layer_ids], dim=-1)
        return out.logits[0], features[0]

    def crop(self, length: int) -> None:
        self.cache.crop(length)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(ids)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.get_output_embeddings()(hidden)


class HFDrafter:
    """The checkpoint's DFlashDraftModel, driven as its spec_generate does."""

    def __init__(self, model, target: HFTarget):
        self.model = model
        self.target = target
        self.mask_id = getattr(model, "mask_token_id", None)
        if self.mask_id is None:
            self.mask_id = model.config.dflash_config["mask_token_id"]
        self.cache = None
        self.cached = 0  # context positions whose keys and values are cached

    def start(self) -> None:
        from transformers import DynamicCache

        self.cache = DynamicCache()
        self.cached = 0

    def draft(self, features: torch.Tensor, anchor: int, start: int, block: int) -> torch.Tensor:
        """Draft logits [block - 1, vocab] for positions start + 1 .. start + block - 1.

        ``features`` are the target features of positions cached .. start - 1
        (the context not given yet); the block is the anchor token at
        ``start`` followed by mask tokens.
        """
        if self.cached + features.shape[0] != start:
            raise ValueError(f"features for {self.cached}..{start - 1} expected, got {features.shape[0]}")
        device = features.device
        ids = torch.full((1, block), self.mask_id, dtype=torch.long, device=device)
        ids[0, 0] = anchor
        hidden = self.model(
            position_ids=torch.arange(self.cached, start + block, device=device)[None],
            noise_embedding=self.target.embed(ids),
            target_hidden=features[None],
            past_key_values=self.cache,
            use_cache=True,
        )
        hidden = getattr(hidden, "last_hidden_state", hidden)
        self.cache.crop(start)  # drop the block's own keys and values
        self.cached = start
        return self.target.head(hidden[0, 1:, :])


def load_awq_weights(model, awq_dir: Path) -> int:
    """Replace the model's weights by the AWQ checkpoint's: every quantized
    projection becomes its dequantized (q - z) * s, the rest is copied."""
    from src.serving.nano_vllm.kernels.w4a16 import awq_unpack
    from src.serving.nano_vllm.model import _load_state_dict

    state = _load_state_dict(awq_dir)
    params = dict(model.named_parameters())
    device = model.device
    replaced = 0
    for name, tensor in state.items():
        base, _, suffix = name.rpartition(".")
        if suffix == "qweight":
            codes, zeros, scales = awq_unpack(
                tensor.to(device), state[f"{base}.qzeros"].to(device), state[f"{base}.scales"].to(device)
            )
            n, k = codes.shape
            groups = scales.shape[1]
            weight = (
                codes.float().view(n, groups, k // groups) - zeros.float().unsqueeze(-1)
            ) * scales.float().unsqueeze(-1)
            params[f"{base}.weight"].data.copy_(weight.view(n, k))
            replaced += 1
        elif suffix not in ("qzeros", "scales") and name in params:
            params[name].data.copy_(tensor.to(device))
    return replaced


def rtn4_(model) -> int:
    """Round every Linear weight to 4 bits in place, as quantize_w4a16 does:
    symmetric, groups of 128 along the input, scale = max|w| / 7.5."""
    count = 0
    for module in model.modules():
        if not isinstance(module, torch.nn.Linear) or module.in_features % 128:
            continue
        weight = module.weight.data
        n, k = weight.shape
        groups = weight.float().view(n, k // 128, 128)
        scales = groups.abs().amax(dim=-1, keepdim=True) / 7.5
        scales = torch.where(scales == 0, torch.ones_like(scales), scales).to(weight.dtype).float()
        codes = torch.round(groups / scales).clamp_(-8, 7)
        weight.copy_((codes * scales).view(n, k))
        count += 1
    return count


# --- the loops -------------------------------------------------------------------


def generate(target, drafter, prompt_ids, max_new, block, sampled, seed, stream, eos_ids):
    """The reference loop -> (tokens, features, accept length per cycle).

    ``tokens`` is the prompt plus the generated stream; ``features`` covers
    at least every position but the last.
    """
    prompt_len = len(prompt_ids)
    target.start()
    drafter.start()
    logits, pending = target.forward(prompt_ids, 0)
    tokens = list(prompt_ids)
    tokens.append(int(draw(logits[-1:], [prompt_len], sampled, seed, stream)))
    features = [pending]
    accepts: list[int] = []
    start = prompt_len
    while len(tokens) - prompt_len < max_new and tokens[-1] not in eos_ids:
        draft_logits = drafter.draft(pending, tokens[start], start, block)
        draft = draw(draft_logits, range(start + 1, start + block), False, seed, stream)
        logits, block_features = target.forward([tokens[start], *draft.tolist()], start)
        posterior = draw(logits, range(start + 1, start + block + 1), sampled, seed, stream)
        accepted = int((draft == posterior[:-1]).long().cumprod(dim=0).sum())
        accepts.append(accepted)
        # The accepted drafts equal posterior[:accepted]; posterior[accepted] is the bonus.
        for token in posterior[: accepted + 1].tolist():
            tokens.append(token)
            if token in eos_ids or len(tokens) - prompt_len >= max_new:
                break
        pending = block_features[: accepted + 1]  # positions start .. start + accepted
        features.append(pending)
        start += accepted + 1
        target.crop(start)
    return tokens, torch.cat(features), accepts


@dataclass
class Cycles:
    """Per-cycle records of one arm; depth j (1-based) is column j - 1."""

    block: int
    accepted: list[int] = field(default_factory=list)
    depths: list[int] = field(default_factory=list)  # verifiable drafts (fewer at a stream's end)
    emitted: list[int] = field(default_factory=list)  # tokens the cycle advances by
    match: list[list[int]] = field(default_factory=list)  # draft == stream; -1 past ``depths``
    q_max: list[list[float]] = field(default_factory=list)  # the drafter's top probability
    second: list[list[int]] = field(default_factory=list)  # its second choice == stream

    def extend(self, other: "Cycles") -> None:
        for name in ("accepted", "depths", "emitted", "match", "q_max", "second"):
            getattr(self, name).extend(getattr(other, name))


def simulate(drafter, tokens, features, prompt_len, block, draft_mode, sampled, seed, stream) -> Cycles:
    """Replay the drafter along a fixed stream (module docstring, step 2)."""
    cycles = Cycles(block)
    drafter.start()
    last = len(tokens) - 1
    start = prompt_len
    draft_sampled = sampled and draft_mode != "greedy"
    draft_stream = stream + INDEPENDENT_STREAM if draft_mode == "independent" else stream
    while start < last:
        logits = drafter.draft(features[drafter.cached : start], tokens[start], start, block)
        draft = draw(logits, range(start + 1, start + block), draft_sampled, seed, draft_stream)
        depths = min(block - 1, last - start)
        truth = torch.tensor(tokens[start + 1 : start + 1 + depths], device=draft.device)
        match = draft[:depths] == truth
        accepted = int(match.long().cumprod(dim=0).sum())
        top = torch.softmax(logits.float(), dim=-1).topk(2, dim=-1)
        pad = [-1] * (block - 1 - depths)
        cycles.accepted.append(accepted)
        cycles.depths.append(depths)
        cycles.match.append(match.int().tolist() + pad)
        cycles.q_max.append(top.values[:, 0].tolist())
        cycles.second.append((top.indices[:depths, 1] == truth).int().tolist() + pad)
        emitted = min(accepted + 1, last - start)  # the drafts, and the bonus if the stream goes on
        cycles.emitted.append(emitted)
        start += emitted
    return cycles


def summarize(cycles: Cycles) -> dict:
    """tau (tokens per cycle) and per-depth rates of one arm."""
    count = len(cycles.accepted)
    if count == 0:
        return {"cycles": 0}
    accepted = torch.tensor(cycles.accepted)
    depths = torch.tensor(cycles.depths)
    match = torch.tensor(cycles.match)
    q_max = torch.tensor(cycles.q_max)
    second = torch.tensor(cycles.second)
    full = depths == cycles.block - 1
    marginal, survival, confidence = [], [], []
    for depth in range(1, cycles.block):
        rows = depths >= depth
        total = max(int(rows.sum()), 1)
        marginal.append(round(float((match[rows, depth - 1] == 1).sum()) / total, 4))
        survival.append(round(float((accepted[rows] >= depth).sum()) / total, 4))
        confidence.append(round(float(q_max[rows, depth - 1].mean()) if rows.any() else 0.0, 4))
    missed = accepted < depths  # a draft was rejected: was the second choice right?
    rows = torch.nonzero(missed).flatten()
    hits = second[rows, accepted[rows]] if len(rows) else torch.zeros(0)
    return {
        "cycles": count,
        "tokens": int(sum(cycles.emitted)),
        "tau": round(sum(cycles.emitted) / count, 3),
        # Cycles with a whole block to verify: accepted drafts + the bonus token.
        "tau_full": round(float(accepted[full].float().mean()) + 1, 3) if full.any() else None,
        "accept_hist": torch.bincount(accepted[full], minlength=cycles.block).tolist() if full.any() else [],
        "marginal": marginal,
        "survival": survival,
        "q_max_mean": confidence,
        "second_at_first_miss": round(float(hits.float().mean()), 4) if len(rows) else None,
    }


def save_raw(path: Path, cycles: Cycles) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        accepted=np.asarray(cycles.accepted, dtype=np.int8),
        depths=np.asarray(cycles.depths, dtype=np.int8),
        emitted=np.asarray(cycles.emitted, dtype=np.int8),
        match=np.asarray(cycles.match, dtype=np.int8),
        q_max=np.asarray(cycles.q_max, dtype=np.float16),
        second=np.asarray(cycles.second, dtype=np.int8),
    )


def arm_name(arm: tuple[str, str, int]) -> str:
    return f"{arm[0]}-{arm[1]}-b{arm[2]}"


def study(target, drafters: dict, prompts: dict, args, target_name: str, emit) -> None:
    """Every (mode, dataset, arm) of one target; ``emit`` takes each summary."""
    for sampled in (False, True):
        mode = "sampled" if sampled else "greedy"
        arms = SAMPLED_ARMS if sampled else GREEDY_ARMS
        for dataset_index, (dataset, prompt_list) in enumerate(prompts.items()):
            totals = {arm: Cycles(arm[2]) for arm in arms}
            same = 0
            began = time.perf_counter()
            for index, prompt_ids in enumerate(prompt_list):
                stream = dataset_index * 10_000 + index
                tokens, features, live = generate(
                    target, drafters["bf16"], prompt_ids, args.max_new_tokens, 10, sampled,
                    args.seed, stream, args.eos_ids,
                )
                for arm in arms:
                    cycles = simulate(
                        drafters[arm[0]], tokens, features, len(prompt_ids), arm[2], arm[1],
                        sampled, args.seed, stream,
                    )
                    totals[arm].extend(cycles)
                    if arm == REFERENCE_ARM:
                        # The live loop's last cycle may run past the stream's end.
                        same += cycles.accepted[:-1] == live[: len(cycles.accepted) - 1]
            for arm in arms:
                summary = {
                    "target": target_name,
                    "mode": mode,
                    "dataset": dataset,
                    "drafter": arm[0],
                    "draw": arm[1],
                    "block": arm[2],
                    "prompts": len(prompt_list),
                    **summarize(totals[arm]),
                }
                if arm == REFERENCE_ARM:
                    summary["replay_matches_live"] = f"{same}/{len(prompt_list)}"
                emit(summary)
                if args.out is not None:
                    name = f"{target_name}_{mode}_{dataset}_{arm_name(arm)}.npz"
                    save_raw(args.out / "raw" / name, totals[arm])
            print(
                f"{target_name} {mode} {dataset}: {len(prompt_list)} prompts in "
                f"{time.perf_counter() - began:.0f} s",
                flush=True,
            )


# --- self-test on stub models ------------------------------------------------------


class _StubTarget:
    """A 'model' whose token at position p is fixed, so every draw is known."""

    vocab = 64

    def truth(self, position: int) -> int:
        return (7 * position + 3) % self.vocab

    def start(self) -> None:
        self.length = 0

    def forward(self, ids, start):
        logits = torch.zeros(len(ids), self.vocab)
        for row in range(len(ids)):
            logits[row, self.truth(start + row + 1)] = 30.0
        self.length = start + len(ids)
        return logits, torch.arange(start, start + len(ids), dtype=torch.float32)[:, None]

    def crop(self, length: int) -> None:
        self.length = length


class _StubDrafter:
    """Right for the first ``good`` depths of a block, wrong after."""

    def __init__(self, target: _StubTarget, good: int):
        self.target, self.good, self.cached = target, good, 0

    def start(self) -> None:
        self.cached = 0

    def draft(self, features, anchor, start, block):
        assert self.cached + features.shape[0] == start, "context features out of step"
        assert features[:, 0].tolist() == list(range(self.cached, start)), "wrong context positions"
        assert anchor == self.target.truth(start) or start == 0
        self.cached = start
        logits = torch.zeros(block - 1, self.target.vocab)
        for depth in range(1, block):
            token = self.target.truth(start + depth)
            logits[depth - 1, token if depth <= self.good else (token + 1) % self.target.vocab] = 30.0
        return logits


def self_test() -> None:
    target = _StubTarget()
    prompt = [target.truth(p) for p in range(5)]
    for sampled in (False, True):
        for good, block in ((9, 10), (3, 10), (0, 10), (2, 4)):
            drafter = _StubDrafter(target, good)
            tokens, features, live = generate(target, drafter, prompt, 60, block, sampled, 0, 1, ())
            assert tokens == [target.truth(p) for p in range(len(tokens))], "stream is not the target's"
            assert len(tokens) == len(prompt) + 60
            for mode in ("greedy", "coupled", "independent"):
                cycles = simulate(drafter, tokens, features, len(prompt), block, mode, sampled, 0, 1)
                expect = min(good, block - 1)
                assert cycles.accepted[:-1] == live[: len(cycles.accepted) - 1], (good, block, mode)
                assert all(a == expect for a in cycles.accepted[:-1]), (good, block, mode, cycles.accepted)
                summary = summarize(cycles)
                assert summary["tau_full"] == expect + 1, summary
                assert sum(cycles.emitted) == 59, "the replay must cover every generated token but the first"
                assert summary["survival"][:expect] == [1.0] * expect
                if expect < block - 1:
                    assert summary["survival"][expect] == 0.0
    # The same key gives the same noise; another position or stream does not.
    a = keyed_noise(0, 3, 17, 64, "cpu")
    assert torch.equal(a, keyed_noise(0, 3, 17, 64, "cpu"))
    assert not torch.equal(a, keyed_noise(0, 3, 18, 64, "cpu"))
    assert not torch.equal(a, keyed_noise(0, 3 + INDEPENDENT_STREAM, 17, 64, "cpu"))
    # Coupled draws of equal logits agree; independent ones mostly do not.
    logits = torch.randn(8, 64, generator=torch.Generator().manual_seed(0))
    positions = list(range(8))
    coupled = draw(logits, positions, True, 0, 5)
    assert torch.equal(coupled, draw(logits.clone(), positions, True, 0, 5))
    assert not torch.equal(coupled, draw(logits, positions, True, 0, 5 + INDEPENDENT_STREAM))
    print("accept_study self-test passed")


# --- main --------------------------------------------------------------------------


def load_prompts(workloads: Path, tokenizer, datasets, count: int) -> dict[str, list[list[int]]]:
    prompts = {}
    for dataset in datasets:
        lines = (workloads / f"{dataset}_c1.jsonl").read_text().splitlines()
        wanted = max(count // 2, 1) if dataset == "long" else count
        prompts[dataset] = [
            tokenizer(json.loads(line)["prompt"], add_special_tokens=True).input_ids
            for line in lines[:wanted]
        ]
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--target", help="BF16 target model dir")
    parser.add_argument("--awq", help="AWQ checkpoint dir of the same model (optional second target)")
    parser.add_argument("--drafter", help="DFlash drafter dir (its dflash.py runs: trust_remote_code)")
    parser.add_argument("--workloads", type=Path, help="make_workloads.py's output dir")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--num-prompts", type=int, default=16, help="per dataset (half for long)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not (args.target and args.drafter and args.workloads and args.out):
        parser.error("--target, --drafter, --workloads and --out are required")

    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    args.eos_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")}
    prompts = load_prompts(args.workloads, tokenizer, args.datasets.split(","), args.num_prompts)
    model = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16).cuda().eval()
    draft_model = (
        AutoModel.from_pretrained(args.drafter, trust_remote_code=True, torch_dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    rounded = copy.deepcopy(draft_model)
    print(f"drafter: {rtn4_(rounded)} weights rounded to 4 bits for the rtn4 arms", flush=True)
    target = HFTarget(model, list(draft_model.target_layer_ids))
    drafters = {"bf16": HFDrafter(draft_model, target), "rtn4": HFDrafter(rounded, target)}

    args.out.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []

    def emit(summary: dict) -> None:
        summaries.append(summary)
        print("P3_ACCEPT " + json.dumps(summary), flush=True)
        (args.out / "accept.json").write_text(json.dumps(summaries, indent=1))

    with torch.inference_mode():
        study(target, drafters, prompts, args, "bf16", emit)
    if args.awq:
        with torch.no_grad():
            replaced = load_awq_weights(model, Path(args.awq))
        print(f"target: {replaced} projections replaced by the AWQ checkpoint's", flush=True)
        with torch.inference_mode():
            study(target, drafters, prompts, args, "awq", emit)


if __name__ == "__main__":
    sys.exit(main())
