#!/usr/bin/env python3
"""Draft parity of nano's DFlashDrafter against the checkpoint's own code (Phase 3, M1).

A drafter ported with a subtle error still gives correct output (the target
verifies every token); it only accepts fewer. So the port is checked directly:
both implementations draft the same blocks and their top-1 tokens are compared.

The target and the drafter are loaded through transformers, the drafter with
trust_remote_code from a pinned revision (its dflash.py; that code imports
`datasets`, so it must be installed). The script then follows the reference
loop, DFlashDraftModel.spec_generate at temperature 0, over real prompts:
prefill, then per cycle one draft block, one target verify pass and the
accepted prefix. In every cycle the same block is also drafted by
DFlashDrafter on a paged KV cache: its context K/V are written from the same
target hidden states (first the prompt's, then each cycle's accepted tokens),
and it shares the transformers target's embedding and lm_head. The loop
continues on the reference's draft, so both always see the same context.

Reports the top-1 agreement over all draft positions (gate: >= 99% over 200
blocks), the share of blocks that agree at every position, the largest logit
difference, and the reference's mean accepted tokens per cycle.

  PYTHONPATH=. python benchmarks/phase3/parity.py --out results/parity.json

Needs a GPU for the real checkpoints (both models in bf16, about 19 GB);
--device cpu --dtype float32 with tiny local checkpoints runs the same code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DEFAULT_TARGET = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DRAFTER = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
# The commit of DEFAULT_DRAFTER whose dflash.py the port was written against.
DEFAULT_DRAFTER_REVISION = "d3af30def9601abdd10810aba220d692f0e803f0"

PROMPTS = [
    "Explain how a hash map handles collisions, with a short example.",
    "Write a Python function that returns the n-th Fibonacci number using memoization.",
    "What are the main causes of the French Revolution? Answer in one paragraph.",
    "A train travels 180 km in 2.5 hours. What is its average speed in km/h? Show the steps.",
    "Write a haiku about debugging at 3 am, then explain the imagery.",
    "Compare TCP and UDP for a real-time multiplayer game.",
    "Implement binary search in C and state its loop invariant.",
    "Summarize the plot of Romeo and Juliet in five sentences.",
    "Give me a three-day itinerary for Kyoto in autumn.",
    "Explain the difference between a process and a thread to a new programmer.",
    "Write a SQL query that returns the top three customers by total order value.",
    "If 3x + 7 = 25, what is x? Then check the answer.",
    "Describe how photosynthesis converts light into chemical energy.",
    "Write a Rust function that reverses the words of a sentence in place.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Draft a polite email asking a colleague to review a pull request by Friday.",
    "Explain gradient descent and why the learning rate matters.",
    "List five ways to reduce the memory use of a Python service.",
    "Translate 'The weather is nice today, shall we go for a walk?' into French and German.",
    "Write a bash one-liner that counts the lines of every .py file under the current directory.",
    "Explain what a KV cache is in transformer inference and why it speeds up decoding.",
    "Tell a short story about a lighthouse keeper who finds a message in a bottle.",
    "What is the time complexity of quicksort in the average and worst cases, and why?",
    "Write a JavaScript debounce function and explain when to use it.",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Top-1 agreement of nano's DFlashDrafter with the checkpoint's remote code."
    )
    parser.add_argument("--target", default=DEFAULT_TARGET, help="target model: Hub id or local directory")
    parser.add_argument("--target-revision", default=None)
    parser.add_argument("--drafter", default=DEFAULT_DRAFTER, help="DFlash drafter: Hub id or local directory")
    parser.add_argument(
        "--drafter-revision", default=DEFAULT_DRAFTER_REVISION,
        help="Hub revision of the drafter's weights and remote code (ignored for a local directory)",
    )
    parser.add_argument("--blocks", type=int, default=200, help="draft blocks to compare")
    parser.add_argument("--max-new-tokens", type=int, default=160, help="generated tokens per prompt")
    parser.add_argument(
        "--prompts-jsonl", default=None,
        help='one JSON object per line with a "prompt" (or "text") string; default: built-in prompts',
    )
    parser.add_argument("--no-chat-template", action="store_true", help="feed the prompts as plain text")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--kv-block-size", type=int, default=16, help="tokens per block of nano's paged cache")
    parser.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8"])
    parser.add_argument("--drafter-weight-quant", default="none", choices=["none", "w4a16"])
    parser.add_argument("--no-triton", action="store_true", help="PyTorch RMSNorm/SiLU in the nano drafter")
    parser.add_argument("--gate", type=float, default=0.99, help="minimum top-1 agreement")
    parser.add_argument("--out", default=None, help="write the summary JSON here")
    args = parser.parse_args(argv)
    if args.blocks < 1 or args.max_new_tokens < 1:
        parser.error("--blocks and --max-new-tokens must be positive")
    return args


def load_prompts(path: str | None) -> list[str]:
    if path is None:
        return list(PROMPTS)
    prompts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            prompts.append(row.get("prompt") or row["text"])
    if not prompts:
        raise ValueError(f"{path} holds no prompts")
    return prompts


def resolve_drafter_dir(drafter: str, revision: str | None) -> Path:
    """A local directory holding config.json, the weights and dflash.py."""
    if Path(drafter).is_dir():
        return Path(drafter)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(drafter, revision=revision))


def encode(tokenizer, prompt: str, chat: bool):
    if chat and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
        )
    return tokenizer(prompt, return_tensors="pt").input_ids


def stop_token_ids(tokenizer) -> set[int]:
    ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")}
    return {i for i in ids if isinstance(i, int) and i >= 0 and i != tokenizer.unk_token_id}


class NanoSide:
    """DFlashDrafter with one sequence on a paged cache."""

    def __init__(self, args, torch, hf_target, drafter_dir: Path, max_len: int) -> None:
        from src.serving.nano_vllm.dflash import DFlashDrafter

        device = torch.device(args.device)
        target_config = hf_target.config
        # What DFlashDrafter reads of a NanoLlamaForCausalLM: its config sizes,
        # device, dtype and backends, the embedding and the lm_head. The
        # transformers target's tensors stand in, so no second copy of the
        # target is loaded.
        target = SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=target_config.hidden_size,
                vocab_size=target_config.vocab_size,
                num_hidden_layers=target_config.num_hidden_layers,
            ),
            device=device,
            dtype=hf_target.lm_head.weight.dtype,
            use_triton=device.type == "cuda" and not args.no_triton,
            attention_backend="torch",
            linear_backend="torch",
            kv_cache_dtype=args.kv_cache_dtype,
            weights={"model.embed_tokens.weight": hf_target.get_input_embeddings().weight.detach()},
            lm_head_weight=hf_target.lm_head.weight.detach(),
        )
        self.torch = torch
        self.device = device
        self.drafter = DFlashDrafter.from_pretrained(
            drafter_dir, target, weight_quant=args.drafter_weight_quant, max_positions=max_len
        )
        self.block_size = args.kv_block_size
        num_blocks = -(-max_len // self.block_size)
        self.cache = self.drafter.new_paged_cache(num_blocks, self.block_size)
        # Blocks in reverse order, so positions and slots differ.
        self.table = list(range(num_blocks - 1, -1, -1))
        self.block_tables = torch.tensor([self.table], dtype=torch.int32, device=device)

    def _slots(self, start: int, end: int):
        slots = [self.table[p // self.block_size] * self.block_size + p % self.block_size for p in range(start, end)]
        return self.torch.tensor(slots, dtype=self.torch.long, device=self.device)

    def write_context(self, taps, start: int) -> None:
        """taps [tokens, layers * hidden] of positions start, start + 1, ..."""
        end = start + taps.shape[0]
        positions = self.torch.arange(start, end, device=self.device)
        self.drafter.write_context(taps, positions, self.cache, self._slots(start, end))

    def draft(self, anchor_id: int, start: int, num_masks: int):
        end = start + num_masks + 1
        return self.drafter.draft(
            self.torch.tensor([anchor_id], device=self.device),
            self.torch.arange(start, end, device=self.device),
            self.cache,
            self.block_tables,
            self.torch.tensor([start], dtype=self.torch.int32, device=self.device),
            self._slots(start, end),
        )


def run(args: argparse.Namespace) -> dict:
    import importlib.util

    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, DynamicCache

    if importlib.util.find_spec("datasets") is None:
        raise SystemExit("the drafter's remote code imports `datasets`: pip install datasets")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    drafter_dir = resolve_drafter_dir(args.drafter, args.drafter_revision)

    tokenizer = AutoTokenizer.from_pretrained(args.target, revision=args.target_revision)
    hf_target = AutoModelForCausalLM.from_pretrained(
        args.target, revision=args.target_revision, torch_dtype=dtype
    ).to(device).eval()
    hf_drafter = AutoModel.from_pretrained(drafter_dir, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
    block_size = hf_drafter.block_size
    num_masks = block_size - 1
    mask_id = hf_drafter.mask_token_id
    layer_ids = list(hf_drafter.target_layer_ids)
    embed = hf_target.get_input_embeddings()

    def taps_of(hidden_states):  # utils.extract_context_feature
        return torch.cat([hidden_states[layer + 1] for layer in layer_ids], dim=-1)

    prompts = [encode(tokenizer, prompt, not args.no_chat_template) for prompt in load_prompts(args.prompts_jsonl)]
    max_len = max(ids.shape[1] for ids in prompts) + args.max_new_tokens + block_size
    nano = NanoSide(args, torch, hf_target, drafter_dir, max_len)
    stops = stop_token_ids(tokenizer)

    blocks = agree = exact_blocks = 0
    max_logit_diff = 0.0
    accepted_per_cycle: list[int] = []
    # Reference top-1 margin (top-1 minus top-2 logit) where the two disagree:
    # small margins are rounding, large ones a porting error.
    disagreement_margins: list[float] = []
    by_position = [0] * num_masks

    with torch.inference_mode():
        for input_ids in prompts:
            if blocks >= args.blocks:
                break
            input_ids = input_ids.to(device)
            start = input_ids.shape[1]
            max_length = start + args.max_new_tokens
            position_ids = torch.arange(max_length + block_size, device=device).unsqueeze(0)
            output_ids = torch.full((1, max_length + block_size), mask_id, dtype=torch.long, device=device)
            target_cache, draft_cache = DynamicCache(), DynamicCache()

            output = hf_target(
                input_ids, position_ids=position_ids[:, :start], past_key_values=target_cache,
                use_cache=True, output_hidden_states=True,
            )
            output_ids[:, :start] = input_ids
            output_ids[:, start] = output.logits[:, -1].argmax(dim=-1)
            target_hidden = taps_of(output.hidden_states)
            nano.write_context(target_hidden[0], 0)

            while start < max_length and blocks < args.blocks:
                block_ids = output_ids[:, start : start + block_size].clone()
                hidden = hf_drafter(
                    target_hidden=target_hidden,
                    noise_embedding=embed(block_ids),
                    position_ids=position_ids[:, draft_cache.get_seq_length() : start + block_size],
                    past_key_values=draft_cache,
                    use_cache=True,
                    is_causal=False,
                )
                ref_logits = hf_target.lm_head(hidden[:, -num_masks:, :])[0].float()
                draft_cache.crop(start)
                nano_logits = nano.draft(int(block_ids[0, 0]), start, num_masks).float()

                ref_top, nano_top = ref_logits.argmax(dim=-1), nano_logits.argmax(dim=-1)
                same = ref_top == nano_top
                blocks += 1
                agree += int(same.sum())
                exact_blocks += int(same.all())
                max_logit_diff = max(max_logit_diff, float((ref_logits - nano_logits).abs().max()))
                top2 = ref_logits.topk(2, dim=-1).values
                for index in (~same).nonzero().flatten().tolist():
                    disagreement_margins.append(float(top2[index, 0] - top2[index, 1]))
                for index, hit in enumerate(same.tolist()):
                    by_position[index] += int(hit)

                # Verify the reference's draft and keep the accepted prefix.
                block_ids[:, 1:] = ref_top
                output = hf_target(
                    block_ids, position_ids=position_ids[:, start : start + block_size],
                    past_key_values=target_cache, use_cache=True, output_hidden_states=True,
                )
                posterior = output.logits.argmax(dim=-1)
                accepted = int((block_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum())
                output_ids[:, start : start + accepted + 1] = block_ids[:, : accepted + 1]
                output_ids[:, start + accepted + 1] = posterior[:, accepted]
                target_hidden = taps_of(output.hidden_states)[:, : accepted + 1, :]
                nano.write_context(target_hidden[0], start)
                new_tokens = output_ids[0, start + 1 : start + accepted + 2].tolist()
                start += accepted + 1
                target_cache.crop(start)
                accepted_per_cycle.append(accepted + 1)
                if stops.intersection(new_tokens):
                    break

    positions = blocks * num_masks
    agreement = agree / positions if positions else 0.0
    return {
        "target": args.target,
        "drafter": args.drafter,
        "drafter_revision": None if Path(args.drafter).is_dir() else args.drafter_revision,
        "dtype": args.dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "drafter_weight_quant": args.drafter_weight_quant,
        "blocks": blocks,
        "blocks_requested": args.blocks,
        "draft_positions": positions,
        "top1_agreement": agreement,
        "top1_agreement_by_position": [hits / blocks for hits in by_position] if blocks else [],
        "blocks_identical": exact_blocks / blocks if blocks else 0.0,
        "max_logit_diff": max_logit_diff,
        "disagreements": len(disagreement_margins),
        "max_reference_margin_at_disagreement": max(disagreement_margins, default=0.0),
        "reference_mean_accepted_per_cycle": (
            sum(accepted_per_cycle) / len(accepted_per_cycle) if accepted_per_cycle else 0.0
        ),
        "gate": args.gate,
        "passed": blocks >= args.blocks and agreement >= args.gate,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if summary["blocks"] < args.blocks:
        print(
            f"FAIL: only {summary['blocks']} of {args.blocks} blocks ran; give more prompts "
            "or a larger --max-new-tokens",
            file=sys.stderr,
        )
    print(
        f"top-1 agreement {summary['top1_agreement']:.4f} over {summary['draft_positions']} draft "
        f"positions ({summary['blocks']} blocks), gate {args.gate}: "
        f"{'PASS' if summary['passed'] else 'FAIL'}"
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
