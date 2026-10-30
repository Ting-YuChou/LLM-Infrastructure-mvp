#!/usr/bin/env python3
"""Teacher-forced quality of nano_vllm configs on the real model.

Every config (label:attention:linear:weight_quant[:kv_cache_dtype[:lm_head_quant]]) runs the serving prefill
path, forward_batch over a whole chunk with every position's logits, on the
same wikitext-2 test chunks. Reports perplexity and, against the first
config, how often the top-1 next token agrees. The default compares the
PyTorch path, the Triton kernels, and the Triton kernels with W4A16 weights;
weight_quant "awq" loads --awq-model instead of --model (same tokenizer).
The chunks (chunks.pt) and each config's top-1 tokens (top1/<label>.pt) stay
in --out for quality_vllm.py, which scores vLLM on the same chunks and adds
its rows to quality.json / quality.md.

  python quality.py --model <dir> --out results/quality
  python quality.py --model <dir> --out d --chunks 8 --configs "a:torch:torch:none b:triton:triton:w4a16"
  python quality.py --model <dir> --awq-model <awq dir> --out d --configs "a:triton:triton:none b:triton:triton:awq"
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.serving.nano_vllm.metadata import build_batch  # noqa: E402
from src.serving.nano_vllm.model import NanoLlamaForCausalLM  # noqa: E402

DEFAULT_CONFIGS = (
    "bf16_torch:torch:torch:none bf16_triton:triton:triton:none w4a16_triton:triton:triton:w4a16"
)
BLOCK_SIZE = 16


def load_text() -> tuple[str, str]:
    """wikitext-2-raw-v1 test split, joined the way perplexity evals usually do."""
    try:
        from datasets import load_dataset

        rows = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
        return "\n\n".join(rows), "wikitext-2-raw-v1 test (datasets)"
    except Exception as e:  # datasets missing or too new for the script loader
        print(f"datasets.load_dataset failed ({type(e).__name__}: {e}); trying the parquet file")
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    rows = pq.read_table(path).column("text").to_pylist()
    return "\n\n".join(rows), "wikitext-2-raw-v1 test (parquet)"


def chunks(model_dir: str, length: int, limit: int) -> list[list[int]]:
    """Non-overlapping chunks of ``length`` tokens, each starting with BOS."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    text, source = load_text()
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    body = length - 1
    out = [
        [tokenizer.bos_token_id] + ids[start : start + body]
        for start in range(0, len(ids) - body + 1, body)
    ]
    print(f"{source}: {len(ids)} tokens -> {len(out)} chunks of {length}")
    return out[:limit] if limit else out


@torch.inference_mode()
def evaluate(model: NanoLlamaForCausalLM, data: list[list[int]]) -> dict:
    length = len(data[0])
    blocks = length // BLOCK_SIZE + 1
    nll, count, top1 = 0.0, 0, []
    for chunk in data:
        cache = model.new_paged_cache(num_blocks=blocks, block_size=BLOCK_SIZE)
        ids, positions, metadata = build_batch(
            input_ids=chunk,
            positions=list(range(length)),
            slots=list(range(length)),
            block_tables=[list(range(blocks))],
            context_lens=[length],
            query_lens=[length],
            is_prefill=True,
            null_block=cache.null_block,
            num_slots=cache.total_slots,
            device=model.device,
            all_logits=True,
        )
        logits = model.forward_batch(ids, positions, cache, metadata)[:-1]
        targets = ids[1:]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        nll -= float(logprobs.gather(1, targets.unsqueeze(1)).sum())
        count += targets.numel()
        top1.append(logits.argmax(dim=-1).to(torch.int32).cpu())
        del cache, logits, logprobs
    return {"nll": nll, "tokens": count, "ppl": math.exp(nll / count), "top1": torch.cat(top1)}


def write_tables(out: Path, results: list[dict], length: int) -> str:
    """quality.json and quality.md from every row so far (quality_vllm.py adds its own)."""
    (out / "quality.json").write_text(json.dumps(results, indent=2))
    lines = [
        f"# Teacher-forced quality ({results[0]['tokens']} tokens, chunks of {length})",
        "",
        "| config | attention | linear | weights | KV cache | weight GiB | perplexity | top-1 agreement |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for r in results:
        agree = f"{100 * r['top1_agreement']:.2f}% vs {r['top1_agreement_vs']}" if "top1_agreement" in r else "-"
        gib = "-" if r.get("weights_gib") is None else f"{r['weights_gib']:.1f}"
        lines.append(
            f"| {r['label']} | {r['attention_backend']} | {r['linear_backend']} | {r['weight_quant']} "
            f"| {r['kv_cache_dtype']} | {gib} | {r['ppl']:.3f} | {agree} |"
        )
    text = "\n".join(lines)
    (out / "quality.md").write_text(text + "\n")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--awq-model", help="AWQ checkpoint for weight_quant=awq configs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--configs", default=DEFAULT_CONFIGS)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--chunks", type=int, default=0, help="0: the whole test split")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "quality.py needs a GPU"

    out = Path(args.out)
    (out / "top1").mkdir(parents=True, exist_ok=True)
    data = chunks(args.model, args.length, args.chunks)
    torch.save(torch.tensor(data, dtype=torch.int32), out / "chunks.pt")
    results, baseline = [], None
    for config in args.configs.split():
        label, attention, linear_backend, weight_quant, *rest = config.split(":")
        kv_cache_dtype = rest[0] if rest else "auto"
        lm_head_quant = rest[1] if len(rest) > 1 else "none"
        if weight_quant == "awq" and not args.awq_model:
            parser.error(f"{label}: weight_quant awq needs --awq-model")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model = NanoLlamaForCausalLM.from_pretrained(
            args.awq_model if weight_quant == "awq" else args.model,
            dtype="bfloat16",
            device="cuda",
            use_triton=True,
            attention_backend=attention,
            max_model_len=args.length,
            linear_backend=linear_backend,
            weight_quant=weight_quant,
            kv_cache_dtype=kv_cache_dtype,
            lm_head_quant=lm_head_quant,
        )
        load_s = time.perf_counter() - t0
        weights_gib = torch.cuda.memory_allocated() / 1024**3
        t0 = time.perf_counter()
        scores = evaluate(model, data)
        torch.cuda.synchronize()
        row = {
            "label": label,
            "attention_backend": attention,
            "linear_backend": linear_backend,
            "weight_quant": weight_quant if lm_head_quant == "none" else f"{weight_quant}+{lm_head_quant} lm_head",
            "kv_cache_dtype": kv_cache_dtype,
            "ppl": scores["ppl"],
            "tokens": scores["tokens"],
            "weights_gib": weights_gib,
            "load_s": load_s,
            "eval_s": time.perf_counter() - t0,
        }
        torch.save(scores["top1"], out / "top1" / f"{label}.pt")
        if baseline is None:
            baseline = (label, scores["top1"])
        else:
            row["top1_agreement_vs"] = baseline[0]
            row["top1_agreement"] = float((scores["top1"] == baseline[1]).float().mean())
        print("QUALITY " + json.dumps(row), flush=True)
        results.append(row)
        del model, scores
        gc.collect()
        torch.cuda.empty_cache()

    print(write_tables(out, results, args.length))


if __name__ == "__main__":
    main()
