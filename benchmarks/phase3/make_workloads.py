#!/usr/bin/env python3
"""Real-prompt workloads for Phase 3 (run_phase3.sh's workloads step).

Speculative decoding only shows on text a drafter can predict, so these
replace the random-token prompts of phases 0-2: Alpaca instructions, GSM8K
questions and HumanEval functions (the three sets the DFlash checkpoint's
card reports for Llama 3.1 8B) plus CNN/DailyMail articles to summarize for a
long context.

Every prompt is rendered once with the model's chat template into a string
and sent to /v1/completions, so every engine tokenizes the same text; the
leading <|begin_of_text|> is dropped because each server adds it. Each
concurrency level of a dataset gets its own slice (<dataset>_c<level>.jsonl,
one {"prompt": ...} per line, the format of `vllm bench serve --dataset-name
custom`), so a level's prompts were not prefilled by an earlier level, except
where a dataset is too small (HumanEval has 164 problems: its c=64 file is
all of them). warmup.jsonl holds Alpaca prompts no level uses.

  python make_workloads.py --model <target dir> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

# The chat template's default date, fixed so the files do not change by the day.
DATE_STRING = "26 Jul 2024"
# Prompts per concurrency level (4 requests per client, at least 24).
LEVELS = {
    "alpaca": {1: 24, 4: 24, 16: 64, 64: 256},
    "humaneval": {1: 24, 4: 24, 16: 64, 64: 164},
    "gsm8k": {1: 24, 16: 64},
    "long": {1: 24, 16: 64},
}
WARMUP_PROMPTS = 16
LONG_TOKENS = (1000, 2500)  # article lengths kept for "long"


def alpaca() -> list[str]:
    from datasets import load_dataset

    rows = load_dataset("tatsu-lab/alpaca", split="train")
    return [row["instruction"] for row in rows.select(range(4000)) if not row["input"].strip()]


def gsm8k() -> list[str]:
    from datasets import load_dataset

    return [row["question"] for row in load_dataset("openai/gsm8k", "main", split="test")]


def humaneval() -> list[str]:
    from datasets import load_dataset

    rows = load_dataset("openai/openai_humaneval", split="test")
    return [
        "Complete the following Python function.\n\n```python\n" + row["prompt"].rstrip() + "\n```"
        for row in rows
    ]


def long_articles(tokenizer, wanted: int) -> list[str]:
    from datasets import load_dataset

    rows = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
    texts = []
    for row in rows:
        tokens = len(tokenizer(row["article"], add_special_tokens=False).input_ids)
        if LONG_TOKENS[0] <= tokens <= LONG_TOKENS[1]:
            texts.append(
                "Summarize the following news article in three sentences.\n\n" + row["article"]
            )
        if len(texts) == wanted:
            break
    return texts


def render(tokenizer, text: str) -> str:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        date_string=DATE_STRING,
    )
    bos = tokenizer.bos_token or ""
    if not bos or not prompt.startswith(bos):
        raise ValueError("the rendered prompt does not start with the BOS token")
    return prompt[len(bos) :]


def slices(name: str, prompts: list[str]) -> dict[int, list[str]]:
    """Disjoint slices per level while the prompts last; a level that would
    run past the end takes the first prompts instead (it overlaps earlier ones)."""
    out, start = {}, 0
    for level, count in LEVELS[name].items():
        if start + count > len(prompts):
            if count > len(prompts):
                raise ValueError(f"{name} has {len(prompts)} prompts, level {level} needs {count}")
            out[level] = prompts[:count]
        else:
            out[level] = prompts[start : start + count]
            start += count
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="target model dir (its tokenizer)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    sources = {
        "alpaca": alpaca(),
        "humaneval": humaneval(),
        "gsm8k": gsm8k(),
        "long": long_articles(tokenizer, sum(LEVELS["long"].values())),
    }
    manifest = {"date_string": DATE_STRING, "files": {}}

    def write(name: str, texts: list[str]) -> None:
        prompts = [render(tokenizer, text) for text in texts]
        body = "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts)
        (args.out / name).write_text(body)
        # What a server sees: the text plus the BOS it adds.
        lengths = [len(tokenizer(prompt, add_special_tokens=True).input_ids) for prompt in prompts]
        manifest["files"][name] = {
            "prompts": len(prompts),
            "tokens_min": min(lengths),
            "tokens_median": int(statistics.median(lengths)),
            "tokens_max": max(lengths),
            "tokens_first": lengths[0],
            "sha256": hashlib.sha256(body.encode()).hexdigest(),
        }

    for name, texts in sources.items():
        for level, chunk in slices(name, texts).items():
            write(f"{name}_c{level}.jsonl", chunk)
    used = sum(count for count in LEVELS["alpaca"].values())
    write("warmup.jsonl", sources["alpaca"][used : used + WARMUP_PROMPTS])
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for name, info in manifest["files"].items():
        print(
            f"P3_WORKLOAD {name}: {info['prompts']} prompts, "
            f"{info['tokens_min']}/{info['tokens_median']}/{info['tokens_max']} tokens (min/median/max)"
        )


if __name__ == "__main__":
    main()
