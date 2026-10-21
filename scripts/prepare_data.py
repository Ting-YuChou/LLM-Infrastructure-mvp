#!/usr/bin/env python3
"""
Data Preparation Script

Writes the SFT, preference and RL prompt files the training configs read:

    <output-dir>/sft/{train,val}.json            SFT (prompt/completion or messages)
    <output-dir>/preferences/{train,val}.json    reward model and DPO
    <output-dir>/prompts/rl_prompts.json         PPO

--source synthetic (the default) repeats a few hand-written examples, for smoke
runs. --source hf converts real data from the Hugging Face Hub, the datasets
Zephyr was trained on:

- SFT: HuggingFaceH4/ultrachat_200k, multi-turn conversations as {"messages": [...]}.
- Reward model / DPO: HuggingFaceH4/ultrafeedback_binarized, single-turn pairs
  with a strictly higher chosen score.
- PPO: UltraFeedback prompts that are not in the preference training set.

Validation rows come from each dataset's test split. With --source hf the
script also tokenizes everything with the chat template (--tokenizer, default
the Llama 3 8B base) and writes data_stats.json: token length percentiles and
how many rows exceed each stage's max_length, using the same rules as the
trainers (SFT cuts after the last reply that fits; RM/DPO drop the pair).

    python scripts/prepare_data.py --source hf --output-dir data \
        --sft-examples 20000 --preference-examples 20000 --rl-prompts 2000
"""

import json
import os
import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from training.chat_format import (  # noqa: E402
    encode,
    encode_conversation,
    encode_prompt_and_completion,
    ensure_chat_tokenizer,
    format_completion,
    format_prompt,
    validate_messages,
)

SFT_DATASET = "HuggingFaceH4/ultrachat_200k"
SFT_SPLITS = ("train_sft", "test_sft")
PREFERENCE_DATASET = "HuggingFaceH4/ultrafeedback_binarized"
PREFERENCE_SPLITS = ("train_prefs", "test_prefs")
DEFAULT_TOKENIZER = "meta-llama/Meta-Llama-3-8B"
# Token lengths reported in data_stats.json besides each stage's own limit.
REPORTED_LIMITS = (512, 1024, 2048, 4096)


def create_sft_dataset(output_dir: str, num_examples: int = 100) -> str:
    """
    Create sample SFT (supervised fine-tuning) dataset
    
    Args:
        output_dir: Directory to save dataset
        num_examples: Number of examples to generate
        
    Returns:
        Path to created dataset file
    """
    print(f"Creating SFT dataset with {num_examples} examples...")
    
    # Sample instruction-response pairs
    base_examples = [
        {
            "prompt": "Explain what machine learning is in simple terms.",
            "completion": "Machine learning is a way for computers to learn from examples without being explicitly programmed. It's like teaching a child by showing them many examples until they understand the pattern. The computer analyzes data, finds patterns, and uses those patterns to make decisions or predictions on new data."
        },
        {
            "prompt": "Write a Python function to reverse a string.",
            "completion": "Here's a simple Python function to reverse a string:\n\n```python\ndef reverse_string(s):\n    return s[::-1]\n```\n\nThis uses Python's slice notation with a step of -1, which reverses the string. You can also use `''.join(reversed(s))` for a more explicit approach."
        },
        {
            "prompt": "What are the main differences between supervised and unsupervised learning?",
            "completion": "Supervised learning uses labeled data where the correct answers are provided during training. The model learns to map inputs to known outputs. Examples include classification and regression.\n\nUnsupervised learning works with unlabeled data, finding hidden patterns or structures without explicit answers. Examples include clustering and dimensionality reduction.\n\nThe key difference is that supervised learning has a 'teacher' (labels), while unsupervised learning discovers patterns independently."
        },
        {
            "prompt": "How do I sort a list in Python?",
            "completion": "You can sort a list in Python in two main ways:\n\n1. In-place sorting (modifies original):\n```python\nmy_list = [3, 1, 4, 1, 5]\nmy_list.sort()\nprint(my_list)  # [1, 1, 3, 4, 5]\n```\n\n2. Return new sorted list:\n```python\nmy_list = [3, 1, 4, 1, 5]\nsorted_list = sorted(my_list)\nprint(sorted_list)  # [1, 1, 3, 4, 5]\n```\n\nBoth support `reverse=True` for descending order and `key=` for custom sorting."
        },
        {
            "prompt": "Explain the concept of recursion with an example.",
            "completion": "Recursion is when a function calls itself to solve a problem by breaking it into smaller, similar sub-problems. Here's a classic example - calculating factorial:\n\n```python\ndef factorial(n):\n    if n <= 1:  # Base case\n        return 1\n    return n * factorial(n - 1)  # Recursive case\n```\n\nFor factorial(5):\n- 5 * factorial(4)\n- 5 * 4 * factorial(3)\n- 5 * 4 * 3 * factorial(2)\n- 5 * 4 * 3 * 2 * factorial(1)\n- 5 * 4 * 3 * 2 * 1 = 120\n\nKey parts: base case (stopping condition) and recursive case (function calling itself)."
        },
    ]
    
    # Replicate examples to reach desired count
    dataset = []
    for i in range(num_examples):
        example = base_examples[i % len(base_examples)].copy()
        example["id"] = i
        dataset.append(example)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dataset
    output_file = os.path.join(output_dir, "train.json")
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✓ SFT dataset saved to: {output_file}")
    print(f"  Total examples: {len(dataset)}")
    
    # Create smaller validation set (10% of training)
    val_size = max(10, num_examples // 10)
    val_dataset = dataset[:val_size]
    val_file = os.path.join(output_dir, "val.json")
    
    with open(val_file, 'w') as f:
        json.dump(val_dataset, f, indent=2)
    
    print(f"✓ Validation dataset saved to: {val_file}")
    print(f"  Total examples: {len(val_dataset)}")
    
    return output_file


def create_preference_dataset(output_dir: str, num_examples: int = 100) -> str:
    """
    Create sample preference dataset for reward model training
    
    Args:
        output_dir: Directory to save dataset
        num_examples: Number of examples to generate
        
    Returns:
        Path to created dataset file
    """
    print(f"\nCreating preference dataset with {num_examples} examples...")
    
    # Sample preference pairs (chosen vs rejected)
    base_examples = [
        {
            "prompt": "How do I learn programming?",
            "chosen": "Start with Python as it's beginner-friendly. Practice daily with small projects, gradually increasing complexity. Use resources like online tutorials, coding platforms (LeetCode, HackerRank), and build real projects. Join programming communities for support and feedback.",
            "rejected": "Just Google it and you'll figure it out eventually. Programming is hard, you might not be cut out for it."
        },
        {
            "prompt": "What's the best way to lose weight?",
            "chosen": "A healthy approach combines balanced nutrition with regular exercise. Focus on whole foods, portion control, and sustainable habits. Aim for 1-2 pounds per week. Consult a healthcare provider for personalized advice.",
            "rejected": "Stop eating. The less you eat, the more weight you'll lose. Crash diets are the fastest way."
        },
        {
            "prompt": "How can I improve my communication skills?",
            "chosen": "Practice active listening, maintain eye contact, and be clear and concise. Read widely to expand vocabulary. Join groups like Toastmasters for public speaking practice. Ask for feedback and reflect on conversations to continuously improve.",
            "rejected": "Just talk more. Volume is what matters. Interrupt people to get your point across faster."
        },
    ]
    
    # Replicate examples
    dataset = []
    for i in range(num_examples):
        example = base_examples[i % len(base_examples)].copy()
        example["id"] = i
        dataset.append(example)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dataset
    output_file = os.path.join(output_dir, "train.json")
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✓ Preference dataset saved to: {output_file}")
    print(f"  Total examples: {len(dataset)}")

    # config/reward_config.yaml and config/dpo_config.yaml read a val split.
    val_dataset = dataset[:max(10, num_examples // 10)]
    val_file = os.path.join(output_dir, "val.json")
    with open(val_file, 'w') as f:
        json.dump(val_dataset, f, indent=2)

    print(f"✓ Preference validation set saved to: {val_file}")
    print(f"  Total examples: {len(val_dataset)}")

    return output_file


def create_rl_prompts(output_dir: str, num_prompts: int = 50) -> str:
    """
    Create sample prompts for RLHF training
    
    Args:
        output_dir: Directory to save prompts
        num_prompts: Number of prompts to generate
        
    Returns:
        Path to created prompts file
    """
    print(f"\nCreating RL prompts with {num_prompts} examples...")
    
    # Sample prompts for RL training
    base_prompts = [
        "Explain the concept of neural networks.",
        "What are the benefits of regular exercise?",
        "How does photosynthesis work?",
        "Describe the water cycle.",
        "What is climate change and why is it important?",
        "How do I start a small business?",
        "What are good study habits for students?",
        "Explain quantum computing in simple terms.",
        "What is the importance of biodiversity?",
        "How can I manage stress effectively?",
    ]
    
    # Replicate prompts
    prompts = []
    for i in range(num_prompts):
        prompts.append({
            "id": i,
            "prompt": base_prompts[i % len(base_prompts)]
        })
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save prompts
    output_file = os.path.join(output_dir, "rl_prompts.json")
    with open(output_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    print(f"✓ RL prompts saved to: {output_file}")
    print(f"  Total prompts: {len(prompts)}")
    
    return output_file


# ============================================================================
# --source hf
# ============================================================================


def load_hf_split(name: str, split: str, seed: int) -> Iterable[Mapping[str, Any]]:
    """One split of a Hub dataset in a seeded random order."""
    from datasets import load_dataset

    # Both datasets store split <s> as data/<s>-*.parquet. Naming the files
    # downloads only this split; otherwise datasets fetches and prepares every
    # split (1.6 GB for UltraChat instead of 0.7 GB for train_sft).
    return load_dataset(name, data_files={split: f"data/{split}-*.parquet"}, split=split).shuffle(seed=seed)


def sft_conversation(row: Mapping[str, Any]) -> Optional[List[Dict[str, str]]]:
    """An UltraChat row as a conversation ending with an assistant reply, or None if unusable."""
    messages = [
        {"role": str(message["role"]), "content": str(message["content"]).strip()}
        for message in row.get("messages") or []
    ]
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    if not messages or not all(message["content"] for message in messages):
        return None
    try:
        return validate_messages(messages, last_role="assistant")
    except ValueError:
        return None


def preference_pair(row: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    """
    An UltraFeedback row as {"prompt", "chosen", "rejected"}, or None.

    Rows are skipped when they are not a single user turn plus one reply on each
    side, when a reply is empty, when both replies are the same text, or when
    the chosen score is not strictly higher (a tie carries no preference).
    """
    chosen, rejected = list(row.get("chosen") or []), list(row.get("rejected") or [])
    if len(chosen) != 2 or len(rejected) != 2 or chosen[0] != rejected[0]:
        return None
    if [chosen[0]["role"], chosen[1]["role"], rejected[1]["role"]] != ["user", "assistant", "assistant"]:
        return None
    prompt, good, bad = (str(text).strip() for text in (chosen[0]["content"], chosen[1]["content"], rejected[1]["content"]))
    if not (prompt and good and bad) or good == bad:
        return None
    score_chosen, score_rejected = row.get("score_chosen"), row.get("score_rejected")
    if score_chosen is not None and score_rejected is not None and score_chosen <= score_rejected:
        return None
    return {"prompt": prompt, "chosen": good, "rejected": bad}


def rl_prompt(row: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    prompt = str(row.get("prompt") or "").strip()
    return {"prompt": prompt} if prompt else None


def _prompt_key(row: Mapping[str, Any]) -> str:
    return " ".join(str(row.get("prompt") or "").split())


def take(
    rows: Iterable[Mapping[str, Any]],
    convert: Callable[[Mapping[str, Any]], Optional[Dict[str, Any]]],
    count: int,
    exclude: Set[str] = frozenset(),
) -> Tuple[List[Dict[str, Any]], Set[str], int]:
    """
    The first `count` rows that convert, one per prompt and none whose prompt is in `exclude`.

    Returns the converted rows (with an "id"), their prompt keys and how many rows were skipped.
    """
    taken: List[Dict[str, Any]] = []
    keys: Set[str] = set()
    skipped = 0
    for row in rows:
        if len(taken) >= count:
            break
        key = _prompt_key(row)
        converted = convert(row) if key not in keys and key not in exclude else None
        if converted is None:
            skipped += 1
            continue
        keys.add(key)
        taken.append({"id": str(row.get("prompt_id") or len(taken)), **converted})
    return taken, keys, skipped


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), ensure_ascii=False), encoding="utf-8")
    print(f"✓ {path}: {len(rows)} rows")
    return str(path)


def build_hf_datasets(
    output_dir: str,
    sft_examples: int,
    preference_examples: int,
    rl_prompts: int,
    seed: int = 0,
    val_fraction: float = 0.05,
    load_split: Callable[[str, str, int], Iterable[Mapping[str, Any]]] = load_hf_split,
) -> Dict[str, Any]:
    """Convert UltraChat and UltraFeedback into the training files; returns a summary."""
    out = Path(output_dir)

    def val_count(count: int) -> int:
        return max(10, int(count * val_fraction))

    sources: Dict[str, Any] = {}
    sft_train, _, sft_skipped = take(load_split(SFT_DATASET, SFT_SPLITS[0], seed), _sft_row, sft_examples)
    sft_val, _, sft_val_skipped = take(load_split(SFT_DATASET, SFT_SPLITS[1], seed), _sft_row, val_count(sft_examples))
    _write_rows(out / "sft" / "train.json", sft_train)
    _write_rows(out / "sft" / "val.json", sft_val)
    sources["sft"] = {"dataset": SFT_DATASET, "splits": list(SFT_SPLITS), "skipped": sft_skipped + sft_val_skipped}

    # The preference split is read twice in the same seeded order: RM/DPO pairs
    # first, then PPO prompts the reward model was not trained on.
    pref_train, pref_keys, pref_skipped = take(
        load_split(PREFERENCE_DATASET, PREFERENCE_SPLITS[0], seed), preference_pair, preference_examples
    )
    pref_val, _, pref_val_skipped = take(
        load_split(PREFERENCE_DATASET, PREFERENCE_SPLITS[1], seed), preference_pair, val_count(preference_examples)
    )
    prompts, _, prompt_skipped = take(
        load_split(PREFERENCE_DATASET, PREFERENCE_SPLITS[0], seed), rl_prompt, rl_prompts, exclude=pref_keys
    )
    _write_rows(out / "preferences" / "train.json", pref_train)
    _write_rows(out / "preferences" / "val.json", pref_val)
    _write_rows(out / "prompts" / "rl_prompts.json", prompts)
    sources["preferences"] = {
        "dataset": PREFERENCE_DATASET,
        "splits": list(PREFERENCE_SPLITS),
        "skipped": pref_skipped + pref_val_skipped,
    }
    sources["prompts"] = {
        "dataset": PREFERENCE_DATASET,
        "splits": [PREFERENCE_SPLITS[0]],
        # Includes the prompts held back because the preference set uses them.
        "skipped": prompt_skipped,
    }

    counts = {
        "sft_train": len(sft_train),
        "sft_val": len(sft_val),
        "preference_train": len(pref_train),
        "preference_val": len(pref_val),
        "rl_prompts": len(prompts),
    }
    for name, wanted in (("sft_train", sft_examples), ("preference_train", preference_examples), ("rl_prompts", rl_prompts)):
        if counts[name] < wanted:
            print(f"! {name}: only {counts[name]} usable rows, {wanted} requested")
    return {"seed": seed, "sources": sources, "counts": counts}


def _sft_row(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    messages = sft_conversation(row)
    return {"messages": messages} if messages else None


def length_summary(lengths: Sequence[int], limits: Iterable[int]) -> Dict[str, Any]:
    """Percentiles of `lengths` and the fraction of rows longer than each limit."""
    ordered = sorted(lengths)
    if not ordered:
        return {"count": 0}

    def percentile(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {
        "count": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 1),
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "over": {str(limit): round(sum(length > limit for length in ordered) / len(ordered), 4) for limit in sorted(set(limits))},
    }


def _stage_limits(config_dir: Path) -> Dict[str, int]:
    def data(name: str) -> Dict[str, Any]:
        return yaml.safe_load((config_dir / name).read_text())["data"]

    return {
        "sft_max_length": int(data("sft_config.yaml")["max_length"]),
        "reward_max_length": int(data("reward_config.yaml")["max_length"]),
        "dpo_max_length": int(data("dpo_config.yaml")["max_length"]),
        "rl_max_prompt_length": int(data("rlhf_config.yaml")["max_prompt_length"]),
    }


def data_stats(output_dir: str, tokenizer: Any, limits: Mapping[str, int]) -> Dict[str, Any]:
    """Chat-templated token lengths of the written files, measured the way the trainers measure them."""
    out = Path(output_dir)
    tokenizer = ensure_chat_tokenizer(tokenizer)

    def read(path: str) -> List[Dict[str, Any]]:
        return json.loads((out / path).read_text(encoding="utf-8"))

    sft_max = limits["sft_max_length"]
    sft_lengths, turns, cut_inside_first_reply = [], [], 0
    for row in read("sft/train.json"):
        encoded = encode_conversation(tokenizer, row["messages"])
        sft_lengths.append(len(encoded.input_ids))
        turns.append(len(encoded.turn_ends))
        if encoded.turn_ends[0] > sft_max:
            cut_inside_first_reply += 1

    pair_lengths, prompt_lengths = [], []
    for row in read("preferences/train.json"):
        prompt_ids, chosen_ids = encode_prompt_and_completion(tokenizer, row["prompt"], row["chosen"])
        rejected_ids = encode(tokenizer, format_completion(row["rejected"]))
        pair_lengths.append(len(prompt_ids) + max(len(chosen_ids), len(rejected_ids)))
    for row in read("prompts/rl_prompts.json"):
        prompt_lengths.append(len(encode(tokenizer, format_prompt(tokenizer, row["prompt"]))))

    sft = length_summary(sft_lengths, (*REPORTED_LIMITS, sft_max))
    sft["turns"] = length_summary(turns, ())
    sft["cut_inside_first_reply"] = round(cut_inside_first_reply / max(1, len(sft_lengths)), 4)
    return {
        "limits": dict(limits),
        "sft_conversation_tokens": sft,
        "preference_pair_tokens": length_summary(
            pair_lengths, (*REPORTED_LIMITS, limits["reward_max_length"], limits["dpo_max_length"])
        ),
        "rl_prompt_tokens": length_summary(prompt_lengths, (*REPORTED_LIMITS, limits["rl_max_prompt_length"])),
    }


def print_stats(stats: Mapping[str, Any]) -> None:
    limits = stats["limits"]
    rows = (
        ("SFT conversation", "sft_conversation_tokens", "sft_max_length",
         "longer ones are cut after the last reply that fits"),
        ("RM/DPO pair (prompt + longer reply)", "preference_pair_tokens", "reward_max_length",
         "longer pairs are dropped"),
        ("PPO prompt", "rl_prompt_tokens", "rl_max_prompt_length", "longer prompts are skipped"),
    )
    print("\nChat-templated token lengths:")
    for title, key, limit_key, rule in rows:
        summary = stats[key]
        if not summary.get("count"):
            print(f"  {title}: no rows")
            continue
        over = ", ".join(f">{limit}: {fraction:.1%}" for limit, fraction in summary["over"].items())
        print(
            f"  {title}: n={summary['count']} p50={summary['p50']} p90={summary['p90']} "
            f"p99={summary['p99']} max={summary['max']}"
        )
        print(f"    {over}  ({limit_key}={limits[limit_key]}; {rule})")
    sft = stats["sft_conversation_tokens"]
    if sft.get("count"):
        print(
            f"  SFT assistant turns: mean={sft['turns']['mean']} max={sft['turns']['max']}; "
            f"first reply alone over sft_max_length: {sft['cut_inside_first_reply']:.1%}"
        )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Prepare datasets for LLM training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--source",
        choices=("synthetic", "hf"),
        default="synthetic",
        help="synthetic: repeated hand-written examples; hf: UltraChat and UltraFeedback from the Hub"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Base output directory for datasets"
    )

    parser.add_argument(
        "--sft-examples",
        type=int,
        default=100,
        help="Number of SFT examples to generate"
    )

    parser.add_argument(
        "--preference-examples",
        type=int,
        default=100,
        help="Number of preference examples to generate"
    )

    parser.add_argument(
        "--rl-prompts",
        type=int,
        default=50,
        help="Number of RL prompts to generate"
    )

    parser.add_argument("--seed", type=int, default=0, help="--source hf: shuffle seed")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="--source hf: validation rows per training row (at least 10)"
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="--source hf: tokenizer for data_stats.json (the model you will train)"
    )
    parser.add_argument("--skip-stats", action="store_true", help="--source hf: do not tokenize for data_stats.json")
    parser.add_argument(
        "--config-dir",
        default=str(ROOT_DIR / "config"),
        help="--source hf: stage YAMLs whose max lengths data_stats.json checks"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Data Preparation for LLM Training")
    print("=" * 80)

    if args.source == "hf":
        summary = build_hf_datasets(
            args.output_dir,
            args.sft_examples,
            args.preference_examples,
            args.rl_prompts,
            seed=args.seed,
            val_fraction=args.val_fraction,
        )
        if not args.skip_stats:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            summary["tokenizer"] = args.tokenizer
            summary["stats"] = data_stats(args.output_dir, tokenizer, _stage_limits(Path(args.config_dir)))
            print_stats(summary["stats"])
        stats_path = Path(args.output_dir) / "data_stats.json"
        stats_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n✓ {stats_path}")
        return

    # Create datasets
    sft_dir = os.path.join(args.output_dir, "sft")
    pref_dir = os.path.join(args.output_dir, "preferences")
    prompt_dir = os.path.join(args.output_dir, "prompts")

    create_sft_dataset(sft_dir, args.sft_examples)
    create_preference_dataset(pref_dir, args.preference_examples)
    create_rl_prompts(prompt_dir, args.rl_prompts)

    print("\n" + "=" * 80)
    print("Data preparation complete!")
    print("=" * 80)
    print(f"\nDatasets created in: {args.output_dir}")
    print("\nNext steps:")
    print("1. Review the generated datasets")
    print("2. Customize with your own data")
    print("3. Run training: python scripts/train_sft.py --config config/sft_config.yaml")


if __name__ == "__main__":
    main()
