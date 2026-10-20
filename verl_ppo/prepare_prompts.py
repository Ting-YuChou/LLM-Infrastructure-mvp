#!/usr/bin/env python3
"""
Convert the PPO prompt JSON used by src/training/rlhf_trainer.py into verl's
RL parquet layout.

    python verl_ppo/prepare_prompts.py --input data/prompts/rl_prompts.json --output-dir data/verl

Input is a JSON list of {"prompt": ...} objects or bare strings (the same rule
as rlhf_trainer.PromptDataset). Each output row is

    {"data_source": "llm_infra_rl_prompts",
     "prompt": [{"role": "user", "content": <prompt>}],
     "ability": "chat",
     "reward_model": {"style": "model", "ground_truth": ""},
     "extra_info": {"split": "train", "index": 0}}

verl applies the actor tokenizer's chat template (the Llama 3 template saved
with our SFT checkpoint) to `prompt`, and scores rollouts with the reward model,
so `ground_truth` stays empty. Only pandas and pyarrow are needed, so this runs
in the verl environment or in ours.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

DATA_SOURCE = "llm_infra_rl_prompts"


def load_prompts(path: Path) -> List[str]:
    """Prompts from a JSON list of {"prompt": ...} objects or bare strings."""
    items = json.loads(path.read_text(encoding="utf-8"))
    return [item if isinstance(item, str) else item["prompt"] for item in items]


def to_rows(prompts: List[str], split: str) -> List[Dict[str, Any]]:
    return [
        {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "chat",
            "reward_model": {"style": "model", "ground_truth": ""},
            "extra_info": {"split": split, "index": index},
        }
        for index, prompt in enumerate(prompts)
    ]


def write_parquet(prompts: List[str], output_dir: Path, val_count: int) -> Dict[str, Path]:
    """Write train.parquet and val.parquet (the last val_count prompts) to output_dir."""
    import pandas as pd

    if not 0 < val_count < len(prompts):
        raise ValueError(f"val_count must be between 1 and {len(prompts) - 1}, got {val_count}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, subset in (("train", prompts[:-val_count]), ("val", prompts[-val_count:])):
        paths[split] = output_dir / f"{split}.parquet"
        pd.DataFrame(to_rows(subset, split)).to_parquet(paths[split], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Prompt JSON file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-count", type=int, default=16, help="Prompts held out for val.parquet")
    args = parser.parse_args()

    prompts = load_prompts(Path(args.input))
    for split, path in write_parquet(prompts, Path(args.output_dir), args.val_count).items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
