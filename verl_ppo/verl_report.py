#!/usr/bin/env python3
"""
Summarize one verl PPO run (verl_ppo/run_ppo.sh) for the GPU smoke report.

Parses verl's console metrics (`step:N - key:value - ...` lines in train.log)
into per-step phase timings in the same shape as rlhf_trainer's
`timing_*_mean` metrics, then reloads the exported actor and generates once to
check it stops on <|eot_id|>. Writes the JSON given by --output.

Timing phases, so they line up with src/training/rlhf_trainer.py:
  gen_s    = timing_s/gen
  update_s = timing_s/update_actor + update_critic + update_weights
  score_s  = timing_s/step - gen_s - update_s   (old/ref log-probs, values, RM, advantages)
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

STEP_LINE = re.compile(r"step:(\d+) - (.*)")
METRIC = re.compile(r"([\w/@.\-]+):(-?[\d.]+(?:e[-+]?\d+)?|nan|inf|-inf)")
UPDATE_TIMERS = ("timing_s/update_actor", "timing_s/update_critic", "timing_s/update_weights")
NON_STEP_TIMERS = ("timing_s/save_checkpoint", "timing_s/testing")
PROMPT = "Explain what machine learning is in simple terms."


def parse_steps(log_text: str) -> List[Dict[str, float]]:
    """Per-step metric dicts from verl's console logger, in step order."""
    steps: Dict[int, Dict[str, float]] = {}
    for line in log_text.splitlines():
        match = STEP_LINE.search(line)
        if not match:
            continue
        metrics = {key: float(value) for key, value in METRIC.findall(match.group(2))}
        if "timing_s/step" in metrics:
            steps[int(match.group(1))] = metrics
    return [steps[step] for step in sorted(steps)]


def summarize(steps: List[Dict[str, float]], batch: int) -> Dict[str, float]:
    """Mean phase timings over steps after the first (warm-up), like summarize_step_timings."""
    measured = steps[1:] if len(steps) > 1 else steps
    if not measured:
        return {}
    phases = []
    for metrics in measured:
        gen = metrics.get("timing_s/gen", 0.0)
        update = sum(metrics.get(key, 0.0) for key in UPDATE_TIMERS)
        # rlhf_trainer's step clock stops before checkpointing; drop verl's save/val time too.
        step = metrics["timing_s/step"] - sum(metrics.get(key, 0.0) for key in NON_STEP_TIMERS)
        phases.append({
            "gen_s": gen,
            "update_s": update,
            "score_s": max(step - gen - update, 0.0),
            "step_s": step,
            "gen_tokens": metrics.get("response_length/mean", 0.0) * batch,
            "response_length": metrics.get("response_length/mean", 0.0),
        })
    summary = {
        f"timing_{key}_mean": sum(phase[key] for phase in phases) / len(phases)
        for key in ("gen_s", "score_s", "update_s", "step_s")
    }
    gen_seconds = sum(phase["gen_s"] for phase in phases)
    summary["gen_tokens_per_s"] = sum(phase["gen_tokens"] for phase in phases) / gen_seconds if gen_seconds else 0.0
    summary["response_length_mean"] = sum(phase["response_length"] for phase in phases) / len(phases)
    summary["timing_steps"] = float(len(phases))
    return summary


def find_adapter(hf_dir: Path) -> Optional[Path]:
    for config in sorted(hf_dir.rglob("adapter_config.json")):
        return config.parent
    return None


def reload_and_generate(hf_dir: Path, sft_ckpt: Path) -> Dict[str, Any]:
    """Load the exported actor (full model, or SFT base + LoRA adapter) and generate once."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if (hf_dir / "config.json").exists() and any(hf_dir.glob("*.safetensors")):
        source, model = "full", AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.bfloat16)
    else:
        adapter = find_adapter(hf_dir)
        if adapter is None:
            return {"ok": False, "error": f"no model weights or LoRA adapter under {hf_dir}"}
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(sft_ckpt, dtype=torch.bfloat16)
        source, model = f"lora:{adapter}", PeftModel.from_pretrained(base, adapter).merge_and_unload()
    model.to(device)
    # Our checkpoints carry the Llama 3 template and generation config (eos = <|eot_id|>).
    tokenizer = AutoTokenizer.from_pretrained(sft_ckpt)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False
    )
    input_ids = torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]], device=device)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=48,
            do_sample=False,
        )
    new_tokens = output[0, input_ids.shape[1]:].tolist()
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    return {
        "ok": True,
        "source": source,
        "generation_config_stops_on_eot": eot_id in eos_ids,
        "ended_with_eot": bool(new_tokens) and new_tokens[-1] == eot_id,
        "num_new_tokens": len(new_tokens),
        "text": tokenizer.decode(new_tokens, skip_special_tokens=False),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="OUTPUT_DIR of verl_ppo/run_ppo.sh")
    parser.add_argument("--sft-ckpt", required=True)
    parser.add_argument("--batch", type=int, required=True, help="Prompts per step (data.train_batch_size)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    log_path = run_dir / "train.log"
    steps = parse_steps(log_path.read_text(errors="replace")) if log_path.exists() else []
    last = steps[-1] if steps else {}
    report: Dict[str, Any] = {
        "steps_logged": len(steps),
        **summarize(steps, args.batch),
        "last_step": {
            key: last.get(key)
            for key in ("critic/score/mean", "critic/rewards/mean", "actor/reward_kl_penalty", "actor/ppo_kl",
                        "response_length/mean", "perf/throughput")
        },
        "metrics_finite": all(math.isfinite(value) for step in steps for value in step.values()),
    }
    try:
        report["reload"] = reload_and_generate(run_dir / "hf", Path(args.sft_ckpt))
    except Exception as exc:  # report, don't abort the summary
        report["reload"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    report["passed"] = bool(steps) and report["metrics_finite"] and report["reload"].get("ok", False)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
