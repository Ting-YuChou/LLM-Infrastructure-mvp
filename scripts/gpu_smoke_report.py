#!/usr/bin/env python3
"""
Check and summarize one GPU smoke run (see scripts/runpod_gpu_smoke.sh).

For every stage in <run>/timings.tsv it checks the exit code, stage_result.json,
the final checkpoint files, FSDP-sharded intermediate checkpoints, finite
training losses and the peak GPU memory from <run>/logs/gpu_<stage>.csv, then
reloads the final checkpoint on one GPU and runs it once (generate for causal
LMs, a chosen/rejected score for the reward model). Writes <run>/summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from training.pipeline import validate_checkpoint_artifact  # noqa: E402

FINAL_DIR = {"sft": "final", "sft_resume": "final", "reward": "final", "dpo": "final", "rlhf": "checkpoint-final"}
PROMPT = "Explain what machine learning is in simple terms."
# bf16 policy (FSDP) vs bf16 reference (plain model): kernel-level noise only.
DPO_LN2_TOLERANCE = 0.015


def read_timings(run_dir: Path) -> List[Dict[str, Any]]:
    """One row per stage; a stage rerun into the same run (SMOKE_RUN_ID) keeps its latest row."""
    rows: Dict[str, Dict[str, Any]] = {}
    path = run_dir / "timings.tsv"
    if path.exists():
        for line in path.read_text().splitlines():
            stage, code, start, end = line.split("\t")
            rows[stage] = {"stage": stage, "exit_code": int(code), "seconds": int(end) - int(start)}
    return list(rows.values())


def stage_output_dirs(run_dir: Path) -> Dict[str, Path]:
    pipeline = yaml.safe_load((run_dir / "configs" / "training_pipeline.yaml").read_text())
    dirs = {name: Path(stage["output_dir"]) for name, stage in pipeline["stages"].items()}
    resume = yaml.safe_load((run_dir / "configs" / "sft_resume.yaml").read_text())
    dirs["sft_resume"] = Path(resume["training"]["output_dir"])
    return dirs


def peak_memory_mib(run_dir: Path, stage: str) -> Dict[str, int]:
    """Max memory.used per GPU index from `nvidia-smi --query-gpu=index,memory.used`."""
    path = run_dir / "logs" / f"gpu_{stage}.csv"
    peaks: Dict[str, int] = {}
    if path.exists():
        for row in csv.reader(path.read_text().splitlines()):
            if len(row) < 2 or not row[1].strip().isdigit():
                continue
            index, used = row[0].strip(), int(row[1].strip())
            peaks[index] = max(peaks.get(index, 0), used)
    return peaks


def training_losses(output_dir: Path) -> List[float]:
    """Training losses logged by the HF Trainer (trainer_state.json of the latest checkpoint)."""
    states = sorted(
        output_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda path: int(path.parent.name.split("-")[-1]) if path.parent.name.split("-")[-1].isdigit() else -1,
    )
    if not states:
        return []
    history = json.loads(states[-1].read_text()).get("log_history", [])
    return [float(entry["loss"]) for entry in history if "loss" in entry]


def sharded_checkpoints(output_dir: Path) -> List[str]:
    """Intermediate checkpoints written as FSDP SHARDED_STATE_DICT (accelerate layout)."""
    return sorted(
        path.name
        for path in output_dir.glob("checkpoint-*")
        if any(child.name.startswith("pytorch_model_fsdp") for child in path.iterdir())
    )


def checkpoint_record_path(run_dir: Path, stage: str) -> Path:
    return run_dir / "logs" / f"checkpoints_{stage}.json"


def prune_stage(run_dir: Path, stage: str) -> None:
    """
    Record a finished stage's intermediate checkpoints, then delete them.

    An 8B sharded checkpoint with optimizer state is ~80 GB, so keeping every
    stage's would fill the pod disk; the report reads the record instead.
    """
    output_dir = stage_output_dirs(run_dir)[stage]
    if not output_dir.is_dir():
        return
    record = {"sharded": sharded_checkpoints(output_dir), "losses": training_losses(output_dir)}
    checkpoint_record_path(run_dir, stage).write_text(json.dumps(record))
    for path in output_dir.glob("checkpoint-*"):
        if path.name != FINAL_DIR[stage]:  # PPO's final model is checkpoint-final
            shutil.rmtree(path)
    # The verl rlhf backend's actor and critic FSDP checkpoints (with optimizer
    # state); its merged actor is already checkpoint-final.
    shutil.rmtree(output_dir / "verl" / "checkpoints", ignore_errors=True)
    print(f"pruned {stage}: {record['sharded'] or 'no'} intermediate checkpoints")


def run_checkpoint(stage: str, checkpoint: Path) -> Dict[str, Any]:
    """Reload the final checkpoint on one device and run it once."""
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    from training.chat_format import encode, encode_prompt_and_completion, format_prompt

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if stage == "reward":
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
        scores = []
        for response in ("Machine learning lets computers learn patterns from examples.", "No."):
            prompt_ids, response_ids = encode_prompt_and_completion(tokenizer, PROMPT, response)
            input_ids = torch.tensor([prompt_ids + response_ids], device=device)
            with torch.no_grad():
                scores.append(float(model(input_ids=input_ids).logits.squeeze()))
        return {"ok": all(math.isfinite(score) for score in scores), "scores": scores}

    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
    input_ids = torch.tensor([encode(tokenizer, format_prompt(tokenizer, PROMPT))], device=device)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=48,
            do_sample=False,
        )
    new_tokens = output[0, input_ids.shape[1]:].tolist()
    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    return {
        "ok": True,
        "generation_config_stops_on_eot": eot_id in eos_ids,
        "ended_with_eot": bool(new_tokens) and new_tokens[-1] == eot_id,
        "num_new_tokens": len(new_tokens),
        "text": tokenizer.decode(new_tokens, skip_special_tokens=False),
    }


def check_stage(run_dir: Path, timing: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    stage = timing["stage"]
    final = output_dir / FINAL_DIR[stage]
    validation = validate_checkpoint_artifact(final)
    record_path = checkpoint_record_path(run_dir, stage)
    if record_path.exists():  # intermediate checkpoints were pruned
        record = json.loads(record_path.read_text())
        losses, sharded = record["losses"], record["sharded"]
    else:
        losses = training_losses(output_dir)
        sharded = sharded_checkpoints(output_dir) if output_dir.is_dir() else []
    report: Dict[str, Any] = {
        **timing,
        "output_dir": str(output_dir),
        "stage_result_json": (output_dir / "stage_result.json").exists(),
        "final_checkpoint": {key: validation[key] for key in ("exists", "config_json", "tokenizer", "weights")},
        "generation_config_json": (final / "generation_config.json").exists(),
        "sharded_intermediate_checkpoints": sharded,
        "losses_first_last": [losses[0], losses[-1]] if losses else [],
        "losses_finite": all(math.isfinite(loss) for loss in losses),
        "peak_gpu_memory_mib": peak_memory_mib(run_dir, stage),
    }
    if stage == "dpo" and losses:
        # Step 1 runs at lr=0 (warmup), so the policy still equals its
        # reference and the loss is ln 2 unless the two forwards disagree
        # numerically (e.g. FSDP casting RoPE buffers to bf16).
        report["initial_loss_minus_ln2"] = losses[0] - math.log(2)
    if stage == "rlhf" and (output_dir / "stage_result.json").exists():
        report["metrics"] = json.loads((output_dir / "stage_result.json").read_text()).get("metrics", {})
    reload: Optional[Dict[str, Any]] = None
    if timing["exit_code"] == 0 and validation["exists"]:
        try:
            reload = run_checkpoint(stage, final)
        except Exception as exc:  # report, don't abort the summary
            reload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    report["reload"] = reload

    checks = [
        timing["exit_code"] == 0,
        validation["ready_for_awq_conversion"],
        report["losses_finite"],
        bool(reload and reload.get("ok")),
    ]
    if stage in ("sft", "reward", "dpo"):
        checks.append(report["stage_result_json"])
    if stage in ("sft", "reward", "dpo", "sft_resume"):
        checks.append(bool(losses))
    if stage != "reward":
        checks.append(report["generation_config_json"])
    if "initial_loss_minus_ln2" in report:
        checks.append(abs(report["initial_loss_minus_ln2"]) < DPO_LN2_TOLERANCE)
    report["passed"] = all(checks)
    return report


PPO_COMPARISON_KEYS = (
    "timing_gen_s_mean",
    "timing_score_s_mean",
    "timing_update_s_mean",
    "timing_step_s_mean",
    "gen_tokens_per_s",
    "response_length_mean",
    "timing_steps",
)


def verl_stage_report(run_dir: Path, timing: Dict[str, Any]) -> Dict[str, Any]:
    """verl_lora / verl_full: merge verl_ppo/verl_report.py's JSON into the stage row."""
    path = run_dir / "logs" / f"{timing['stage']}.json"
    report = json.loads(path.read_text()) if path.exists() else {"passed": False, "error": f"missing {path}"}
    return {
        **timing,
        **report,
        "peak_gpu_memory_mib": peak_memory_mib(run_dir, timing["stage"]),
        "passed": timing["exit_code"] == 0 and bool(report.get("passed")),
    }


def ppo_comparison(stages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per-step PPO timings of our trainer (rlhf) next to the verl runs, same prompts/lengths."""
    table = {}
    for stage in stages:
        if stage["stage"] in ("rlhf", "verl_lora", "verl_full"):
            source = stage.get("metrics", {}) if stage["stage"] == "rlhf" else stage
            row = {key: source.get(key) for key in PPO_COMPARISON_KEYS}
            row["peak_gpu_memory_mib"] = max((stage.get("peak_gpu_memory_mib") or {}).values(), default=None)
            table[stage["stage"]] = row
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--fsdp", action="store_true", help="Require sharded intermediate checkpoints for FSDP stages")
    parser.add_argument("--prune-stage", help="Record and delete this stage's intermediate checkpoints, then exit")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if args.prune_stage:
        prune_stage(run_dir, args.prune_stage)
        return 0
    output_dirs = stage_output_dirs(run_dir)
    stages = []
    for timing in read_timings(run_dir):
        if timing["stage"] in ("verl_lora", "verl_full"):
            stages.append(verl_stage_report(run_dir, timing))
            continue
        if timing["stage"] not in output_dirs:
            stages.append({**timing, "passed": timing["exit_code"] == 0})
            continue
        report = check_stage(run_dir, timing, output_dirs[timing["stage"]])
        if args.fsdp and timing["stage"] in ("sft", "reward", "dpo") and not report["sharded_intermediate_checkpoints"]:
            report["passed"] = False
        stages.append(report)

    environment = {}
    env_path = run_dir / "environment.json"
    if env_path.exists():
        environment = json.loads(env_path.read_text())
    summary = {
        "run_dir": str(run_dir),
        "environment": environment,
        "stages": stages,
        "passed": bool(stages) and all(stage["passed"] for stage in stages),
    }
    comparison = ppo_comparison(stages)
    if len(comparison) > 1:
        summary["ppo_comparison"] = comparison
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
