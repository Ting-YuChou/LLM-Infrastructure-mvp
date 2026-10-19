"""
Worker script for the multi-process training tests (run under torchrun, gloo on CPU).

Each rank writes result-rank{N}.json into --tmp; test_training_distributed.py
asserts on those files from the parent process. MLflow is replaced by the
fakes from test_mlflow_lifecycle.py so the tests can count runs per rank.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from test_mlflow_lifecycle import FakeClient, FakeMLflow  # noqa: E402
from tiny_llama import build_tiny_llama, build_tiny_tokenizer  # noqa: E402
from training import distributed  # noqa: E402
from training import mlflow_lifecycle  # noqa: E402

EVENTS = []


def _install_fake_mlflow():
    fake, client = FakeMLflow(), FakeClient()
    start_run, create_version = fake.start_run, client.create_model_version

    def recording_start_run(*args, **kwargs):
        EVENTS.append("start_run")
        return start_run(*args, **kwargs)

    def recording_create_version(*args, **kwargs):
        EVENTS.append("create_model_version")
        return create_version(*args, **kwargs)

    fake.start_run = recording_start_run
    client.create_model_version = recording_create_version
    mlflow_lifecycle.MLflowTrainingLifecycle._require_mlflow = lambda self: fake
    mlflow_lifecycle.MLflowTrainingLifecycle._new_client = lambda self: client


def _use_cpu(module):
    training_arguments_cls = module.TrainingArguments
    module.TrainingArguments = lambda **kwargs: training_arguments_cls(use_cpu=True, **kwargs)


def _write_json(path, rows):
    path.write_text(json.dumps(rows))
    return str(path)


def _raw_config(config_name, model_dir, output_dir, train_file, val_file, strategy):
    raw = yaml.safe_load((ROOT / "config" / config_name).read_text())
    raw["model"].update(name=str(model_dir), precision="fp32", use_flash_attention=False)
    raw["model"].pop("cache_dir", None)
    raw["data"].update(train_file=train_file, val_file=val_file, max_length=24)
    raw["training"].update(
        output_dir=str(output_dir),
        num_train_epochs=1,
        max_steps=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=1,
        eval_steps=1,
        save_steps=1,
        save_total_limit=1,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    raw["tracking"].update(mlflow_enabled=True, tensorboard_enabled=False, wandb_enabled=False)
    raw["distributed"] = {"strategy": strategy, "timeout_minutes": 5}
    return raw


def _prepare_tiny_llama(tmp, causal_lm=True, name="tiny-llama"):
    model_dir = tmp / name
    if distributed.is_main_process():
        tokenizer = build_tiny_tokenizer()
        build_tiny_llama(len(tokenizer), causal_lm=causal_lm).save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    distributed.barrier()
    return model_dir


def run_sft(tmp, use_lora):
    from training import sft_trainer

    _use_cpu(sft_trainer)
    model_dir = _prepare_tiny_llama(tmp)
    rows = [{"prompt": "explain tests", "completion": "they catch regressions"} for _ in range(8)]
    if distributed.is_main_process():
        _write_json(tmp / "train.json", rows)
        _write_json(tmp / "val.json", rows[:4])
    distributed.barrier()
    raw = _raw_config(
        "sft_config.yaml",
        model_dir,
        tmp / "sft",
        str(tmp / "train.json"),
        str(tmp / "val.json"),
        strategy="ddp",
    )
    raw["lora"]["enabled"] = use_lora
    config = sft_trainer.SFTConfig(
        model_name=str(model_dir),
        train_file=str(tmp / "train.json"),
        val_file=str(tmp / "val.json"),
        output_dir=str(tmp / "sft"),
        max_length=24,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=1e-3,
        raw_config=raw,
    )
    result = sft_trainer.SFTTrainer(config, use_lora=use_lora).train()
    return {"checkpoint_path": result.checkpoint_path, "eval_loss": result.metrics.get("eval_loss")}


CASES = {
    "sft_ddp": lambda tmp: run_sft(tmp, use_lora=False),
    "sft_lora_ddp": lambda tmp: run_sft(tmp, use_lora=True),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--tmp", required=True)
    args = parser.parse_args()

    tmp = Path(args.tmp)
    distributed.init_distributed(timeout_minutes=5)
    _install_fake_mlflow()
    result = CASES[args.case](tmp)
    result.update(rank=distributed.rank(), world_size=distributed.world_size(), events=EVENTS)
    (tmp / f"result-rank{distributed.rank()}.json").write_text(json.dumps(result))
    distributed.cleanup_distributed()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
