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


def _prepare_chat_models(tmp):
    """Tiny SFT policy + reward model that already carry the chat-token vocabulary."""
    from training.chat_format import ensure_chat_tokenizer

    policy_dir, reward_dir = tmp / "sft-final", tmp / "reward-final"
    if distributed.is_main_process():
        tokenizer = ensure_chat_tokenizer(build_tiny_tokenizer())
        for model_dir, causal_lm in ((policy_dir, True), (reward_dir, False)):
            build_tiny_llama(len(tokenizer), causal_lm=causal_lm).save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
        _write_json(tmp / "prompts.json", [{"prompt": f"explain tests {i}"} for i in range(8)])
    distributed.barrier()
    return policy_dir, reward_dir


def _checksum(parameters):
    return float(sum(param.detach().double().sum() for param in parameters))


def _merged_logit_diff(trainer, merged_path):
    """Largest logit gap between the trained LoRA policy and the merged checkpoint."""
    import torch
    from transformers import AutoModelForCausalLM

    merged = AutoModelForCausalLM.from_pretrained(merged_path, torch_dtype=torch.float32)
    input_ids = torch.tensor([[1, 5, 6, 7, 8, 9]])
    trainer.policy_model.eval()
    with torch.no_grad():
        expected = trainer.policy_model(input_ids=input_ids).logits
        actual = merged(input_ids=input_ids).logits
    return float((expected - actual).abs().max())


def run_ppo(tmp):
    from training import rlhf_trainer

    policy_dir, reward_dir = _prepare_chat_models(tmp)
    raw = yaml.safe_load((ROOT / "config" / "rlhf_config.yaml").read_text())
    raw["model"]["policy"].update(name=str(policy_dir), precision="fp32", use_flash_attention=False)
    raw["model"]["reward"]["name"] = str(reward_dir)
    raw["lora"].update(r=4, alpha=8)
    raw["distributed"]["timeout_minutes"] = 5
    raw["tracking"].update(mlflow_enabled=True, tensorboard_enabled=False, wandb_enabled=False)
    config = rlhf_trainer.rlhf_config_from_dict(
        raw,
        output_dir=str(tmp / "aligned"),
        num_epochs=1,
        ppo_epochs=2,
        num_steps=2,
        batch_size=2,
        max_response_length=4,
        save_steps=1,
        learning_rate=1e-2,
        value_learning_rate=1e-2,
    )

    kl_means = []
    masked_global_mean = rlhf_trainer.masked_global_mean

    def recording_masked_global_mean(values, mask):
        result = masked_global_mean(values, mask)
        kl_means.append(float(result))
        return result

    rlhf_trainer.masked_global_mean = recording_masked_global_mean
    trainer = rlhf_trainer.RLHFTrainer(config)
    lora_before = _checksum(trainer._trainable_policy_parameters())
    value_before = _checksum(trainer.value_head.parameters())
    result = trainer.train(str(tmp / "prompts.json"))
    merge_max_diff = None
    if distributed.is_main_process():
        merge_max_diff = _merged_logit_diff(trainer, result.checkpoint_path)
    return {
        "merge_max_diff": merge_max_diff,
        "checkpoint_path": result.checkpoint_path,
        "timing": {key: value for key, value in result.metrics.items() if key.startswith(("timing_", "gen_tokens"))},
        "kl_means": kl_means,
        "kl_coef": trainer.kl_coef,
        "lora_before": lora_before,
        "lora_after": _checksum(trainer._trainable_policy_parameters()),
        "value_before": value_before,
        "value_after": _checksum(trainer.value_head.parameters()),
    }


def run_dpo(tmp):
    """DDP DPO with lr=0, plus the sharded reference pass checked against per-row scores."""
    import torch
    from training import dpo_trainer
    from training.chat_format import PaddingCollator

    _use_cpu(dpo_trainer)
    policy_dir, _ = _prepare_chat_models(tmp)
    # 7 pairs: the reference pass splits them 4/3 across the two ranks.
    pairs = [
        {"prompt": f"answer question {i}", "chosen": "good answer", "rejected": "bad " * (i + 1)}
        for i in range(7)
    ]
    if distributed.is_main_process():
        _write_json(tmp / "pairs.json", pairs)
    distributed.barrier()
    raw = _raw_config(
        "dpo_config.yaml",
        policy_dir,
        tmp / "dpo",
        str(tmp / "pairs.json"),
        str(tmp / "pairs.json"),
        strategy="ddp",
    )
    config = dpo_trainer.DPOConfig(
        model_name=str(policy_dir),
        train_file=str(tmp / "pairs.json"),
        val_file=str(tmp / "pairs.json"),
        output_dir=str(tmp / "dpo"),
        max_length=24,
        per_device_train_batch_size=2,
        learning_rate=0.0,
        reference_batch_size=2,
        raw_config=raw,
    )
    trainer = dpo_trainer.DPOModelTrainer(config)
    result = trainer.train()

    tokenizer = trainer.load_tokenizer()
    preprocessor = dpo_trainer.RewardDataPreprocessor(tokenizer, 24, include_labels=True)
    dataset, _ = preprocessor.prepare_datasets(str(tmp / "pairs.json"))
    model = trainer.load_model(tokenizer)
    (scored,) = dpo_trainer.compute_reference_log_probs(
        model, [dataset], 2, torch.device("cpu"), pad_token_id=tokenizer.pad_token_id
    )
    expected = []
    with torch.no_grad():
        for index in range(len(dataset)):
            row = PaddingCollator(tokenizer.pad_token_id)([dataset[index]])
            chosen, rejected = dpo_trainer.concatenated_log_probs(model, row)
            expected += [float(chosen), float(rejected)]
    return {
        "checkpoint_path": result.checkpoint_path,
        "eval_loss": result.metrics.get("eval_loss"),
        # Flat [chosen0, rejected0, chosen1, ...] for pytest.approx.
        "reference": [
            logp for pair in zip(scored["ref_chosen_logps"], scored["ref_rejected_logps"]) for logp in pair
        ],
        "expected_reference": expected,
    }


def run_pipeline_sft(tmp):
    """Plain-python orchestrator that launches the SFT stage as a 2-process torchrun job."""
    from training.pipeline import TrainingPipeline

    model_dir = _prepare_tiny_llama(tmp)
    rows = [{"prompt": "explain tests", "completion": "they catch regressions"} for _ in range(8)]
    raw = _raw_config(
        "sft_config.yaml",
        model_dir,
        tmp / "pipeline" / "models" / "sft",
        _write_json(tmp / "train.json", rows),
        _write_json(tmp / "val.json", rows[:4]),
        strategy="ddp",
    )
    raw["training"]["use_cpu"] = True
    raw["tracking"]["mlflow_enabled"] = False
    (tmp / "sft.yaml").write_text(yaml.safe_dump(raw))
    pipeline_config = {
        "output_dir": str(tmp / "pipeline"),
        "distributed": {"launcher": "torchrun", "nproc_per_node": 2},
        "stages": {
            "sft": {"config": str(tmp / "sft.yaml"), "output_dir": str(tmp / "pipeline" / "models" / "sft")},
            # Only referenced by the execution plan; these stages do not run.
            "reward": {"config": str(ROOT / "config" / "reward_config.yaml")},
            "rlhf": {"config": str(ROOT / "config" / "rlhf_config.yaml")},
        },
    }
    (tmp / "pipeline.yaml").write_text(yaml.safe_dump(pipeline_config))

    result = TrainingPipeline(str(tmp / "pipeline.yaml"), stages="sft").run()
    return {"sft": result["stages"]["sft"]}


CASES = {
    "pipeline_sft": run_pipeline_sft,
    "sft_ddp": lambda tmp: run_sft(tmp, use_lora=False),
    "sft_lora_ddp": lambda tmp: run_sft(tmp, use_lora=True),
    "ppo_lora_ddp": run_ppo,
    "dpo_ddp": run_dpo,
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
