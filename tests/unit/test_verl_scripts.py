"""verl_ppo/: prompt conversion, launch overrides and report parsing (no verl install needed)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "verl_ppo"))

import prepare_prompts  # noqa: E402
import verl_report  # noqa: E402


def test_load_prompts_accepts_objects_and_bare_strings(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps([{"prompt": "explain tests", "id": 0}, "write code"]))

    assert prepare_prompts.load_prompts(path) == ["explain tests", "write code"]


def test_write_parquet_uses_verl_rl_schema(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    paths = prepare_prompts.write_parquet([f"prompt {i}" for i in range(5)], tmp_path, val_count=2)

    train, val = pd.read_parquet(paths["train"]), pd.read_parquet(paths["val"])
    assert len(train) == 3 and len(val) == 2
    assert set(train.columns) == {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    row = train.iloc[0]
    assert list(row["prompt"]) == [{"role": "user", "content": "prompt 0"}]
    assert row["data_source"] == prepare_prompts.DATA_SOURCE
    assert row["reward_model"]["style"] == "model"
    assert val.iloc[0]["extra_info"]["split"] == "val"


def test_write_parquet_rejects_val_count_that_leaves_no_train(tmp_path):
    with pytest.raises(ValueError):
        prepare_prompts.write_parquet(["only one"], tmp_path, val_count=1)


def _dry_run(mode, **env):
    completed = subprocess.run(
        ["bash", str(ROOT / "verl_ppo" / "run_ppo.sh")],
        env={
            **os.environ,
            "MODE": mode,
            "SFT_CKPT": "/ckpt/sft",
            "RM_CKPT": "/ckpt/rm",
            "DATA_DIR": "/data",
            "OUTPUT_DIR": "/out",
            "VERL_DIR": "/verl",
            "NGPUS": "4",
            "VERL_DRY_RUN": "1",
            **env,
        },
        capture_output=True,
        text=True,
    )
    return completed


def _overrides(mode, **env):
    completed = _dry_run(mode, **env)
    assert completed.returncode == 0, completed.stderr
    train, merge = completed.stdout.split("---\n")
    return dict(line.split("=", 1) for line in train.splitlines() if "=" in line), merge.splitlines()


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_run_ppo_points_actor_at_sft_and_reward_and_critic_at_rm(mode):
    overrides, merge = _overrides(mode, BATCH="64", STEPS="6", RESPONSE_LEN="512")

    assert overrides["actor_rollout_ref.model.path"] == "/ckpt/sft"
    assert overrides["critic.model.path"] == "/ckpt/rm"
    assert overrides["reward.reward_model.enable"] == "True"
    assert overrides["reward.reward_model.model_path"] == "/ckpt/rm"
    assert overrides["reward.reward_model.n_gpus_per_node"] == "4"
    assert overrides["trainer.n_gpus_per_node"] == "4"
    assert overrides["algorithm.adv_estimator"] == "gae"
    assert overrides["algorithm.use_kl_in_reward"] == "True"
    # Same prompts per step, lengths and steps as the rlhf stage it is compared with.
    assert overrides["data.train_batch_size"] == "64"
    assert overrides["actor_rollout_ref.actor.ppo_mini_batch_size"] == "64"
    assert overrides["data.max_response_length"] == "512"
    assert overrides["trainer.total_training_steps"] == "6"
    assert overrides["data.train_files"] == "/data/train.parquet"
    assert "/out/checkpoints/global_step_6/actor" in merge


def test_run_ppo_lora_matches_rlhf_config_adapter():
    overrides, _ = _overrides("lora")

    assert overrides["actor_rollout_ref.model.lora_rank"] == "16"
    assert overrides["actor_rollout_ref.model.lora_alpha"] == "32"
    assert overrides["actor_rollout_ref.model.target_modules"] == "all-linear"
    # verl's LoRA rollout needs vLLM to load the real base weights.
    assert overrides["actor_rollout_ref.rollout.load_format"] == "safetensors"
    # V1 trainer bug: the critic inherits the reference pass's no_lora_adapter flag.
    assert overrides["trainer.use_v1"] == "False"
    assert overrides["actor_rollout_ref.rollout.layered_summon"] == "True"
    # The critic trains all parameters in both modes, so it is offloaded in both.
    assert overrides["critic.fsdp.optimizer_offload"] == "True"


def test_run_ppo_full_trains_every_parameter_with_offload():
    overrides, _ = _overrides("full")

    assert "actor_rollout_ref.model.lora_rank" not in overrides
    assert overrides["actor_rollout_ref.actor.fsdp_config.param_offload"] == "True"
    assert overrides["critic.fsdp.optimizer_offload"] == "True"


def test_run_ppo_runs_the_venv_python_not_uv_run():
    completed = _dry_run("lora")
    train, merge = completed.stdout.split("---\n")

    # Under `uv run`, Ray launches its workers through `uv run` as well.
    assert train.splitlines()[:3] == ["/verl/.venv/bin/python", "-m", "verl.trainer.main_ppo"]
    assert merge.splitlines()[0] == "/verl/.venv/bin/python"
    assert "uv" not in completed.stdout.split()


def test_run_ppo_rejects_unknown_mode():
    assert _dry_run("dpo").returncode == 2


def test_verl_report_summarizes_console_metrics_like_rlhf_trainer():
    log = "\n".join(
        f"(TaskRunner pid=1) step:{step} - timing_s/gen:{gen} - timing_s/update_actor:2.0 - "
        f"timing_s/update_critic:1.0 - timing_s/update_weights:0.5 - timing_s/step:{gen + 6.5} - "
        f"response_length/mean:100.0 - critic/score/mean:-0.25"
        for step, gen in ((1, 30.0), (2, 10.0), (3, 12.0))
    )

    steps = verl_report.parse_steps(log)
    summary = verl_report.summarize(steps, batch=64)

    assert [step["timing_s/gen"] for step in steps] == [30.0, 10.0, 12.0]
    # The warm-up step is dropped, as in rlhf_trainer.summarize_step_timings.
    assert summary["timing_steps"] == 2.0
    assert summary["timing_gen_s_mean"] == pytest.approx(11.0)
    assert summary["timing_update_s_mean"] == pytest.approx(3.5)
    assert summary["timing_score_s_mean"] == pytest.approx(3.0)
    assert summary["gen_tokens_per_s"] == pytest.approx(2 * 100 * 64 / 22.0)


def test_verl_report_leaves_checkpoint_saves_out_of_step_time():
    steps = [
        {"timing_s/gen": 10.0, "timing_s/step": 20.0},
        {"timing_s/gen": 10.0, "timing_s/step": 20.0},
        {"timing_s/gen": 10.0, "timing_s/step": 190.0, "timing_s/save_checkpoint": 170.0},
    ]

    summary = verl_report.summarize(steps, batch=1)

    assert summary["timing_step_s_mean"] == pytest.approx(20.0)
    assert summary["timing_score_s_mean"] == pytest.approx(10.0)


def test_smoke_report_puts_rlhf_and_verl_timings_side_by_side(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    import gpu_smoke_report

    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "verl_lora.json").write_text(json.dumps({"passed": True, "timing_gen_s_mean": 4.0}))
    (tmp_path / "logs" / "gpu_verl_lora.csv").write_text("0, 50000\n1, 61000\n")
    verl = gpu_smoke_report.verl_stage_report(tmp_path, {"stage": "verl_lora", "exit_code": 0, "seconds": 900})
    rlhf = {"stage": "rlhf", "metrics": {"timing_gen_s_mean": 40.0}, "peak_gpu_memory_mib": {"0": 34000}}

    table = gpu_smoke_report.ppo_comparison([{"stage": "sft"}, rlhf, verl])

    assert verl["passed"] is True
    assert set(table) == {"rlhf", "verl_lora"}
    assert table["rlhf"]["timing_gen_s_mean"] == 40.0
    assert table["verl_lora"]["timing_gen_s_mean"] == 4.0
    assert table["verl_lora"]["peak_gpu_memory_mib"] == 61000
