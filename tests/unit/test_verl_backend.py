"""stages.rlhf.backend=verl: the pipeline runs verl_ppo/run_ppo.sh and hands on a native-shaped result."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from training import verl_backend  # noqa: E402
from training.pipeline import TrainingPipeline  # noqa: E402

STEP_LOG = "\n".join(
    f"(TaskRunner pid=1) step:{step} - timing_s/gen:{gen} - timing_s/update_actor:2.0 - "
    f"timing_s/update_critic:1.0 - timing_s/update_weights:0.5 - timing_s/step:{gen + 6.5} - "
    f"response_length/mean:100.0 - critic/score/mean:{score} - actor/reward_kl_penalty:{kl} - "
    f"actor/reward_kl_penalty_coeff:0.05"
    for step, gen, score, kl in ((1, 30.0, -2.0, 0.0), (2, 10.0, -1.0, 0.02), (3, 12.0, 0.0, 0.04))
)


def _checkpoint(path: Path, tokenizer: bool = True) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_text("weights")
    if tokenizer:
        (path / "tokenizer.json").write_text('{"sft": true}')
        (path / "tokenizer_config.json").write_text("{}")
        (path / "generation_config.json").write_text('{"eos_token_id": [128009]}')
    return path


def _pipeline_config(tmp_path: Path, backend: str = "verl", **verl) -> Path:
    rlhf = yaml.safe_load((ROOT / "config" / "rlhf_config.yaml").read_text())
    rlhf["tracking"]["mlflow_enabled"] = False
    rlhf["training"].update({"per_device_batch_size": 16, "total_steps": 3})
    rlhf["data"].update({"max_prompt_length": 256, "max_response_length": 512})
    (tmp_path / "rlhf.yaml").write_text(yaml.safe_dump(rlhf))
    (tmp_path / "prompts.json").write_text(json.dumps([f"prompt {i}" for i in range(40)]))
    out = tmp_path / "out"
    config = {
        "output_dir": str(out),
        "stages": {
            "sft": {"config": "config/sft_config.yaml", "output_dir": str(out / "sft")},
            "reward": {"config": "config/reward_config.yaml", "output_dir": str(out / "reward")},
            "rlhf": {
                "config": str(tmp_path / "rlhf.yaml"),
                "output_dir": str(out / "aligned"),
                "prompts_file": str(tmp_path / "prompts.json"),
                "nproc_per_node": 4,
                "backend": backend,
                "verl": {"mode": "lora", "dir": str(tmp_path / "verl"), **verl},
            },
        },
        "promotion": {"metrics_path": str(out / "eval" / "metrics.json"), "gate_config": "config/eval_gate.training.json"},
    }
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_native_is_the_default_backend_and_unknown_backends_fail(tmp_path):
    plan = TrainingPipeline("config/training_pipeline.yaml", dry_run=True).run()["plan"]
    assert plan["rlhf_backend"] == "native"

    with pytest.raises(ValueError, match="backend"):
        TrainingPipeline(str(_pipeline_config(tmp_path, backend="ray")), dry_run=True)


def test_verl_settings_need_a_checkout_and_a_known_mode(monkeypatch):
    monkeypatch.delenv("VERL_DIR", raising=False)
    with pytest.raises(ValueError, match="VERL_DIR"):
        verl_backend.verl_settings({"verl": {"mode": "full"}}, nproc=4)
    with pytest.raises(ValueError, match="mode"):
        verl_backend.verl_settings({"verl": {"mode": "dpo", "dir": "/v"}}, nproc=4)

    monkeypatch.setenv("VERL_DIR", "/env/verl")
    assert verl_backend.verl_settings({}, nproc=8) == verl_backend.VerlSettings("full", Path("/env/verl"), 8)


def test_verl_backend_refuses_to_run_under_torchrun(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    pipeline = TrainingPipeline(str(_pipeline_config(tmp_path)), stages="rlhf")

    with pytest.raises(ValueError, match="torchrun"):
        pipeline.run()
    with pytest.raises(ValueError, match="torchrun"):
        pipeline.run_gpu_stage("rlhf")


def test_pipeline_rlhf_stage_runs_verl_and_produces_a_native_shaped_result(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    config = _pipeline_config(tmp_path)
    out = tmp_path / "out"
    sft = _checkpoint(out / "sft" / "final")
    _checkpoint(out / "reward" / "final")
    (tmp_path / "verl" / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / "verl" / ".venv" / "bin" / "python").write_text("")
    (out / "eval").mkdir(parents=True)
    (out / "eval" / "metrics.json").write_text(json.dumps({"sft_eval_loss": 1.25}))
    calls = []

    def fake_run(command, check, env):
        calls.append((command, env))
        work = Path(env["OUTPUT_DIR"])
        _checkpoint(work / "hf", tokenizer=False)  # verl's merger output, without our tokenizer
        (work / "train.log").write_text(STEP_LOG)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(verl_backend.subprocess, "run", fake_run)
    results = TrainingPipeline(str(config), stages="rlhf,training_eval").run()

    command, env = calls[0]
    assert command == ["bash", str(ROOT / "verl_ppo" / "run_ppo.sh")]
    assert env["MODE"] == "lora" and env["NGPUS"] == "4" and env["VERL_DIR"] == str(tmp_path / "verl")
    assert env["SFT_CKPT"] == str(sft) and env["RM_CKPT"] == str(out / "reward" / "final")
    # Same prompts per step, lengths and steps the native backend reads from rlhf.yaml.
    assert (env["BATCH"], env["STEPS"], env["PROMPT_LEN"], env["RESPONSE_LEN"]) == ("64", "3", "256", "512")
    assert (Path(env["DATA_DIR"]) / "train.parquet").exists()

    final = out / "aligned" / "checkpoint-final"
    rlhf = results["stages"]["rlhf"]
    assert rlhf["checkpoint_path"] == str(final)
    stage_result = json.loads((out / "aligned" / "stage_result.json").read_text())
    assert stage_result["checkpoint_path"] == str(final)
    assert stage_result["metrics"]["rlhf_mean_reward"] == pytest.approx(-1.0)
    assert (final / "tokenizer.json").read_text() == '{"sft": true}'
    assert (final / "generation_config.json").exists()
    metrics = rlhf["metrics"]
    assert metrics["rlhf_mean_reward"] == pytest.approx(-1.0)
    assert metrics["rlhf_mean_kl"] == pytest.approx(0.02)
    assert metrics["kl_coef"] == 0.05 and metrics["total_steps"] == 3
    assert metrics["timing_gen_s_mean"] == pytest.approx(11.0)

    promotion = json.loads((out / "eval" / "metrics.json").read_text())
    # training_eval ran without sft/reward in this process, so it keeps the earlier SFT metric.
    assert promotion["sft_eval_loss"] == 1.25
    assert promotion["rlhf_mean_reward"] == pytest.approx(-1.0)
    assert promotion["artifact_ready"] == 1.0
