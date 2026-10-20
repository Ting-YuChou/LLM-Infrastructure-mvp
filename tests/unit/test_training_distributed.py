"""
Distributed helpers and two-process (gloo, CPU) training runs under torchrun.

The multi-process cases launch tests/unit/distributed_workers.py in fresh
interpreters, so they use the real transformers/datasets/peft regardless of
the stubs other test modules install.
"""

import importlib.util
import json
import math
import socket
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from training import distributed  # noqa: E402
from training.mlflow_lifecycle import MLflowTrainingLifecycle  # noqa: E402

WORKER = Path(__file__).parent / "distributed_workers.py"


@pytest.fixture
def single_process(monkeypatch):
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)


def test_helpers_default_to_a_single_process(single_process):
    assert distributed.rank() == 0
    assert distributed.world_size() == 1
    assert distributed.is_main_process()
    assert not distributed.is_distributed()
    distributed.barrier()
    with distributed.main_process_first():
        pass
    assert distributed.init_distributed().type in {"cpu", "cuda"}


def test_main_process_only_skips_other_ranks(monkeypatch):
    calls = []
    record = distributed.main_process_only(lambda value: calls.append(value) or value)

    monkeypatch.setenv("RANK", "1")
    assert record("rank1") is None
    monkeypatch.setenv("RANK", "0")
    assert record("rank0") == "rank0"
    assert calls == ["rank0"]


def test_single_process_reductions_are_local(single_process):
    values = torch.tensor([[1.0, 3.0, 100.0]])
    mask = torch.tensor([[1, 1, 0]])

    assert distributed.masked_global_mean(values, mask).item() == pytest.approx(2.0)
    whitened = distributed.masked_global_whiten(values, mask)
    assert whitened.tolist()[0][2] == 0.0
    assert whitened[mask.bool()].mean().item() == pytest.approx(0.0, abs=1e-6)
    assert distributed.reduce_scalar_metrics({"kl": 0.5}) == {"kl": 0.5}


@pytest.mark.parametrize("rank,enabled", [("0", True), ("1", False)])
def test_mlflow_lifecycle_only_runs_on_rank_zero(monkeypatch, rank, enabled):
    monkeypatch.setenv("RANK", rank)
    monkeypatch.setenv("WORLD_SIZE", "2")

    lifecycle = MLflowTrainingLifecycle.from_training_config(
        {"tracking": {"mlflow_enabled": True}}, model_type="sft"
    )

    assert lifecycle.enabled is enabled
    assert lifecycle.config.tags["world_size"] == "2"


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_workers(case, tmp_path, nproc=2):
    for required in ("transformers", "datasets", "tokenizers"):
        if importlib.util.find_spec(required) is None:
            pytest.skip(f"{required} is not installed")
    if nproc == 0:  # plain python, no torchrun
        launcher = [sys.executable]
        nproc = 1
    else:
        # A static rendezvous on loopback: --standalone resolves the host name,
        # which hangs on some macOS setups.
        launcher = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            f"--nproc_per_node={nproc}",
            "--master_addr=127.0.0.1",
            f"--master_port={_free_port()}",
        ]
    command = launcher + [
        str(WORKER),
        "--case",
        case,
        "--tmp",
        str(tmp_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
    return [
        json.loads((tmp_path / f"result-rank{rank}.json").read_text())
        for rank in range(nproc)
    ]


@pytest.mark.parametrize("case,weights", [("sft_ddp", "model.safetensors"), ("sft_lora_ddp", "adapter_model.safetensors")])
def test_sft_under_torchrun_logs_and_saves_once(tmp_path, case, weights):
    if case == "sft_lora_ddp" and importlib.util.find_spec("peft") is None:
        pytest.skip("peft is not installed")

    rank0, rank1 = _run_workers(case, tmp_path)

    assert rank0["world_size"] == rank1["world_size"] == 2
    assert rank0["events"] == ["start_run", "create_model_version"]
    assert rank1["events"] == []
    assert rank0["eval_loss"] == pytest.approx(rank1["eval_loss"])
    final = Path(rank0["checkpoint_path"])
    assert (final / weights).exists()
    assert (final / "generation_config.json").exists()
    assert (final / "tokenizer_config.json").exists()


def test_ppo_lora_ddp_keeps_ranks_in_sync_and_merges_final_model(tmp_path):
    if importlib.util.find_spec("peft") is None:
        pytest.skip("peft is not installed")

    rank0, rank1 = _run_workers("ppo_lora_ddp", tmp_path)

    # LoRA B starts at zero, so the adapter-disabled reference equals the policy.
    assert rank0["kl_means"][0] == pytest.approx(0.0, abs=1e-6)
    # Globally reduced KL drives one kl_coef on every rank.
    assert rank0["kl_means"] == pytest.approx(rank1["kl_means"])
    assert rank0["kl_coef"] == pytest.approx(rank1["kl_coef"])
    # DDP keeps adapter and value head identical while they train.
    assert rank0["lora_after"] == pytest.approx(rank1["lora_after"])
    assert rank0["value_after"] == pytest.approx(rank1["value_after"])
    assert rank0["lora_after"] != pytest.approx(rank0["lora_before"])
    assert rank0["value_after"] != pytest.approx(rank0["value_before"])
    assert rank0["events"] == ["start_run", "create_model_version"]
    assert rank1["events"] == []

    output_dir = tmp_path / "aligned"
    final = output_dir / "checkpoint-final"
    assert (final / "config.json").exists()
    assert (final / "model.safetensors").exists()
    assert (final / "value_head.pt").exists()
    assert (output_dir / "checkpoint-final-adapter" / "adapter_config.json").exists()
    assert (output_dir / "checkpoint-1" / "optimizer.pt").exists()
    assert rank0["merge_max_diff"] < 1e-4


def test_dpo_ddp_merges_sharded_reference_pass(tmp_path):
    rank0, rank1 = _run_workers("dpo_ddp", tmp_path)

    # Rows scored on different ranks land back at their own index on every rank.
    assert rank0["reference"] == pytest.approx(rank0["expected_reference"], abs=1e-5)
    assert rank1["reference"] == pytest.approx(rank0["reference"])
    # lr=0: the policy stays equal to its reference, so the DPO loss is ln 2.
    assert rank0["eval_loss"] == pytest.approx(math.log(2), abs=1e-4)
    assert rank1["eval_loss"] == pytest.approx(rank0["eval_loss"])
    assert rank0["events"] == ["start_run", "create_model_version"]
    assert rank1["events"] == []
    final = Path(rank0["checkpoint_path"])
    assert (final / "model.safetensors").exists()
    assert (final / "generation_config.json").exists()


def test_pipeline_launches_each_training_stage_as_a_torchrun_job(tmp_path):
    (result,) = _run_workers("pipeline_sft", tmp_path, nproc=0)

    sft = result["sft"]
    final = Path(sft["checkpoint_path"])
    assert (final / "model.safetensors").exists()
    assert "eval_loss" in sft["metrics"]
    # Written by rank 0 of the torchrun job and read back by the orchestrator.
    stage_result = json.loads((tmp_path / "pipeline" / "models" / "sft" / "stage_result.json").read_text())
    assert stage_result["checkpoint_path"] == sft["checkpoint_path"]
