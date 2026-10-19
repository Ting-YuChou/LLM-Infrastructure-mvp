"""
Distributed helpers and two-process (gloo, CPU) training runs under torchrun.

The multi-process cases launch tests/unit/distributed_workers.py in fresh
interpreters, so they use the real transformers/datasets/peft regardless of
the stubs other test modules install.
"""

import importlib.util
import json
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
    # A static rendezvous on loopback: --standalone resolves the host name,
    # which hangs on some macOS setups.
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        "--master_addr=127.0.0.1",
        f"--master_port={_free_port()}",
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
