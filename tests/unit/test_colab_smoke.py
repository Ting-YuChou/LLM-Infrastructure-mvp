"""Colab smoke driver: the kernel-side launcher and the driver script's shell syntax."""

import importlib.util
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("colab_smoke_launcher", ROOT / "scripts" / "colab_smoke_launcher.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    smoke_root = tmp_path / "smoke"
    monkeypatch.setattr(module, "SMOKE_ROOT", smoke_root)
    monkeypatch.setattr(module, "DRIVER_LOG", smoke_root / "driver.log")
    monkeypatch.setattr(module, "REPORT", smoke_root / "report.tgz")
    monkeypatch.setattr(module, "REPO_DIR", tmp_path / "repo")
    monkeypatch.setattr(module, "VENV_DIR", tmp_path / "venv")
    monkeypatch.setattr(module, "POLL_SECONDS", 0.05)
    return module


def test_smoke_env_puts_the_venv_first_and_exits_when_done(launcher):
    env = launcher.smoke_env()
    assert env["PATH"].startswith(f"{launcher.VENV_DIR / 'bin'}:")
    assert env["VIRTUAL_ENV"] == str(launcher.VENV_DIR)
    assert env["SMOKE_IDLE_STOP"] == "0"


def test_stream_smoke_logs_output_and_returns_the_exit_code(launcher, capsys):
    script = launcher.REPO_DIR / "scripts" / "runpod_gpu_smoke.sh"
    script.parent.mkdir(parents=True)
    script.write_text('echo "stage sft exit 0"; echo "$SMOKE_IDLE_STOP"; exit 3\n')

    assert launcher.stream_smoke() == 3
    assert launcher.DRIVER_LOG.read_text() == "stage sft exit 0\n0\n"
    assert "stage sft exit 0" in capsys.readouterr().out


def test_report_keeps_logs_and_summaries_but_no_checkpoints_or_data(launcher):
    run_dir = launcher.SMOKE_ROOT / "runs" / "abc123-1b"
    for name in ("logs/sft.log", "summary.json", "timings.tsv", "configs/sft.yaml",
                 "outputs/sft/final/model.safetensors", "data/sft/train.json"):
        (run_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (run_dir / name).write_text("x")
    launcher.DRIVER_LOG.write_text("log")

    launcher.write_report()

    with tarfile.open(launcher.REPORT) as tar:
        names = set(tar.getnames())
    assert {"driver.log", "abc123-1b/summary.json", "abc123-1b/timings.tsv",
            "abc123-1b/logs/sft.log", "abc123-1b/configs/sft.yaml"} <= names
    assert not any("/outputs" in name or "/data" in name for name in names)


@pytest.mark.parametrize("script", ["colab_gpu_smoke.sh", "runpod_gpu_smoke.sh"])
def test_shell_scripts_parse(script):
    subprocess.run(["bash", "-n", str(ROOT / "scripts" / script)], check=True)
