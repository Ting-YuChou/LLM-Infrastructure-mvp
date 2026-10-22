"""
Kernel side of scripts/colab_gpu_smoke.sh: runs inside the Colab VM's Jupyter
kernel (`colab exec -f`), not on this machine.

Unpacks the repo tarball the driver uploaded, creates a Python 3.11 venv with
uv (Colab's own Python is 3.12, and the training pins such as pandas 2.0.3 have
no 3.12 wheels), then runs scripts/runpod_gpu_smoke.sh with SMOKE_IDLE_STOP=0
and streams its output. The kernel stays busy for the whole run, which is what
keeps a Colab VM assigned, and a heartbeat line every minute keeps `colab exec`
from timing out while a stage writes only to its own log. The smoke script runs
in its own process group, so a dropped `colab exec` connection or a kernel
interrupt does not kill it.

When it finishes, <SMOKE_ROOT>/report.tgz holds every run's summary, timings,
configs and logs (no checkpoints or data) for the driver to download.
"""

import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

TARBALL = Path(os.environ.get("COLAB_REPO_TARBALL", "/content/repo.tgz"))
# KEY=VALUE lines (HF tokens) the driver uploads instead of `colab exec --env`.
SECRETS = Path("/content/.smoke_secrets")
REPO_DIR = Path("/content/repo")
VENV_DIR = Path("/content/venv")
SMOKE_ROOT = Path(os.environ.setdefault("SMOKE_ROOT", "/content/smoke"))
DRIVER_LOG = SMOKE_ROOT / "driver.log"
REPORT = SMOKE_ROOT / "report.tgz"
# Run-directory entries left out of the report: checkpoints and datasets.
REPORT_EXCLUDE = {"outputs", "data", "verl_data"}
HEARTBEAT_SECONDS = 60
POLL_SECONDS = 5


def say(message: str) -> None:
    print(f"[colab-launcher {time.strftime('%H:%M:%S')}] {message}", flush=True)


def run(cmd: list) -> None:
    say("+ " + " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def unpack_repo() -> None:
    shutil.rmtree(REPO_DIR, ignore_errors=True)
    REPO_DIR.mkdir(parents=True)
    with tarfile.open(TARBALL) as tar:
        tar.extractall(REPO_DIR, filter="data")
    say(f"unpacked {TARBALL} -> {REPO_DIR}")


def load_secrets() -> None:
    if not SECRETS.exists():
        return
    for line in SECRETS.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep:
            os.environ[key] = value
    SECRETS.unlink()
    say("loaded the uploaded tokens")


def make_venv() -> None:
    try:
        import uv  # noqa: F401
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    if not (VENV_DIR / "bin" / "python").exists():
        # --seed: the smoke script installs with `python -m pip`.
        run([sys.executable, "-m", "uv", "venv", "--seed", "--python", "3.11", str(VENV_DIR)])


def smoke_env() -> dict:
    env = dict(os.environ)
    env["PATH"] = f"{VENV_DIR / 'bin'}:{env.get('PATH', '')}"
    env["VIRTUAL_ENV"] = str(VENV_DIR)
    env["SMOKE_IDLE_STOP"] = "0"
    return env


def stream_smoke() -> int:
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    with open(DRIVER_LOG, "wb") as log:
        process = subprocess.Popen(
            ["bash", "scripts/runpod_gpu_smoke.sh"],
            cwd=REPO_DIR,
            env=smoke_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    say(f"smoke pid {process.pid}, log {DRIVER_LOG}")
    started = last_output = time.time()
    with open(DRIVER_LOG, "rb") as log:
        while True:
            finished = process.poll() is not None
            chunk = log.read()
            if chunk:
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
                last_output = time.time()
            elif time.time() - last_output >= HEARTBEAT_SECONDS:
                say(f"still running ({int(time.time() - started)}s)")
                last_output = time.time()
            if finished and not chunk:
                return process.returncode
            time.sleep(POLL_SECONDS)


def write_report() -> None:
    with tarfile.open(REPORT, "w:gz") as tar:
        for run_dir in sorted((SMOKE_ROOT / "runs").glob("*")):
            for entry in sorted(run_dir.iterdir()):
                if entry.name not in REPORT_EXCLUDE:
                    tar.add(entry, arcname=f"{run_dir.name}/{entry.name}")
        tar.add(DRIVER_LOG, arcname=DRIVER_LOG.name)
    say(f"report {REPORT}")


def main() -> None:
    load_secrets()
    unpack_repo()
    make_venv()
    rc = stream_smoke()
    say(f"smoke exit {rc}")
    write_report()


# The Colab kernel runs `colab exec -f` code as __main__ too.
if __name__ == "__main__":
    main()
