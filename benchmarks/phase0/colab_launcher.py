"""
Kernel side of benchmarks/phase0/colab_phase0.sh: runs inside the Colab VM's
Jupyter kernel (`colab exec -f`), not on this machine.

Unpacks the code tarball the driver uploaded, runs
benchmarks/$BENCH_PHASE/run_$BENCH_PHASE.sh (default phase0) in its own process
group (a dropped `colab exec` connection does not kill it) and streams its
console. The kernel stays busy for the whole run, which keeps the VM
assigned; a heartbeat every minute keeps `colab exec` from timing out while a
step writes only to its own log. At the end <ROOT>/report.tgz holds the run
directory (minus the nsys SQLite exports, which the VM already analyzed).
"""

import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

TARBALL = Path("/content/p0code.tgz")
CODE_DIR = Path("/content/p0repo")
ROOT = Path("/content/p0")
CONSOLE = ROOT / "console.log"
REPORT = ROOT / "report.tgz"
HEARTBEAT_SECONDS = 60
POLL_SECONDS = 5


def say(message: str) -> None:
    print(f"[colab-launcher {time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    shutil.rmtree(CODE_DIR, ignore_errors=True)
    CODE_DIR.mkdir(parents=True)
    with tarfile.open(TARBALL) as tar:
        tar.extractall(CODE_DIR, filter="data")
    ROOT.mkdir(parents=True, exist_ok=True)
    run_id = os.environ["P0_RUN_ID"]
    phase = os.environ.get("BENCH_PHASE", "phase0")
    script = CODE_DIR / f"benchmarks/{phase}/run_{phase}.sh"
    env = dict(os.environ)
    # The kernel's MPLBACKEND (matplotlib_inline) does not exist in the venvs and
    # makes any matplotlib import fail (vLLM imports it); /env/python is Colab's.
    env.pop("PYTHONPATH", None)
    env.update(
        MPLBACKEND="Agg",
        P0_ROOT=str(ROOT),
        P0_REPO_DIR=str(CODE_DIR),
        P0_IDLE_STOP="0",
    )
    env.setdefault("P0_CONCURRENCY", "1 2 4 8 16 32 64 128")
    with open(CONSOLE, "wb") as console:
        process = subprocess.Popen(
            ["bash", str(script)],
            env=env,
            stdout=console,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    say(f"{script.name} pid {process.pid}, run {run_id}, console {CONSOLE}")
    started = last_output = time.time()
    with open(CONSOLE, "rb") as console:
        while True:
            finished = process.poll() is not None
            chunk = console.read()
            if chunk:
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
                last_output = time.time()
            elif time.time() - last_output >= HEARTBEAT_SECONDS:
                say(f"still running ({int(time.time() - started)}s)")
                last_output = time.time()
            if finished and not chunk:
                break
            time.sleep(POLL_SECONDS)
    say(f"{script.name} exit {process.returncode}")

    run_dir = ROOT / "results" / run_id
    with tarfile.open(REPORT, "w:gz") as tar:
        tar.add(run_dir, arcname=run_id, filter=lambda t: None if t.name.endswith(".sqlite") else t)
        tar.add(CONSOLE, arcname=f"{run_id}/console.log")
    say(f"report {REPORT} ({REPORT.stat().st_size / 1e6:.0f} MB)")


# The Colab kernel runs `colab exec -f` code as __main__ too.
if __name__ == "__main__":
    main()
