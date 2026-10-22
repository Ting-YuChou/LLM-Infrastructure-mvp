#!/usr/bin/env bash
# GPU smoke test of the training stages on one Google Colab GPU, driven from
# this machine with the Colab CLI (`uv tool install google-colab-cli`; the
# first command asks you to sign in once in a browser).
#
# Colab runtimes have a single GPU, so this runs the same
# scripts/runpod_gpu_smoke.sh with one process per stage. The 1b preset fits
# one A100/H100; the 8b presets need 4x 80 GB and stay on RunPod.
#
#   bash scripts/colab_gpu_smoke.sh            # new VM, upload, run, fetch the report, stop the VM
#   bash scripts/colab_gpu_smoke.sh tail       # last lines of the running smoke log
#   bash scripts/colab_gpu_smoke.sh fetch      # download <SMOKE_ROOT>/report.tgz again
#   bash scripts/colab_gpu_smoke.sh stop       # release the VM
#
# The repo is uploaded from this checkout (tracked and untracked, not ignored,
# files plus .git), so it does not need to be pushed. The VM's /content disk
# is gone once the VM is released; the report (summaries, timings, logs) lands
# in COLAB_OUT_DIR.
#
# Env: COLAB_SESSION (default llm-smoke), COLAB_GPU (A100 | H100 | L4 | T4,
# default A100), COLAB_HIGH_MEM=1, COLAB_KEEP=1 (do not stop the VM at the
# end), COLAB_OUT_DIR (default outputs/colab_smoke/<session>), COLAB_TIMEOUT
# (seconds without output before `colab exec` gives up, default 900; the run
# itself goes on and `tail` / `fetch` still work). SMOKE_* (see
# runpod_gpu_smoke.sh), HF_TOKEN, HF_UPLOAD_REPO_PREFIX and HF_WRITE_TOKEN
# are forwarded to the VM when set. SMOKE_MODEL defaults to the ungated
# unsloth/Llama-3.2-1B copy of meta-llama/Llama-3.2-1B when HF_TOKEN is unset.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${COLAB_SESSION:-llm-smoke}"
GPU="${COLAB_GPU:-A100}"
OUT_DIR="${COLAB_OUT_DIR:-${REPO_DIR}/outputs/colab_smoke/${SESSION}}"
SMOKE_ROOT="${SMOKE_ROOT:-/content/smoke}"
SMOKE_PRESET="${SMOKE_PRESET:-1b}"
FORWARDED_ENV=(SMOKE_PRESET SMOKE_MODEL SMOKE_STAGES SMOKE_STAGE_TIMEOUT SMOKE_PRUNE SMOKE_RUN_ID
  HF_TOKEN HF_UPLOAD_REPO_PREFIX HF_WRITE_TOKEN)

log() { echo "[colab-smoke $(date +%H:%M:%S)] $*"; }

remote_shell() {  # command: run in the VM's tty, which works while the kernel is busy
  echo "$1" | colab console -s "${SESSION}" | grep -a -v -E '^\s*$|\[exited\]|logout'
}

fetch() {
  mkdir -p "${OUT_DIR}"
  colab download -s "${SESSION}" "${SMOKE_ROOT}/report.tgz" "${OUT_DIR}/report.tgz"
  tar -xzf "${OUT_DIR}/report.tgz" -C "${OUT_DIR}"
  log "report in ${OUT_DIR}"
  find "${OUT_DIR}" -name summary.json -maxdepth 2 -exec cat {} \;
}

start() {
  if [[ -z "${HF_TOKEN:-}" && -z "${SMOKE_MODEL:-}" ]]; then
    if [[ "${SMOKE_PRESET}" != "1b" ]]; then
      log "preset ${SMOKE_PRESET} needs HF_TOKEN (gated Meta Llama 3 weights)"; exit 1
    fi
    export SMOKE_MODEL=unsloth/Llama-3.2-1B
  fi
  export SMOKE_PRESET
  local env_args=() name
  for name in "${FORWARDED_ENV[@]}"; do
    if [[ -n "${!name:-}" ]]; then env_args+=(--env "${name}=${!name}"); fi
  done
  env_args+=(--env "SMOKE_ROOT=${SMOKE_ROOT}")

  local tarball
  tarball="$(mktemp -d)/repo.tgz"
  # COPYFILE_DISABLE / --no-mac-metadata: no macOS ._ files or xattrs in the
  # archive; git's fsmonitor socket cannot be archived.
  (cd "${REPO_DIR}" && { git ls-files -co --exclude-standard; echo .git; } |
    COPYFILE_DISABLE=1 tar --no-mac-metadata --exclude '.git/fsmonitor--daemon.ipc' -czf "${tarball}" -T -)
  log "repo tarball $(du -h "${tarball}" | cut -f1)"

  log "allocating ${GPU} session ${SESSION}"
  colab new -s "${SESSION}" --gpu "${GPU}" ${COLAB_HIGH_MEM:+--high-mem}
  colab status -s "${SESSION}"
  colab upload -s "${SESSION}" "${tarball}" /content/repo.tgz
  rm -f "${tarball}"

  mkdir -p "${OUT_DIR}"
  log "running the smoke (preset ${SMOKE_PRESET}); output also in ${OUT_DIR}/launcher.log"
  local rc=0
  colab exec -s "${SESSION}" --timeout "${COLAB_TIMEOUT:-900}" "${env_args[@]}" \
    -f "${REPO_DIR}/scripts/colab_smoke_launcher.py" 2>&1 | tee "${OUT_DIR}/launcher.log" || rc=$?
  if ! grep -q 'colab-launcher .*\] report ' "${OUT_DIR}/launcher.log"; then
    log "colab exec ended (exit ${rc}) before the smoke finished; it keeps running on the VM."
    log "check it with '$0 tail', then '$0 fetch' and '$0 stop'"
    exit 1
  fi
  fetch
  if [[ "${COLAB_KEEP:-0}" != "1" ]]; then colab stop -s "${SESSION}"; fi
  # summary.json's last line is the run's overall result.
  if ! tail -n 2 "${OUT_DIR}"/*/summary.json | grep -q '"passed": true'; then
    log "smoke failed; see ${OUT_DIR}"; exit 1
  fi
}

case "${1:-start}" in
  start) start ;;
  tail) remote_shell "tail -n ${TAIL_LINES:-40} ${SMOKE_ROOT}/driver.log; ls ${SMOKE_ROOT}/report.tgz 2>/dev/null" ;;
  fetch) fetch ;;
  stop) colab stop -s "${SESSION}" ;;
  *) echo "usage: $0 [start|tail|fetch|stop]" >&2; exit 2 ;;
esac
