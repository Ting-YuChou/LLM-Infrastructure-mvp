#!/usr/bin/env bash
# Phase 0 on one Google Colab A100 (40GB, ncu allowed), driven from this
# machine with the Colab CLI (see scripts/colab_gpu_smoke.sh for sign-in).
#
# Uploads benchmarks/phase0 and src/serving/nano_vllm from this checkout (no
# push needed), runs run_phase0.sh on the VM through colab_launcher.py, then
# downloads the run directory into benchmarks/phase0/results/<run_id>/ and
# releases the VM.
#
#   bash benchmarks/phase0/colab_phase0.sh          # new VM, upload, run, fetch, stop
#   bash benchmarks/phase0/colab_phase0.sh rerun    # same VM (venvs, model kept), upload, run, fetch, stop
#   bash benchmarks/phase0/colab_phase0.sh tail     # last lines of the run's console
#   bash benchmarks/phase0/colab_phase0.sh fetch    # download the results again
#   bash benchmarks/phase0/colab_phase0.sh stop     # release the VM
#
# Env: COLAB_SESSION (default phase0), COLAB_GPU (default A100), COLAB_KEEP=1
# (do not stop the VM at the end), COLAB_TIMEOUT (seconds without output before
# `colab exec` gives up, default 900; the run goes on and tail/fetch still
# work), P0_RUN_ID, P0_STEPS, P0_CONCURRENCY (default 1..128: 40 GB leaves room
# for ~145k KV tokens, so 256 concurrent 1280-token requests would just queue).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${HERE}/../.." && pwd)"
SESSION="${COLAB_SESSION:-phase0}"
GPU="${COLAB_GPU:-A100}"
RUN_ID="${P0_RUN_ID:-$(date -u +%Y%m%d-%H%M%S)-colab}"
REMOTE_ROOT=/content/p0
RESULTS="${HERE}/results"

log() { echo "[colab-phase0 $(date +%H:%M:%S)] $*"; }

fetch() {
  mkdir -p "${RESULTS}"
  colab download -s "${SESSION}" "${REMOTE_ROOT}/report.tgz" "${RESULTS}/report.tgz"
  tar -xzf "${RESULTS}/report.tgz" -C "${RESULTS}" && rm -f "${RESULTS}/report.tgz"
  log "results in ${RESULTS}:"; ls -1 "${RESULTS}"
}

upload_code() {
  local tarball
  tarball="$(mktemp -d)/p0code.tgz"
  (cd "${REPO_DIR}" &&
    COPYFILE_DISABLE=1 tar --no-mac-metadata --exclude results --exclude __pycache__ \
      -czf "${tarball}" benchmarks/phase0 src/serving/nano_vllm)
  log "code tarball $(du -h "${tarball}" | cut -f1)"

  colab upload -s "${SESSION}" "${tarball}" /content/p0code.tgz
  rm -f "${tarball}"
}

run() {
  local env_args=(--env "P0_RUN_ID=${RUN_ID}") name
  for name in P0_STEPS P0_CONCURRENCY P0_SGLANG_SPEC; do
    if [[ -n "${!name:-}" ]]; then env_args+=(--env "${name}=${!name}"); fi
  done
  mkdir -p "${RESULTS}"
  log "running phase 0 (run ${RUN_ID}); console also in ${RESULTS}/${RUN_ID}.launcher.log"
  local rc=0
  colab exec -s "${SESSION}" --timeout "${COLAB_TIMEOUT:-900}" "${env_args[@]}" \
    -f "${HERE}/colab_launcher.py" 2>&1 | tee "${RESULTS}/${RUN_ID}.launcher.log" || rc=$?
  if ! grep -q 'colab-launcher .*\] report ' "${RESULTS}/${RUN_ID}.launcher.log"; then
    log "colab exec ended (exit ${rc}) before the run finished; it keeps running on the VM."
    log "check it with '$0 tail', then '$0 fetch' and '$0 stop'"
    exit 1
  fi
  fetch
  if [[ "${COLAB_KEEP:-0}" != "1" ]]; then colab stop -s "${SESSION}"; fi
}

start() {
  log "allocating ${GPU} session ${SESSION}"
  colab new -s "${SESSION}" --gpu "${GPU}"
  colab status -s "${SESSION}"
  upload_code
  run
}

rerun() {  # the VM already has the venvs and the model
  export P0_STEPS="${P0_STEPS:-bandwidth vllm_serve sglang_serve nsys_vllm nano ncu summarize}"
  upload_code
  run
}

case "${1:-start}" in
  start) start ;;
  rerun) rerun ;;
  tail) echo "tail -n ${TAIL_LINES:-40} ${REMOTE_ROOT}/console.log" | colab console -s "${SESSION}" ;;
  fetch) fetch ;;
  stop) colab stop -s "${SESSION}" ;;
  *) echo "usage: $0 [start|rerun|tail|fetch|stop]" >&2; exit 2 ;;
esac
