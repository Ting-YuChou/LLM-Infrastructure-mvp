#!/usr/bin/env bash
# Phase 0 on one Google Colab A100 (40GB, ncu allowed), driven from this
# machine with the Colab CLI (see scripts/colab_gpu_smoke.sh for sign-in).
# BENCH_PHASE=phase1 / phase2 drives benchmarks/phase{1,2}/run_phase*.sh the
# same way.
#
# Uploads benchmarks/phase0, benchmarks/${BENCH_PHASE}, src/serving/nano_vllm
# and the nano_vllm tests from this checkout (no push needed), runs
# run_${BENCH_PHASE}.sh on the VM through colab_launcher.py, then downloads the
# run directory into benchmarks/${BENCH_PHASE}/results/<run_id>/ and releases
# the VM.
#
#   bash benchmarks/phase0/colab_phase0.sh          # new VM, upload, run, fetch, stop
#   bash benchmarks/phase0/colab_phase0.sh rerun    # same VM (venvs, model kept), upload, run, fetch, stop
#   bash benchmarks/phase0/colab_phase0.sh tail     # last lines of the run's console
#   bash benchmarks/phase0/colab_phase0.sh fetch    # download the results again
#   bash benchmarks/phase0/colab_phase0.sh stop     # release the VM
#
# Env: BENCH_PHASE (default phase0), COLAB_SESSION (default phase0; both
# phases share the VM's venvs and model), COLAB_GPU (default A100), COLAB_KEEP=1
# (do not stop the VM at the end), COLAB_TIMEOUT (seconds without output before
# `colab exec` gives up, default 900; the run goes on and tail/fetch still
# work), P0_RUN_ID, P0_STEPS, P0_CONCURRENCY (default 1..128: 40 GB leaves room
# for ~145k KV tokens, so 256 concurrent 1280-token requests would just queue),
# and every P1_* / P2_* variable that is set (see benchmarks/phase{1,2}/run_phase*.sh).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${HERE}/../.." && pwd)"
PHASE="${BENCH_PHASE:-phase0}"
SESSION="${COLAB_SESSION:-phase0}"
GPU="${COLAB_GPU:-A100}"
RUN_ID="${P0_RUN_ID:-$(date -u +%Y%m%d-%H%M%S)-colab}"
REMOTE_ROOT=/content/p0
RESULTS="${REPO_DIR}/benchmarks/${PHASE}/results"
[[ -f "${REPO_DIR}/benchmarks/${PHASE}/run_${PHASE}.sh" ]] || {
  echo "no benchmarks/${PHASE}/run_${PHASE}.sh for BENCH_PHASE=${PHASE}" >&2; exit 2; }

log() { echo "[colab-phase0 $(date +%H:%M:%S)] $*"; }

fetch() {
  mkdir -p "${RESULTS}"
  colab download -s "${SESSION}" "${REMOTE_ROOT}/report.tgz" "${RESULTS}/report.tgz"
  tar -xzf "${RESULTS}/report.tgz" -C "${RESULTS}" && rm -f "${RESULTS}/report.tgz"
  log "results in ${RESULTS}:"; ls -1 "${RESULTS}"
}

upload_code() {
  local tarball paths=(benchmarks/phase0 src/serving/nano_vllm)
  [[ "${PHASE}" == phase0 ]] || paths+=("benchmarks/${PHASE}")
  # The nano_vllm tests and the repo files they import or read outside nano_vllm.
  paths+=(tests/unit/tiny_llama.py src/monitoring/metrics.py src/api/model_routing.py
    config/model_routing.nano.json config/nano_vllm_config.yaml)
  if [[ -d "${REPO_DIR}/tests/gpu" ]]; then paths+=(tests/gpu); fi
  tarball="$(mktemp -d)/p0code.tgz"
  (cd "${REPO_DIR}" &&
    COPYFILE_DISABLE=1 tar --no-mac-metadata --exclude results --exclude __pycache__ \
      -czf "${tarball}" "${paths[@]}" tests/unit/test_nano_vllm_*.py)
  log "code tarball $(du -h "${tarball}" | cut -f1)"

  colab upload -s "${SESSION}" "${tarball}" /content/p0code.tgz
  rm -f "${tarball}"
}

run() {
  local env_args=(--env "P0_RUN_ID=${RUN_ID}" --env "BENCH_PHASE=${PHASE}") name
  for name in P0_STEPS P0_CONCURRENCY P0_SGLANG_SPEC ${!P1_@} ${!P2_@}; do
    if [[ -n "${!name:-}" ]]; then env_args+=(--env "${name}=${!name}"); fi
  done
  mkdir -p "${RESULTS}"
  log "running ${PHASE} (run ${RUN_ID}); console also in ${RESULTS}/${RUN_ID}.launcher.log"
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
  case "${PHASE}" in
    phase0) export P0_STEPS="${P0_STEPS:-bandwidth vllm_serve sglang_serve nsys_vllm nano ncu summarize}" ;;
    phase1) export P1_STEPS="${P1_STEPS:-tests kbench ncu e2e quality summarize}" ;;
    phase2) export P2_STEPS="${P2_STEPS:-tests prefix_bench serve nsys summarize}" ;;
  esac
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
