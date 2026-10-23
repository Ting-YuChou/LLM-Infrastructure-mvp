#!/usr/bin/env bash
# Phase 2 on one A100, once per engine-systems milestone: nano_vllm's tests on
# the GPU, then nano_vllm's OpenAI server, vLLM and SGLang under the same
# `vllm bench serve` workloads, and nsys timelines of the nano server. Every
# step logs to <run>/logs/<step>.log and appends to status.tsv; a failed step
# does not stop the ones after it.
#
# On Colab: BENCH_PHASE=phase2 bash benchmarks/phase0/colab_phase0.sh [rerun]
# (colab_launcher.py sets P0_ROOT=/content/p0, P0_REPO_DIR and P0_RUN_ID).
#
# Workloads (every run uses --ignore-eos, --temperature 0 and seed 0):
#   w1     random 1024/256 at each P2_CONCURRENCY (throughput, TTFT, ITL)
#   w1c32  w1 at c=32 only (the GIL diagnostic run)
#   w1u    random 1024/128 at c=1 and c=16 with their own seeds: prompts no
#          earlier run sent, so TTFT includes the whole prefill (vllm bench's
#          random prompt i is the same in every run with seed 0, so w1's
#          prompts are mostly prefix-cache hits after the warmup)
#   w1c1   w1 at c=1, 10 requests 1024/128 (nsys)
#   w1c1c  w1c1 twice: the second pass's prompts hit the prefix cache, so its
#          prefill steps are small (nsys: eager vs piecewise host time)
#   w2a    one shared 2048-token prefix + 128 random tokens, output 128, c=16
#   w2b    prefix_repetition: 48 prefixes of 4096 tokens (more than the KV
#          cache holds) + 256-token suffixes, output 128, c=16
#   w3     prefill interference: background 256/1024 at c=32 with 4096/8
#          prompts arriving at 0.25 req/s on top; the background's ITL
#          percentiles show the decode stalls
#
# Steps (P2_STEPS, default all; colab_phase0.sh rerun drops installs):
#   installs  vLLM venv (shared with phases 0-1) + pytest/ninja, SGLang venv, nsys,
#             the model, and CUDA 12 nvcc for the CUDA kernels' tests (cuda129)
#   tests     tests/gpu + tests/unit/test_nano_vllm_*.py; any SKIPPED fails it
#   prefix_bench  host cost of the radix-tree prefix cache against Phase 1's hash
#             chain at serving scale (CPU only; prefix_bench.py)
#   serve     each P2_RUNS entry: start its server, run its workloads, stop it
#   nsys      the nano server under nsys for each P2_NSYS workload, with the
#             engine's NVTX phase ranges (NANO_NVTX=1) and P2_NSYS_ENV settings
#   offline   nano_bench.py (decode ms/step at b=1/16/64) synchronous and with
#             async scheduling, plus nsys of both at b=16 (not in the default)
#   summarize summary.md
#
# P2_RUNS entries are engine:label:workloads[:extra], workloads comma-separated,
# extra comma-separated: KEY=VALUE environment settings for nano, server
# arguments for vllm and sglang. Default: the M0 baseline (below).
#
# Env: P0_ROOT (default /workspace/p0), P0_RUN_ID (default: UTC timestamp),
# P0_REPO_DIR (default: the checkout holding this script), P2_STEPS, P2_RUNS,
# P2_NSYS (nano workloads to capture, default "w1c16 w1c64 w3"), P2_NSYS_ENV
# (KEY=VALUE,... for those servers), P2_OFFLINE_RUNS (nano_bench --runs),
# P2_CONCURRENCY (w1 levels, default "1 2 4 8 16 32 64"), P2_NANO_KV_BLOCKS
# (default: vLLM's logged KV token count / 16 from this run, else 9000),
# P2_MODEL (default unsloth/Meta-Llama-3.1-8B-Instruct), P0_VLLM_VERSION.
set -uo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P0_DIR="$(cd "${CODE_DIR}/../phase0" && pwd)"
ROOT="${P0_ROOT:-/workspace/p0}"
RUN_ID="${P0_RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
OUT="${ROOT}/results/${RUN_ID}"
MODEL_ID="${P2_MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}"
MODEL="${ROOT}/models/$(basename "${MODEL_ID}")"
VENVS="${ROOT}/venvs"
PY="${VENVS}/vllm/bin/python"
VLLM_VERSION="${P0_VLLM_VERSION:-0.19.1}"
SGLANG_SPEC="${P0_SGLANG_SPEC:-sglang[all]}"
REPO_DIR="${P0_REPO_DIR:-$(cd "${CODE_DIR}/../.." && pwd)}"
STEPS="${P2_STEPS:-installs tests prefix_bench serve nsys summarize}"
read -r -a CONCURRENCY <<< "${P2_CONCURRENCY:-1 2 4 8 16 32 64}"
# M0 baseline: vLLM first (nano's KV size comes from its log), then nano as
# it is (prefill-first, 512-token chunks, hash prefix cache), nano without the
# prefix cache, without chunking, and with a 0.5 ms GIL switch interval; vLLM
# at nano's chunk size and without chunked prefill; SGLang.
RUNS="${P2_RUNS:-vllm:vllm:w1,w2a,w2b,w3 \
vllm:vllm_mbt512:w3:--max-num-batched-tokens=512 \
vllm:vllm_nochunk:w3:--no-enable-chunked-prefill \
nano:nano:w1,w2a,w2b,w3 \
nano:nano_nocache:w2a:ENABLE_PREFIX_CACHE=false \
nano:nano_nochunk:w3:MAX_NUM_BATCHED_TOKENS=8192 \
nano:nano_gil:w1c32:NANO_SWITCH_INTERVAL=0.0005 \
sglang:sglang:w1,w2a,w2b,w3}"
NSYS_WORKLOADS="${P2_NSYS:-w1c16 w1c64 w3}"
NANO_PORT=8100
CUDA_ENV="${ROOT}/cuda129.env"

export HF_HOME="${ROOT}/hf_cache" UV_CACHE_DIR="${ROOT}/uv_cache" UV_PYTHON_INSTALL_DIR="${ROOT}/uv_python"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg SERVE_TAG=P2_SERVE BENCH_TEMPERATURE=0
export NANO_EXT_DIR="${ROOT}/nano_ext"  # the C++ radix tree's build (radix_cpp.py)
mkdir -p "${OUT}"/{logs,tests,serve,nsys,offline}
exec 3>&1  # console; step output goes to ${OUT}/logs/<step>.log

log() { echo "[p2 $(date -u +%H:%M:%S)] $*" >&3; }
# run_step, want, installs, gpu_tests, start_server/bench_serve/stop_server, nsys_analyze
# shellcheck source-path=SCRIPTDIR source=../phase0/lib.sh
source "${P0_DIR}/lib.sh"

# --- installs ----------------------------------------------------------------
install_venv() {  # reuse the VM's vLLM venv when an earlier phase made it
  local have
  have="$("${PY}" -c "import importlib.metadata as m; print(m.version('vllm'))" 2>/dev/null)"
  if [[ "${have}" != "${VLLM_VERSION}" ]]; then
    rm -rf "${VENVS}/vllm"
    install_vllm || return 1
  fi
  uv pip install -p "${PY}" pytest ninja nanobind &&
  "${PY}" -c "import torch, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__)"
}

installs() {
  local rc=0
  command -v uv > /dev/null || python3 -m pip install -q uv > "${OUT}/logs/install_uv.log" 2>&1
  run_step install_venv install_venv || rc=1
  [[ -x "${VENVS}/sglang/bin/python" ]] || run_step install_sglang install_sglang || rc=1
  run_step install_nsys install_nsys || rc=1
  [[ -f "${MODEL}/config.json" ]] || run_step download_model download_model || rc=1
  [[ -f "${CUDA_ENV}" ]] || run_step cuda129 cuda129 || rc=1  # the CUDA kernels' tests
  return "${rc}"
}

# --- workloads ---------------------------------------------------------------
random_args() {  # input_len output_len concurrency
  echo --dataset-name random --random-input-len "$1" --random-output-len "$2" \
    --random-range-ratio 0.0 --max-concurrency "$3"
}

w1() {  # name port
  bench_serve "$1" "$2" warmup 32 64 $(random_args 1024 64 16)
  rm -f "${OUT}/serve/$1_warmup.json"
  local c n
  for c in "${CONCURRENCY[@]}"; do
    n=$(( c * 4 > 16 ? c * 4 : 16 )); (( c == 1 )) && n=10
    bench_serve "$1" "$2" "w1_c${c}" "${n}" 256 $(random_args 1024 256 "${c}")
  done
}

w1c32() { bench_serve "$1" "$2" w1_c32 128 256 $(random_args 1024 256 32); }
w1c1() { bench_serve "$1" "$2" w1_c1 10 128 $(random_args 1024 128 1); }  # short: nsys
w1c1c() {
  bench_serve "$1" "$2" w1_c1_first 10 8 $(random_args 1024 8 1) &&
  bench_serve "$1" "$2" w1_c1 10 128 $(random_args 1024 128 1)
}

w1u() {  # seeds no other run uses: the prompts miss the prefix cache
  bench_serve "$1" "$2" w1u_c1 10 128 $(random_args 1024 128 1) --seed 101 &&
  bench_serve "$1" "$2" w1u_c16 64 128 $(random_args 1024 128 16) --seed 116
}
w1c16() { bench_serve "$1" "$2" w1_c16 32 128 $(random_args 1024 128 16); }  # short: nsys
w1c64() { bench_serve "$1" "$2" w1_c64 64 128 $(random_args 1024 128 64); }

w2a() {
  bench_serve "$1" "$2" w2a 64 128 $(random_args 128 128 16) --random-prefix-len 2048
}

w2b() {
  bench_serve "$1" "$2" w2b 192 128 --dataset-name prefix_repetition \
    --prefix-repetition-prefix-len 4096 --prefix-repetition-suffix-len 256 \
    --prefix-repetition-num-prefixes 48 --prefix-repetition-output-len 128 --max-concurrency 16
}

w3() {  # background decodes with long prompts arriving on top (seed 1: distinct tokens)
  local bg rc=0
  SERVE_KEEP_RAW=1 bench_serve "$1" "$2" w3_bg 96 1024 $(random_args 256 1024 32) \
    --metric-percentiles 50,90,99,99.9,100 &
  bg=$!
  sleep 15
  SERVE_KEEP_RAW=1 bench_serve "$1" "$2" w3_fg 8 8 --dataset-name random --random-input-len 4096 \
    --random-output-len 8 --random-range-ratio 0.0 --request-rate 0.25 --seed 1 \
    --metric-percentiles 50,90,99,100 || rc=1
  wait "${bg}" || rc=1
  return "${rc}"
}

# --- servers -----------------------------------------------------------------
nano_kv_blocks() {  # vLLM's KV capacity in 16-token blocks, so both hold the same tokens
  local tokens
  if [[ -n "${P2_NANO_KV_BLOCKS:-}" ]]; then echo "${P2_NANO_KV_BLOCKS}"; return; fi
  tokens="$(grep -ohE 'GPU KV cache size: [0-9,]+ tokens' "${OUT}"/logs/vllm*_server.log 2>/dev/null \
    | head -1 | grep -oE '[0-9,]+' | tr -d ,)"
  echo $(( ${tokens:-144000} / 16 ))
}

nano_cmd() {  # extra KEY=VALUE,...: the nano server command with the benchmark settings
  local extra=()
  [[ -n "$1" ]] && IFS=, read -r -a extra <<< "$1"
  echo env PYTHONPATH="${REPO_DIR}" MODEL_NAME="${MODEL}" DTYPE=bfloat16 HOST=127.0.0.1 \
    PORT="${NANO_PORT}" MAX_MODEL_LEN=8192 MAX_NEW_TOKENS=4096 \
    NUM_KVCACHE_BLOCKS="$(nano_kv_blocks)" MAX_NUM_SEQS=64 MAX_CONCURRENT_REQUESTS=256 \
    MAX_NUM_BATCHED_TOKENS=512 CUDA_GRAPH_MAX_BS=64 ${extra[@]+"${extra[@]}"} \
    "${PY}" -m src.serving.nano_vllm.server --config "${REPO_DIR}/config/nano_vllm_config.yaml"
}

run_server() {  # engine label workloads extra
  local engine=$1 label=$2 workloads=$3 extra=$4 port ready cmd=() args=() w rc=0
  [[ -n "${extra}" ]] && IFS=, read -r -a args <<< "${extra}"
  case "${engine}" in
    nano)
      port="${NANO_PORT}"; ready=/ready
      read -r -a cmd <<< "$(nano_cmd "${extra}")" ;;
    vllm)
      port=8000; ready=/health
      cmd=("${VENVS}/vllm/bin/vllm" serve "${MODEL}" --host 127.0.0.1 --port 8000
        --max-model-len 8192 --seed 0 ${args[@]+"${args[@]}"}) ;;
    sglang)
      port=30000; ready=/health_generate
      cmd=(env "PATH=${VENVS}/sglang/bin:${PATH}" "${VENVS}/sglang/bin/python" -m sglang.launch_server
        --model-path "${MODEL}" --host 127.0.0.1 --port 30000 --context-length 8192 --random-seed 0
        ${args[@]+"${args[@]}"}) ;;
    *) log "unknown engine ${engine}"; return 1 ;;
  esac
  if ! start_server "${label}" "${port}" "${ready}" "${cmd[@]}"; then
    [[ "${engine}" == sglang ]] || return 1
    # flashinfer's JIT needs an nvcc that matches torch's CUDA; Triton kernels need none.
    log "sglang failed with its default backends; retrying with triton attention"
    mv "${OUT}/logs/${label}_server.log" "${OUT}/logs/${label}_server_flashinfer.log"
    start_server "${label}" "${port}" "${ready}" "${cmd[@]}" \
      --attention-backend triton --sampling-backend pytorch || return 1
  fi
  for w in ${workloads//,/ }; do
    snapshot "${label}_${w}_before" "${port}"
    "${w}" "${label}" "${port}" || rc=1
    snapshot "${label}_${w}_after" "${port}"
  done
  stop_server
  return "${rc}"
}

snapshot() {  # name port: the server's /metrics (and nano's /stats) around a workload
  curl -sf "http://127.0.0.1:$2/metrics" > "${OUT}/serve/$1.metrics.txt" 2>/dev/null || rm -f "${OUT}/serve/$1.metrics.txt"
  curl -sf "http://127.0.0.1:$2/stats" > "${OUT}/serve/$1.stats.json" 2>/dev/null || rm -f "${OUT}/serve/$1.stats.json"
}

serve() {
  local entry engine label workloads extra rc=0
  for entry in ${RUNS}; do
    IFS=: read -r engine label workloads extra <<< "${entry}"
    log "serve ${label}: ${workloads}${extra:+ (${extra})}"
    run_server "${engine}" "${label}" "${workloads}" "${extra}" || rc=1
  done
  grep -h P2_SERVE "${OUT}/logs/serve.log" >&3
  return "${rc}"
}

# --- nsys: the nano server with its NVTX phase ranges ------------------------
nano_nsys() {  # workload: capture between /start_profile and /stop_profile
  local workload=$1 label="nano_${1}" nsys cmd=() rc=0
  nsys="$(find_nsys)"
  [[ -n "${nsys}" ]] || { echo "nsys not installed"; return 1; }
  read -r -a cmd <<< "$(nano_cmd "NANO_NVTX=1${P2_NSYS_ENV:+,${P2_NSYS_ENV}}")"
  # stop-shutdown: cudaProfilerStop ends the capture and the server, then nsys writes the report.
  start_server "${label}" "${NANO_PORT}" /ready "${nsys}" profile -t cuda,nvtx --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --sample=none --cpuctxsw=none \
    -f true -o "${OUT}/nsys/${label}" "${cmd[@]}" || return 1
  curl -sf -X POST "http://127.0.0.1:${NANO_PORT}/start_profile" > /dev/null || rc=1
  "${workload}" "${label}" "${NANO_PORT}" || rc=1
  curl -sf -X POST "http://127.0.0.1:${NANO_PORT}/stop_profile" > /dev/null
  local i
  for i in $(seq 120); do kill -0 "${SERVER_PID}" 2>/dev/null || break; sleep 2; done
  stop_server
  [[ -f "${OUT}/nsys/${label}.nsys-rep" ]] || { echo "no report for ${label}"; return 1; }
  nsys_analyze "${label}" || rc=1
  return "${rc}"
}

nsys_step() {
  local w rc=0
  for w in ${NSYS_WORKLOADS}; do nano_nsys "${w}" || rc=1; done
  return "${rc}"
}

offline() {
  local rc=0 mode flag runs="${P2_OFFLINE_RUNS:-paged:1:128:256,paged:16:1024:256,paged:64:1024:256}"
  export PYTHONPATH="${REPO_DIR}"
  for mode in sync async; do
    # The server's kernels (config/nano_vllm_config.yaml); nano_bench defaults to torch.
    flag="--attention-backend triton --linear-backend triton"
    [[ "${mode}" == async ]] && flag+=" --async-scheduling"
    "${PY}" "${P0_DIR}/nano_bench.py" --model "${MODEL}" --runs "${runs}" ${flag} \
      --out "${OUT}/offline/nano_${mode}.json" || rc=1
    NANO_NVTX=1 nsys_run "nano_offline_${mode}_b16" "${PY}" "${P0_DIR}/nano_bench.py" --model "${MODEL}" \
      --runs paged:16:1024:64 ${flag} --out "${OUT}/offline/nano_${mode}_b16_nsys.json" || rc=1
  done
  grep -h NANO_BENCH "${OUT}/logs/offline.log" >&3
  return "${rc}"
}

prefix_bench() {
  PYTHONPATH="${REPO_DIR}" "${PY}" "${CODE_DIR}/prefix_bench.py" --out "${OUT}/prefix_bench" &&
  cat "${OUT}/prefix_bench/prefix_bench.md" >&3
}

summarize() {
  "${PY}" "${CODE_DIR}/summarize.py" "${OUT}" && cat "${OUT}/summary.md" >&3
}

log "run ${RUN_ID}: steps '${STEPS}', model ${MODEL_ID}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader >&3
want installs && installs
# ninja (cpp_extension JIT) and pytest come from the venv.
export PATH="${VENVS}/vllm/bin:${PATH}"
# shellcheck source=/dev/null
if [[ -f "${CUDA_ENV}" ]]; then source "${CUDA_ENV}"; fi
# nanobind builds the C++ radix tree; a rerun skips installs, so check here.
want tests && { "${PY}" -c "import nanobind" 2>/dev/null || uv pip install -p "${PY}" nanobind > /dev/null; }
want tests && run_step tests gpu_tests
want prefix_bench && { "${PY}" -c "import nanobind" 2>/dev/null || uv pip install -p "${PY}" nanobind > /dev/null; }
want prefix_bench && run_step prefix_bench prefix_bench
want serve && run_step serve serve
want nsys && run_step nsys nsys_step
want offline && run_step offline offline
want summarize && run_step summarize summarize
log "done; results in ${OUT}"
