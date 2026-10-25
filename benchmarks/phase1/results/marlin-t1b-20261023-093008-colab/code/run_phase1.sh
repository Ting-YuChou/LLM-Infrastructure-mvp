#!/usr/bin/env bash
# Phase 1 on one A100, once per kernel milestone: nano_vllm's tests on the GPU,
# kernel micro-benchmarks (kbench.py), ncu on every kbench kernel, and nano_vllm
# end to end against vLLM with nsys timelines. Every step logs to
# <run>/logs/<step>.log and appends to status.tsv; a failed step does not stop
# the ones after it.
#
# On Colab: BENCH_PHASE=phase1 bash benchmarks/phase0/colab_phase0.sh [rerun]
# (colab_launcher.py sets P0_ROOT=/content/p0, P0_REPO_DIR and P0_RUN_ID).
#
# Steps (P1_STEPS, default all; colab_phase0.sh rerun drops installs, cuda129):
#   installs  vLLM venv (shared with phase 0) + pytest + ninja, nsys, the model
#   cuda129   CUDA 12 nvcc matching the venv's torch for cpp_extension builds,
#             written to <P0_ROOT>/cuda129.env, which every later step sources
#   tests     tests/gpu + tests/unit/test_nano_vllm_*.py; any SKIPPED fails it
#   kbench    kbench.py: correctness, then time / GB/s / % of peak per shape
#   tune      tune.py: sweep the kernels' launch settings (not in the default)
#   ncu       ncu on each kbench kernel (kbench.py --ncu-mode), analyze_ncu.py
#   e2e       nano_bench.py per nano config and vllm_offline.py at the same
#             points (also on P1_VLLM_W4_MODEL when a config is w4a16); nsys on
#             the last nano config
#   quality   quality.py: wikitext-2 perplexity and top-1 agreement of the torch,
#             Triton and W4A16 paths on the real model
#   summarize summary.md
#
# Env: P0_ROOT (default /workspace/p0), P0_RUN_ID (default: UTC timestamp),
# P0_REPO_DIR (default: the checkout holding this script), P1_STEPS,
# P1_MODEL (default unsloth/Meta-Llama-3.1-8B-Instruct), P1_TUNE (tune.py
# kernels: decode,decode_fp8,prefill,gemv,cuda_gemv,w4a16,w4a16_tiled), P1_TESTS (pytest paths for
# the tests step instead of all of tests/gpu and the nano_vllm unit tests), P1_KERNELS (kbench and
# ncu filter, comma-separated), P1_KBENCH_SHAPES (regex on the shape label,
# same two steps), P1_E2E_RUNS (nano_bench kind:batch:in:out list;
# vLLM runs the paged ones), P1_NANO_CONFIGS (label:attention:linear:eager|graph[:weight_quant[:flags]]
# per nano_bench run, default: the Phase 1 milestones in order; flags joined by +: async for async
# scheduling, sampled for P1_SAMPLING instead of greedy (with any sampled config vLLM runs its
# points sampled too), tiled / rows for the W4A16 weight layout, fp8kv for an fp8 KV cache
# (with any fp8kv config vLLM runs its points with --kv-cache-dtype fp8 and
# P1_VLLM_FP8_BACKEND, default TRITON_ATTN: FLASH_ATTN refuses fp8 on sm80)), P1_QUALITY_CONFIGS
# (quality.py --configs), P1_SAMPLING ("temperature top_p", default "0.7 0.9": the nano server's
# default sampling), P1_NCU_LAUNCHES (cap per kbench kernel, default
# 300), P1_CUDA_VERSION (cuda129 step, default: the venv's torch.version.cuda),
# P0_VLLM_VERSION (default 0.19.1).
set -uo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P0_DIR="$(cd "${CODE_DIR}/../phase0" && pwd)"
ROOT="${P0_ROOT:-/workspace/p0}"
RUN_ID="${P0_RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
OUT="${ROOT}/results/${RUN_ID}"
MODEL_ID="${P1_MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}"
MODEL="${ROOT}/models/$(basename "${MODEL_ID}")"
# vLLM's baseline for the w4a16 nano configs: the same model as AWQ int4
# (group 128), which vLLM runs with its Marlin kernel.
W4_MODEL_ID="${P1_VLLM_W4_MODEL:-hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4}"
W4_MODEL="${ROOT}/models/$(basename "${W4_MODEL_ID}")"
VENVS="${ROOT}/venvs"
PY="${VENVS}/vllm/bin/python"
VLLM_VERSION="${P0_VLLM_VERSION:-0.19.1}"
REPO_DIR="${P0_REPO_DIR:-$(cd "${CODE_DIR}/../.." && pwd)}"
STEPS="${P1_STEPS:-installs cuda129 tests kbench ncu e2e quality summarize}"
KERNELS="${P1_KERNELS:-}"
KBENCH_SHAPES="${P1_KBENCH_SHAPES:-}"
E2E_RUNS="${P1_E2E_RUNS:-static:1:128:256,paged:1:128:256,paged:16:1024:256,paged:64:1024:256}"
# torch_eager: the M1 refactor alone; triton_eager: Triton attention and fused
# kernels (M2, M4, M5); triton_graph: + decode CUDA graphs (M3); gemv_graph:
# + the Triton GEMV (M6); cuda_gemv_graph: the CUDA GEMV instead (M6);
# w4a16_graph: 4-bit decoder weights (M7).
NANO_CONFIGS="${P1_NANO_CONFIGS:-torch_eager:torch:torch:eager triton_eager:triton:torch:eager triton_graph:triton:torch:graph gemv_graph:triton:triton:graph cuda_gemv_graph:triton:cuda:graph w4a16_graph:triton:triton:graph:w4a16}"
read -r SAMPLED_TEMPERATURE SAMPLED_TOP_P <<< "${P1_SAMPLING:-0.7 0.9}"
VLLM_FP8_BACKEND="${P1_VLLM_FP8_BACKEND:-TRITON_ATTN}"
CUDA_ENV="${ROOT}/cuda129.env"

export HF_HOME="${ROOT}/hf_cache" UV_CACHE_DIR="${ROOT}/uv_cache" UV_PYTHON_INSTALL_DIR="${ROOT}/uv_python"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
mkdir -p "${OUT}"/{logs,tests,kbench,ncu,nsys,offline,quality}
exec 3>&1  # console; step output goes to ${OUT}/logs/<step>.log

log() { echo "[p1 $(date -u +%H:%M:%S)] $*" >&3; }
# run_step, want, install_vllm/nsys, download_model, nsys_run, ncu_run
# shellcheck source-path=SCRIPTDIR source=../phase0/lib.sh
source "${P0_DIR}/lib.sh"

# --- installs ----------------------------------------------------------------
install_venv() {  # reuse the VM's vLLM venv when phase 0 already made it
  local have
  have="$("${PY}" -c "import importlib.metadata as m; print(m.version('vllm'))" 2>/dev/null)"
  if [[ "${have}" != "${VLLM_VERSION}" ]]; then
    rm -rf "${VENVS}/vllm"
    install_vllm || return 1
  fi
  uv pip install -p "${PY}" pytest ninja &&
  "${PY}" -c "import torch, triton, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__, 'cuda', torch.version.cuda, '| triton', triton.__version__)"
}


# --- tests: gpu_tests (lib.sh) -----------------------------------------------

# --- kernels -----------------------------------------------------------------
kbench() {
  local rc=0
  "${PY}" "${CODE_DIR}/kbench.py" --out "${OUT}/kbench" ${KERNELS:+--kernels "${KERNELS}"} \
    ${KBENCH_SHAPES:+--shapes "${KBENCH_SHAPES}"} || rc=1
  cat "${OUT}/kbench/kbench.md" >&3
  return "${rc}"
}

tune() {
  "${PY}" "${CODE_DIR}/tune.py" --out "${OUT}/tune" ${P1_TUNE:+--kernels "${P1_TUNE}"} &&
  cat "${OUT}/tune/tune.md" >&3
}

ncu_kbench() {  # one ncu report per kbench kernel, limited to its --kernel-name regex
  local rc=0 listing name regex
  listing="$("${PY}" "${CODE_DIR}/kbench.py" --list ${KERNELS:+--kernels "${KERNELS}"})" || return 1
  while IFS=$'\t' read -r name regex <&4; do
    [[ -n "${name}" ]] || continue
    ncu_run "kbench_${name}" "${P1_NCU_LAUNCHES:-300}" --kernel-name "regex:${regex}" --nvtx -- \
      "${PY}" "${CODE_DIR}/kbench.py" --ncu-mode --kernels "${name}" ${KBENCH_SHAPES:+--shapes "${KBENCH_SHAPES}"} || rc=1
  done 4<<< "${listing}"
  return "${rc}"
}

# --- end to end --------------------------------------------------------------
nano_config_args() {  # label:attention:linear:eager|graph[:weight_quant[:flags]] -> nano_bench flags
  local label attention linear mode quant flags args
  IFS=: read -r label attention linear mode quant flags <<< "$1"
  args="--attention-backend ${attention} --linear-backend ${linear} --weight-quant ${quant:-none}"
  [[ "${mode}" == eager ]] && args+=" --enforce-eager"
  [[ "+${flags}+" == *+async+* ]] && args+=" --async-scheduling"
  [[ "+${flags}+" == *+sampled+* ]] && args+=" $(sampled_args)"
  [[ "+${flags}+" == *+fp8kv+* ]] && args+=" --kv-cache-dtype fp8"
  [[ "+${flags}+" == *+tiled+* ]] && args+=" --w4a16-layout tiled"
  [[ "+${flags}+" == *+rows+* ]] && args+=" --w4a16-layout rows"
  echo "${args}"
}

sampled_args() { echo "--temperature ${SAMPLED_TEMPERATURE} --top-p ${SAMPLED_TOP_P}"; }

e2e() {
  local rc=0 run kind batch input output config label last modes=greedy mode tag args
  export PYTHONPATH="${REPO_DIR}"
  for config in ${NANO_CONFIGS}; do
    label="${config%%:*}"
    # shellcheck disable=SC2046  # the flags are meant to split
    "${PY}" "${P0_DIR}/nano_bench.py" --model "${MODEL}" --runs "${E2E_RUNS}" \
      $(nano_config_args "${config}") --out "${OUT}/offline/nano_${label}.json" || rc=1
    last="${config}"
  done
  [[ "${NANO_CONFIGS}" == *sampled* ]] && modes+=" sampled"
  [[ "${NANO_CONFIGS}" == *fp8kv* ]] && modes+=" fp8kv"
  for run in ${E2E_RUNS//,/ }; do
    IFS=: read -r kind batch input output <<< "${run}"
    [[ "${kind}" == paged ]] || continue
    for mode in ${modes}; do
      case "${mode}" in
        greedy) tag="" args="" ;;
        sampled) tag="sampled_" args="$(sampled_args)" ;;
        fp8kv) tag="fp8kv_" args="--kv-cache-dtype fp8 --attention-backend ${VLLM_FP8_BACKEND}" ;;
      esac
      # shellcheck disable=SC2086  # the flags are meant to split
      "${PY}" "${P0_DIR}/vllm_offline.py" --model "${MODEL}" --batch "${batch}" \
        --input-len "${input}" --output-len "${output}" ${args} \
        --out "${OUT}/offline/vllm_${tag}b${batch}_in${input}_out${output}.json" || rc=1
      if [[ "${NANO_CONFIGS}" == *:w4a16* ]]; then
        # shellcheck disable=SC2086
        { [[ -d "${W4_MODEL}" ]] || download_w4_model; } &&
        "${PY}" "${P0_DIR}/vllm_offline.py" --model "${W4_MODEL}" --batch "${batch}" \
          --input-len "${input}" --output-len "${output}" ${args} \
          --out "${OUT}/offline/vllmw4_${tag}b${batch}_in${input}_out${output}.json" || rc=1
      fi
    done
  done
  grep -h 'NANO_BENCH\|VLLM_OFFLINE' "${OUT}/logs/e2e.log" >&3
  label="${last%%:*}"
  # shellcheck disable=SC2046
  nsys_run "nano_${label}_b1" "${PY}" "${P0_DIR}/nano_bench.py" --model "${MODEL}" \
    --runs paged:1:128:64 $(nano_config_args "${last}") \
    --out "${OUT}/offline/nano_${label}_b1_nsys.json" || rc=1
  # shellcheck disable=SC2046
  nsys_run "nano_${label}_b16" "${PY}" "${P0_DIR}/nano_bench.py" --model "${MODEL}" \
    --runs paged:16:1024:64 $(nano_config_args "${last}") \
    --out "${OUT}/offline/nano_${label}_b16_nsys.json" || rc=1
  return "${rc}"
}

download_w4_model() {  # download_model reads MODEL_ID and MODEL
  local MODEL_ID="${W4_MODEL_ID}" MODEL="${W4_MODEL}"
  download_model
}

quality() {
  # wikitext-2 comes as parquet; the vLLM venv has neither datasets nor pyarrow.
  "${PY}" -c "import pyarrow" 2>/dev/null || uv pip install -p "${PY}" pyarrow || return 1
  PYTHONPATH="${REPO_DIR}" "${PY}" "${CODE_DIR}/quality.py" --model "${MODEL}" --out "${OUT}/quality" \
    ${P1_QUALITY_CONFIGS:+--configs "${P1_QUALITY_CONFIGS}"} &&
  cat "${OUT}/quality/quality.md" >&3
}

environment() {
  {
    echo "run_id: ${RUN_ID}"
    nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv
    grep RmProfilingAdminOnly /proc/driver/nvidia/params
    "${PY}" -c "import importlib.metadata as m, torch, triton; print('vllm', m.version('vllm'), '| torch', torch.__version__, 'cuda', torch.version.cuda, '| triton', triton.__version__)"
    echo "CUDA_HOME=${CUDA_HOME:-unset}; nvcc: $(nvcc --version 2>/dev/null | tail -1)"
    "$(find_nsys)" --version
    "$(find_ncu)" --version | tail -1
  } > "${OUT}/environment.txt" 2>&1
  cat "${OUT}/environment.txt" >&3
}

summarize() {
  "${PY}" "${CODE_DIR}/summarize.py" "${OUT}" >&3
}

# --- main --------------------------------------------------------------------
log "phase1 run ${RUN_ID} -> ${OUT} (steps: ${STEPS})"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader >&3
rm -rf "${OUT}/code" && cp -r "${CODE_DIR}" "${OUT}/code"

if want installs; then
  python3 -m pip install -q uv > "${OUT}/logs/install_uv.log" 2>&1 || log "pip install uv failed"
  run_step install_nsys install_nsys & nsys_pid=$!
  run_step download_model download_model & model_pid=$!
  run_step install_vllm install_venv
  wait "${nsys_pid}" "${model_pid}"
fi
# After the installs: download_model's python3 must not resolve to a half-built venv.
# ninja (cpp_extension JIT) and pytest come from the venv.
export PATH="${VENVS}/vllm/bin:${PATH}"
want cuda129 && run_step cuda129 cuda129
# shellcheck source=/dev/null
if [[ -f "${CUDA_ENV}" ]]; then source "${CUDA_ENV}"; fi
# The C++ radix tree's tests (Phase 2) need nanobind; a rerun skips installs.
want tests && { "${PY}" -c "import nanobind" 2>/dev/null || uv pip install -p "${PY}" nanobind > /dev/null; }
want tests && run_step tests gpu_tests
want kbench && run_step kbench kbench
want tune && run_step tune tune
want ncu && run_step ncu ncu_kbench
want e2e && run_step e2e e2e
want quality && run_step quality quality
run_step environment environment
want summarize && run_step summarize summarize
log "P1_DONE ${RUN_ID}; status:"
cat "${OUT}/status.tsv" >&3
