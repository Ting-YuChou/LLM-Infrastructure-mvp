#!/usr/bin/env bash
# Phase 3 (speculative decoding) on one A100: vLLM and SGLang with and without
# the DFlash drafter and nano with and without it, all under the same
# real-prompt workloads; the acceptance study of the drafter in plain PyTorch
# (M0); nano's tests on the GPU and the parity of its drafter with the
# checkpoint's own code (M1). Every step logs to <run>/logs/<step>.log and
# appends to status.tsv; a failed step does not stop the ones after it.
#
# On Colab: BENCH_PHASE=phase3 bash benchmarks/phase0/colab_phase0.sh [rerun]
# (colab_launcher.py sets P0_ROOT=/content/p0, P0_REPO_DIR and P0_RUN_ID).
#
# Workloads (make_workloads.py): prompts rendered with the chat template and
# sent as text to /v1/completions, max_tokens 256, requests stop at EOS, one
# file per dataset and concurrency level:
#   all      alpaca and humaneval at c=1, 4, 16, 64; gsm8k and long (CNN/DM
#            articles to summarize, 1000-2500 tokens) at c=1, 16; temperature 0
#   sampled  alpaca and humaneval at c=1 and 16 with temperature 0.7, top-p 0.9
#            (labels end in "s"), after "all" on the same server
#   c1       every dataset at c=1 only
#   texts    texts.py: 8 prompts per dataset at temperature 0, one at a time
#            and 16 at a time, the completions kept to compare servers
# Speculative servers are measured against the same server without a drafter;
# their accepted tokens per draft come from /metrics around each run.
#
# Steps (P3_STEPS, default all; colab_phase0.sh rerun drops installs):
#   installs   the phase 0-2 vLLM venv (vllm bench serve and nano), a venv with
#              the vLLM that has "method": "dflash" (P3_VLLM_SPEC), one with
#              SGLang (P3_SGLANG_SPEC), one for the workloads and the
#              acceptance study, the models, the drafter at
#              P3_DRAFT_REVISION, CUDA 12 nvcc for nano's kernels
#   workloads  make_workloads.py into <root>/workloads (kept across reruns)
#   accept     accept_study.py: accepted tokens per cycle by depth, BF16 and
#              AWQ target, greedy and sampled, drafter bf16 and 4-bit
#   tests      tests/gpu + tests/unit/test_nano_vllm_*.py; any SKIPPED fails it
#              (not in the default)
#   parity     parity.py: nano's DFlashDrafter against the checkpoint's own
#              code, top-1 agreement over 200 blocks (not in the default)
#   spec_check spec_check.py: every token nano emits, with and without the
#              drafter, against the target's argmax in one teacher-forced
#              pass; AWQ and BF16 target (not in the default)
#   kbench     phase1/kbench.py on P3_KERNELS (default: the verify rows'
#              attention by the block, prefill and decode kernels and vLLM's
#              flash-attention, and the drafter's norm + RoPE + KV write)
#              (not in the default)
#   tune       phase1/tune.py on P3_TUNE (default block): the block kernel's
#              launch settings and split count (not in the default)
#   serve      each P3_RUNS entry: start its server, run its workloads, stop it
#   summarize  summary.md
#
# P3_RUNS entries are engine:label:workloads[:extra], workloads and extra
# comma-separated; extra is KEY=VALUE environment settings for nano and
# server arguments for the others. Engines: vllm, vllmw4 (the AWQ checkpoint
# served under the BF16 model's name), sglang, each also as <engine>_dflash,
# nano (WEIGHT_QUANT=awq loads the AWQ checkpoint), and vllm019 (the pinned
# vLLM of phases 0-2, no drafter), and nano_dflash (nano with
# SPECULATIVE_MODEL set to the drafter).
#
# Env: P0_ROOT (default /workspace/p0), P0_RUN_ID (default: UTC timestamp),
# P0_REPO_DIR, P3_STEPS, P3_RUNS, P3_MODEL, P3_W4_MODEL, P3_DRAFT_MODEL,
# P3_DRAFT_REVISION, P3_SPEC_TOKENS (vLLM's num_speculative_tokens, default 9:
# the drafter's block of 10; SGLang reads the block size from the drafter),
# P3_VLLM_SPEC (pip spec, default vllm: the latest release), P3_SGLANG_SPEC
# (default sglang[all]), P3_VLLM_DFLASH_ARGS, P3_SGLANG_DFLASH_ARGS,
# P3_SGLANG_DFLASH_ENV, P3_ACCEPT_PROMPTS (per dataset, default 16),
# P3_ACCEPT_AWQ=0 (BF16 target only), P3_NANO_KV_BLOCKS, P0_VLLM_VERSION.
set -uo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P0_DIR="$(cd "${CODE_DIR}/../phase0" && pwd)"
ROOT="${P0_ROOT:-/workspace/p0}"
RUN_ID="${P0_RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
OUT="${ROOT}/results/${RUN_ID}"
MODEL_ID="${P3_MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}"
MODEL="${ROOT}/models/$(basename "${MODEL_ID}")"
W4_MODEL_ID="${P3_W4_MODEL:-hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4}"
W4_MODEL="${ROOT}/models/$(basename "${W4_MODEL_ID}")"
DRAFT_ID="${P3_DRAFT_MODEL:-z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat}"
DRAFT_REVISION="${P3_DRAFT_REVISION:-d3af30def9601abdd10810aba220d692f0e803f0}"
DRAFT="${ROOT}/models/$(basename "${DRAFT_ID}")"
SPEC_TOKENS="${P3_SPEC_TOKENS:-9}"
VENVS="${ROOT}/venvs"
PY="${VENVS}/vllm/bin/python"
VLLM_VERSION="${P0_VLLM_VERSION:-0.19.1}"
VLLM_SPEC="${P3_VLLM_SPEC:-vllm}"
SGLANG_SPEC="${P3_SGLANG_SPEC:-sglang[all]}"
SPEC_VLLM="${VENVS}/vllm_spec/bin"
SPEC_SGLANG="${VENVS}/sglang_spec/bin"
STUDY_PY="${VENVS}/study/bin/python"
REPO_DIR="${P0_REPO_DIR:-$(cd "${CODE_DIR}/../.." && pwd)}"
STEPS="${P3_STEPS:-installs workloads accept serve summarize}"
WORKLOADS="${ROOT}/workloads"
RUNS="${P3_RUNS:-vllm:vllm:all,sampled vllm_dflash:vllm_dflash:all,sampled \
vllmw4:vllmw4:all,sampled vllmw4_dflash:vllmw4_dflash:all,sampled \
sglang:sglang:all,sampled sglang_dflash:sglang_dflash:all,sampled \
nano:nano:all,sampled:ENGINE_MULTIPROCESS=true,NUM_KVCACHE_BLOCKS=0 \
nano:nano_awq:all,sampled:WEIGHT_QUANT=awq,ENGINE_MULTIPROCESS=true,NUM_KVCACHE_BLOCKS=0}"
# The drafter card's vLLM flags; a server that refuses them is retried without.
VLLM_DFLASH_ARGS="${P3_VLLM_DFLASH_ARGS:---attention-backend flash_attn --max-num-batched-tokens 32768}"
# The card uses fa3, which needs sm_90; on the A100 SGLang picks its own backend.
SGLANG_DFLASH_ARGS="${P3_SGLANG_DFLASH_ARGS:---dtype bfloat16 --mem-fraction-static 0.75 --trust-remote-code}"
SGLANG_DFLASH_ENV="${P3_SGLANG_DFLASH_ENV:-SGLANG_ENABLE_SPEC_V2=1 SGLANG_ENABLE_DFLASH_SPEC_V2=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1}"
NANO_PORT=8100
CUDA_ENV="${ROOT}/cuda129.env"

export HF_HOME="${ROOT}/hf_cache" UV_CACHE_DIR="${ROOT}/uv_cache" UV_PYTHON_INSTALL_DIR="${ROOT}/uv_python"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
export SERVE_TAG=P3_SERVE BENCH_TEMPERATURE=0 BENCH_IGNORE_EOS=0
export NANO_EXT_DIR="${ROOT}/nano_ext"  # the C++ radix tree's build (radix_cpp.py)
mkdir -p "${OUT}"/{logs,serve,accept,tests,parity,spec_check,kbench,tune}
exec 3>&1  # console; step output goes to ${OUT}/logs/<step>.log

log() { echo "[p3 $(date -u +%H:%M:%S)] $*" >&3; }
# run_step, want, installs, start_server/bench_serve/stop_server
# shellcheck source-path=SCRIPTDIR source=../phase0/lib.sh
source "${P0_DIR}/lib.sh"

# --- installs ----------------------------------------------------------------
install_venv() {  # the load generator's and nano's venv; reuse an earlier phase's
  local have
  have="$("${PY}" -c "import importlib.metadata as m; print(m.version('vllm'))" 2>/dev/null)"
  if [[ "${have}" != "${VLLM_VERSION}" ]]; then
    rm -rf "${VENVS}/vllm"
    install_vllm || return 1
  fi
  uv pip install -p "${PY}" pytest ninja nanobind pandas &&  # pandas: vllm bench's custom dataset
  "${PY}" -c "import torch, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__)"
}

install_study() {  # the versions the drafter's card names: its dflash.py runs here
  { [[ -x "${STUDY_PY}" ]] || uv venv --seed --python 3.12 "${VENVS}/study"; } &&
  uv pip install -p "${STUDY_PY}" torch==2.9.0 transformers==4.57.3 accelerate safetensors numpy datasets &&
  "${STUDY_PY}" -c "import torch, transformers; print('torch', torch.__version__, torch.version.cuda, '| transformers', transformers.__version__)"
}

install_vllm_spec() {
  { [[ -x "${SPEC_VLLM}/python" ]] || uv venv --seed --python 3.12 "${VENVS}/vllm_spec"; } &&
  uv pip install -p "${SPEC_VLLM}/python" "${VLLM_SPEC}" &&
  "${SPEC_VLLM}/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda)"
}

install_sglang_spec() {
  # ninja: flashinfer JIT-compiles kernels when SGLang captures its CUDA graphs.
  { [[ -x "${SPEC_SGLANG}/python" ]] || uv venv --seed --python 3.12 "${VENVS}/sglang_spec"; } &&
  uv pip install -p "${SPEC_SGLANG}/python" "${SGLANG_SPEC}" ninja &&
  "${SPEC_SGLANG}/python" -c "import sglang, torch; print('sglang', sglang.__version__, 'torch', torch.__version__)"
}

download_w4_model() {  # download_model reads MODEL_ID and MODEL
  local MODEL_ID="${W4_MODEL_ID}" MODEL="${W4_MODEL}"
  download_model
}

download_drafter() {  # a fixed revision: the acceptance study runs its dflash.py
  python3 - <<EOF
import os
from huggingface_hub import snapshot_download
snapshot_download("${DRAFT_ID}", revision="${DRAFT_REVISION}", local_dir="${DRAFT}",
                  token=os.environ.get("HF_TOKEN"),
                  allow_patterns=["*.json", "*.safetensors", "*.py"])
EOF
  du -sh "${DRAFT}" && sha256sum "${DRAFT}"/*.py
}

installs() {
  local rc=0
  command -v uv > /dev/null || python3 -m pip install -q uv > "${OUT}/logs/install_uv.log" 2>&1
  run_step install_venv install_venv || rc=1
  run_step install_study install_study || rc=1
  if [[ " ${RUNS}" == *" vllm"* ]]; then run_step install_vllm_spec install_vllm_spec || rc=1; fi
  if [[ " ${RUNS}" == *" sglang"* ]]; then run_step install_sglang_spec install_sglang_spec || rc=1; fi
  [[ -f "${MODEL}/config.json" ]] || run_step download_model download_model || rc=1
  [[ -f "${W4_MODEL}/config.json" ]] || run_step download_w4_model download_w4_model || rc=1
  [[ -f "${DRAFT}/config.json" ]] || run_step download_drafter download_drafter || rc=1
  [[ -f "${CUDA_ENV}" ]] || run_step cuda129 cuda129 || rc=1  # nano's CUDA kernels
  return "${rc}"
}

# --- workloads ---------------------------------------------------------------
workloads() {
  if [[ ! -f "${WORKLOADS}/manifest.json" ]]; then
    "${STUDY_PY}" "${CODE_DIR}/make_workloads.py" --model "${MODEL}" --out "${WORKLOADS}" || return 1
  fi
  mkdir -p "${OUT}/workloads" && cp "${WORKLOADS}"/* "${OUT}/workloads/"
  "${STUDY_PY}" -c "import json; [print('P3_WORKLOAD', k, v['prompts'], 'prompts', v['tokens_median'], 'median tokens') \
for k, v in json.load(open('${WORKLOADS}/manifest.json'))['files'].items()]" >&3
}

levels() {  # dataset: its concurrency levels
  case "$1" in
    alpaca|humaneval) echo 1 4 16 64 ;;
    *) echo 1 16 ;;
  esac
}

real() {  # name port dataset concurrency [label suffix]
  local name=$1 port=$2 dataset=$3 c=$4 suffix=${5:-} file n label
  file="${WORKLOADS}/${dataset}_c${c}.jsonl"
  n="$(wc -l < "${file}")"
  label="${dataset}_c${c}${suffix}"
  spec_metrics "${name}_${label}_before" "${port}"
  bench_serve "${name}" "${port}" "${label}" "${n}" 256 --dataset-name custom --dataset-path "${file}" \
    --custom-output-len 256 --skip-chat-template --disable-shuffle --no-oversample --max-concurrency "${c}"
  local rc=$?
  spec_metrics "${name}_${label}_after" "${port}"
  return "${rc}"
}

warmup() {  # name port
  bench_serve "$1" "$2" warmup 16 32 --dataset-name custom --dataset-path "${WORKLOADS}/warmup.jsonl" \
    --custom-output-len 32 --skip-chat-template --disable-shuffle --no-oversample --max-concurrency 4
  rm -f "${OUT}/serve/$1_warmup.json"
}

all() {  # name port
  local dataset c rc=0
  for dataset in alpaca humaneval gsm8k long; do
    for c in $(levels "${dataset}"); do real "$1" "$2" "${dataset}" "${c}" || rc=1; done
  done
  return "${rc}"
}

c1() {  # name port
  local dataset rc=0
  for dataset in alpaca humaneval gsm8k long; do real "$1" "$2" "${dataset}" 1 || rc=1; done
  return "${rc}"
}

texts() {  # name port
  "${PY}" "${CODE_DIR}/texts.py" --base-url "http://127.0.0.1:$2" --model "${MODEL}" \
    --workloads "${WORKLOADS}" --out "${OUT}/serve/$1_texts.json" >&3
}

sampled() {
  local dataset c rc=0
  for dataset in alpaca humaneval; do
    for c in 1 16; do
      BENCH_TEMPERATURE=0.7 BENCH_TOP_P=0.9 real "$1" "$2" "${dataset}" "${c}" s || rc=1
    done
  done
  return "${rc}"
}

spec_metrics() {  # name port: the server's speculative-decoding counters
  curl -sf "http://127.0.0.1:$2/metrics" 2>/dev/null | grep -i 'spec\|accept\|draft' | grep -v '^#' \
    > "${OUT}/serve/$1.metrics.txt"
  [[ -s "${OUT}/serve/$1.metrics.txt" ]] || rm -f "${OUT}/serve/$1.metrics.txt"
}

token_check() {  # label port: the prompt tokens each server counts for one prompt per dataset
  local label=$1 port=$2
  "${PY}" - "${label}" "${port}" "${MODEL}" "${WORKLOADS}" <<'EOF' >&3
import json, sys, urllib.request
label, port, model, workloads = sys.argv[1:]
counts = {}
for dataset in ("alpaca", "humaneval", "gsm8k", "long"):
    prompt = json.loads(open(f"{workloads}/{dataset}_c1.jsonl").readline())["prompt"]
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", body, {"Content-Type": "application/json"})
    try:
        counts[dataset] = json.load(urllib.request.urlopen(request, timeout=120))["usage"]["prompt_tokens"]
    except Exception as exc:  # reported, not fatal
        counts[dataset] = f"error: {exc}"
print("P3_TOKENS " + json.dumps({"engine": label, **counts}))
EOF
}

# --- servers -----------------------------------------------------------------
nano_kv_blocks() {  # vLLM's KV capacity in 16-token blocks, so both hold the same tokens
  local tokens
  if [[ -n "${P3_NANO_KV_BLOCKS:-}" ]]; then echo "${P3_NANO_KV_BLOCKS}"; return; fi
  tokens="$(grep -ohE 'GPU KV cache size: [0-9,]+ tokens' "${OUT}"/logs/vllm_server.log 2>/dev/null \
    | head -1 | grep -oE '[0-9,]+' | tr -d ,)"
  echo $(( ${tokens:-144000} / 16 ))
}

nano_cmd() {  # extra KEY=VALUE,...: the nano server command with the benchmark settings
  local extra=() model="${MODEL}"
  [[ -n "$1" ]] && IFS=, read -r -a extra <<< "$1"
  if [[ ",$1," == *,WEIGHT_QUANT=awq,* ]]; then model="${W4_MODEL}"; fi
  # cuda129's toolkit (nvcc, cudart, cccl only) is for nano's kernels alone: under it
  # flashinfer's JIT in the vLLM and SGLang servers finds no curand.h.
  local cuda
  cuda="$(sed -n 's/^export CUDA_HOME=//p' "${CUDA_ENV}" 2>/dev/null)"
  echo env ${cuda:+CUDA_HOME="${cuda}" PATH="${cuda}/bin:${BASE_PATH}" TORCH_CUDA_ARCH_LIST=8.0} \
    TORCH_EXTENSIONS_DIR="${ROOT}/torch_ext" \
    PYTHONPATH="${REPO_DIR}" MODEL_NAME="${model}" DTYPE=bfloat16 HOST=127.0.0.1 \
    PORT="${NANO_PORT}" MAX_MODEL_LEN=8192 MAX_NEW_TOKENS=4096 \
    NUM_KVCACHE_BLOCKS="$(nano_kv_blocks)" MAX_NUM_SEQS=64 MAX_CONCURRENT_REQUESTS=256 \
    MAX_NUM_BATCHED_TOKENS=2048 CUDA_GRAPH_MAX_BS=64 ${extra[@]+"${extra[@]}"} \
    "${PY}" -m src.serving.nano_vllm.server --config "${REPO_DIR}/config/nano_vllm_config.yaml"
}

spec_json() {  # vLLM's --speculative-config for the drafter
  echo "{\"method\": \"dflash\", \"model\": \"${DRAFT}\", \"num_speculative_tokens\": ${SPEC_TOKENS}}"
}

run_server() {  # engine label workloads extra
  local engine=$1 label=$2 workloads=$3 extra=$4 port ready cmd=() args=() dflash=() fallback=() w rc=0
  [[ -n "${extra}" ]] && IFS=, read -r -a args <<< "${extra}"
  case "${engine}" in
    nano)
      port="${NANO_PORT}"; ready=/ready
      read -r -a cmd <<< "$(nano_cmd "${extra}")" ;;
    nano_dflash)
      port="${NANO_PORT}"; ready=/ready
      read -r -a cmd <<< "$(nano_cmd "${extra:+${extra},}SPECULATIVE_MODEL=${DRAFT}")" ;;
    vllm019)
      port=8000; ready=/health
      cmd=("${VENVS}/vllm/bin/vllm" serve "${MODEL}" --host 127.0.0.1 --port 8000
        --max-model-len 8192 --seed 0 ${args[@]+"${args[@]}"}) ;;
    vllm|vllm_dflash|vllmw4|vllmw4_dflash)
      port=8000; ready=/health
      cmd=("${SPEC_VLLM}/vllm" serve)
      if [[ "${engine}" == vllmw4* ]]; then
        cmd+=("${W4_MODEL}" --served-model-name "${MODEL}")
      else
        cmd+=("${MODEL}")
      fi
      cmd+=(--host 127.0.0.1 --port 8000 --max-model-len 8192 --seed 0 ${args[@]+"${args[@]}"})
      if [[ "${engine}" == *_dflash ]]; then
        cmd+=(--speculative-config "$(spec_json)")
        read -r -a dflash <<< "${VLLM_DFLASH_ARGS}"
      fi ;;
    sglang|sglang_dflash)
      port=30000; ready=/health_generate
      cmd=(env "PATH=${SPEC_SGLANG}:${PATH}")
      if [[ "${engine}" == *_dflash ]]; then
        read -r -a dflash <<< "${SGLANG_DFLASH_ENV}"
        cmd+=(${dflash[@]+"${dflash[@]}"})
        read -r -a dflash <<< "${SGLANG_DFLASH_ARGS}"
        dflash+=(--speculative-algorithm DFLASH --speculative-draft-model-path "${DRAFT}")
      fi
      cmd+=("${SPEC_SGLANG}/python" -m sglang.launch_server --model-path "${MODEL}" --host 127.0.0.1
        --port 30000 --context-length 8192 --random-seed 0 --enable-metrics ${args[@]+"${args[@]}"})
      # flashinfer's JIT needs an nvcc that matches torch's CUDA; Triton kernels need none.
      fallback=(--attention-backend triton --sampling-backend pytorch) ;;
    *) log "unknown engine ${engine}"; return 1 ;;
  esac
  if ! start_server "${label}" "${port}" "${ready}" "${cmd[@]}" ${dflash[@]+"${dflash[@]}"}; then
    mv "${OUT}/logs/${label}_server.log" "${OUT}/logs/${label}_server_first.log"
    case "${engine}" in
      vllm_dflash|vllmw4_dflash)
        log "${label} failed with '${VLLM_DFLASH_ARGS}'; retrying without"
        start_server "${label}" "${port}" "${ready}" "${cmd[@]}" || return 1 ;;
      sglang|sglang_dflash)
        log "${label} failed with its default backends; retrying with triton attention"
        start_server "${label}" "${port}" "${ready}" "${cmd[@]}" ${dflash[@]+"${dflash[@]}"} \
          "${fallback[@]}" || return 1 ;;
      *) return 1 ;;
    esac
  fi
  token_check "${label}" "${port}"
  warmup "${label}" "${port}"
  for w in ${workloads//,/ }; do
    "${w}" "${label}" "${port}" || rc=1
  done
  # What the server itself logged about speculation (accept lengths, the config it chose).
  grep -i 'accept\|speculat\|dflash\|KV cache size' "${OUT}/logs/${label}_server.log" | tail -40 \
    > "${OUT}/serve/${label}_spec_log.txt"
  stop_server
  return "${rc}"
}

serve() {
  local entry engine label workloads extra rc=0
  for entry in ${RUNS}; do
    IFS=: read -r engine label workloads extra <<< "${entry}"
    log "serve ${label}: ${workloads}${extra:+ (${extra})}"
    run_server "${engine}" "${label}" "${workloads}" "${extra}" || rc=1
  done
  return "${rc}"
}

# --- acceptance study ----------------------------------------------------------
accept() {
  local awq=()
  [[ "${P3_ACCEPT_AWQ:-1}" == 1 ]] && awq=(--awq "${W4_MODEL}")
  PYTHONPATH="${REPO_DIR}" "${STUDY_PY}" "${CODE_DIR}/accept_study.py" --self-test &&
  PYTHONPATH="${REPO_DIR}" "${STUDY_PY}" "${CODE_DIR}/accept_study.py" --target "${MODEL}" --drafter "${DRAFT}" \
    --workloads "${WORKLOADS}" --out "${OUT}/accept" --num-prompts "${P3_ACCEPT_PROMPTS:-16}" \
    ${awq[@]+"${awq[@]}"}
}

parity() {  # in the study venv: the drafter's remote code needs its transformers
  PYTHONPATH="${REPO_DIR}" "${STUDY_PY}" "${CODE_DIR}/parity.py" --target "${MODEL}" --drafter "${DRAFT}" \
    --out "${OUT}/parity/parity.json" ${P3_PARITY_ARGS:-}
}

nano_env() {  # the nano server's build environment (see nano_cmd), for an in-process run
  local cuda
  cuda="$(sed -n 's/^export CUDA_HOME=//p' "${CUDA_ENV}" 2>/dev/null)"
  echo env ${cuda:+CUDA_HOME="${cuda}" PATH="${cuda}/bin:${BASE_PATH}" TORCH_CUDA_ARCH_LIST=8.0} \
    TORCH_EXTENSIONS_DIR="${ROOT}/torch_ext" PYTHONPATH="${REPO_DIR}"
}

spec_check() {
  local label model quant rc=0 cmd=()
  read -r -a cmd <<< "$(nano_env)"
  for label in ${P3_SPEC_CHECK:-awq bf16}; do
    model="${MODEL}"; quant=none
    if [[ "${label}" == awq ]]; then model="${W4_MODEL}"; quant=awq; fi
    "${cmd[@]}" "${PY}" "${CODE_DIR}/spec_check.py" --model "${model}" --weight-quant "${quant}" \
      --drafter "${DRAFT}" --workloads "${WORKLOADS}" --label "${label}" \
      --out "${OUT}/spec_check/${label}.json" | tee /dev/fd/3 || rc=1
  done
  return "${rc}"
}

kbench() {  # vLLM's flash-attention comes from the load generator's venv
  local rc=0
  PYTHONPATH="${REPO_DIR}" "${PY}" "${CODE_DIR}/../phase1/kbench.py" --out "${OUT}/kbench" \
    --kernels "${P3_KERNELS:-paged_block_attention,norm_rope_cache_write}" \
    ${P3_KBENCH_SHAPES:+--shapes "${P3_KBENCH_SHAPES}"} || rc=1
  cat "${OUT}/kbench/kbench.md" >&3
  return "${rc}"
}

tune() {
  PYTHONPATH="${REPO_DIR}" "${PY}" "${CODE_DIR}/../phase1/tune.py" --out "${OUT}/tune" \
    --kernels "${P3_TUNE:-block}" &&
  cat "${OUT}/tune/tune.md" >&3
}

summarize() {
  "${PY}" "${CODE_DIR}/summarize.py" "${OUT}" && cat "${OUT}/summary.md" >&3
}

log "run ${RUN_ID}: steps '${STEPS}', model ${MODEL_ID}, drafter ${DRAFT_ID}@${DRAFT_REVISION:0:8}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader >&3
want installs && installs
# ninja (cpp_extension JIT) comes from the venv.
export PATH="${VENVS}/vllm/bin:${PATH}"
BASE_PATH="${PATH}"
{
  "${PY}" -c "import importlib.metadata as m; print('load generator / nano: vllm', m.version('vllm'))"
  "${SPEC_VLLM}/python" -c "import importlib.metadata as m; print('vllm baseline:', m.version('vllm'), '| torch', m.version('torch'))"
  "${SPEC_SGLANG}/python" -c "import importlib.metadata as m; print('sglang baseline:', m.version('sglang'), '| torch', m.version('torch'))"
  "${STUDY_PY}" -c "import importlib.metadata as m; print('study:', 'torch', m.version('torch'), '| transformers', m.version('transformers'))"
  echo "drafter: ${DRAFT_ID} @ ${DRAFT_REVISION}"
} > "${OUT}/environment.txt" 2>&1
cat "${OUT}/environment.txt" >&3
want workloads && run_step workloads workloads
want accept && run_step accept accept
want tests && run_step tests gpu_tests
want parity && run_step parity parity
want spec_check && run_step spec_check spec_check
want kbench && run_step kbench kbench
want tune && run_step tune tune
want serve && run_step serve serve
want summarize && run_step summarize summarize
log "done; results in ${OUT}"
