#!/usr/bin/env bash
# Phase 0 on one A100: vLLM and SGLang serving baselines for Llama 3.1 8B
# Instruct, nsys decode timelines, ncu metrics of the decode kernels,
# achievable HBM bandwidth and cuBLAS floors, and where this repo's nano_vllm
# starts. Every step logs to <run>/logs/<step>.log and appends to status.tsv; a
# failed step does not stop the ones after it.
#
# On Colab, benchmarks/phase0/colab_phase0.sh uploads the code and runs this
# with P0_ROOT=/content/p0 P0_REPO_DIR=<the uploaded repo> P0_IDLE_STOP=0.
#
# Env: P0_ROOT (default /workspace/p0), P0_RUN_ID (default: UTC timestamp),
# P0_REPO_DIR (repo checkout for nano_vllm; default: clone the public repo),
# P0_MODEL (default: meta-llama/Llama-3.1-8B-Instruct with HF_TOKEN, else the
# ungated copy unsloth/Meta-Llama-3.1-8B-Instruct), P0_STEPS (subset of the
# steps below), P0_CONCURRENCY (sweep levels), P0_VLLM_VERSION (default: the
# requirements.txt pin), P0_SGLANG_SPEC, P0_REPO + HF_WRITE_TOKEN (also upload
# the run to that private HF dataset repo after every step), P0_IDLE_STOP (keep
# a pod up this long after the last step, then stop it with runpodctl; 0 exits).
set -uo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${P0_ROOT:-/workspace/p0}"
RUN_ID="${P0_RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
OUT="${ROOT}/results/${RUN_ID}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  MODEL_ID="${P0_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
else
  MODEL_ID="${P0_MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}"  # same weights, not gated
fi
MODEL="${ROOT}/models/$(basename "${MODEL_ID}")"
VENVS="${ROOT}/venvs"
PY="${VENVS}/vllm/bin/python"
VLLM_VERSION="${P0_VLLM_VERSION:-0.19.1}"
SGLANG_SPEC="${P0_SGLANG_SPEC:-sglang[all]}"
REPO_URL=https://github.com/Ting-YuChou/LLM-Infrastructure-mvp.git
REPO_DIR="${P0_REPO_DIR:-${ROOT}/repo}"
STEPS="${P0_STEPS:-installs bandwidth vllm_serve sglang_serve nsys_vllm nano ncu summarize}"
read -r -a CONCURRENCY <<< "${P0_CONCURRENCY:-1 2 4 8 16 32 64 128 256}"

export HF_HOME="${ROOT}/hf_cache" UV_CACHE_DIR="${ROOT}/uv_cache" UV_PYTHON_INSTALL_DIR="${ROOT}/uv_python"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
mkdir -p "${OUT}"/{logs,serve,nsys,ncu,offline}
exec 3>&1  # console (pod logs); step output goes to ${OUT}/logs/<step>.log

log() { echo "[p0 $(date -u +%H:%M:%S)] $*" >&3; }

upload() {  # everything under ${OUT} -> P0_REPO/runs/<run id>, when P0_REPO is set
  [[ -n "${P0_REPO:-}" ]] || return 0
  python3 - "${OUT}" "${RUN_ID}" >> "${OUT}/logs/upload.log" 2>&1 <<'EOF' || log "upload failed (logs/upload.log)"
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_WRITE_TOKEN"])
api.create_repo(os.environ["P0_REPO"], repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(
    folder_path=sys.argv[1], path_in_repo=f"runs/{sys.argv[2]}",
    repo_id=os.environ["P0_REPO"], repo_type="dataset", commit_message=f"phase0 run {sys.argv[2]}")
EOF
}

run_step() {  # name command...: log to logs/<name>.log, record status.tsv
  local name=$1; shift
  local start rc
  start=$(date +%s)
  log "step ${name} started"
  "$@" > "${OUT}/logs/${name}.log" 2>&1
  rc=$?
  printf '%s\t%s\t%s\n' "${name}" "${rc}" "$(( $(date +%s) - start ))" >> "${OUT}/status.tsv"
  log "step ${name} rc=${rc} in $(( $(date +%s) - start ))s"
  if [[ ${rc} -ne 0 ]]; then tail -30 "${OUT}/logs/${name}.log" >&3; fi
  return "${rc}"
}

want() { [[ " ${STEPS} " == *" $1 "* ]]; }

find_nsys() {
  command -v nsys || ls -1 /usr/local/cuda*/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | tail -1
}

find_ncu() {
  command -v ncu || ls -1 /usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | tail -1
}

# --- installs (run in parallel) --------------------------------------------
install_vllm() {
  uv venv --seed --python 3.12 "${VENVS}/vllm" &&
  uv pip install -p "${PY}" "vllm==${VLLM_VERSION}" huggingface_hub matplotlib &&
  "${PY}" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda)"
}

install_sglang() {
  # ninja: flashinfer JIT-compiles kernels when SGLang captures its CUDA graphs.
  { [[ -x "${VENVS}/sglang/bin/python" ]] || uv venv --seed --python 3.12 "${VENVS}/sglang"; } &&
  { uv pip install -p "${VENVS}/sglang/bin/python" "${SGLANG_SPEC}" ninja ||
    uv pip install -p "${VENVS}/sglang/bin/python" sglang ninja; } &&
  "${VENVS}/sglang/bin/python" -c "import sglang, torch; print('sglang', sglang.__version__, 'torch', torch.__version__)"
}

install_nsys() {
  if [[ -n "$(find_nsys)" ]]; then find_nsys; return 0; fi
  apt-get update -qq
  if ! apt-cache policy cuda-nsight-systems-12-8 | grep -q 'Candidate: [0-9]'; then
    local ubuntu  # NVIDIA's CUDA apt repo is not configured in this image
    ubuntu="$(. /etc/os-release && echo "${VERSION_ID//./}")"
    wget -q -O /tmp/cuda-keyring.deb \
      "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${ubuntu}/x86_64/cuda-keyring_1.1-1_all.deb" &&
      dpkg -i /tmp/cuda-keyring.deb && apt-get update -qq
  fi
  local v
  for v in 12-9 13-0 12-8; do
    if apt-cache policy "cuda-nsight-systems-${v}" | grep -q 'Candidate: [0-9]'; then
      DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        "cuda-nsight-systems-${v}" && break
    fi
  done
  [[ -n "$(find_nsys)" ]] && "$(find_nsys)" --version
}

download_model() {
  python3 - <<EOF
import os
from huggingface_hub import snapshot_download
snapshot_download("${MODEL_ID}", local_dir="${MODEL}", token=os.environ.get("HF_TOKEN"),
                  allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
EOF
  du -sh "${MODEL}"
}

clone_repo() {
  if [[ -n "${P0_REPO_DIR:-}" ]]; then ls "${REPO_DIR}/src/serving/nano_vllm"; return; fi
  rm -rf "${REPO_DIR}" && git clone --depth 1 "${REPO_URL}" "${REPO_DIR}" && git -C "${REPO_DIR}" log -1 --oneline
}

# --- serving benchmarks ----------------------------------------------------
SERVER_PID=""

wait_ready() {  # url timeout_s
  local deadline=$(( $(date +%s) + $2 ))
  until curl -sf -o /dev/null "$1"; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then return 1; fi
    if (( $(date +%s) > deadline )); then return 1; fi
    sleep 5
  done
}

stop_server() {
  [[ -n "${SERVER_PID}" ]] || return 0
  kill -TERM -- "-${SERVER_PID}" 2>/dev/null
  local i
  for i in $(seq 60); do kill -0 "${SERVER_PID}" 2>/dev/null || break; sleep 2; done
  kill -KILL -- "-${SERVER_PID}" 2>/dev/null
  SERVER_PID=""
  for i in $(seq 30); do  # wait until the GPU memory is back
    (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) < 2000 )) && break
    sleep 2
  done
}

bench() {  # engine port label input_len output_len num_prompts concurrency
  timeout 1200 "${VENVS}/vllm/bin/vllm" bench serve \
    --backend vllm --base-url "http://127.0.0.1:$2" --model "${MODEL}" \
    --dataset-name random --random-input-len "$4" --random-output-len "$5" --random-range-ratio 0.0 \
    --ignore-eos --num-prompts "$6" --max-concurrency "$7" --seed 0 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "${OUT}/serve" --result-filename "$1_$3.json"
  local rc=$?
  "${PY}" - "${OUT}/serve/$1_$3.json" "$1" "$3" "$6" "$5" <<'EOF' >&3
import json, sys
path, engine, label, n, out_len = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
try:
    r = json.load(open(path))
except (OSError, ValueError) as e:
    print(f"P0_SERVE {engine} {label} missing result: {e}"); sys.exit()
slim = {k: v for k, v in r.items() if not isinstance(v, (list, dict))}  # drop per-request arrays
json.dump(slim, open(path, "w"), indent=2)
ok = slim.get("completed") == n and slim.get("total_output_tokens") == n * out_len
print("P0_SERVE " + json.dumps({
    "engine": engine, "run": label, "ok": ok, "completed": slim.get("completed"),
    "out_tok_s": round(slim.get("output_throughput", 0), 1),
    "ttft_p50_ms": round(slim.get("median_ttft_ms", 0), 1),
    "tpot_p50_ms": round(slim.get("median_tpot_ms", 0), 2),
    "itl_p50_ms": round(slim.get("median_itl_ms", 0), 2),
    "itl_p99_ms": round(slim.get("p99_itl_ms", 0), 2)}))
EOF
  return "${rc}"
}

sweep() {  # engine port
  bench "$1" "$2" warmup 1024 64 32 16
  rm -f "${OUT}/serve/$1_warmup.json"
  bench "$1" "$2" latency_c1 128 256 10 1
  local c n
  for c in "${CONCURRENCY[@]}"; do
    n=$(( c * 4 > 16 ? c * 4 : 16 )); (( c == 1 )) && n=10
    bench "$1" "$2" "sweep_c${c}" 1024 256 "${n}" "${c}"
  done
}

serve_engine() {  # engine port health_path command...
  local engine=$1 port=$2 health=$3; shift 3
  local start=$(date +%s)
  setsid "$@" > "${OUT}/logs/${engine}_server.log" 2>&1 &
  SERVER_PID=$!
  if ! wait_ready "http://127.0.0.1:${port}${health}" 1200; then
    log "${engine} server did not become ready"; tail -40 "${OUT}/logs/${engine}_server.log" >&3
    stop_server; return 1
  fi
  echo "{\"engine\": \"${engine}\", \"startup_s\": $(( $(date +%s) - start )), \"args\": \"${*//\"/}\"}" \
    > "${OUT}/serve/${engine}_startup.json"
  log "${engine} ready after $(( $(date +%s) - start ))s; $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) in use"
  sweep "${engine}" "${port}"
  stop_server
}

vllm_serve() {
  serve_engine vllm 8000 /health \
    "${VENVS}/vllm/bin/vllm" serve "${MODEL}" --host 127.0.0.1 --port 8000 --max-model-len 8192 --seed 0
}

sglang_serve() {
  local bin="${VENVS}/sglang/bin"
  [[ -x "${bin}/ninja" ]] || uv pip install -p "${bin}/python" ninja
  local cmd=(env "PATH=${bin}:${PATH}" "${bin}/python" -m sglang.launch_server --model-path "${MODEL}"
    --host 127.0.0.1 --port 30000 --context-length 8192 --random-seed 0)
  serve_engine sglang 30000 /health_generate "${cmd[@]}" && return 0
  # flashinfer's JIT needs an nvcc that matches torch's CUDA; Triton kernels need none.
  log "sglang failed with its default flashinfer backends; retrying with triton attention"
  mv "${OUT}/logs/sglang_server.log" "${OUT}/logs/sglang_server_flashinfer.log"
  serve_engine sglang 30000 /health_generate "${cmd[@]}" --attention-backend triton --sampling-backend pytorch
}

# --- nsys ------------------------------------------------------------------
nsys_run() {  # label command...: capture the cudaProfilerStart/Stop range, export, analyze
  local label=$1; shift
  local nsys
  if [[ -f "${OUT}/nsys/${label}.json" ]]; then echo "${label}: already analyzed, skipped"; return 0; fi
  nsys="$(find_nsys)"
  [[ -n "${nsys}" ]] || { echo "nsys not installed"; return 1; }
  timeout 1800 "${nsys}" profile -t cuda,nvtx --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop --sample=none --cpuctxsw=none \
    -f true -o "${OUT}/nsys/${label}" "$@" || return 1
  "${nsys}" export --type sqlite -f true -o "${OUT}/nsys/${label}.sqlite" "${OUT}/nsys/${label}.nsys-rep" || return 1
  "${nsys}" stats -q -f csv --report cuda_gpu_kern_sum --report cuda_api_sum \
    -o "${OUT}/nsys/${label}" "${OUT}/nsys/${label}.sqlite"
  "${PY}" "${CODE_DIR}/analyze_nsys.py" "${OUT}/nsys/${label}.sqlite" --label "${label}" \
    --out-dir "${OUT}/nsys" >&3
}

nsys_vllm() {
  local rc=0
  nsys_run vllm_graph_b1 "${PY}" "${CODE_DIR}/vllm_offline.py" --model "${MODEL}" \
    --batch 1 --input-len 128 --output-len 256 --out "${OUT}/offline/vllm_graph_b1.json" || rc=1
  nsys_run vllm_eager_b1 "${PY}" "${CODE_DIR}/vllm_offline.py" --model "${MODEL}" --eager \
    --batch 1 --input-len 128 --output-len 256 --out "${OUT}/offline/vllm_eager_b1.json" || rc=1
  nsys_run vllm_graph_b64 "${PY}" "${CODE_DIR}/vllm_offline.py" --model "${MODEL}" \
    --batch 64 --input-len 1024 --output-len 256 --out "${OUT}/offline/vllm_graph_b64.json" || rc=1
  grep -h VLLM_OFFLINE "${OUT}"/logs/nsys_vllm.log >&3
  return "${rc}"
}

nano() {
  local rc=0
  export PYTHONPATH="${REPO_DIR}"
  "${PY}" "${CODE_DIR}/nano_bench.py" --model "${MODEL}" --out "${OUT}/offline/nano.json" || rc=1
  grep -h NANO_BENCH "${OUT}/logs/nano.log" >&3
  nsys_run nano_static_b1 "${PY}" "${CODE_DIR}/nano_bench.py" --model "${MODEL}" \
    --runs static:1:128:64 --out "${OUT}/offline/nano_static_b1_nsys.json" || rc=1
  nsys_run nano_paged_b1 "${PY}" "${CODE_DIR}/nano_bench.py" --model "${MODEL}" \
    --runs paged:1:128:64 --out "${OUT}/offline/nano_paged_b1_nsys.json" || rc=1
  nsys_run nano_paged_b16 "${PY}" "${CODE_DIR}/nano_bench.py" --model "${MODEL}" \
    --runs paged:16:1024:64 --out "${OUT}/offline/nano_paged_b16_nsys.json" || rc=1
  return "${rc}"
}

# --- ncu: hardware counters for one decode step (needs CAP_SYS_ADMIN; Colab yes, RunPod no)
ncu_run() {  # label launch_count command...
  local label=$1 count=$2; shift 2
  local ncu
  if [[ -f "${OUT}/ncu/${label}.json" ]]; then echo "${label}: already analyzed, skipped"; return 0; fi
  ncu="$(find_ncu)"
  [[ -n "${ncu}" ]] || { echo "ncu not installed"; return 1; }
  # --clock-control none: real boost clocks, so durations line up with the nsys/serve timings.
  timeout 2400 "${ncu}" --profile-from-start off --graph-profiling node --launch-count "${count}" \
    --clock-control none \
    --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
    --section LaunchStats --section WarpStateStats \
    -f -o "${OUT}/ncu/${label}" "$@" || return 1
  "${ncu}" --import "${OUT}/ncu/${label}.ncu-rep" --csv --page raw > "${OUT}/ncu/${label}_raw.csv" || return 1
  "${PY}" "${CODE_DIR}/analyze_ncu.py" "${OUT}/ncu/${label}_raw.csv" --label "${label}" \
    --out-dir "${OUT}/ncu" >&3
}

ncu_kernels() {
  local rc=0
  # ncu backs up the memory a kernel may write before each of its ~13 passes; with
  # vLLM's default 0.85 of a 40 GB GPU taken the backup goes to the host and one
  # kernel takes ~20 s. At 0.45 (weights + a small KV cache) it stays on the GPU.
  # b=64 needs a bigger KV cache, so its breakdown comes from nsys only.
  ncu_run vllm_graph_b1 400 "${PY}" "${CODE_DIR}/vllm_offline.py" --model "${MODEL}" \
    --batch 1 --input-len 128 --output-len 32 --profile-steps 1 --gpu-memory-utilization 0.45 \
    --out "${OUT}/offline/vllm_graph_b1_ncu.json" || rc=1
  # nano launches ~30 kernels per layer; 300 covers ~10 identical layers of one step.
  export PYTHONPATH="${REPO_DIR}"
  ncu_run nano_static_b1 300 "${PY}" "${CODE_DIR}/nano_bench.py" \
    --model "${MODEL}" --runs static:1:128:4 --out "${OUT}/offline/nano_static_b1_ncu.json" || rc=1
  return "${rc}"
}

environment() {
  {
    echo "run_id: ${RUN_ID}"
    nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv
    nvidia-smi topo -m | head -5
    grep RmProfilingAdminOnly /proc/driver/nvidia/params
    lscpu | grep -E 'Model name|^CPU\(s\)'
    free -g | head -2
    "${PY}" -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__)"
    "${VENVS}/sglang/bin/python" -c "import sglang, torch; print('sglang', sglang.__version__, '| torch', torch.__version__)
try:
    import flashinfer; print('flashinfer', flashinfer.__version__)
except Exception as e: print('flashinfer', e)"
    "$(find_nsys)" --version
    git -C "${REPO_DIR}" log -1 --format='repo %h %s' 2>/dev/null || echo "repo: uploaded working tree"
    "$(find_ncu)" --version | tail -1
  } > "${OUT}/environment.txt" 2>&1
  cat "${OUT}/environment.txt" >&3
}

summarize() {
  local gpu=A100-SXM4-40GB
  nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 80GB && gpu=A100-SXM4-80GB
  "${PY}" "${CODE_DIR}/roofline.py" --gpu "${gpu}" --bandwidth-json "${OUT}/bandwidth.json" \
    --out "${OUT}/roofline.json" > "${OUT}/roofline.md"
  "${PY}" "${CODE_DIR}/summarize.py" "${OUT}" >&3
}

# --- main ------------------------------------------------------------------
log "phase0 run ${RUN_ID} -> ${OUT} (steps: ${STEPS})"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader >&3
rm -rf "${OUT}/code" && cp -r "${CODE_DIR}" "${OUT}/code"

if want installs; then
  python3 -m pip install -q uv > "${OUT}/logs/install_uv.log" 2>&1 || log "pip install uv failed"
  run_step install_sglang install_sglang & sglang_pid=$!
  run_step install_nsys install_nsys & nsys_pid=$!
  run_step download_model download_model & model_pid=$!
  run_step clone_repo clone_repo
  run_step install_vllm install_vllm
  wait "${model_pid}"
  upload
fi
want bandwidth && { run_step bandwidth "${PY}" "${CODE_DIR}/bandwidth.py" --out "${OUT}/bandwidth.json"; grep -h -A3 copy: "${OUT}/logs/bandwidth.log" >&3; upload; }
want vllm_serve && { run_step vllm_serve vllm_serve; upload; }
if want installs; then wait "${sglang_pid}" "${nsys_pid}"; fi
want sglang_serve && { run_step sglang_serve sglang_serve; upload; }
want nsys_vllm && { run_step nsys_vllm nsys_vllm; upload; }
want nano && { run_step nano nano; upload; }
want ncu && { run_step ncu ncu_kernels; upload; }
run_step environment environment
want summarize && run_step summarize summarize
upload
log "P0_DONE ${RUN_ID}; status:"
cat "${OUT}/status.tsv" >&3

idle="${P0_IDLE_STOP:-20m}"
[[ "${idle}" == "0" ]] && exit 0
log "idle for ${idle}, then stop the pod"
sleep "${idle}"
if command -v runpodctl >/dev/null 2>&1 && [[ -n "${RUNPOD_POD_ID:-}" ]]; then
  runpodctl stop pod "${RUNPOD_POD_ID}"
fi
sleep infinity
