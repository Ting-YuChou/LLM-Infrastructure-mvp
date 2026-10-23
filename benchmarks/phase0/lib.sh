#!/usr/bin/env bash
# Shared by run_phase0.sh and benchmarks/phase{1,2}/run_phase*.sh: step
# runner, installs, serving benchmarks, nsys and ncu captures. Source it after
# setting
#   OUT          run directory (logs/, nsys/, ncu/ go under it)
#   STEPS        space-separated steps this run does (for want)
#   VENVS, PY    venv root and the vLLM venv's python
#   VLLM_VERSION vllm pin for install_vllm
#   MODEL_ID     HF repo to download; MODEL: where it goes
# and defining log() (writes to the console on fd 3). The analyzers come from
# this directory, whichever phase sources it.

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# --- installs --------------------------------------------------------------
install_vllm() {
  uv venv --seed --python 3.12 "${VENVS}/vllm" &&
  uv pip install -p "${PY}" "vllm==${VLLM_VERSION}" huggingface_hub matplotlib &&
  "${PY}" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda)"
}

# SGLANG_SPEC: the pip spec (phase 0: sglang[all]).
install_sglang() {
  # ninja: flashinfer JIT-compiles kernels when SGLang captures its CUDA graphs.
  { [[ -x "${VENVS}/sglang/bin/python" ]] || uv venv --seed --python 3.12 "${VENVS}/sglang"; } &&
  { uv pip install -p "${VENVS}/sglang/bin/python" "${SGLANG_SPEC}" ninja ||
    uv pip install -p "${VENVS}/sglang/bin/python" sglang ninja; } &&
  "${VENVS}/sglang/bin/python" -c "import sglang, torch; print('sglang', sglang.__version__, 'torch', torch.__version__)"
}

cuda_apt_repo() {  # probe_package: add NVIDIA's CUDA apt repo unless apt already offers the package
  apt-get update -qq
  if ! apt-cache policy "$1" | grep -q 'Candidate: [0-9]'; then
    local ubuntu  # NVIDIA's CUDA apt repo is not configured in this image
    ubuntu="$(. /etc/os-release && echo "${VERSION_ID//./}")"
    wget -q -O /tmp/cuda-keyring.deb \
      "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${ubuntu}/x86_64/cuda-keyring_1.1-1_all.deb" &&
      dpkg -i /tmp/cuda-keyring.deb && apt-get update -qq
  fi
}

# CUDA 12 nvcc for torch.utils.cpp_extension (nano_vllm's CUDA kernels); needs
# ROOT, PY and CUDA_ENV (the env file it writes; later steps source it).
cuda129() {
  # torch.utils.cpp_extension builds with $CUDA_HOME/bin/nvcc and only checks the
  # CUDA major version, so Colab's system nvcc 13.0 would build against cudart 13
  # while torch runs on CUDA 12. Install the toolkit torch was built with.
  local version cuda pkg
  version="${P1_CUDA_VERSION:-$("${PY}" -c "import torch; print(torch.version.cuda)")}" || return 1
  [[ "${version}" == 12.* ]] || { echo "unexpected CUDA version '${version}'"; return 1; }
  [[ "${version}" == 12.9 ]] || echo "NOTE: the venv's torch is CUDA ${version}, not 12.9; installing ${version}"
  cuda="/usr/local/cuda-${version}"
  pkg="${version/./-}"
  echo "/usr/local/cuda -> $(readlink -f /usr/local/cuda) before"
  if [[ ! -x "${cuda}/bin/nvcc" ]]; then
    cuda_apt_repo "cuda-nvcc-${pkg}" &&
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      "cuda-nvcc-${pkg}" "cuda-cudart-dev-${pkg}" "cuda-cccl-${pkg}" || return 1
  fi
  echo "/usr/local/cuda -> $(readlink -f /usr/local/cuda) after"
  "${cuda}/bin/nvcc" --version || return 1
  cat > "${CUDA_ENV}" <<EOF
export CUDA_HOME=${cuda}
export PATH=${cuda}/bin:\${PATH}
export TORCH_CUDA_ARCH_LIST=8.0
export TORCH_EXTENSIONS_DIR=${ROOT}/torch_ext
EOF
  cat "${CUDA_ENV}"
}

install_nsys() {
  if [[ -n "$(find_nsys)" ]]; then find_nsys; return 0; fi
  cuda_apt_repo cuda-nsight-systems-12-8
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

# --- tests: nano_vllm on the GPU (REPO_DIR, OUT/tests) ------------------------
gpu_tests() {
  local paths=() rc=0
  if [[ -d "${REPO_DIR}/tests/gpu" ]]; then paths+=(tests/gpu); else echo "no tests/gpu yet: nano_vllm unit tests only"; fi
  (cd "${REPO_DIR}" && PYTHONPATH="${REPO_DIR}" "${PY}" -m pytest -q -rs --junitxml="${OUT}/tests/junit.xml" \
    ${paths[@]+"${paths[@]}"} tests/unit/test_nano_vllm_*.py) | tee "${OUT}/tests/pytest.log" || rc=1
  # Same rule as CI's training job: on the GPU every test must run.
  if grep -q '^SKIPPED' "${OUT}/tests/pytest.log"; then echo "tests were skipped on the GPU"; rc=1; fi
  { tail -1 "${OUT}/tests/pytest.log"; grep '^SKIPPED\|^FAILED\|^ERROR' "${OUT}/tests/pytest.log"; } >&3
  return "${rc}"
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

start_server() {  # name port ready_path command...: own process group, wait until ready
  local name=$1 port=$2 ready=$3; shift 3
  local start
  start=$(date +%s)
  setsid "$@" > "${OUT}/logs/${name}_server.log" 2>&1 &
  SERVER_PID=$!
  if ! wait_ready "http://127.0.0.1:${port}${ready}" 1200; then
    log "${name} server did not become ready"; tail -40 "${OUT}/logs/${name}_server.log" >&3
    stop_server; return 1
  fi
  echo "{\"engine\": \"${name}\", \"startup_s\": $(( $(date +%s) - start )), \"args\": \"${*//\"/}\"}" \
    > "${OUT}/serve/${name}_startup.json"
  log "${name} ready after $(( $(date +%s) - start ))s; $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) in use"
}

# BENCH_TEMPERATURE, BENCH_TOP_P: sent with every request when set. vllm bench
# serve 0.19 sends none by default, so each server applies its own default
# (vLLM: the model's generation_config).
bench_serve() {  # name port label num_prompts output_len [vllm bench serve args...]
  local name=$1 port=$2 label=$3 n=$4 out_len=$5; shift 5
  timeout "${BENCH_TIMEOUT:-1200}" "${VENVS}/vllm/bin/vllm" bench serve \
    --backend vllm --base-url "http://127.0.0.1:${port}" --model "${MODEL}" \
    --ignore-eos --num-prompts "${n}" --seed 0 ${BENCH_TEMPERATURE:+--temperature "${BENCH_TEMPERATURE}"} \
    ${BENCH_TOP_P:+--top-p "${BENCH_TOP_P}"} \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "${OUT}/serve" --result-filename "${name}_${label}.json" "$@"
  local rc=$?
  "${PY}" "${LIB_DIR}/serve_summary.py" "${OUT}/serve/${name}_${label}.json" "${name}" "${label}" \
    "${n}" "${out_len}" --tag "${SERVE_TAG:-P0_SERVE}" ${SERVE_KEEP_RAW:+--keep-raw} >&3
  return "${rc}"
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
  nsys_analyze "${label}"
}

nsys_analyze() {  # label: export <label>.nsys-rep to SQLite, stats CSVs, analyze_nsys.py
  local label=$1 nsys
  nsys="$(find_nsys)"
  "${nsys}" export --type sqlite -f true -o "${OUT}/nsys/${label}.sqlite" "${OUT}/nsys/${label}.nsys-rep" || return 1
  "${nsys}" stats -q -f csv --report cuda_gpu_kern_sum --report cuda_api_sum \
    -o "${OUT}/nsys/${label}" "${OUT}/nsys/${label}.sqlite"
  "${PY}" "${LIB_DIR}/analyze_nsys.py" "${OUT}/nsys/${label}.sqlite" --label "${label}" \
    --out-dir "${OUT}/nsys" >&3
}

# --- ncu: hardware counters (needs CAP_SYS_ADMIN; Colab yes, RunPod no) -------
ncu_run() {  # label launch_count [ncu options... --] command...
  local label=$1 count=$2; shift 2
  local ncu arg extra=()
  for arg in "$@"; do  # options before a "--" go to ncu, e.g. --kernel-name regex:...
    if [[ "${arg}" == -- ]]; then
      while [[ "$1" != -- ]]; do extra+=("$1"); shift; done
      shift
      break
    fi
  done
  if [[ -f "${OUT}/ncu/${label}.json" ]]; then echo "${label}: already analyzed, skipped"; return 0; fi
  ncu="$(find_ncu)"
  [[ -n "${ncu}" ]] || { echo "ncu not installed"; return 1; }
  # --clock-control none: real boost clocks, so durations line up with the nsys/serve timings.
  timeout 2400 "${ncu}" --profile-from-start off --graph-profiling node --launch-count "${count}" \
    --clock-control none \
    --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
    --section LaunchStats --section WarpStateStats ${extra[@]+"${extra[@]}"} \
    -f -o "${OUT}/ncu/${label}" "$@" || return 1
  "${ncu}" --import "${OUT}/ncu/${label}.ncu-rep" --csv --page raw > "${OUT}/ncu/${label}_raw.csv" || return 1
  "${PY}" "${LIB_DIR}/analyze_ncu.py" "${OUT}/ncu/${label}_raw.csv" --label "${label}" \
    --out-dir "${OUT}/ncu" >&3
}
