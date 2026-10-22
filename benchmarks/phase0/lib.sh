#!/usr/bin/env bash
# Shared by run_phase0.sh and benchmarks/phase1/run_phase1.sh: step runner,
# installs, nsys and ncu captures. Source it after setting
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
