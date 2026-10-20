#!/usr/bin/env bash
# GPU smoke test of the training stages on one RunPod pod (the pod's start command).
#
# The pod clones this repo at REPO_REF and runs this script, which installs the
# pinned training stack, runs sft -> reward -> dpo -> rlhf (+ an SFT resume from
# the sharded checkpoint) for a few steps each with scripts/launch_training.sh,
# and writes <run>/summary.json via scripts/gpu_smoke_report.py. Everything
# lands on the volume at SMOKE_ROOT, one run directory per commit and preset,
# so a pod restart after a new push starts a fresh run and a restart without
# one does not repeat a finished run.
#
# Pod start command:
#   bash -c 'git clone --depth 1 --branch "$REPO_REF" https://github.com/Ting-YuChou/LLM-Infrastructure-mvp.git /root/repo && exec bash /root/repo/scripts/runpod_gpu_smoke.sh'
#
# Env: HF_TOKEN (gated Llama weights), SMOKE_PRESET (1b | 8b), SMOKE_MODEL,
# SMOKE_ROOT (default /workspace/smoke), SMOKE_STAGES, SMOKE_STAGE_TIMEOUT,
# SMOKE_PRUNE (1: record then delete each stage's intermediate checkpoints
# once it is done; an 8B one is ~80 GB),
# SMOKE_IDLE_STOP (how long to keep the pod up after finishing so the logs can
# be read, then stop it with runpodctl when available).
#
# verl comparison (SMOKE_PRESET=8b_ppo_bench): add the stages
# `verl_setup verl_lora verl_full` after `sft reward rlhf`. verl gets its own uv
# environment (verl_ppo/setup_env.sh, VERL_REF, default v0.9.1) and trains PPO
# from the same SFT/RM checkpoints with the rlhf stage's batch and lengths.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

SMOKE_ROOT="${SMOKE_ROOT:-/workspace/smoke}"
SMOKE_PRESET="${SMOKE_PRESET:-1b}"
# sft_resume runs right after sft so sft's checkpoints can be pruned early.
SMOKE_STAGES="${SMOKE_STAGES:-sft sft_resume reward dpo rlhf}"
SMOKE_PRUNE="${SMOKE_PRUNE:-1}"
SMOKE_STAGE_TIMEOUT="${SMOKE_STAGE_TIMEOUT:-90m}"
SMOKE_IDLE_STOP="${SMOKE_IDLE_STOP:-30m}"
VERL_REF="${VERL_REF:-v0.9.1}"
VERL_DIR="${SMOKE_ROOT}/verl-${VERL_REF}"
# SMOKE_RUN_ID reruns stages inside an earlier run (e.g. verl stages on its
# SFT/RM checkpoints); new rows are appended and the report keeps the latest.
RUN_ID="${SMOKE_RUN_ID:-$(git rev-parse --short HEAD)-${SMOKE_PRESET}}"
RUN_DIR="${SMOKE_ROOT}/runs/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"

log() { echo "[smoke $(date -u +%H:%M:%S)] $*"; }

idle_then_stop() {
  log "idle for ${SMOKE_IDLE_STOP} so the logs can be read"
  sleep "${SMOKE_IDLE_STOP}"
  if command -v runpodctl >/dev/null 2>&1 && [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    log "stopping pod ${RUNPOD_POD_ID}"
    runpodctl stop pod "${RUNPOD_POD_ID}" || true
  fi
  sleep infinity
}

if [[ -n "${SMOKE_RUN_ID:-}" ]]; then rm -f "${RUN_DIR}/DONE"; fi
if [[ -f "${RUN_DIR}/DONE" ]]; then
  log "run ${RUN_ID} already finished; summary:"
  cat "${RUN_DIR}/summary.json" || true
  idle_then_stop
fi

mkdir -p "${LOG_DIR}"
[[ -n "${SMOKE_RUN_ID:-}" ]] || : > "${RUN_DIR}/timings.tsv"
log "run ${RUN_ID} -> ${RUN_DIR}"

export HF_HOME="${SMOKE_ROOT}/hf_cache"
export PIP_CACHE_DIR="${SMOKE_ROOT}/pip_cache"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export UV_CACHE_DIR="${SMOKE_ROOT}/uv_cache"
export UV_PYTHON_INSTALL_DIR="${SMOKE_ROOT}/uv_python"
export UV_PYTHON=3.12  # verl's uv.lock needs >= 3.12; the image has 3.11

record() {  # stage exit_code start_epoch
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(date +%s)" >> "${RUN_DIR}/timings.tsv"
}

# --- 1. Pinned training stack (CUDA 12.9 torch wheel, then the CI pins) -----
log "installing torch 2.10.0+cu129 and requirements.training-ci.txt"
start=$(date +%s)
{
  python -m pip install --upgrade pip &&
  # Base images ship torchvision/torchaudio built for their own torch; with
  # torch 2.10 transformers' torchvision import fails, and training needs neither.
  python -m pip uninstall -y torchvision torchaudio &&
  python -m pip install --index-url https://download.pytorch.org/whl/cu129 \
    "$(grep '^torch==' requirements.training-ci.txt)" &&
  python -m pip install -r requirements.training-ci.txt
} > "${LOG_DIR}/install.log" 2>&1
rc=$?
record install "${rc}" "${start}"
if [[ ${rc} -ne 0 ]]; then
  log "install failed:"; tail -50 "${LOG_DIR}/install.log"
  touch "${RUN_DIR}/DONE"; idle_then_stop
fi

# --- 2. Environment and NCCL check ------------------------------------------
NPROC="$(nvidia-smi -L | wc -l | tr -d ' ')"
nvidia-smi
start=$(date +%s)
python - <<'EOF' > "${RUN_DIR}/environment.json"
import json, platform, torch, transformers, accelerate, datasets, peft
print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "nccl": ".".join(map(str, torch.cuda.nccl.version())),
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "datasets": datasets.__version__,
    "peft": peft.__version__,
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "bf16": torch.cuda.is_bf16_supported(),
}))
EOF
rc=$?
record environment "${rc}" "${start}"
cat "${RUN_DIR}/environment.json"
if [[ ${rc} -ne 0 ]]; then
  log "training stack does not import; see the traceback above"
  touch "${RUN_DIR}/DONE"; idle_then_stop
fi

cat > /tmp/nccl_check.py <<'EOF'
import os, torch, torch.distributed as dist
dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
x = torch.ones(1 << 20, device="cuda") * (rank + 1)
dist.all_reduce(x)
expected = sum(range(1, dist.get_world_size() + 1))
assert torch.allclose(x, torch.full_like(x, expected)), x[:4]
if rank == 0:
    print(f"NCCL all_reduce ok across {dist.get_world_size()} GPUs")
dist.destroy_process_group()
EOF
nccl_check() {
  timeout 120 torchrun --standalone --nproc_per_node "${NPROC}" /tmp/nccl_check.py 2>&1 | tee -a "${LOG_DIR}/nccl.log"
  return "${PIPESTATUS[0]}"
}
start=$(date +%s)
nccl_check
rc=$?
if [[ ${rc} -ne 0 ]]; then
  # PCIe hosts where GPU peer-to-peer hangs: route NCCL through host memory.
  log "NCCL check failed (exit ${rc}); retrying with NCCL_P2P_DISABLE=1"
  export NCCL_P2P_DISABLE=1
  nccl_check
  rc=$?
fi
record nccl_check "${rc}" "${start}"
echo "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}" >> "${LOG_DIR}/nccl.log"
if [[ ${rc} -ne 0 ]]; then
  log "NCCL all_reduce fails even without P2P"
  touch "${RUN_DIR}/DONE"; idle_then_stop
fi

# --- 3. Data, configs, gated-model access -----------------------------------
RL_PROMPTS="$(python -c "import sys; sys.path.insert(0, 'scripts'); import make_smoke_configs as m; print(m.PRESETS['${SMOKE_PRESET}']['rl_prompts'])")"
python scripts/prepare_data.py --output-dir "${RUN_DIR}/data" \
  --sft-examples 400 --preference-examples 400 --rl-prompts "${RL_PROMPTS}" > "${LOG_DIR}/data.log" 2>&1
python scripts/make_smoke_configs.py --preset "${SMOKE_PRESET}" --root "${RUN_DIR}" \
  ${SMOKE_MODEL:+--model "${SMOKE_MODEL}"}
export PIPELINE_CONFIG="${RUN_DIR}/configs/training_pipeline.yaml"
BASE_MODEL="$(python -c "import yaml; print(yaml.safe_load(open('${RUN_DIR}/configs/sft.yaml'))['model']['name'])")"
start=$(date +%s)
# model_info is public even for gated repos; downloading a file proves access.
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('${BASE_MODEL}', 'config.json'); print('HF access ok: ${BASE_MODEL}')" \
  2>&1 | tail -3
rc=${PIPESTATUS[0]}
record hf_access "${rc}" "${start}"
if [[ ${rc} -ne 0 ]]; then
  log "cannot read ${BASE_MODEL}: set HF_TOKEN and accept the model license on huggingface.co"
  touch "${RUN_DIR}/DONE"; idle_then_stop
fi

# --- 4. Stages ---------------------------------------------------------------
prune() {
  [[ "${SMOKE_PRUNE}" == "1" ]] && python scripts/gpu_smoke_report.py --run-dir "${RUN_DIR}" --prune-stage "$1"
}

# verl PPO from the SFT/RM checkpoints, with the rlhf stage's prompts per step,
# lengths and step count so its timings line up with ours.
verl_stage() {  # lora | full
  local mode="$1" output="${RUN_DIR}/outputs/verl_$1"
  local bench
  bench="$(python - "${RUN_DIR}/configs/rlhf.yaml" "${NPROC}" <<'EOF'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
training, data = config["training"], config["data"]
print(training["per_device_batch_size"] * int(sys.argv[2]), training["total_steps"],
      data["max_prompt_length"], data["max_response_length"])
EOF
)"
  read -r BATCH STEPS PROMPT_LEN RESPONSE_LEN <<< "${bench}"
  python verl_ppo/prepare_prompts.py --input "${RUN_DIR}/data/prompts/rl_prompts.json" \
    --output-dir "${RUN_DIR}/verl_data" &&
  MODE="${mode}" SFT_CKPT="${RUN_DIR}/outputs/sft/final" RM_CKPT="${RUN_DIR}/outputs/reward/final" \
    DATA_DIR="${RUN_DIR}/verl_data" OUTPUT_DIR="${output}" VERL_DIR="${VERL_DIR}" NGPUS="${NPROC}" \
    BATCH="${BATCH}" STEPS="${STEPS}" PROMPT_LEN="${PROMPT_LEN}" RESPONSE_LEN="${RESPONSE_LEN}" \
    bash verl_ppo/run_ppo.sh &&
  (cd "${VERL_DIR}" && uv run --frozen --all-packages --extra vllm --extra fsdp \
    python "${REPO_DIR}/verl_ppo/verl_report.py" --run-dir "${output}" \
    --sft-ckpt "${RUN_DIR}/outputs/sft/final" --batch "${BATCH}" --output "${LOG_DIR}/verl_${mode}.json")
  local rc=$?
  # Actor + critic FSDP checkpoints with optimizer state; the merged actor stays.
  if [[ "${SMOKE_PRUNE}" == "1" ]]; then rm -rf "${output}/checkpoints"; fi
  return "${rc}"
}

monitor_pid=""
start_monitor() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 5 \
    > "${LOG_DIR}/gpu_$1.csv" 2>/dev/null &
  monitor_pid=$!
}
stop_monitor() { [[ -n "${monitor_pid}" ]] && kill "${monitor_pid}" 2>/dev/null; monitor_pid=""; }

for stage in ${SMOKE_STAGES}; do
  log "=== stage ${stage} on ${NPROC} GPUs ==="
  start=$(date +%s)
  start_monitor "${stage}"
  case "${stage}" in
    sft_resume)
      timeout "${SMOKE_STAGE_TIMEOUT}" torchrun --standalone --nproc_per_node "${NPROC}" \
        -m training.sft_trainer --config "${RUN_DIR}/configs/sft_resume.yaml" 2>&1 | tee "${LOG_DIR}/${stage}.log"
      rc=${PIPESTATUS[0]} ;;
    verl_setup)
      VERL_DIR="${VERL_DIR}" VERL_REF="${VERL_REF}" timeout "${SMOKE_STAGE_TIMEOUT}" \
        bash verl_ppo/setup_env.sh 2>&1 | tee "${LOG_DIR}/${stage}.log"
      rc=${PIPESTATUS[0]} ;;
    verl_lora|verl_full)
      verl_stage "${stage#verl_}" 2>&1 | tee "${LOG_DIR}/${stage}.log"
      rc=${PIPESTATUS[0]} ;;
    *)
      timeout "${SMOKE_STAGE_TIMEOUT}" bash scripts/launch_training.sh "${stage}" 2>&1 | tee "${LOG_DIR}/${stage}.log"
      rc=${PIPESTATUS[0]} ;;
  esac
  stop_monitor
  record "${stage}" "${rc}" "${start}"
  log "stage ${stage} exit ${rc} after $(( $(date +%s) - start ))s"
  if [[ "${stage}" == "sft" && ${rc} -ne 0 ]]; then
    log "every later stage starts from the SFT checkpoint; stopping here"
    break
  fi
  case "${stage}" in
    sft) [[ " ${SMOKE_STAGES} " == *" sft_resume "* ]] || prune sft ;;  # resume reads them
    sft_resume) prune sft; prune sft_resume ;;
    verl_*) ;;  # verl_stage prunes its own checkpoints
    *) prune "${stage}" ;;
  esac
done

# --- 5. Report ---------------------------------------------------------------
log "=== SMOKE SUMMARY ${RUN_ID} ==="
python scripts/gpu_smoke_report.py --run-dir "${RUN_DIR}" --fsdp 2>&1 | tee "${LOG_DIR}/report.log"
log "=== SMOKE SUMMARY END (exit ${PIPESTATUS[0]}) ==="
cp "${RUN_DIR}/summary.json" "${SMOKE_ROOT}/latest_summary.json" 2>/dev/null || true
touch "${RUN_DIR}/DONE"
idle_then_stop
