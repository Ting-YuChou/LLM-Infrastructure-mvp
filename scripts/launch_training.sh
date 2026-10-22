#!/usr/bin/env bash
# Launch training on every GPU of this node (e.g. a RunPod 8x H100/A100 pod).
#
#   scripts/launch_training.sh all              # sft -> reward -> rlhf -> handoff
#   scripts/launch_training.sh sft              # one stage under torchrun
#   scripts/launch_training.sh sft,reward       # several stages via the pipeline
#   scripts/launch_training.sh dpo              # DPO from the SFT checkpoint
#   scripts/launch_training.sh rlhf             # PPO (stages.rlhf.backend: native | verl)
#
# Env: HF_TOKEN (needed for gated Llama 3), NPROC_PER_NODE (default: all
# GPUs), PIPELINE_CONFIG, and any extra train_pipeline.py flags after the stage.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGES="${1:-all}"
shift || true
PIPELINE_CONFIG="${PIPELINE_CONFIG:-${ROOT_DIR}/config/training_pipeline.yaml}"

# A warning, not an error: ungated copies (e.g. unsloth/Llama-3.2-1B) need no token.
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set; gated models such as meta-llama/Meta-Llama-3-8B will not download." >&2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
  GPU_COUNT=1
fi
NPROC="${NPROC_PER_NODE:-${GPU_COUNT}}"

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "Stages: ${STAGES} | GPUs: ${NPROC} | config: ${PIPELINE_CONFIG}"

case "${STAGES}" in
  # rlhf goes through the pipeline, which starts torchrun for the native backend
  # and runs verl (stages.rlhf.backend: verl) as one process.
  sft|reward|dpo)
    exec torchrun --standalone --nproc_per_node "${NPROC}" \
      -m training.stage_runner --pipeline-config "${PIPELINE_CONFIG}" --stage "${STAGES}" "$@"
    ;;
  *)
    exec python "${ROOT_DIR}/scripts/train_pipeline.py" --config "${PIPELINE_CONFIG}" \
      --stages "${STAGES}" --nproc-per-node "${NPROC}" "$@"
    ;;
esac
