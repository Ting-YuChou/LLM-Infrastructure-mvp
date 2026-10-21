#!/usr/bin/env bash
# PPO with verl (Ray + vLLM rollout + FSDP actor/critic) from our SFT and reward
# checkpoints. Runs in verl's own uv environment (see verl_ppo/README.md); the
# overrides mirror config/rlhf_config.yaml so timings compare with
# src/training/rlhf_trainer.py.
#
#   MODE=lora SFT_CKPT=... RM_CKPT=... DATA_DIR=... OUTPUT_DIR=... VERL_DIR=... verl_ppo/run_ppo.sh
#
# MODE      lora (policy LoRA, like our PPO) | full (full-parameter FSDP actor)
# SFT_CKPT  actor / reference / rollout init (HF checkpoint with our chat template)
# RM_CKPT   reward model (LlamaForSequenceClassification) and critic init
# DATA_DIR  train.parquet / val.parquet from verl_ppo/prepare_prompts.py
# VERL_DIR  verl checkout at the pinned tag, already `uv sync`ed
# NGPUS, BATCH (prompts per step), PROMPT_LEN, RESPONSE_LEN, STEPS override the defaults.
# VERL_DRY_RUN=1 prints the commands instead of running them.
set -euo pipefail

MODE="${MODE:?MODE=lora|full}"
SFT_CKPT="${SFT_CKPT:?SFT_CKPT}"
RM_CKPT="${RM_CKPT:?RM_CKPT}"
DATA_DIR="${DATA_DIR:?DATA_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR}"
VERL_DIR="${VERL_DIR:?VERL_DIR}"
NGPUS="${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}"
BATCH="${BATCH:-64}"
PROMPT_LEN="${PROMPT_LEN:-256}"
RESPONSE_LEN="${RESPONSE_LEN:-512}"
STEPS="${STEPS:-6}"
# The venv's own python, not `uv run`: under `uv run`, Ray starts each worker
# through `uv run` as well, and the LoRA run's workers died with SIGBUS while
# importing native libraries from the shared venv.
VERL_PY="${VERL_DIR}/.venv/bin/python"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

case "${MODE}" in
  lora)
    # config/rlhf_config.yaml: r=16, alpha=32 on all seven Llama projections.
    MODE_ARGS=(
      actor_rollout_ref.model.lora_rank=16
      actor_rollout_ref.model.lora_alpha=32
      actor_rollout_ref.model.target_modules=all-linear
      actor_rollout_ref.rollout.load_format=safetensors
      actor_rollout_ref.actor.optim.lr=1e-5
      # verl v0.9.1's V1 trainer leaves the reference pass's no_lora_adapter
      # flag on the shared batch, so the (non-LoRA) critic then calls
      # disable_adapter and fails. The legacy trainer builds fresh metadata per call.
      trainer.use_v1=False
    )
    ;;
  full)
    MODE_ARGS=(
      actor_rollout_ref.actor.optim.lr=1e-6
      actor_rollout_ref.actor.fsdp_config.param_offload=True
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
      actor_rollout_ref.ref.fsdp_config.param_offload=True
      critic.fsdp.param_offload=True
      critic.fsdp.optimizer_offload=True
    )
    ;;
  *)
    echo "MODE must be lora or full, got ${MODE}" >&2
    exit 2
    ;;
esac

OVERRIDES=(
  # Data: chat-message prompts; verl applies the SFT tokenizer's chat template.
  data.train_files="${DATA_DIR}/train.parquet"
  data.val_files="${DATA_DIR}/val.parquet"
  data.train_batch_size="${BATCH}"
  data.max_prompt_length="${PROMPT_LEN}"
  data.max_response_length="${RESPONSE_LEN}"
  data.filter_overlong_prompts=True
  data.truncation=error
  # Actor / reference / rollout from the SFT checkpoint.
  actor_rollout_ref.model.path="${SFT_CKPT}"
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.actor.strategy=fsdp
  # One full-batch update per PPO epoch, two epochs (our ppo_update loop).
  actor_rollout_ref.actor.ppo_mini_batch_size="${BATCH}"
  actor_rollout_ref.actor.ppo_epochs=2
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
  actor_rollout_ref.actor.clip_ratio=0.2
  actor_rollout_ref.actor.entropy_coeff=0.01
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.temperature=0.7
  actor_rollout_ref.rollout.top_p=0.9
  actor_rollout_ref.rollout.n=1
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
  # Critic starts from the reward model (its score head is the value head).
  critic.model.path="${RM_CKPT}"
  critic.model.enable_gradient_checkpointing=True
  critic.optim.lr=5e-5
  critic.ppo_micro_batch_size_per_gpu=4
  critic.cliprange_value=0.2
  # Our reward model, served by vLLM's classify endpoint (raw logit).
  reward.reward_model.enable=True
  reward.reward_model.model_path="${RM_CKPT}"
  reward.reward_model.n_gpus_per_node="${NGPUS}"
  reward.reward_model.rollout.name=vllm
  reward.reward_model.rollout.tensor_model_parallel_size=1
  reward.reward_model.rollout.gpu_memory_utilization=0.3
  reward.reward_model.rollout.prompt_length=1024
  reward.reward_model.rollout.response_length=1024
  # Per-token KL penalty in the reward with an adaptive coefficient, GAE.
  algorithm.adv_estimator=gae
  algorithm.gamma=0.99
  algorithm.lam=0.95
  algorithm.use_kl_in_reward=True
  algorithm.kl_ctrl.type=adaptive
  algorithm.kl_ctrl.kl_coef=0.05
  algorithm.kl_ctrl.target_kl=0.1
  trainer.logger=[console]
  trainer.project_name=llm_infra_verl
  trainer.experiment_name="ppo_${MODE}"
  trainer.default_local_dir="${OUTPUT_DIR}/checkpoints"
  trainer.n_gpus_per_node="${NGPUS}"
  trainer.nnodes=1
  trainer.total_training_steps="${STEPS}"
  trainer.save_freq="${STEPS}"
  trainer.test_freq=-1
  trainer.val_before_train=False
  trainer.resume_mode=disable
)

MERGE=("${VERL_PY}" -m verl.model_merger merge --backend fsdp
  --local_dir "${OUTPUT_DIR}/checkpoints/global_step_${STEPS}/actor"
  --target_dir "${OUTPUT_DIR}/hf")

if [[ "${VERL_DRY_RUN:-0}" == "1" ]]; then
  printf '%s\n' "${VERL_PY}" -m verl.trainer.main_ppo "${MODE_ARGS[@]}" "${OVERRIDES[@]}"
  echo "---"
  printf '%s\n' "${MERGE[@]}"
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"
cd "${VERL_DIR}"
PYTHONUNBUFFERED=1 "${VERL_PY}" -m verl.trainer.main_ppo "${MODE_ARGS[@]}" "${OVERRIDES[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/train.log"
"${MERGE[@]}" 2>&1 | tee "${OUTPUT_DIR}/merge.log"
