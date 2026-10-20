"""
RLHF Trainer using PPO (Proximal Policy Optimization)
Implements reinforcement learning from human feedback for LLM alignment
"""

import os
import logging
from functools import partial
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    get_scheduler,
)
import numpy as np

try:
    from .results import TrainingRunResult
    from .chat_format import (
        encode,
        ensure_chat_tokenizer,
        format_prompt,
        pad_to_length,
        resize_embeddings_if_needed,
        stop_token_ids,
    )
    from .distributed import (
        all_reduce_mean,
        barrier,
        distributed_timeout_minutes,
        init_distributed,
        is_distributed,
        is_main_process,
        local_rank,
        main_process_only,
        masked_global_mean,
        masked_global_whiten,
        rank,
        reduce_scalar_metrics,
        world_size,
    )
    from .export import merge_adapter
    from .hf_config import (
        apply_lora,
        attention_implementation,
        model_load_dtype,
        resolve_distributed_strategy,
        resolve_precision,
    )
    from .mlflow_lifecycle import MLflowTrainingLifecycle, PromotionConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from results import TrainingRunResult  # type: ignore[no-redef]
    from chat_format import (  # type: ignore[no-redef]
        encode,
        ensure_chat_tokenizer,
        format_prompt,
        pad_to_length,
        resize_embeddings_if_needed,
        stop_token_ids,
    )
    from distributed import (  # type: ignore[no-redef]
        all_reduce_mean,
        barrier,
        distributed_timeout_minutes,
        init_distributed,
        is_distributed,
        is_main_process,
        local_rank,
        main_process_only,
        masked_global_mean,
        masked_global_whiten,
        rank,
        reduce_scalar_metrics,
        world_size,
    )
    from export import merge_adapter  # type: ignore[no-redef]
    from hf_config import (  # type: ignore[no-redef]
        apply_lora,
        attention_implementation,
        model_load_dtype,
        resolve_distributed_strategy,
        resolve_precision,
    )
    from mlflow_lifecycle import (  # type: ignore[no-redef]
        MLflowTrainingLifecycle,
        PromotionConfig,
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RLHFConfig:
    """RLHF training configuration"""
    policy_model_path: str
    reward_model_path: str
    output_dir: str

    # PPO hyperparameters. num_epochs counts passes over the prompts;
    # ppo_epochs counts optimizer passes per rollout batch (defaults to num_epochs).
    num_epochs: int = 4
    num_steps: int = 10000
    batch_size: int = 4  # prompts per GPU per step
    learning_rate: float = 1e-6
    value_learning_rate: Optional[float] = None  # defaults to 5x learning_rate
    ppo_epochs: Optional[int] = None
    max_grad_norm: float = 1.0
    save_steps: int = 500
    seed: int = 42

    # KL penalty
    init_kl_coef: float = 0.05
    target_kl: float = 0.1
    kl_coef_min: float = 0.001
    kl_coef_max: float = 1.0

    # PPO specific
    clip_range: float = 0.2
    value_loss_coef: float = 0.1
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 1.0  # Standard for RLHF (no discounting within response)

    # Generation
    max_prompt_length: int = 512
    max_response_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # MLflow lifecycle
    raw_config: Dict[str, Any] = field(default_factory=dict)
    config_path: Optional[str] = None
    promotion_metrics_path: Optional[str] = None
    baseline_metrics_path: Optional[str] = None
    gate_config_path: Optional[str] = None
    skip_promotion: bool = False


class PromptDataset(Dataset):
    """Chat-templated RL prompts, kept unpadded; collate_prompts pads each batch."""

    def __init__(self, prompts_file: str, tokenizer: AutoTokenizer, max_length: int = 512):
        """
        Initialize dataset

        Prompts longer than max_length are skipped: truncating a templated
        prompt would cut its BOS or its assistant header.

        Args:
            prompts_file: Path to JSON file with prompts
            tokenizer: Tokenizer (with the chat template applied)
            max_length: Maximum prompt length in tokens
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load prompts
        with open(prompts_file, 'r') as f:
            data = json.load(f)

        self.prompts: List[str] = []
        self.input_ids: List[List[int]] = []
        skipped = 0
        for item in data:
            # Objects with a "prompt" key, or bare strings
            # (Anthropic.Dataset/rl_prompts.json).
            prompt = item if isinstance(item, str) else item['prompt']
            input_ids = encode(tokenizer, format_prompt(tokenizer, prompt))
            if len(input_ids) > max_length:
                skipped += 1
                continue
            self.prompts.append(prompt)
            self.input_ids.append(input_ids)

        if skipped:
            logger.warning("Skipped %d prompts longer than %d tokens", skipped, max_length)
        logger.info(f"Loaded {len(self.prompts)} prompts from {prompts_file}")

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            'input_ids': self.input_ids[idx],
            'prompt_text': self.prompts[idx],
        }


def collate_prompts(examples: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    """
    Left-pad a batch of prompts.

    generate() appends new tokens after the last column, so right padding
    would make the model continue from pad tokens instead of the prompt.
    """
    length = max(len(example['input_ids']) for example in examples)
    input_ids = pad_to_length(
        [example['input_ids'] for example in examples], length, pad_token_id, left=True
    )
    attention_mask = pad_to_length(
        [[1] * len(example['input_ids']) for example in examples], length, 0, left=True
    )
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'prompt_text': [example['prompt_text'] for example in examples],
    }


def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Position ids that skip left padding, as generate() computes them.

    Llama's RoPE is relative, so a uniform left-pad offset barely changes the
    scores; passing the same positions keeps scoring passes consistent with
    sampling and stays correct for models with absolute position embeddings.
    """
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)


def response_mask_from_stops(response_ids: torch.Tensor, stop_ids: List[int]) -> torch.Tensor:
    """
    1 for generated tokens up to and including the first stop token, 0 after it.

    The stop token itself stays in the mask so PPO credits the decision to end
    the turn, and the reward lands on it.
    """
    stops = torch.tensor(stop_ids, device=response_ids.device)
    is_stop = torch.isin(response_ids, stops).long()
    stops_before = is_stop.cumsum(-1) - is_stop
    return (stops_before == 0).long()


class ValueHead(nn.Module):
    """
    Value function head for PPO
    Estimates state value V(s) for each token position (per-token values)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.value_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.value_head.weight, mean=0.0, std=0.01)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - outputs value for each token position

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            values: [batch_size, seq_len] - per-token value estimates
        """
        # The head stays in fp32 while the policy runs in bf16/fp16, and with
        # device_map="auto" the last hidden state can sit on another GPU.
        weight = self.value_head.weight
        hidden_states = hidden_states.to(device=weight.device, dtype=weight.dtype)
        values = self.value_head(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        return values


class ActorCritic(nn.Module):
    """
    Policy + value head as one module, so one DDP wrapper syncs both.

    One forward returns the policy logits and per-token values from the same
    last hidden state; the PPO update runs exactly one forward and one
    backward through it per optimizer step.
    """

    def __init__(self, policy: nn.Module, value_head: ValueHead):
        super().__init__()
        self.policy = policy
        self.value_head = value_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits, self.value_head(outputs.hidden_states[-1])


class RLHFTrainer:
    """
    RLHF Trainer using PPO

    Training loop:
    1. Generate responses from policy model
    2. Compute rewards using reward model
    3. Compute advantages with GAE
    4. Update policy with PPO objective
    5. Update value function
    """

    def __init__(self, config: RLHFConfig):
        """
        Initialize trainer

        Args:
            config: RLHF configuration
        """
        self.config = config
        raw_config = config.raw_config or {}
        self.device = init_distributed(distributed_timeout_minutes(raw_config))
        strategy = resolve_distributed_strategy(raw_config, world_size())
        if strategy == "fsdp":
            raise ValueError(
                "PPO supports distributed.strategy ddp (LoRA policy) or none; "
                "full-parameter PPO with FSDP is not implemented"
            )
        # Identical LoRA/value-head init on every rank (DDP also broadcasts rank 0's).
        torch.manual_seed(config.seed)

        logger.info("Initializing RLHF Trainer")
        logger.info(f"Device: {self.device} (rank {rank()}/{world_size()})")
        logger.info(f"Policy model: {config.policy_model_path}")
        logger.info(f"Reward model: {config.reward_model_path}")

        # Create output directory
        if is_main_process():
            os.makedirs(config.output_dir, exist_ok=True)

        # Load models
        self.tokenizer = ensure_chat_tokenizer(
            AutoTokenizer.from_pretrained(config.policy_model_path)
        )
        self.stop_token_ids = stop_token_ids(self.tokenizer)

        model_config = raw_config.get("model") or {}
        policy_config = model_config.get("policy") or {}
        lora_settings = raw_config.get("lora") or {}
        self.use_lora = bool(lora_settings.get("enabled", False))
        if is_distributed() and not self.use_lora:
            raise ValueError(
                "Distributed PPO needs lora.enabled: true; full-parameter PPO only "
                "runs as a single process"
            )

        # The policy is updated with a plain torch optimizer and no GradScaler,
        # so precision "fp16" loads fp32 policy weights for full-parameter PPO.
        # With LoRA the base is frozen and only the (fp32) adapter trains.
        self.precision = resolve_precision(policy_config)
        self.policy_dtype = model_load_dtype(self.precision, full_fine_tune=not self.use_lora)
        frozen_dtype = model_load_dtype(self.precision, full_fine_tune=False)
        logger.info(
            "Precision: %s (policy dtype=%s, frozen model dtype=%s, lora=%s)",
            self.precision,
            self.policy_dtype,
            frozen_dtype,
            self.use_lora,
        )
        # One full copy of each model per GPU (DDP), never split across GPUs.
        device_map = {"": self.device}

        # Policy model (model being optimized)
        logger.info("Loading policy model...")
        policy_model = AutoModelForCausalLM.from_pretrained(
            config.policy_model_path,
            torch_dtype=self.policy_dtype,
            attn_implementation=attention_implementation(policy_config),
            device_map=device_map,
        )
        resize_embeddings_if_needed(policy_model, self.tokenizer)
        if policy_config.get("gradient_checkpointing", False):
            policy_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            policy_model.config.use_cache = False

        if self.use_lora:
            # Dropout would make the rollout's old log-probs differ from the
            # update's, so step-0 ratios and the reference KL would not start at 1/0.
            if float(lora_settings.get("dropout", 0.0)) != 0.0:
                logger.warning("lora.dropout should be 0 for PPO; got %s", lora_settings["dropout"])
            policy_model = apply_lora(policy_model, lora_settings, default_dropout=0.0)
            # The reference model is the same weights with the adapter disabled.
            self.ref_model = None
        else:
            logger.info("Loading reference model...")
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                config.policy_model_path,
                torch_dtype=frozen_dtype,
                attn_implementation=attention_implementation(policy_config),
                device_map=device_map,
            )
            resize_embeddings_if_needed(self.ref_model, self.tokenizer)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        self.policy_model = policy_model
        self.policy_model.train()

        # Reward model (frozen)
        logger.info("Loading reward model...")
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            config.reward_model_path,
            num_labels=1,
            torch_dtype=frozen_dtype,
            attn_implementation=attention_implementation(policy_config),
            device_map=device_map,
        )
        self.reward_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.reward_model.eval()
        for param in self.reward_model.parameters():
            param.requires_grad = False

        # Value function: kept in fp32 next to the policy's final hidden state.
        hidden_size = self.policy_model.config.hidden_size
        self.value_head = ValueHead(hidden_size).to(self.device)
        self.actor_critic = ActorCritic(self.policy_model, self.value_head)
        if is_distributed():
            self.ac_ddp = DistributedDataParallel(
                self.actor_critic,
                device_ids=[local_rank()] if self.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            self.ac_ddp = self.actor_critic
        # Ranks must sample different responses from here on.
        torch.manual_seed(config.seed + rank())

        # One optimizer, two learning rates: the trainable policy weights (LoRA
        # adapters, or everything for full-parameter PPO) and the value head.
        value_learning_rate = config.value_learning_rate or config.learning_rate * 5
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self._trainable_policy_parameters(), "lr": config.learning_rate},
                {"params": list(self.value_head.parameters()), "lr": value_learning_rate},
            ],
            weight_decay=0.0,
        )

        # KL coefficient (adaptive)
        self.kl_coef = config.init_kl_coef

        logger.info("RLHF Trainer initialized successfully")

    def _trainable_policy_parameters(self) -> List[nn.Parameter]:
        return [param for param in self.policy_model.parameters() if param.requires_grad]

    def _build_lifecycle(self) -> MLflowTrainingLifecycle:
        """Build the MLflow lifecycle controller for this RLHF run."""
        return MLflowTrainingLifecycle.from_training_config(
            self.config.raw_config,
            model_type="rlhf",
            config_path=self.config.config_path,
            promotion=PromotionConfig(
                metrics_path=self.config.promotion_metrics_path,
                baseline_metrics_path=self.config.baseline_metrics_path,
                gate_config_path=self.config.gate_config_path,
                skip_promotion=self.config.skip_promotion,
            ),
            extra_tags={
                "artifact_type": "full_model",
                "algorithm": "ppo",
            },
        )

    @main_process_only
    def _write_promotion_metrics(
        self,
        checkpoint_path: str,
        final_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge final RLHF metrics into the pipeline promotion metrics file."""
        if not self.config.promotion_metrics_path:
            return final_metrics

        metrics_path = self.config.promotion_metrics_path
        metrics: Dict[str, Any] = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

        metrics.update(final_metrics)
        metrics["artifact_ready"] = 1.0 if os.path.isdir(checkpoint_path) else 0.0

        os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        return metrics

    @property
    def logit_temperature(self) -> float:
        """Temperature applied to logits wherever log-probs are computed."""
        return self.config.temperature if self.config.temperature > 0 else 1.0

    @torch.no_grad()
    def generate_response(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample responses from the policy model

        Log-probs and values are not taken from generate(): its scores are
        already temperature/top-p processed, which skews the PPO ratio. The
        rollout recomputes them with a forward pass instead.

        Args:
            prompt_ids: Left-padded prompt token IDs [batch_size, prompt_len]
            attention_mask: Prompt attention mask [batch_size, prompt_len]

        Returns:
            response_ids: Generated token IDs [batch_size, response_len]
            response_mask: 1 up to and including the first stop token [batch_size, response_len]
        """
        # Generate on the unwrapped policy (DDP only wraps forward). Eval mode
        # keeps the KV cache on when gradient checkpointing is enabled.
        was_training = self.policy_model.training
        self.policy_model.eval()
        try:
            sequences = self.policy_model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_response_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                use_cache=True,
                eos_token_id=self.stop_token_ids,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        finally:
            self.policy_model.train(was_training)
        response_ids = sequences[:, prompt_ids.shape[1]:]
        return response_ids, response_mask_from_stops(response_ids, self.stop_token_ids)

    @torch.no_grad()
    def compute_reward(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        response_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute scalar reward using reward model

        Args:
            prompt_ids: Prompt token IDs [batch_size, prompt_len]
            response_ids: Response token IDs [batch_size, response_len]

        Returns:
            rewards: Scalar rewards [batch_size]
        """
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        if prompt_attention_mask is None:
            prompt_attention_mask = (prompt_ids != self.tokenizer.pad_token_id).long()
        if response_attention_mask is None:
            response_attention_mask = (response_ids != self.tokenizer.pad_token_id).long()
        full_attention_mask = torch.cat(
            [
                prompt_attention_mask.long(),
                response_attention_mask.long(),
            ],
            dim=1,
        )

        outputs = self.reward_model(
            input_ids=full_ids,
            attention_mask=full_attention_mask,
            position_ids=position_ids_from_mask(full_attention_mask),
        )
        rewards = outputs.logits.squeeze(-1)

        return rewards

    def compute_rewards_per_token(
        self,
        rm_scores: torch.Tensor,
        kl_per_token: torch.Tensor,
        response_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine RM scores and KL penalty into per-token rewards

        Per-token reward structure:
        - All tokens: -kl_coef * kl_per_token (KL penalty)
        - Last valid token: += rm_score (RM reward only at end)

        Args:
            rm_scores: Scalar RM scores [batch_size]
            kl_per_token: Per-token KL penalty [batch_size, response_len]
            response_mask: Mask for valid tokens [batch_size, response_len]

        Returns:
            rewards: Per-token rewards [batch_size, response_len]
        """
        batch_size, response_len = kl_per_token.shape

        # Start with KL penalty as negative reward (we want to minimize KL)
        rewards = -self.kl_coef * kl_per_token  # [batch_size, response_len]

        # Find the last valid token index for each sample
        # response_mask: 1 for valid tokens, 0 for padding
        # Sum the mask to get the number of valid tokens, then subtract 1 for 0-indexed
        valid_lengths = response_mask.sum(dim=-1).long()  # [batch_size]
        last_valid_indices = (valid_lengths - 1).clamp(min=0)  # [batch_size]

        # Add RM score to the last valid token position
        # Create a one-hot mask for last valid positions
        batch_indices = torch.arange(batch_size, device=rewards.device)
        rewards[batch_indices, last_valid_indices] += rm_scores

        return rewards

    def _response_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Per-token log-probs of response_ids under `model`.

        Logits are divided by the sampling temperature so rollout, PPO update
        and reference all score the distribution the tokens were drawn from.

        Returns:
            log_probs: [batch_size, response_len]
            log_probs_all: [batch_size, response_len, vocab_size]
            outputs: the raw model outputs
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        prompt_length = input_ids.shape[1] - response_ids.shape[1]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids_from_mask(attention_mask),
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        log_probs, log_probs_all = self._log_probs_from_logits(
            outputs.logits, response_ids, prompt_length
        )
        return log_probs, log_probs_all, outputs

    def _log_probs_from_logits(
        self,
        logits: torch.Tensor,
        response_ids: torch.Tensor,
        prompt_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Temperature-scaled log-probs of response_ids; logits at i predict token i+1."""
        logits = logits[:, prompt_length - 1:-1, :].float() / self.logit_temperature
        log_probs_all = F.log_softmax(logits, dim=-1)
        log_probs = log_probs_all.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
        return log_probs, log_probs_all

    @torch.no_grad()
    def compute_ref_log_probs(
        self,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Per-token log-probs of the response under the frozen reference model

        The per-token KL estimate is policy_log_probs - ref_log_probs on the
        sampled tokens.

        Args:
            input_ids: Prompt + response token IDs [batch_size, seq_len]
            response_ids: Response token IDs [batch_size, response_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            ref_log_probs: [batch_size, response_len]
        """
        ref_model = getattr(self, "ref_model", None)
        if ref_model is not None:
            ref_log_probs, _, _ = self._response_log_probs(
                ref_model, input_ids, response_ids, attention_mask
            )
            return ref_log_probs
        # LoRA: the reference is the base weights, i.e. the policy with its
        # adapter switched off (no second 8B copy on the GPU).
        with self.policy_model.disable_adapter():
            ref_log_probs, _, _ = self._response_log_probs(
                self.policy_model, input_ids, response_ids, attention_mask
            )
        return ref_log_probs

    def compute_gae_per_token(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        response_mask: torch.Tensor,
        gamma: float = 1.0,
        gae_lambda: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE) per token within each response

        This computes advantages by iterating backward through each response's tokens,
        treating each token position as a timestep in the MDP.

        Args:
            rewards: Per-token rewards [batch_size, response_len]
            values: Per-token value estimates [batch_size, response_len]
            response_mask: Mask for valid tokens [batch_size, response_len]
            gamma: Discount factor (default 1.0 for RLHF)
            gae_lambda: GAE lambda parameter

        Returns:
            advantages: Per-token advantages [batch_size, response_len]
            returns: Per-token returns [batch_size, response_len]
        """
        batch_size, response_len = rewards.shape
        device = rewards.device

        # Initialize advantages tensor
        advantages = torch.zeros_like(rewards)

        # We need to iterate backward through each response
        # For the last token in each sequence, next_value = 0 (terminal state)
        # For earlier tokens, next_value = values[:, t+1]

        # Get the last valid index for each sample to handle padding correctly
        valid_lengths = response_mask.sum(dim=-1).long()  # [batch_size]

        # Initialize GAE accumulator
        gae = torch.zeros(batch_size, device=device)

        # Iterate backward through response positions
        for t in reversed(range(response_len)):
            # Check if this position is valid for each sample
            is_valid = response_mask[:, t]  # [batch_size]

            # Determine if this is the last valid token for each sample
            is_last_token = (t == (valid_lengths - 1))  # [batch_size]

            # Get next value (0 for last token or padding, values[:, t+1] otherwise)
            if t == response_len - 1:
                next_value = torch.zeros(batch_size, device=device)
            else:
                next_value = values[:, t + 1]
                # Zero out next_value for positions where t is the last valid token
                next_value = next_value * (~is_last_token).float()

            # TD error: delta = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[:, t] + gamma * next_value - values[:, t]

            # GAE accumulation: A_t = delta_t + gamma * lambda * A_{t+1}
            # Reset GAE to 0 for positions after the last valid token
            gae = delta + gamma * gae_lambda * gae * (~is_last_token).float()

            # Apply mask: set advantage to 0 for padding tokens
            advantages[:, t] = gae * is_valid

        # Compute returns: returns = advantages + values
        returns = advantages + values

        # Apply mask to returns as well
        returns = returns * response_mask

        return advantages, returns

    def _actor_critic(self, train: bool) -> nn.Module:
        """DDP-wrapped module for the update, the plain module for no-grad scoring."""
        if train and getattr(self, "ac_ddp", None) is not None:
            return self.ac_ddp
        actor_critic = getattr(self, "actor_critic", None)
        if actor_critic is None:
            actor_critic = ActorCritic(self.policy_model, self.value_head)
        return actor_critic

    def compute_log_probs_and_entropy_per_token(
        self,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        train: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute per-token log probabilities, entropy, and values for given sequences

        Used both for the rollout's old log-probs/values (under no_grad) and in
        the PPO update (train=True, through DDP), so the first update's ratio
        is exactly 1.

        Args:
            input_ids: Full input sequence (prompt + response) [batch_size, seq_len]
            response_ids: Response tokens [batch_size, response_len]
            attention_mask: Attention mask [batch_size, seq_len]
            train: Run through the DDP wrapper so gradients are synchronized

        Returns:
            log_probs: Per-token log probabilities [batch_size, response_len]
            entropy: Per-token entropy [batch_size, response_len]
            values: Per-token value estimates [batch_size, response_len]
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        prompt_length = input_ids.shape[1] - response_ids.shape[1]
        logits, all_values = self._actor_critic(train)(
            input_ids, attention_mask, position_ids_from_mask(attention_mask)
        )
        log_probs, log_probs_all = self._log_probs_from_logits(logits, response_ids, prompt_length)

        # Compute per-token entropy: -sum(p * log(p))
        entropy = -torch.sum(log_probs_all.exp() * log_probs_all, dim=-1)

        # Value at position t estimates V(s_t) before generating token t+1
        values = all_values.to(log_probs.device)[:, prompt_length - 1:-1]  # [batch_size, response_len]

        return log_probs, entropy, values

    def ppo_update(
        self,
        batch: Dict[str, torch.Tensor],
        num_ppo_epochs: int = 4
    ) -> Dict[str, float]:
        """
        PPO policy update with multiple epochs over the batch (per-token version)

        Args:
            batch: Batch of experiences containing:
                - input_ids: Full sequences (prompt + response) [batch_size, seq_len]
                - response_ids: Response tokens only [batch_size, response_len]
                - attention_mask: Attention mask [batch_size, seq_len]
                - response_mask: Mask for valid response tokens [batch_size, response_len]
                - old_log_probs: Per-token log probs from rollout [batch_size, response_len]
                - old_values: Per-token value estimates from rollout [batch_size, response_len]
                - advantages: Per-token computed advantages [batch_size, response_len]
                - returns: Per-token computed returns [batch_size, response_len]
            num_ppo_epochs: Number of PPO epochs per batch

        Returns:
            metrics: Training metrics
        """
        input_ids = batch['input_ids'].to(self.device)
        response_ids = batch['response_ids'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        response_mask = batch['response_mask'].to(self.device)
        old_log_probs = batch['old_log_probs'].to(self.device).detach()
        old_values = batch['old_values'].to(self.device).detach()
        advantages = batch['advantages'].to(self.device).detach()
        returns = batch['returns'].to(self.device).detach()

        # Normalize advantages over valid tokens, pooled across ranks so every
        # rank scales its gradients the same way. Padding positions stay 0.
        advantages = masked_global_whiten(advantages, response_mask)

        # Track metrics across PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0

        for ppo_epoch in range(num_ppo_epochs):
            # Compute new per-token log probs, entropy, and values
            new_log_probs, entropy, values = self.compute_log_probs_and_entropy_per_token(
                input_ids, response_ids, attention_mask, train=True
            )

            # Compute per-token ratio: exp(new_log_prob - old_log_prob)
            ratio = torch.exp(new_log_probs - old_log_probs)  # [batch_size, response_len]

            # Clipped surrogate objective (per-token)
            surr1 = ratio * advantages  # [batch_size, response_len]
            surr2 = torch.clamp(
                ratio,
                1.0 - self.config.clip_range,
                1.0 + self.config.clip_range
            ) * advantages

            # Policy loss: masked mean over valid tokens
            policy_loss_per_token = -torch.min(surr1, surr2)
            num_valid_tokens = response_mask.sum() + 1e-8
            policy_loss = (policy_loss_per_token * response_mask).sum() / num_valid_tokens

            # Value loss with clipping (per-token, masked)
            values_clipped = old_values + torch.clamp(
                values - old_values,
                -self.config.clip_range,
                self.config.clip_range
            )
            value_loss_unclipped = (values - returns) ** 2
            value_loss_clipped = (values_clipped - returns) ** 2
            value_loss_per_token = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped)
            value_loss = (value_loss_per_token * response_mask).sum() / num_valid_tokens

            # Entropy bonus: masked mean over valid tokens
            entropy_bonus = (entropy * response_mask).sum() / num_valid_tokens

            # Total loss
            loss = (
                policy_loss
                + self.config.value_loss_coef * value_loss
                - self.config.entropy_coef * entropy_bonus
            )

            # One backward per DDP forward; DDP all-reduces the gradients, so
            # the clipped norms below are identical on every rank.
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._trainable_policy_parameters(), self.config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            # Compute metrics (masked)
            with torch.no_grad():
                approx_kl_per_token = (ratio - 1) - torch.log(ratio)
                approx_kl = (approx_kl_per_token * response_mask).sum() / num_valid_tokens
                clip_frac_per_token = ((ratio - 1.0).abs() > self.config.clip_range).float()
                clip_frac = (clip_frac_per_token * response_mask).sum() / num_valid_tokens

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_bonus.item()
            total_kl += approx_kl.item()
            total_clip_frac += clip_frac.item()

        # Average metrics over PPO epochs, then across ranks
        num_epochs = num_ppo_epochs
        return reduce_scalar_metrics({
            'policy_loss': total_policy_loss / num_epochs,
            'value_loss': total_value_loss / num_epochs,
            'entropy': total_entropy / num_epochs,
            'approx_kl': total_kl / num_epochs,
            'clip_frac': total_clip_frac / num_epochs,
            'total_loss': (total_policy_loss + total_value_loss) / num_epochs,
        })

    def train(self, prompts_file: str) -> TrainingRunResult:
        lifecycle = self._build_lifecycle()
        with lifecycle:
            return self._train(prompts_file, lifecycle)

    def _train(self, prompts_file: str, lifecycle: MLflowTrainingLifecycle):
        """
        Main training loop with per-token RLHF

        This implements the industry-standard per-token value and GAE computation,
        where advantages and returns are computed for each token within a response.

        Args:
            prompts_file: Path to prompts JSON file
        """
        logger.info("Starting RLHF training (per-token version)")
        logger.info(f"Config: num_epochs={self.config.num_epochs}, num_steps={self.config.num_steps}")
        logger.info(f"Batch size: {self.config.batch_size}, Learning rate: {self.config.learning_rate}")
        logger.info(f"Gamma: {self.config.gamma}, GAE Lambda: {self.config.gae_lambda}")

        # Create dataset. Each rank gets a disjoint shard of equal size, so all
        # ranks run the same number of steps and reach every collective together.
        dataset = PromptDataset(prompts_file, self.tokenizer, self.config.max_prompt_length)
        if is_distributed():
            sampler = DistributedSampler(dataset, shuffle=True, seed=self.config.seed, drop_last=True)
        else:
            generator = torch.Generator().manual_seed(self.config.seed)
            sampler = RandomSampler(dataset, generator=generator)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            collate_fn=partial(collate_prompts, pad_token_id=self.tokenizer.pad_token_id),
        )
        ppo_epochs = self.config.ppo_epochs or self.config.num_epochs

        # Training metrics
        global_step = 0
        running_reward = 0.0
        running_kl = 0.0
        num_batches = 0
        training_reward_total = 0.0
        training_kl_total = 0.0
        training_metric_batches = 0

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(dataloader):
                if global_step >= self.config.num_steps:
                    break

                prompt_ids = batch['input_ids'].to(self.device)
                prompt_attention_mask = batch['attention_mask'].to(self.device)

                # ==================== Rollout Phase ====================
                with torch.no_grad():
                    response_ids, response_attention_mask = self.generate_response(
                        prompt_ids, prompt_attention_mask
                    )
                    # response_mask covers tokens up to and including the first
                    # stop token, so the EOS decision is trained and rewarded.
                    response_mask = response_attention_mask.float()

                    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
                    full_attention_mask = torch.cat(
                        [prompt_attention_mask, response_attention_mask], dim=1
                    )

                    # Old log-probs/values from the same forward pass the PPO
                    # update uses, not from generate()'s processed scores.
                    old_log_probs, _, old_values = self.compute_log_probs_and_entropy_per_token(
                        input_ids, response_ids, full_attention_mask
                    )
                    ref_log_probs = self.compute_ref_log_probs(
                        input_ids, response_ids, full_attention_mask
                    )
                    # Per-token KL estimate on the sampled tokens
                    kl_per_token = (old_log_probs - ref_log_probs) * response_mask

                    # Compute scalar RM scores
                    rm_scores = self.compute_reward(
                        prompt_ids,
                        response_ids,
                        prompt_attention_mask=prompt_attention_mask,
                        response_attention_mask=response_attention_mask,
                    ).to(self.device)
                    # rm_scores: [batch_size]

                    # Combine into per-token rewards
                    # rewards = -kl_coef * kl_per_token + rm_score (at last valid token)
                    rewards = self.compute_rewards_per_token(rm_scores, kl_per_token, response_mask)

                    # Compute per-token GAE advantages and returns
                    advantages, returns = self.compute_gae_per_token(
                        rewards=rewards,
                        values=old_values,
                        response_mask=response_mask,
                        gamma=self.config.gamma,
                        gae_lambda=self.config.gae_lambda
                    )

                # ==================== PPO Update Phase ====================
                ppo_batch = {
                    'input_ids': input_ids,
                    'response_ids': response_ids,
                    'attention_mask': full_attention_mask,
                    'response_mask': response_mask,
                    'old_log_probs': old_log_probs,
                    'old_values': old_values,
                    'advantages': advantages,
                    'returns': returns,
                }

                # PPO update
                metrics = self.ppo_update(ppo_batch, num_ppo_epochs=ppo_epochs)

                # ==================== Logging and Metrics ====================
                # Mean KL/reward over every rank's batch: the adaptive KL
                # coefficient must stay identical across ranks.
                with torch.no_grad():
                    mean_kl = masked_global_mean(kl_per_token, response_mask)
                    mean_reward = all_reduce_mean(rm_scores.float().mean())

                running_reward += mean_reward.item()
                running_kl += mean_kl.item()
                num_batches += 1
                training_reward_total += mean_reward.item()
                training_kl_total += mean_kl.item()
                training_metric_batches += 1

                # Log metrics
                if global_step % 10 == 0 or global_step == 0:
                    avg_reward = running_reward / num_batches if num_batches > 0 else 0
                    avg_kl = running_kl / num_batches if num_batches > 0 else 0
                    step_metrics = {
                        "reward_mean": avg_reward,
                        "kl_mean": avg_kl,
                        "policy_loss": metrics["policy_loss"],
                        "value_loss": metrics["value_loss"],
                        "entropy": metrics["entropy"],
                        "approx_kl": metrics["approx_kl"],
                        "clip_frac": metrics["clip_frac"],
                        "kl_coef": self.kl_coef,
                    }
                    logger.info(
                        f"Step {global_step}: "
                        f"reward={avg_reward:.3f}, "
                        f"kl={avg_kl:.4f}, "
                        f"policy_loss={metrics['policy_loss']:.4f}, "
                        f"value_loss={metrics['value_loss']:.4f}, "
                        f"entropy={metrics['entropy']:.4f}, "
                        f"clip_frac={metrics['clip_frac']:.3f}, "
                        f"kl_coef={self.kl_coef:.4f}"
                    )
                    lifecycle.log_metrics(step_metrics, step=global_step)
                    # Reset running metrics
                    running_reward = 0.0
                    running_kl = 0.0
                    num_batches = 0

                # ==================== Adaptive KL Coefficient ====================
                current_kl = mean_kl.item()
                if current_kl > self.config.target_kl * 1.5:
                    self.kl_coef = min(self.kl_coef * 1.5, self.config.kl_coef_max)
                elif current_kl < self.config.target_kl / 1.5:
                    self.kl_coef = max(self.kl_coef / 1.5, self.config.kl_coef_min)

                global_step += 1

                # Save checkpoint periodically
                if self.config.save_steps and global_step % self.config.save_steps == 0:
                    self.save_checkpoint(global_step)

            # End of epoch
            if global_step >= self.config.num_steps:
                break

        # Save final model
        logger.info("Saving final model")
        final_checkpoint_path = self.save_final_model()
        final_metrics = {
            "total_steps": global_step,
            "kl_coef": self.kl_coef,
            "rlhf_mean_reward": (
                training_reward_total / training_metric_batches
                if training_metric_batches
                else 0.0
            ),
            "rlhf_mean_kl": (
                training_kl_total / training_metric_batches
                if training_metric_batches
                else 0.0
            ),
            "task_eval_score": 0.0,
            "safety_score": 1.0,
            "latency_p95_ms": 0.0,
            "cost_per_1k_tokens_usd": 0.0,
            "artifact_checksum_ready": 1.0 if os.path.isdir(final_checkpoint_path) else 0.0,
        }
        # Rank 0 writes the metrics file; other ranks keep their local copy.
        final_metrics = self._write_promotion_metrics(
            final_checkpoint_path,
            final_metrics,
        ) or final_metrics
        lifecycle_result = lifecycle.register_checkpoint(
            final_checkpoint_path,
            final_metrics=final_metrics,
            extra_tags={
                "artifact_type": "full_model",
                "algorithm": "ppo",
            },
        )
        logger.info(f"Training complete! Total steps: {global_step}")
        return TrainingRunResult(
            checkpoint_path=final_checkpoint_path,
            metrics=final_metrics,
            lifecycle_result=lifecycle_result,
        )

    def save_checkpoint(self, step):
        """
        Save a resumable checkpoint (rank 0 writes; every rank waits for it).

        LoRA runs save only the adapter plus optimizer state, which is small;
        save_final_model turns the final adapter into a full model.
        """
        save_path = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        if is_main_process():
            os.makedirs(save_path, exist_ok=True)
            self.policy_model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            torch.save(self.value_head.state_dict(), os.path.join(save_path, "value_head.pt"))
            if getattr(self, "use_lora", False) and getattr(self, "optimizer", None) is not None:
                torch.save(self.optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
            self._write_run_config(save_path, step)
            logger.info(f"Checkpoint saved: {save_path}")
        barrier()
        return save_path

    def save_final_model(self) -> str:
        """
        Save checkpoint-final as a full Hugging Face model.

        For LoRA, rank 0 saves the adapter next to it and merges it into the
        base weights on CPU, so the AWQ/vLLM handoff keeps getting a complete
        model. The merge is not followed by a barrier: other ranks are done.
        """
        if not getattr(self, "use_lora", False):
            return self.save_checkpoint("final")

        adapter_path = self.save_checkpoint("final-adapter")
        final_path = os.path.join(self.config.output_dir, "checkpoint-final")
        if is_main_process():
            merge_adapter(
                self.config.policy_model_path,
                adapter_path,
                final_path,
                dtype=model_load_dtype(self.precision, full_fine_tune=False),
            )
            torch.save(self.value_head.state_dict(), os.path.join(final_path, "value_head.pt"))
            self._write_run_config(final_path, "final")
        return final_path

    def _write_run_config(self, save_path: str, step) -> None:
        with open(os.path.join(save_path, "rlhf_config.json"), 'w') as f:
            json.dump({
                'policy_model_path': self.config.policy_model_path,
                'reward_model_path': self.config.reward_model_path,
                'num_epochs': self.config.num_epochs,
                'ppo_epochs': self.config.ppo_epochs or self.config.num_epochs,
                'num_steps': self.config.num_steps,
                'batch_size': self.config.batch_size,
                'world_size': world_size(),
                'learning_rate': self.config.learning_rate,
                'value_learning_rate': self.config.value_learning_rate,
                'lora': bool(getattr(self, "use_lora", False)),
                'step': step,
                'kl_coef': self.kl_coef,
                'target_kl': self.config.target_kl,
                'clip_range': self.config.clip_range,
            }, f, indent=2)

    @torch.no_grad()
    def evaluate(
        self,
        eval_prompts: List[str],
        num_samples: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate current policy on a set of prompts

        Args:
            eval_prompts: List of evaluation prompts
            num_samples: Number of samples per prompt

        Returns:
            Evaluation results including rewards and sample generations
        """
        self.policy_model.eval()

        results = {
            'rewards': [],
            'kl_divergences': [],
            'samples': []
        }

        for prompt in eval_prompts[:num_samples]:
            prompt_ids = torch.tensor(
                [encode(self.tokenizer, format_prompt(self.tokenizer, prompt))],
                device=self.device,
            )
            attention_mask = torch.ones_like(prompt_ids)

            response_ids, response_attention_mask = self.generate_response(prompt_ids, attention_mask)
            response_mask = response_attention_mask.float()
            input_ids = torch.cat([prompt_ids, response_ids], dim=1)
            full_attention_mask = torch.cat([attention_mask, response_attention_mask], dim=1)

            reward = self.compute_reward(
                prompt_ids,
                response_ids,
                prompt_attention_mask=attention_mask,
                response_attention_mask=response_attention_mask,
            )

            policy_log_probs, _, _ = self.compute_log_probs_and_entropy_per_token(
                input_ids, response_ids, full_attention_mask
            )
            ref_log_probs = self.compute_ref_log_probs(input_ids, response_ids, full_attention_mask)
            kl_per_token = policy_log_probs - ref_log_probs
            kl = (kl_per_token * response_mask).sum() / response_mask.sum().clamp(min=1.0)

            response_text = self.tokenizer.decode(
                response_ids[0][response_attention_mask[0].bool()],
                skip_special_tokens=True,
            )

            results['rewards'].append(reward.item())
            results['kl_divergences'].append(kl.item())
            results['samples'].append({
                'prompt': prompt,
                'response': response_text,
                'reward': reward.item(),
                'kl': kl.item()
            })

        # Compute statistics
        results['mean_reward'] = np.mean(results['rewards'])
        results['std_reward'] = np.std(results['rewards'])
        results['mean_kl'] = np.mean(results['kl_divergences'])

        self.policy_model.train()

        return results


def rlhf_config_from_dict(
    config_dict: Dict[str, Any],
    config_path: Optional[str] = None,
    **overrides: Any,
) -> RLHFConfig:
    """Build RLHFConfig from the rlhf YAML mapping (shared by the CLI and the pipeline)."""
    ppo_config = (config_dict.get('algorithm') or {}).get('ppo') or {}
    adaptive_kl = (config_dict.get('advanced') or {}).get('adaptive_kl') or {}
    training_config = config_dict.get('training') or {}
    generation_config = config_dict.get('generation') or {}
    data_config = config_dict.get('data') or {}

    values: Dict[str, Any] = dict(
        # Model paths
        policy_model_path=config_dict['model']['policy']['name'],
        reward_model_path=config_dict['model']['reward']['name'],
        output_dir=training_config.get('output_dir', './outputs/rlhf'),

        # PPO hyperparameters
        num_epochs=ppo_config.get('num_epochs', 4),
        ppo_epochs=ppo_config.get('ppo_epochs'),
        num_steps=training_config.get('total_steps', 10000),
        batch_size=training_config.get('per_device_batch_size', 4),
        learning_rate=training_config.get('policy_learning_rate', 1e-6),
        value_learning_rate=training_config.get('value_learning_rate'),
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        save_steps=training_config.get('save_steps', 500),
        seed=training_config.get('seed', 42),

        # KL penalty
        init_kl_coef=ppo_config.get('init_kl_coef', ppo_config.get('kl_coef', 0.05)),
        target_kl=ppo_config.get('target_kl', 0.1),
        kl_coef_min=adaptive_kl.get('kl_coef_min', 0.001),
        kl_coef_max=adaptive_kl.get('kl_coef_max', 1.0),

        # PPO specific
        clip_range=ppo_config.get('clip_range', 0.2),
        value_loss_coef=ppo_config.get('value_loss_coef', 0.1),
        entropy_coef=ppo_config.get('entropy_coef', 0.01),
        gae_lambda=ppo_config.get('gae_lambda', 0.95),
        gamma=ppo_config.get('gamma', 0.99),

        # Generation
        max_prompt_length=data_config.get('max_prompt_length', 512),
        max_response_length=data_config.get('max_response_length', 512),
        temperature=generation_config.get('temperature', 0.7),
        top_p=generation_config.get('top_p', 0.9),
        raw_config=config_dict,
        config_path=config_path,
    )
    values.update(overrides)
    return RLHFConfig(**values)


def main():
    """Main entry point"""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="RLHF Training with PPO")
    parser.add_argument("--config", type=str, required=True, help="Config YAML file")
    parser.add_argument("--prompts", type=str, required=True, help="Prompts JSON file")
    parser.add_argument("--eval-prompts", type=str, default=None, help="Evaluation prompts JSON file")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with reduced steps")
    parser.add_argument("--promotion-metrics", help="Candidate eval metrics JSON for promotion")
    parser.add_argument("--baseline-metrics", help="Baseline eval metrics JSON for delta gates")
    parser.add_argument("--gate-config", help="Promotion gate config JSON")
    parser.add_argument(
        "--skip-promotion",
        action="store_true",
        help="Register candidate but do not evaluate champion promotion",
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = rlhf_config_from_dict(
        config_dict,
        config_path=args.config,
        promotion_metrics_path=args.promotion_metrics,
        baseline_metrics_path=args.baseline_metrics,
        gate_config_path=args.gate_config,
        skip_promotion=args.skip_promotion,
    )

    # Debug mode: reduce steps
    if args.debug:
        logger.info("Running in DEBUG mode with reduced steps")
        config.num_steps = 100
        config.num_epochs = 1

    # Log configuration
    logger.info("=" * 60)
    logger.info("RLHF Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Policy model: {config.policy_model_path}")
    logger.info(f"Reward model: {config.reward_model_path}")
    logger.info(f"Output dir: {config.output_dir}")
    logger.info(f"Total steps: {config.num_steps}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"KL coef: {config.init_kl_coef}, Target KL: {config.target_kl}")
    logger.info(f"Clip range: {config.clip_range}")
    logger.info("=" * 60)

    # Initialize trainer
    trainer = RLHFTrainer(config)

    # Train
    trainer.train(args.prompts)

    # Optional evaluation (rank 0 only; it needs no other rank)
    if args.eval_prompts and is_main_process():
        logger.info("Running evaluation...")
        with open(args.eval_prompts, 'r') as f:
            eval_data = json.load(f)
        eval_prompts = [item['prompt'] for item in eval_data]

        results = trainer.evaluate(eval_prompts, num_samples=10)

        logger.info("=" * 60)
        logger.info("Evaluation Results")
        logger.info("=" * 60)
        logger.info(f"Mean reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
        logger.info(f"Mean KL: {results['mean_kl']:.4f}")
        logger.info("Sample generations:")
        for i, sample in enumerate(results['samples'][:3]):
            logger.info(f"\n[Sample {i+1}]")
            logger.info(f"Prompt: {sample['prompt'][:100]}...")
            logger.info(f"Response: {sample['response'][:200]}...")
            logger.info(f"Reward: {sample['reward']:.4f}, KL: {sample['kl']:.4f}")

    logger.info("Training and evaluation complete!")


if __name__ == "__main__":
    main()
