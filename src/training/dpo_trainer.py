"""
DPO Trainer
Direct Preference Optimization (Rafailov et al., 2023) from the SFT checkpoint.

Full-parameter training under FSDP. The reference model is the SFT checkpoint
itself; its per-sequence log-probs are computed once before training (each rank
holds a full copy for a data-parallel pass over its shard of the pairs), so no
second model stays resident while the policy trains.
"""

import gc
import os
import yaml
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
)

try:
    from .results import TrainingRunResult
    from .chat_format import (
        apply_generation_config,
        ensure_chat_tokenizer,
        resize_embeddings_if_needed,
    )
    from .distributed import (
        all_gather_objects,
        barrier,
        distributed_timeout_minutes,
        init_distributed,
        is_main_process,
        main_process_first,
        rank,
        save_trainer_model,
        world_size,
    )
    from .hf_config import (
        attention_implementation,
        build_training_arguments_kwargs,
        early_stopping_settings,
        fsdp_environment,
        keep_fsdp_buffers_in_fp32,
        model_load_dtype,
        precision_dtype,
        resolve_precision,
    )
    from .mlflow_lifecycle import (
        MLflowMetricsCallback,
        MLflowTrainingLifecycle,
        PromotionConfig,
    )
    from .reward_trainer import RewardDataPreprocessor
    from .sft_trainer import limit_debug_samples
except ImportError:  # pragma: no cover - supports direct script execution
    from results import TrainingRunResult  # type: ignore[no-redef]
    from chat_format import (  # type: ignore[no-redef]
        apply_generation_config,
        ensure_chat_tokenizer,
        resize_embeddings_if_needed,
    )
    from distributed import (  # type: ignore[no-redef]
        all_gather_objects,
        barrier,
        distributed_timeout_minutes,
        init_distributed,
        is_main_process,
        main_process_first,
        rank,
        save_trainer_model,
        world_size,
    )
    from hf_config import (  # type: ignore[no-redef]
        attention_implementation,
        build_training_arguments_kwargs,
        early_stopping_settings,
        fsdp_environment,
        keep_fsdp_buffers_in_fp32,
        model_load_dtype,
        precision_dtype,
        resolve_precision,
    )
    from mlflow_lifecycle import (  # type: ignore[no-redef]
        MLflowMetricsCallback,
        MLflowTrainingLifecycle,
        PromotionConfig,
    )
    from reward_trainer import RewardDataPreprocessor  # type: ignore[no-redef]
    from sft_trainer import limit_debug_samples  # type: ignore[no-redef]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

_LOSS_TYPES = ("sigmoid",)


@dataclass
class DPOConfig:
    """Configuration for DPO training"""
    model_name: str
    train_file: str
    val_file: Optional[str]
    output_dir: str
    max_length: int = 1024
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    learning_rate: float = 5e-7
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"
    reference_batch_size: int = 4
    raw_config: Dict[str, Any] = field(default_factory=dict)
    config_path: Optional[str] = None
    promotion_metrics_path: Optional[str] = None
    baseline_metrics_path: Optional[str] = None
    gate_config_path: Optional[str] = None
    skip_promotion: bool = False

    @classmethod
    def from_yaml(cls, config_path: str) -> 'DPOConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        dpo = config.get('dpo') or {}
        loss_type = str(dpo.get('loss_type', 'sigmoid'))
        if loss_type not in _LOSS_TYPES:
            raise ValueError(f"Unsupported dpo.loss_type {loss_type!r}; expected one of {_LOSS_TYPES}")
        return cls(
            model_name=config['model']['name'],
            train_file=config['data']['train_file'],
            val_file=config['data'].get('val_file'),
            output_dir=config['training']['output_dir'],
            max_length=config['data'].get('max_length', 1024),
            num_train_epochs=config['training'].get('num_train_epochs', 1),
            per_device_train_batch_size=config['training'].get('per_device_train_batch_size', 2),
            learning_rate=config['training'].get('learning_rate', 5e-7),
            beta=float(dpo.get('beta', 0.1)),
            label_smoothing=float(dpo.get('label_smoothing', 0.0)),
            loss_type=loss_type,
            reference_batch_size=int(dpo.get('reference_batch_size', 4)),
            raw_config=config,
            config_path=config_path,
        )


def sequence_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Summed log-prob of each row's labelled tokens; logits at i predict token i+1."""
    logits = logits[:, :-1, :].float()
    labels = labels[:, 1:]
    mask = labels != -100
    targets = labels.masked_fill(~mask, 0)
    token_log_probs = (
        logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
    )
    return (token_log_probs * mask).sum(-1)


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sigmoid DPO loss per pair plus the implicit rewards beta * (log pi - log pi_ref).

    label_smoothing > 0 is the conservative DPO variant (assumes that fraction
    of preference labels are flipped).
    """
    logits = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
    losses = (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
    )
    chosen_rewards = beta * (policy_chosen - reference_chosen).detach()
    rejected_rewards = beta * (policy_rejected - reference_rejected).detach()
    return losses, chosen_rewards, rejected_rewards


def concatenated_log_probs(
    model: PreTrainedModel,
    inputs: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chosen and rejected sequence log-probs from one forward over both halves."""
    batch_size = inputs['chosen_input_ids'].shape[0]
    outputs = model(
        input_ids=torch.cat([inputs['chosen_input_ids'], inputs['rejected_input_ids']]),
        attention_mask=torch.cat([inputs['chosen_attention_mask'], inputs['rejected_attention_mask']]),
        use_cache=False,
    )
    log_probs = sequence_log_probs(
        outputs.logits,
        torch.cat([inputs['chosen_labels'], inputs['rejected_labels']]),
    )
    return log_probs[:batch_size], log_probs[batch_size:]


@contextmanager
def _plain_model_loading() -> Iterator[None]:
    """
    Load a full model on every rank even when FSDP CPU-RAM-efficient loading is on.

    With it on, transformers puts non-zero local ranks' weights on the meta
    device and relies on FSDP's sync_module_states to broadcast them; the
    reference model is never FSDP-wrapped, so it must load normally.
    """
    previous = os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING")
    os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "false"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("FSDP_CPU_RAM_EFFICIENT_LOADING", None)
        else:
            os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = previous


@torch.no_grad()
def compute_reference_log_probs(
    model: PreTrainedModel,
    datasets: List[Dataset],
    batch_size: int,
    device: torch.device,
) -> List[Dataset]:
    """
    Add ref_chosen_logps/ref_rejected_logps columns to each dataset.

    Every rank scores rows rank, rank + world_size, ... and the results are
    all-gathered by row index, so every rank ends up with identical columns.
    """
    model.eval()
    results = []
    for dataset in datasets:
        indices = list(range(rank(), len(dataset), world_size()))
        local: Dict[int, Tuple[float, float]] = {}
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            rows = dataset.select(chunk).with_format("torch")
            inputs = {
                key: rows[key].to(device)
                for key in (
                    'chosen_input_ids', 'chosen_attention_mask', 'chosen_labels',
                    'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels',
                )
            }
            chosen, rejected = concatenated_log_probs(model, inputs)
            for index, chosen_logp, rejected_logp in zip(chunk, chosen.tolist(), rejected.tolist()):
                local[index] = (chosen_logp, rejected_logp)

        merged: Dict[int, Tuple[float, float]] = {}
        for part in all_gather_objects(local):
            merged.update(part)
        if len(merged) != len(dataset):
            raise RuntimeError(
                f"Reference pass scored {len(merged)} of {len(dataset)} preference pairs"
            )
        dataset = dataset.add_column('ref_chosen_logps', [merged[i][0] for i in range(len(dataset))])
        dataset = dataset.add_column('ref_rejected_logps', [merged[i][1] for i in range(len(dataset))])
        results.append(dataset)
    return results


class DPOTrainer(Trainer):
    """
    Trainer for the sigmoid DPO objective.

    Batches carry chosen_*/rejected_* input ids, attention masks and labels
    (prompt and padding at -100) plus precomputed reference log-probs.
    """

    def __init__(self, *args: Any, beta: float = 0.1, label_smoothing: float = 0.0, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.label_smoothing = label_smoothing

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        policy_chosen, policy_rejected = concatenated_log_probs(model, inputs)
        losses, chosen_rewards, rejected_rewards = dpo_loss(
            policy_chosen,
            policy_rejected,
            inputs['ref_chosen_logps'].to(policy_chosen.dtype),
            inputs['ref_rejected_logps'].to(policy_chosen.dtype),
            beta=self.beta,
            label_smoothing=self.label_smoothing,
        )
        loss = losses.mean()
        if return_outputs:
            return loss, {'chosen_rewards': chosen_rewards, 'rejected_rewards': rejected_rewards}
        return loss

    def prediction_step(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Evaluate one batch; logits are the [chosen, rejected] implicit rewards."""
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.detach()
        if prediction_loss_only:
            return loss, None, None

        logits = torch.stack([outputs['chosen_rewards'], outputs['rejected_rewards']], dim=-1)
        # compute_metrics only runs when labels are present; chosen is index 0.
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return loss, logits, labels


def compute_dpo_metrics(eval_prediction: Any) -> Dict[str, float]:
    """Implicit-reward accuracy, margin and means from [chosen, rejected] rewards."""
    rewards = np.asarray(eval_prediction.predictions, dtype=np.float64)
    if rewards.size == 0:
        return {"rewards_accuracy": 0.0, "rewards_margin": 0.0}
    margins = rewards[:, 0] - rewards[:, 1]
    return {
        "rewards_accuracy": float((margins > 0).mean()),
        "rewards_margin": float(margins.mean()),
        "rewards_chosen": float(rewards[:, 0].mean()),
        "rewards_rejected": float(rewards[:, 1].mean()),
    }


class DPOModelTrainer:
    """Main trainer for DPO"""

    def __init__(self, config: DPOConfig):
        self.config = config
        self.precision = resolve_precision(config.raw_config.get("model"))

        logger.info("Initializing DPO Trainer")
        logger.info(f"Policy/reference model: {config.model_name}")
        logger.info(f"Output directory: {config.output_dir}")

        os.makedirs(config.output_dir, exist_ok=True)

    def load_tokenizer(self) -> AutoTokenizer:
        logger.info(f"Loading tokenizer from {self.config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=False)
        return ensure_chat_tokenizer(tokenizer)

    def load_model(self, tokenizer: AutoTokenizer, device_map: Optional[Dict[str, Any]] = None) -> PreTrainedModel:
        """Load the SFT checkpoint as a causal LM (policy, or the reference with device_map)."""
        model_config = self.config.raw_config.get("model") or {}
        torch_dtype = model_load_dtype(self.precision, full_fine_tune=True)
        logger.info("Loading model from %s (dtype=%s)", self.config.model_name, torch_dtype)
        # No device_map for the policy: Trainer/accelerate place it, which keeps
        # torchrun and FSDP launches working.
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attention_implementation(model_config),
            cache_dir=model_config.get("cache_dir"),
            device_map=device_map,
            trust_remote_code=False,
        )
        if model_config.get("gradient_checkpointing", False):
            model.config.use_cache = False
        resize_embeddings_if_needed(model, tokenizer)
        apply_generation_config(model, tokenizer)
        return model

    def add_reference_log_probs(
        self,
        tokenizer: AutoTokenizer,
        datasets: List[Dataset],
        device: torch.device,
    ) -> List[Dataset]:
        """Score every pair with the frozen SFT model, then free it."""
        device_map = {"": device} if device.type == "cuda" else None
        with _plain_model_loading():
            reference = self.load_model(tokenizer, device_map=device_map)
        reference.to(device)
        logger.info("Computing reference log-probs for %s pairs", [len(d) for d in datasets])
        datasets = compute_reference_log_probs(
            reference,
            datasets,
            batch_size=self.config.reference_batch_size,
            device=device,
        )
        del reference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return datasets

    def _build_lifecycle(self) -> MLflowTrainingLifecycle:
        return MLflowTrainingLifecycle.from_training_config(
            self.config.raw_config,
            model_type="dpo",
            config_path=self.config.config_path,
            promotion=PromotionConfig(
                metrics_path=self.config.promotion_metrics_path,
                baseline_metrics_path=self.config.baseline_metrics_path,
                gate_config_path=self.config.gate_config_path,
                skip_promotion=self.config.skip_promotion,
            ),
            extra_tags={"artifact_type": "full_model"},
        )

    def _training_arguments_kwargs(self, has_eval: bool) -> Dict[str, Any]:
        kwargs = build_training_arguments_kwargs(
            self.config.raw_config,
            output_dir=self.config.output_dir,
            has_eval=has_eval,
        )
        # The dataclass fields can be overridden after loading the YAML.
        kwargs["num_train_epochs"] = self.config.num_train_epochs
        kwargs["per_device_train_batch_size"] = self.config.per_device_train_batch_size
        kwargs.setdefault("per_device_eval_batch_size", self.config.per_device_train_batch_size)
        kwargs["learning_rate"] = self.config.learning_rate
        return kwargs

    def train(self) -> TrainingRunResult:
        """Execute DPO training (single process or one rank under torchrun)."""
        device = init_distributed(distributed_timeout_minutes(self.config.raw_config))
        lifecycle = self._build_lifecycle()
        with lifecycle:
            tokenizer = self.load_tokenizer()
            has_eval = bool(self.config.val_file)

            # Rank 0 tokenizes first and fills the datasets cache for the rest.
            preprocessor = RewardDataPreprocessor(tokenizer, self.config.max_length, include_labels=True)
            with main_process_first():
                train_dataset, val_dataset = preprocessor.prepare_datasets(
                    self.config.train_file,
                    self.config.val_file,
                )
            train_dataset, val_dataset = limit_debug_samples(
                self.config.raw_config,
                train_dataset,
                val_dataset,
            )
            if has_eval and not val_dataset:
                raise ValueError(
                    f"Validation file {self.config.val_file} has no usable pairs "
                    f"within max_length={self.config.max_length}"
                )

            # Reference pass first, before TrainingArguments switches on the
            # accelerate FSDP state and before the policy takes GPU memory.
            scored = self.add_reference_log_probs(
                tokenizer,
                [train_dataset] + ([val_dataset] if val_dataset else []),
                device,
            )
            train_dataset = scored[0]
            val_dataset = scored[1] if val_dataset else None

            # TrainingArguments sets up the accelerate state (FSDP/DDP), so build
            # it before loading the policy (see SFTTrainer.train).
            os.environ.update(fsdp_environment(self.config.raw_config, world_size()))
            training_args = TrainingArguments(**self._training_arguments_kwargs(has_eval=has_eval))
            model = self.load_model(tokenizer)

            callbacks = [MLflowMetricsCallback(lifecycle)]
            early_stopping = early_stopping_settings(self.config.raw_config, has_eval)
            if early_stopping is not None:
                callbacks.append(
                    EarlyStoppingCallback(early_stopping_patience=early_stopping["patience"])
                )

            trainer = DPOTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_dpo_metrics,
                callbacks=callbacks,
                beta=self.config.beta,
                label_smoothing=self.config.label_smoothing,
            )
            keep_fsdp_buffers_in_fp32(trainer)

            logger.info("Starting DPO training...")
            resume_from_checkpoint = (
                self.config.raw_config.get("training") or {}
            ).get("resume_from_checkpoint")
            train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            final_metrics = dict(getattr(train_output, "metrics", {}) or {})
            if val_dataset:
                eval_metrics = trainer.evaluate()
                final_metrics.update(eval_metrics)
                # Promotion gates read the unprefixed names.
                final_metrics["dpo_reward_accuracy"] = eval_metrics["eval_rewards_accuracy"]
                final_metrics["dpo_reward_margin"] = eval_metrics["eval_rewards_margin"]

            final_model_path = f"{self.config.output_dir}/final"
            logger.info(f"Saving final model to {final_model_path}")
            # Every rank joins the save (FSDP gather); only rank 0 writes files.
            save_trainer_model(trainer, final_model_path, dtype=precision_dtype(self.precision))
            if is_main_process():
                tokenizer.save_pretrained(final_model_path)
                model.generation_config.save_pretrained(final_model_path)
            barrier()

            # Rank 0 only; no barrier after the upload (see SFTTrainer.train).
            lifecycle_result = lifecycle.register_checkpoint(
                final_model_path,
                final_metrics=final_metrics,
                extra_tags={"artifact_type": "full_model"},
            )

            logger.info("DPO training complete!")
            return TrainingRunResult(
                checkpoint_path=final_model_path,
                metrics=final_metrics,
                lifecycle_result=lifecycle_result,
            )


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Direct Preference Optimization")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--promotion-metrics", help="Candidate eval metrics JSON for promotion")
    parser.add_argument("--baseline-metrics", help="Baseline eval metrics JSON for delta gates")
    parser.add_argument("--gate-config", help="Promotion gate config JSON")
    parser.add_argument(
        "--skip-promotion",
        action="store_true",
        help="Register candidate but do not evaluate champion promotion",
    )
    parser.add_argument("--debug", action="store_true", help="Train on debug.max_*_samples rows")

    args = parser.parse_args()

    config = DPOConfig.from_yaml(args.config)
    config.promotion_metrics_path = args.promotion_metrics
    config.baseline_metrics_path = args.baseline_metrics
    config.gate_config_path = args.gate_config
    config.skip_promotion = args.skip_promotion
    if args.debug:
        config.raw_config.setdefault("debug", {})["enabled"] = True

    DPOModelTrainer(config).train()


if __name__ == "__main__":
    main()
