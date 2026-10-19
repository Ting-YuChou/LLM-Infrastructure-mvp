"""
Reward Model Trainer
Trains a reward model from human preference data for RLHF
"""

import os
import json
import yaml
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
)

try:
    from .results import TrainingRunResult
    from .chat_format import (
        encode,
        encode_prompt_and_completion,
        ensure_chat_tokenizer,
        format_completion,
        resize_embeddings_if_needed,
    )
    from .distributed import (
        barrier,
        distributed_timeout_minutes,
        init_distributed,
        is_main_process,
        main_process_first,
        save_trainer_model,
        world_size,
    )
    from .hf_config import (
        attention_implementation,
        build_training_arguments_kwargs,
        early_stopping_settings,
        fsdp_environment,
        model_load_dtype,
        precision_dtype,
        resolve_precision,
    )
    from .mlflow_lifecycle import (
        MLflowMetricsCallback,
        MLflowTrainingLifecycle,
        PromotionConfig,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from results import TrainingRunResult  # type: ignore[no-redef]
    from chat_format import (  # type: ignore[no-redef]
        encode,
        encode_prompt_and_completion,
        ensure_chat_tokenizer,
        format_completion,
        resize_embeddings_if_needed,
    )
    from distributed import (  # type: ignore[no-redef]
        barrier,
        distributed_timeout_minutes,
        init_distributed,
        is_main_process,
        main_process_first,
        save_trainer_model,
        world_size,
    )
    from hf_config import (  # type: ignore[no-redef]
        attention_implementation,
        build_training_arguments_kwargs,
        early_stopping_settings,
        fsdp_environment,
        model_load_dtype,
        precision_dtype,
        resolve_precision,
    )
    from mlflow_lifecycle import (  # type: ignore[no-redef]
        MLflowMetricsCallback,
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
class RewardModelConfig:
    """Configuration for reward model training"""
    model_name: str
    train_file: str
    val_file: Optional[str]
    output_dir: str
    max_length: int = 512
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    learning_rate: float = 1e-5
    raw_config: Dict[str, Any] = field(default_factory=dict)
    config_path: Optional[str] = None
    promotion_metrics_path: Optional[str] = None
    baseline_metrics_path: Optional[str] = None
    gate_config_path: Optional[str] = None
    skip_promotion: bool = False

    @classmethod
    def from_yaml(cls, config_path: str) -> 'RewardModelConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return cls(
            model_name=config['model']['name'],
            train_file=config['data']['train_file'],
            val_file=config['data'].get('val_file'),
            output_dir=config['training']['output_dir'],
            max_length=config['data'].get('max_length', 512),
            num_train_epochs=config['training'].get('num_train_epochs', 3),
            per_device_train_batch_size=config['training'].get('per_device_train_batch_size', 4),
            learning_rate=config['training'].get('learning_rate', 1e-5),
            raw_config=config,
            config_path=config_path,
        )


class RewardDataPreprocessor:
    """Handles data loading and preprocessing for reward model training"""

    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 512):
        """
        Initialize preprocessor

        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = ensure_chat_tokenizer(tokenizer)
        self.max_length = max_length

        logger.info(f"Initialized RewardDataPreprocessor with max_length={max_length}")

    def load_dataset(self, data_file: str) -> Dataset:
        """
        Load preference dataset from JSON/JSONL file

        Expected format:
        [
            {
                "prompt": "question or instruction",
                "chosen": "preferred response",
                "rejected": "less preferred response"
            },
            ...
        ]

        Args:
            data_file: Path to data file

        Returns:
            HuggingFace Dataset
        """
        logger.info(f"Loading preference dataset from {data_file}")

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # Load based on file extension
        if data_file.endswith('.jsonl'):
            dataset = load_dataset('json', data_files=data_file, split='train')
        elif data_file.endswith('.json'):
            with open(data_file, 'r') as f:
                data = json.load(f)
            dataset = Dataset.from_list(data)
        else:
            raise ValueError(f"Unsupported file format: {data_file}")

        logger.info(f"Loaded {len(dataset)} preference pairs")
        return dataset

    def create_comparison_pairs(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        Create paired comparisons for training

        For each example, we create:
        - chosen_input_ids: chat-templated prompt + chosen response + <|eot_id|>
        - rejected_input_ids: chat-templated prompt + rejected response + <|eot_id|>

        This is the same token layout the PPO policy produces, so the reward
        head is read at the response's closing <|eot_id|> in both places.
        Pairs where either side exceeds max_length are dropped rather than
        truncated: right truncation cuts the response tail the reward should
        judge, and a long prompt can leave chosen and rejected identical.

        The model will be trained so that:
        reward(chosen) > reward(rejected)

        Args:
            examples: Batch of examples with 'prompt', 'chosen', 'rejected'

        Returns:
            Tokenized comparison pairs (possibly fewer rows than the input)
        """
        pad_token_id = self.tokenizer.pad_token_id
        batch: Dict[str, List] = {
            'chosen_input_ids': [],
            'chosen_attention_mask': [],
            'rejected_input_ids': [],
            'rejected_attention_mask': [],
        }
        dropped = 0

        for prompt, chosen, rejected in zip(
            examples['prompt'],
            examples['chosen'],
            examples['rejected']
        ):
            prompt_ids, chosen_ids = encode_prompt_and_completion(self.tokenizer, prompt, chosen)
            rejected_ids = encode(self.tokenizer, format_completion(rejected))
            chosen_full = prompt_ids + chosen_ids
            rejected_full = prompt_ids + rejected_ids
            if max(len(chosen_full), len(rejected_full)) > self.max_length:
                dropped += 1
                continue

            for side, ids in (('chosen', chosen_full), ('rejected', rejected_full)):
                padding = self.max_length - len(ids)
                batch[f'{side}_input_ids'].append(ids + [pad_token_id] * padding)
                batch[f'{side}_attention_mask'].append([1] * len(ids) + [0] * padding)

        if dropped:
            logger.warning(
                "Dropped %d preference pairs longer than max_length=%d",
                dropped,
                self.max_length,
            )
        return batch

    def prepare_datasets(
        self,
        train_file: str,
        val_file: Optional[str] = None
    ) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Load and preprocess training and validation datasets

        Args:
            train_file: Path to training data
            val_file: Path to validation data (optional)

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Load datasets
        train_dataset = self.load_dataset(train_file)
        val_dataset = self.load_dataset(val_file) if val_file else None

        # Process datasets
        logger.info("Creating comparison pairs for training...")
        train_dataset = train_dataset.map(
            self.create_comparison_pairs,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Creating train pairs"
        )

        if val_dataset:
            logger.info("Creating comparison pairs for validation...")
            val_dataset = val_dataset.map(
                self.create_comparison_pairs,
                batched=True,
                remove_columns=val_dataset.column_names,
                desc="Creating val pairs"
            )

        return train_dataset, val_dataset


class RewardTrainer(Trainer):
    """
    Custom Trainer for reward model

    Implements pairwise ranking loss:
    Loss = -log(sigmoid(reward_chosen - reward_rejected))

    This encourages the model to assign higher rewards to chosen responses
    """

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute pairwise ranking loss

        Args:
            model: Reward model
            inputs: Batch of chosen/rejected pairs
            return_outputs: Whether to return model outputs

        Returns:
            Loss tensor (and optionally outputs)
        """
        # Get chosen and rejected inputs
        chosen_input_ids = inputs['chosen_input_ids']
        chosen_attention_mask = inputs['chosen_attention_mask']
        rejected_input_ids = inputs['rejected_input_ids']
        rejected_attention_mask = inputs['rejected_attention_mask']

        # Compute rewards for chosen responses. The model is a standard
        # AutoModelForSequenceClassification checkpoint with num_labels=1.
        chosen_outputs = model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask
        )
        rewards_chosen = chosen_outputs.logits.squeeze(-1)

        # Compute rewards for rejected responses
        rejected_outputs = model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask
        )
        rewards_rejected = rejected_outputs.logits.squeeze(-1)

        # Pairwise ranking loss
        # We want: reward_chosen > reward_rejected
        # Loss = -log(sigmoid(reward_chosen - reward_rejected)), computed with
        # logsigmoid so large negative margins don't underflow to -log(0) = inf.
        loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()

        if return_outputs:
            return loss, {
                'rewards_chosen': rewards_chosen,
                'rewards_rejected': rewards_rejected
            }

        return loss

    def prediction_step(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Evaluate one chosen/rejected batch.

        The batches carry no `labels`, so the default Trainer.prediction_step
        would call model(**inputs) with the chosen_*/rejected_* keys and crash.
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.detach()
        if prediction_loss_only:
            return loss, None, None

        logits = torch.stack(
            [outputs['rewards_chosen'], outputs['rewards_rejected']],
            dim=-1,
        ).detach()
        # compute_metrics only runs when labels are present; the chosen
        # response is always index 0.
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return loss, logits, labels


def compute_reward_metrics(eval_prediction: Any) -> Dict[str, float]:
    """Pairwise accuracy and mean margin from [chosen, rejected] reward logits."""
    rewards = np.asarray(eval_prediction.predictions, dtype=np.float64)
    if rewards.size == 0:
        return {"reward_accuracy": 0.0, "reward_margin": 0.0}
    margins = rewards[:, 0] - rewards[:, 1]
    return {
        "reward_accuracy": float((margins > 0).mean()),
        "reward_margin": float(margins.mean()),
    }


class RewardModelTrainer:
    """Main trainer for reward models"""

    def __init__(self, config: RewardModelConfig):
        """
        Initialize trainer

        Args:
            config: Training configuration
        """
        self.config = config

        logger.info(f"Initializing Reward Model Trainer")
        logger.info(f"Base model: {config.model_name}")
        logger.info(f"Output directory: {config.output_dir}")

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def load_model_and_tokenizer(self) -> Tuple[PreTrainedModel, AutoTokenizer]:
        """Load a registry-safe sequence-classification reward model."""
        logger.info(f"Loading tokenizer from {self.config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=False,
        )

        tokenizer = ensure_chat_tokenizer(tokenizer)

        model_config = self.config.raw_config.get("model") or {}
        precision = resolve_precision(model_config)
        torch_dtype = model_load_dtype(precision, full_fine_tune=True)
        logger.info(
            "Loading base model from %s (dtype=%s)",
            self.config.model_name,
            torch_dtype,
        )
        # No device_map: Trainer/accelerate place the model, which keeps
        # torchrun, FSDP and DeepSpeed launches working.
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=1,  # Single scalar output
            torch_dtype=torch_dtype,
            attn_implementation=attention_implementation(model_config),
            cache_dir=model_config.get("cache_dir"),
            trust_remote_code=False,
        )
        resize_embeddings_if_needed(base_model, tokenizer)
        # LlamaForSequenceClassification scores the last non-pad token, which
        # is the closing <|eot_id|> now that pad is a dedicated token.
        base_model.config.pad_token_id = tokenizer.pad_token_id
        if model_config.get("gradient_checkpointing", False):
            base_model.config.use_cache = False

        return base_model, tokenizer

    def _build_lifecycle(self) -> MLflowTrainingLifecycle:
        """Build the MLflow lifecycle controller for this reward run."""
        return MLflowTrainingLifecycle.from_training_config(
            self.config.raw_config,
            model_type="reward",
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
        """Build TrainingArguments kwargs from the YAML config."""
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
        """Execute reward model training (single process or one rank under torchrun)."""
        init_distributed(distributed_timeout_minutes(self.config.raw_config))
        lifecycle = self._build_lifecycle()
        with lifecycle:
            # Build TrainingArguments (and the accelerate FSDP/DDP state) before
            # loading the model; see SFTTrainer.train.
            has_eval = bool(self.config.val_file)
            os.environ.update(fsdp_environment(self.config.raw_config, world_size()))
            training_args = TrainingArguments(**self._training_arguments_kwargs(has_eval=has_eval))

            # Load model and tokenizer
            model, tokenizer = self.load_model_and_tokenizer()

            # Rank 0 tokenizes first and fills the datasets cache for the rest.
            preprocessor = RewardDataPreprocessor(tokenizer, self.config.max_length)
            with main_process_first():
                train_dataset, val_dataset = preprocessor.prepare_datasets(
                    self.config.train_file,
                    self.config.val_file
                )
            if has_eval and not val_dataset:
                raise ValueError(
                    f"Validation file {self.config.val_file} has no usable pairs "
                    f"within max_length={self.config.max_length}"
                )

            callbacks = [MLflowMetricsCallback(lifecycle)]
            early_stopping = early_stopping_settings(self.config.raw_config, has_eval)
            if early_stopping is not None:
                callbacks.append(
                    EarlyStoppingCallback(early_stopping_patience=early_stopping["patience"])
                )

            # Initialize trainer
            trainer = RewardTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_reward_metrics,
                callbacks=callbacks,
            )

            # Train
            logger.info("Starting reward model training...")
            resume_from_checkpoint = (
                self.config.raw_config.get("training") or {}
            ).get("resume_from_checkpoint")
            train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            final_metrics = dict(getattr(train_output, "metrics", {}) or {})
            if val_dataset:
                eval_metrics = trainer.evaluate()
                final_metrics.update(eval_metrics)
                # Promotion gates read the unprefixed names.
                final_metrics["reward_accuracy"] = eval_metrics["eval_reward_accuracy"]
                final_metrics["reward_margin"] = eval_metrics["eval_reward_margin"]

            # Save final model
            final_model_path = f"{self.config.output_dir}/final"
            logger.info(f"Saving final model to {final_model_path}")
            # Every rank joins the save (FSDP gather); only rank 0 writes files.
            save_trainer_model(
                trainer,
                final_model_path,
                dtype=precision_dtype(resolve_precision(self.config.raw_config.get("model"))),
            )
            if is_main_process():
                tokenizer.save_pretrained(final_model_path)
            barrier()

            # Rank 0 only; no barrier after the upload (see SFTTrainer.train).
            lifecycle_result = lifecycle.register_checkpoint(
                final_model_path,
                final_metrics=final_metrics,
                extra_tags={"artifact_type": "full_model"},
            )

            logger.info("Reward model training complete!")
            return TrainingRunResult(
                checkpoint_path=final_model_path,
                metrics=final_metrics,
                lifecycle_result=lifecycle_result,
            )


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Reward Model Training for RLHF")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument("--promotion-metrics", help="Candidate eval metrics JSON for promotion")
    parser.add_argument("--baseline-metrics", help="Baseline eval metrics JSON for delta gates")
    parser.add_argument("--gate-config", help="Promotion gate config JSON")
    parser.add_argument(
        "--skip-promotion",
        action="store_true",
        help="Register candidate but do not evaluate champion promotion",
    )

    args = parser.parse_args()

    # Load configuration
    config = RewardModelConfig.from_yaml(args.config)
    config.promotion_metrics_path = args.promotion_metrics
    config.baseline_metrics_path = args.baseline_metrics
    config.gate_config_path = args.gate_config
    config.skip_promotion = args.skip_promotion

    # Initialize and run trainer
    trainer = RewardModelTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
