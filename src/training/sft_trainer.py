"""
Supervised Fine-Tuning (SFT) Trainer
Implements instruction fine-tuning for LLMs using HuggingFace Transformers
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

try:
    from .results import TrainingRunResult
    from .chat_format import (
        apply_generation_config,
        encode_prompt_and_completion,
        ensure_chat_tokenizer,
        format_prompt,
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
        apply_lora,
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
        apply_generation_config,
        encode_prompt_and_completion,
        ensure_chat_tokenizer,
        format_prompt,
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
        apply_lora,
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
class SFTConfig:
    """Configuration for SFT training"""
    model_name: str
    train_file: str
    val_file: str
    output_dir: str
    max_length: int = 2048
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    learning_rate: float = 2e-5
    raw_config: Dict[str, Any] = field(default_factory=dict)
    config_path: Optional[str] = None
    promotion_metrics_path: Optional[str] = None
    baseline_metrics_path: Optional[str] = None
    gate_config_path: Optional[str] = None
    skip_promotion: bool = False

    @classmethod
    def from_yaml(cls, config_path: str) -> 'SFTConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return cls(
            model_name=config['model']['name'],
            train_file=config['data']['train_file'],
            val_file=config['data']['val_file'],
            output_dir=config['training']['output_dir'],
            max_length=config['data'].get('max_length', 2048),
            num_train_epochs=config['training'].get('num_train_epochs', 3),
            per_device_train_batch_size=config['training'].get('per_device_train_batch_size', 4),
            learning_rate=config['training'].get('learning_rate', 2e-5),
            raw_config=config,
            config_path=config_path,
        )


class SFTDataPreprocessor:
    """Handles data loading and preprocessing for SFT"""

    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 2048):
        """
        Initialize preprocessor

        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = ensure_chat_tokenizer(tokenizer)
        self.max_length = max_length

        logger.info(f"Initialized SFTDataPreprocessor with max_length={max_length}")

    def load_dataset(self, data_file: str) -> Dataset:
        """
        Load dataset from JSON/JSONL file

        Expected format:
        [
            {"prompt": "instruction", "completion": "response"},
            ...
        ]

        Args:
            data_file: Path to data file

        Returns:
            HuggingFace Dataset
        """
        logger.info(f"Loading dataset from {data_file}")

        # Check if file exists
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # Load based on file extension
        if data_file.endswith('.jsonl'):
            dataset = load_dataset('json', data_files=data_file, split='train')
        elif data_file.endswith('.json'):
            # Load JSON array
            with open(data_file, 'r') as f:
                data = json.load(f)
            dataset = Dataset.from_list(data)
        else:
            raise ValueError(f"Unsupported file format: {data_file}")

        logger.info(f"Loaded {len(dataset)} examples")
        return dataset

    def format_prompt(self, prompt: str) -> str:
        """Render the chat-templated prompt that is masked from the loss."""
        return format_prompt(self.tokenizer, prompt)

    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        Tokenize prompt/completion pairs with the chat template.

        Prompt tokens and padding get label -100. Examples whose completion is
        cut off entirely by max_length are dropped: an all -100 row has no
        tokens to average over and turns the loss into NaN.

        Args:
            examples: Batch of examples

        Returns:
            Tokenized batch (possibly with fewer rows than the input)
        """
        pad_token_id = self.tokenizer.pad_token_id
        batch: Dict[str, List] = {"input_ids": [], "attention_mask": [], "labels": []}
        dropped = 0

        for prompt, completion in zip(examples.get('prompt', []), examples.get('completion', [])):
            prompt_ids, completion_ids = encode_prompt_and_completion(
                self.tokenizer,
                str(prompt or ""),
                str(completion or ""),
            )
            input_ids = (prompt_ids + completion_ids)[:self.max_length]
            labels = ([-100] * len(prompt_ids) + completion_ids)[:self.max_length]
            if all(label == -100 for label in labels):
                dropped += 1
                continue

            padding = self.max_length - len(input_ids)
            batch["input_ids"].append(input_ids + [pad_token_id] * padding)
            batch["attention_mask"].append([1] * len(input_ids) + [0] * padding)
            batch["labels"].append(labels + [-100] * padding)

        if dropped:
            logger.warning(
                "Dropped %d SFT examples whose prompt fills max_length=%d",
                dropped,
                self.max_length,
            )
        return batch

    def prepare_datasets(
        self,
        train_file: str,
        val_file: Optional[str] = None
    ) -> tuple:
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

        # Tokenize datasets
        logger.info("Tokenizing training dataset...")
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train data"
        )

        if val_dataset:
            logger.info("Tokenizing validation dataset...")
            val_dataset = val_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=val_dataset.column_names,
                desc="Tokenizing val data"
            )

        return train_dataset, val_dataset


def limit_debug_samples(
    raw_config: Dict[str, Any],
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
) -> tuple:
    """Trim datasets to debug.max_train_samples/max_eval_samples when debug is enabled."""
    debug = raw_config.get("debug") or {}
    if not debug.get("enabled", False):
        return train_dataset, val_dataset

    max_train = debug.get("max_train_samples")
    if max_train is not None:
        train_dataset = train_dataset.select(range(min(int(max_train), len(train_dataset))))
    max_eval = debug.get("max_eval_samples")
    if val_dataset is not None and max_eval is not None:
        val_dataset = val_dataset.select(range(min(int(max_eval), len(val_dataset))))
    logger.info(
        "Debug mode: using %d train / %s eval examples",
        len(train_dataset),
        len(val_dataset) if val_dataset is not None else 0,
    )
    return train_dataset, val_dataset


class MLflowCallback(MLflowMetricsCallback):
    """Backward-compatible name for the shared MLflow metrics callback."""


class SFTTrainer:
    """Supervised Fine-Tuning Trainer for LLMs"""

    def __init__(self, config: SFTConfig, use_lora: bool = False):
        """
        Initialize SFT trainer

        Args:
            config: Training configuration
            use_lora: Whether to use LoRA for parameter-efficient training
        """
        self.config = config
        lora_config = config.raw_config.get("lora") or {}
        self.use_lora = use_lora or bool(lora_config.get("enabled", False))
        self.precision = resolve_precision(config.raw_config.get("model"))

        logger.info(f"Initializing SFT Trainer for model: {config.model_name}")
        logger.info(f"Output directory: {config.output_dir}")

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def load_model_and_tokenizer(self):
        """Load pre-trained model and tokenizer"""
        logger.info(f"Loading tokenizer from {self.config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=False,
        )

        tokenizer = ensure_chat_tokenizer(tokenizer)

        model_config = self.config.raw_config.get("model") or {}
        torch_dtype = model_load_dtype(self.precision, full_fine_tune=not self.use_lora)
        attn_implementation = attention_implementation(model_config)
        logger.info(
            "Loading model from %s (dtype=%s, attention=%s)",
            self.config.model_name,
            torch_dtype,
            attn_implementation,
        )
        # No device_map: Trainer/accelerate place the model, which keeps
        # torchrun, FSDP and DeepSpeed launches working.
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            cache_dir=model_config.get("cache_dir"),
            trust_remote_code=False,
        )
        if model_config.get("gradient_checkpointing", False):
            model.config.use_cache = False
        resize_embeddings_if_needed(model, tokenizer)
        apply_generation_config(model, tokenizer)

        if self.use_lora:
            model = self._apply_lora(model)

        return model, tokenizer

    def _apply_lora(self, model):
        """Wrap the model with a LoRA adapter configured from the `lora:` YAML section."""
        lora_settings = self.config.raw_config.get("lora") or {}
        if not {"embed_tokens", "lm_head"} <= set(lora_settings.get("modules_to_save") or ()):
            logger.warning(
                "LoRA without embed_tokens/lm_head in lora.modules_to_save: the Llama 3 "
                "base embeddings for the chat header/<|eot_id|> tokens stay untrained"
            )
        return apply_lora(model, lora_settings)

    def _build_lifecycle(self) -> MLflowTrainingLifecycle:
        """Build the MLflow lifecycle controller for this training run."""
        artifact_type = "peft_adapter" if self.use_lora else "full_model"
        return MLflowTrainingLifecycle.from_training_config(
            self.config.raw_config,
            model_type="sft",
            config_path=self.config.config_path,
            promotion=PromotionConfig(
                metrics_path=self.config.promotion_metrics_path,
                baseline_metrics_path=self.config.baseline_metrics_path,
                gate_config_path=self.config.gate_config_path,
                skip_promotion=self.config.skip_promotion,
            ),
            extra_tags={
                "artifact_type": artifact_type,
                "lora_enabled": str(self.use_lora).lower(),
            },
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
        """Execute SFT training (single process or one rank under torchrun)."""
        init_distributed(distributed_timeout_minutes(self.config.raw_config))
        lifecycle = self._build_lifecycle()
        with lifecycle:
            # TrainingArguments sets up the accelerate state (FSDP/DDP), so build
            # it before loading the model. FSDP settings it cannot take go
            # through the environment first.
            has_eval = bool(self.config.val_file)
            os.environ.update(fsdp_environment(self.config.raw_config, world_size()))
            training_args = TrainingArguments(**self._training_arguments_kwargs(has_eval=has_eval))

            # Load model and tokenizer
            model, tokenizer = self.load_model_and_tokenizer()

            # Rank 0 tokenizes first and fills the datasets cache for the rest.
            preprocessor = SFTDataPreprocessor(tokenizer, self.config.max_length)
            with main_process_first():
                train_dataset, val_dataset = preprocessor.prepare_datasets(
                    self.config.train_file,
                    self.config.val_file
                )
            train_dataset, val_dataset = limit_debug_samples(
                self.config.raw_config,
                train_dataset,
                val_dataset,
            )
            if has_eval and not val_dataset:
                raise ValueError(
                    f"Validation file {self.config.val_file} has no usable examples "
                    f"within max_length={self.config.max_length}"
                )

            callbacks = [MLflowMetricsCallback(lifecycle)]
            early_stopping = early_stopping_settings(self.config.raw_config, has_eval)
            if early_stopping is not None:
                callbacks.append(
                    EarlyStoppingCallback(early_stopping_patience=early_stopping["patience"])
                )

            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=default_data_collator,
                callbacks=callbacks,
            )

            # Train
            logger.info("Starting training...")
            resume_from_checkpoint = (
                self.config.raw_config.get("training") or {}
            ).get("resume_from_checkpoint")
            train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            final_metrics = dict(getattr(train_output, "metrics", {}) or {})
            if val_dataset:
                final_metrics.update(trainer.evaluate())

            # Save final model. Every rank joins the save (it gathers FSDP shards);
            # only rank 0 writes files.
            final_model_path = f"{self.config.output_dir}/final"
            logger.info(f"Saving final model to {final_model_path}")
            save_trainer_model(
                trainer,
                final_model_path,
                dtype=precision_dtype(resolve_precision(self.config.raw_config.get("model"))),
            )
            if is_main_process():
                tokenizer.save_pretrained(final_model_path)
                # Adapter-only saves skip generation_config.json; vLLM and generate()
                # read the <|eot_id|> stop token from it.
                model.generation_config.save_pretrained(final_model_path)
            barrier()

            # Rank 0 only (the lifecycle is disabled elsewhere). No barrier after
            # it: a long artifact upload must not hold the other ranks in NCCL.
            lifecycle_result = lifecycle.register_checkpoint(
                final_model_path,
                final_metrics=final_metrics,
                extra_tags={
                    "artifact_type": "peft_adapter" if self.use_lora else "full_model",
                    "lora_enabled": str(self.use_lora).lower(),
                },
            )

            logger.info("Training complete!")
            return TrainingRunResult(
                checkpoint_path=final_model_path,
                metrics=final_metrics,
                lifecycle_result=lifecycle_result,
            )


def main():
    """Main entry point for SFT training"""
    import argparse

    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for LLMs")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA for parameter-efficient training"
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
    config = SFTConfig.from_yaml(args.config)
    config.promotion_metrics_path = args.promotion_metrics
    config.baseline_metrics_path = args.baseline_metrics
    config.gate_config_path = args.gate_config
    config.skip_promotion = args.skip_promotion

    # Initialize and run trainer
    trainer = SFTTrainer(config, use_lora=args.use_lora)
    trainer.train()


if __name__ == "__main__":
    main()
