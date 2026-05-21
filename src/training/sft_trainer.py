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

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

try:
    from .mlflow_lifecycle import (
        MLflowMetricsCallback,
        MLflowTrainingLifecycle,
        PromotionConfig,
    )
except ImportError:  # pragma: no cover - supports direct script execution
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
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

    def format_instruction(self, example: Dict[str, str]) -> str:
        """
        Format instruction-response pair into a single training sequence

        Uses Alpaca-style formatting:
        ### Instruction:
        {prompt}

        ### Response:
        {completion}

        Args:
            example: Dict with 'prompt' and 'completion' keys

        Returns:
            Formatted text string
        """
        instruction = example.get('prompt', '')
        response = example.get('completion', '')

        # Alpaca formatting
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{response}{self.tokenizer.eos_token}"
        )

        return text

    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        Tokenize examples for training

        Args:
            examples: Batch of examples

        Returns:
            Tokenized batch
        """
        # Format all examples in batch
        texts = [self.format_instruction(ex) for ex in examples]

        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors=None,  # Return lists, not tensors
        )

        # For causal LM, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()

        return tokenized

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
        self.use_lora = use_lora

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

        # Ensure tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loading model from {self.config.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=False,
        )

        # Apply LoRA if enabled
        if self.use_lora:
            logger.info("Applying LoRA configuration")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,  # LoRA rank
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        return model, tokenizer

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

    def train(self) -> None:
        """Execute SFT training"""
        lifecycle = self._build_lifecycle()
        with lifecycle:
            # Load model and tokenizer
            model, tokenizer = self.load_model_and_tokenizer()

            # Prepare datasets
            preprocessor = SFTDataPreprocessor(tokenizer, self.config.max_length)
            train_dataset, val_dataset = preprocessor.prepare_datasets(
                self.config.train_file,
                self.config.val_file
            )

            # Define training arguments
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_train_epochs,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_train_batch_size,
                learning_rate=self.config.learning_rate,
                lr_scheduler_type="cosine",
                warmup_ratio=0.03,
                logging_steps=10,
                save_steps=500,
                eval_steps=500 if val_dataset else None,
                evaluation_strategy="steps" if val_dataset else "no",
                save_total_limit=3,
                fp16=True,
                gradient_accumulation_steps=8,
                remove_unused_columns=False,
                report_to=["tensorboard"],
                logging_dir=f"{self.config.output_dir}/logs",
            )

            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,  # Causal LM, not masked LM
            )

            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                callbacks=[MLflowMetricsCallback(lifecycle)],
            )

            # Train
            logger.info("Starting training...")
            train_output = trainer.train()
            final_metrics = dict(getattr(train_output, "metrics", {}) or {})
            if val_dataset:
                final_metrics.update(trainer.evaluate())

            # Save final model
            final_model_path = f"{self.config.output_dir}/final"
            logger.info(f"Saving final model to {final_model_path}")
            trainer.save_model(final_model_path)
            tokenizer.save_pretrained(final_model_path)

            lifecycle.register_checkpoint(
                final_model_path,
                final_metrics=final_metrics,
                extra_tags={
                    "artifact_type": "peft_adapter" if self.use_lora else "full_model",
                    "lora_enabled": str(self.use_lora).lower(),
                },
            )

            logger.info("Training complete!")


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
