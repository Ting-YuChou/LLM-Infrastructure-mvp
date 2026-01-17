"""
Reward Model Trainer
Trains a reward model from human preference data for RLHF
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
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
        )


class RewardModel(nn.Module):
    """
    Reward Model that outputs a scalar reward for a given input
    
    Architecture:
    - Base LLM (frozen or fine-tunable)
    - Linear head that outputs single scalar value
    """
    
    def __init__(self, base_model: PreTrainedModel):
        """
        Initialize reward model
        
        Args:
            base_model: Pre-trained language model
        """
        super().__init__()
        self.base_model = base_model
        
        # Get hidden size from base model
        hidden_size = base_model.config.hidden_size
        
        # Reward head: maps hidden states to single scalar
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)
        
        # Initialize reward head with small values
        nn.init.normal_(self.reward_head.weight, mean=0.0, std=0.01)
        
        logger.info(f"Initialized reward model with hidden_size={hidden_size}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Rewards: Scalar rewards [batch_size, 1]
        """
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get last hidden state
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # Use the last token's hidden state (or mean pool)
        # For causal LM, last token is the most informative
        last_token_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        
        # Compute reward
        rewards = self.reward_head(last_token_hidden)  # [batch_size, 1]
        
        return rewards


class RewardDataPreprocessor:
    """Handles data loading and preprocessing for reward model training"""
    
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 512):
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
        - chosen_input_ids: tokenized prompt + chosen response
        - rejected_input_ids: tokenized prompt + rejected response
        
        The model will be trained so that:
        reward(chosen) > reward(rejected)
        
        Args:
            examples: Batch of examples with 'prompt', 'chosen', 'rejected'
            
        Returns:
            Tokenized comparison pairs
        """
        chosen_texts = []
        rejected_texts = []
        
        # Combine prompt with responses
        for prompt, chosen, rejected in zip(
            examples['prompt'],
            examples['chosen'],
            examples['rejected']
        ):
            # Format: "Prompt: {prompt}\n\nResponse: {response}"
            chosen_text = f"Prompt: {prompt}\n\nResponse: {chosen}"
            rejected_text = f"Prompt: {prompt}\n\nResponse: {rejected}"
            
            chosen_texts.append(chosen_text)
            rejected_texts.append(rejected_text)
        
        # Tokenize both chosen and rejected
        chosen_encodings = self.tokenizer(
            chosen_texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors=None,
        )
        
        rejected_encodings = self.tokenizer(
            rejected_texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors=None,
        )
        
        # Return paired data
        return {
            'chosen_input_ids': chosen_encodings['input_ids'],
            'chosen_attention_mask': chosen_encodings['attention_mask'],
            'rejected_input_ids': rejected_encodings['input_ids'],
            'rejected_attention_mask': rejected_encodings['attention_mask'],
        }
    
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
        model: RewardModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
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
        
        # Compute rewards for chosen responses
        rewards_chosen = model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask
        )
        
        # Compute rewards for rejected responses
        rewards_rejected = model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask
        )
        
        # Pairwise ranking loss
        # We want: reward_chosen > reward_rejected
        # Loss = -log(sigmoid(reward_chosen - reward_rejected))
        loss = -torch.log(torch.sigmoid(rewards_chosen - rewards_rejected)).mean()
        
        if return_outputs:
            return loss, {
                'rewards_chosen': rewards_chosen,
                'rewards_rejected': rewards_rejected
            }
        
        return loss


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
    
    def load_model_and_tokenizer(self) -> Tuple[RewardModel, AutoTokenizer]:
        """Load base model and tokenizer, wrap in RewardModel"""
        logger.info(f"Loading tokenizer from {self.config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=False,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Loading base model from {self.config.model_name}")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=1,  # Single scalar output
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=False,
        )
        
        # Wrap in reward model
        reward_model = RewardModel(base_model)
        
        return reward_model, tokenizer
    
    def train(self) -> None:
        """Execute reward model training"""
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Prepare datasets
        preprocessor = RewardDataPreprocessor(tokenizer, self.config.max_length)
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
            warmup_ratio=0.1,
            logging_steps=10,
            save_steps=500,
            eval_steps=500 if val_dataset else None,
            evaluation_strategy="steps" if val_dataset else "no",
            save_total_limit=3,
            fp16=True,
            gradient_accumulation_steps=4,
            remove_unused_columns=False,
            report_to=["tensorboard"],
            logging_dir=f"{self.config.output_dir}/logs",
        )
        
        # Initialize trainer
        trainer = RewardTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train
        logger.info("Starting reward model training...")
        trainer.train()
        
        # Save final model
        logger.info(f"Saving final model to {self.config.output_dir}/final")
        trainer.save_model(f"{self.config.output_dir}/final")
        tokenizer.save_pretrained(f"{self.config.output_dir}/final")
        
        logger.info("Reward model training complete!")


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
    
    args = parser.parse_args()
    
    # Load configuration
    config = RewardModelConfig.from_yaml(args.config)
    
    # Initialize and run trainer
    trainer = RewardModelTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
