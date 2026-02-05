"""
RLHF Trainer using PPO (Proximal Policy Optimization)
Implements reinforcement learning from human feedback for LLM alignment
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
)
import numpy as np

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
    
    # PPO hyperparameters
    num_epochs: int = 4
    num_steps: int = 10000
    batch_size: int = 4
    learning_rate: float = 1e-6
    
    # KL penalty
    init_kl_coef: float = 0.05
    target_kl: float = 0.1
    
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


class PromptDataset(Dataset):
    """Dataset of prompts for RL training"""
    
    def __init__(self, prompts_file: str, tokenizer: AutoTokenizer, max_length: int = 512):
        """
        Initialize dataset
        
        Args:
            prompts_file: Path to JSON file with prompts
            tokenizer: Tokenizer
            max_length: Maximum prompt length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load prompts
        with open(prompts_file, 'r') as f:
            data = json.load(f)
        
        self.prompts = [item['prompt'] for item in data]
        logger.info(f"Loaded {len(self.prompts)} prompts from {prompts_file}")
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt = self.prompts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'prompt_text': prompt
        }


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
        # Apply value head to each token position
        values = self.value_head(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        return values


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("Initializing RLHF Trainer")
        logger.info(f"Device: {self.device}")
        logger.info(f"Policy model: {config.policy_model_path}")
        logger.info(f"Reward model: {config.reward_model_path}")
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Load models
        self.tokenizer = AutoTokenizer.from_pretrained(config.policy_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Policy model (model being optimized)
        logger.info("Loading policy model...")
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            config.policy_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.policy_model.train()
        
        # Reference model (frozen, for KL penalty)
        logger.info("Loading reference model...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.policy_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Reward model (frozen)
        logger.info("Loading reward model...")
        self.reward_model = AutoModelForCausalLM.from_pretrained(
            config.reward_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.reward_model.eval()
        for param in self.reward_model.parameters():
            param.requires_grad = False
        
        # Value function
        hidden_size = self.policy_model.config.hidden_size
        self.value_head = ValueHead(hidden_size).to(self.device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy_model.parameters(),
            lr=config.learning_rate
        )
        
        self.value_optimizer = torch.optim.Adam(
            self.value_head.parameters(),
            lr=config.learning_rate * 5  # Higher LR for value function
        )
        
        # KL coefficient (adaptive)
        self.kl_coef = config.init_kl_coef
        
        logger.info("RLHF Trainer initialized successfully")
    
    @torch.no_grad()
    def generate_response(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate response from policy model
        
        Args:
            prompt_ids: Prompt token IDs [batch_size, prompt_len]
            attention_mask: Attention mask [batch_size, prompt_len]
            
        Returns:
            response_ids: Generated token IDs [batch_size, response_len]
            log_probs: Per-token log probabilities [batch_size, response_len]
            values: Per-token value estimates [batch_size, response_len]
        """
        batch_size = prompt_ids.shape[0]
        prompt_length = prompt_ids.shape[1]
        
        # Generate with scores
        outputs = self.policy_model.generate(
            prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )
        
        # Extract response tokens (exclude prompt)
        response_ids = outputs.sequences[:, prompt_length:]
        response_length = response_ids.shape[1]
        
        # Compute log probabilities from scores
        # outputs.scores is a tuple of tensors, one for each generated token
        # Each tensor has shape [batch_size, vocab_size]
        if len(outputs.scores) > 0:
            # Stack scores: [response_len, batch_size, vocab_size]
            stacked_scores = torch.stack(outputs.scores, dim=0)
            # Transpose to [batch_size, response_len, vocab_size]
            stacked_scores = stacked_scores.transpose(0, 1)
            
            # Apply softmax to get probabilities
            log_probs_all = F.log_softmax(stacked_scores, dim=-1)
            
            # Gather log probs for selected tokens
            # response_ids: [batch_size, response_len]
            # We need to gather the log prob for each selected token
            response_ids_expanded = response_ids.unsqueeze(-1)  # [batch_size, response_len, 1]
            log_probs = log_probs_all.gather(-1, response_ids_expanded).squeeze(-1)  # [batch_size, response_len]
        else:
            log_probs = torch.zeros(batch_size, response_length, device=self.device)
        
        # Compute per-token value estimates using value head
        # We need to get hidden states from the full sequence
        full_ids = outputs.sequences
        with torch.no_grad():
            model_outputs = self.policy_model(
                full_ids,
                output_hidden_states=True,
                return_dict=True
            )
            # Get last hidden state for all positions
            hidden_states = model_outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            
            # Apply value head to get per-token values
            all_values = self.value_head(hidden_states)  # [batch_size, seq_len]
            
            # Extract values for response positions only
            # The value at position t estimates V(s_t), which is the value before generating token t+1
            # For response tokens, we want values from positions [prompt_length-1, seq_len-2]
            # because the value at position i predicts the value of state after seeing tokens 0..i
            values = all_values[:, prompt_length-1:-1]  # [batch_size, response_len]
            
            # Handle case where response_len doesn't match
            if values.shape[1] != response_length:
                # Pad or truncate to match response_length
                if values.shape[1] < response_length:
                    padding = torch.zeros(batch_size, response_length - values.shape[1], device=self.device)
                    values = torch.cat([values, padding], dim=1)
                else:
                    values = values[:, :response_length]
        
        return response_ids, log_probs, values
    
    @torch.no_grad()
    def compute_reward(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute scalar reward using reward model
        
        Args:
            prompt_ids: Prompt token IDs [batch_size, prompt_len]
            response_ids: Response token IDs [batch_size, response_len]
            
        Returns:
            rewards: Scalar rewards [batch_size]
        """
        # Combine prompt and response
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        
        # Get reward from reward model
        # Assuming reward model outputs scalar via custom head
        outputs = self.reward_model(full_ids)
        
        # Extract reward (simplified - depends on reward model architecture)
        rewards = outputs.logits[:, -1, 0]  # Last token, first logit
        
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
    
    @torch.no_grad()
    def compute_kl_penalty_per_token(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-token KL divergence from reference model
        
        KL(policy || reference) for each token in the response
        
        Args:
            prompt_ids: Prompt token IDs [batch_size, prompt_len]
            response_ids: Response token IDs [batch_size, response_len]
            
        Returns:
            kl_per_token: Per-token KL penalty [batch_size, response_len]
        """
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        prompt_length = prompt_ids.shape[1]
        
        # Policy model logits
        policy_outputs = self.policy_model(full_ids)
        policy_logits = policy_outputs.logits
        
        # Reference model logits
        ref_outputs = self.ref_model(full_ids)
        ref_logits = ref_outputs.logits
        
        # Get logits for response positions only
        # Logits at position i predict token at position i+1
        # So for response tokens, we need logits from positions [prompt_length-1, seq_len-2]
        policy_logits_response = policy_logits[:, prompt_length-1:-1, :]  # [batch_size, response_len, vocab_size]
        ref_logits_response = ref_logits[:, prompt_length-1:-1, :]  # [batch_size, response_len, vocab_size]
        
        # Compute log probabilities
        policy_log_probs = F.log_softmax(policy_logits_response, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits_response, dim=-1)
        
        # Get log probs for the actual response tokens
        # response_ids: [batch_size, response_len]
        response_ids_expanded = response_ids.unsqueeze(-1)  # [batch_size, response_len, 1]
        
        policy_token_log_probs = policy_log_probs.gather(-1, response_ids_expanded).squeeze(-1)  # [batch_size, response_len]
        ref_token_log_probs = ref_log_probs.gather(-1, response_ids_expanded).squeeze(-1)  # [batch_size, response_len]
        
        # Per-token KL: log(policy/ref) = log_policy - log_ref
        # This is the simple KL estimator: E[log(p/q)] where we use the sampled token
        kl_per_token = policy_token_log_probs - ref_token_log_probs  # [batch_size, response_len]
        
        return kl_per_token
    
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
    
    def compute_log_probs_and_entropy_per_token(
        self,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute per-token log probabilities, entropy, and values for given sequences
        
        Args:
            input_ids: Full input sequence (prompt + response) [batch_size, seq_len]
            response_ids: Response tokens [batch_size, response_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            log_probs: Per-token log probabilities [batch_size, response_len]
            entropy: Per-token entropy [batch_size, response_len]
            values: Per-token value estimates [batch_size, response_len]
        """
        batch_size = input_ids.shape[0]
        prompt_length = input_ids.shape[1] - response_ids.shape[1]
        response_length = response_ids.shape[1]
        
        # Forward pass through policy model
        outputs = self.policy_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get logits for response positions
        # Logits at position i predict token at position i+1
        # So for response tokens, we need logits from positions [prompt_length-1, seq_len-2]
        logits = outputs.logits[:, prompt_length-1:-1, :]  # [batch_size, response_len, vocab_size]
        
        # Compute log probabilities
        log_probs_all = F.log_softmax(logits, dim=-1)  # [batch_size, response_len, vocab_size]
        
        # Gather log probs for actual response tokens
        response_ids_expanded = response_ids.unsqueeze(-1)  # [batch_size, response_len, 1]
        log_probs = log_probs_all.gather(-1, response_ids_expanded).squeeze(-1)  # [batch_size, response_len]
        
        # Compute per-token entropy: -sum(p * log(p))
        probs = F.softmax(logits, dim=-1)  # [batch_size, response_len, vocab_size]
        entropy = -torch.sum(probs * log_probs_all, dim=-1)  # [batch_size, response_len]
        
        # Compute per-token values from hidden states
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        all_values = self.value_head(hidden_states)  # [batch_size, seq_len]
        
        # Extract values for response positions
        # Value at position t estimates V(s_t) before generating token t+1
        values = all_values[:, prompt_length-1:-1]  # [batch_size, response_len]
        
        # Handle case where dimensions don't match
        if values.shape[1] != response_length:
            if values.shape[1] < response_length:
                padding = torch.zeros(batch_size, response_length - values.shape[1], device=values.device)
                values = torch.cat([values, padding], dim=1)
            else:
                values = values[:, :response_length]
        
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
        
        # Normalize advantages (only over valid tokens)
        valid_advantages = advantages[response_mask.bool()]
        if valid_advantages.numel() > 0:
            adv_mean = valid_advantages.mean()
            adv_std = valid_advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std
            # Apply mask again to ensure padding positions are 0
            advantages = advantages * response_mask
        
        # Track metrics across PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0
        
        for ppo_epoch in range(num_ppo_epochs):
            # Compute new per-token log probs, entropy, and values
            new_log_probs, entropy, values = self.compute_log_probs_and_entropy_per_token(
                input_ids, response_ids, attention_mask
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
            
            # Backward pass
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), 1.0)
            
            # Optimizer steps
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
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
        
        # Average metrics over PPO epochs
        num_epochs = num_ppo_epochs
        return {
            'policy_loss': total_policy_loss / num_epochs,
            'value_loss': total_value_loss / num_epochs,
            'entropy': total_entropy / num_epochs,
            'approx_kl': total_kl / num_epochs,
            'clip_frac': total_clip_frac / num_epochs,
            'total_loss': (total_policy_loss + total_value_loss) / num_epochs,
        }
    
    def train(self, prompts_file: str):
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
        
        # Create dataset
        dataset = PromptDataset(prompts_file, self.tokenizer, self.config.max_prompt_length)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Training metrics
        global_step = 0
        running_reward = 0.0
        running_kl = 0.0
        num_batches = 0
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            for batch_idx, batch in enumerate(dataloader):
                if global_step >= self.config.num_steps:
                    break
                
                prompt_ids = batch['input_ids'].to(self.device)
                prompt_attention_mask = batch['attention_mask'].to(self.device)
                batch_size = prompt_ids.shape[0]
                
                # ==================== Rollout Phase ====================
                # Generate responses and get per-token log_probs and values
                with torch.no_grad():
                    response_ids, old_log_probs, old_values = self.generate_response(
                        prompt_ids, prompt_attention_mask
                    )
                    # response_ids: [batch_size, response_len]
                    # old_log_probs: [batch_size, response_len] (per-token)
                    # old_values: [batch_size, response_len] (per-token)
                    
                    # Create response mask for valid (non-padding) tokens
                    pad_token_id = self.tokenizer.pad_token_id
                    response_mask = (response_ids != pad_token_id).float()  # [batch_size, response_len]
                    
                    # Compute per-token KL penalty
                    kl_per_token = self.compute_kl_penalty_per_token(prompt_ids, response_ids)
                    # kl_per_token: [batch_size, response_len]
                    
                    # Compute scalar RM scores
                    rm_scores = self.compute_reward(prompt_ids, response_ids)
                    # rm_scores: [batch_size]
                    
                    # Combine into per-token rewards
                    # rewards = -kl_coef * kl_per_token + rm_score (at last valid token)
                    rewards = self.compute_rewards_per_token(rm_scores, kl_per_token, response_mask)
                    # rewards: [batch_size, response_len]
                    
                    # Compute per-token GAE advantages and returns
                    advantages, returns = self.compute_gae_per_token(
                        rewards=rewards,
                        values=old_values,
                        response_mask=response_mask,
                        gamma=self.config.gamma,
                        gae_lambda=self.config.gae_lambda
                    )
                    # advantages: [batch_size, response_len]
                    # returns: [batch_size, response_len]
                
                # ==================== PPO Update Phase ====================
                # Create full input_ids (prompt + response)
                input_ids = torch.cat([prompt_ids, response_ids], dim=1)
                
                # Create full attention mask
                full_attention_mask = (input_ids != pad_token_id).long()
                
                # Create PPO batch with per-token data
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
                metrics = self.ppo_update(ppo_batch, num_ppo_epochs=self.config.num_epochs)
                
                # ==================== Logging and Metrics ====================
                # Compute mean KL for this batch (over valid tokens)
                with torch.no_grad():
                    mean_kl = (kl_per_token * response_mask).sum() / (response_mask.sum() + 1e-8)
                    mean_reward = rm_scores.mean()
                
                running_reward += mean_reward.item()
                running_kl += mean_kl.item()
                num_batches += 1
                
                # Log metrics
                if global_step % 10 == 0 or global_step == 0:
                    avg_reward = running_reward / num_batches if num_batches > 0 else 0
                    avg_kl = running_kl / num_batches if num_batches > 0 else 0
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
                    # Reset running metrics
                    running_reward = 0.0
                    running_kl = 0.0
                    num_batches = 0
                
                # ==================== Adaptive KL Coefficient ====================
                current_kl = mean_kl.item()
                if current_kl > self.config.target_kl * 1.5:
                    self.kl_coef = min(self.kl_coef * 1.5, 1.0)  # Cap at 1.0
                elif current_kl < self.config.target_kl / 1.5:
                    self.kl_coef = max(self.kl_coef / 1.5, 0.001)  # Floor at 0.001
                
                global_step += 1
                
                # Save checkpoint periodically
                if global_step % 500 == 0:
                    self.save_checkpoint(global_step)
            
            # End of epoch
            if global_step >= self.config.num_steps:
                break
        
        # Save final model
        logger.info("Saving final model")
        self.save_checkpoint("final")
        logger.info(f"Training complete! Total steps: {global_step}")
    
    def save_checkpoint(self, step):
        """Save model checkpoint"""
        save_path = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(save_path, exist_ok=True)
        
        self.policy_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        torch.save(self.value_head.state_dict(), os.path.join(save_path, "value_head.pt"))
        
        # Save config
        config_path = os.path.join(save_path, "rlhf_config.json")
        with open(config_path, 'w') as f:
            json.dump({
                'policy_model_path': self.config.policy_model_path,
                'reward_model_path': self.config.reward_model_path,
                'num_epochs': self.config.num_epochs,
                'num_steps': self.config.num_steps,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'kl_coef': self.kl_coef,
                'target_kl': self.config.target_kl,
                'clip_range': self.config.clip_range,
            }, f, indent=2)
        
        logger.info(f"Checkpoint saved: {save_path}")
    
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
            # Tokenize prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=self.config.max_prompt_length,
                padding='max_length'
            )
            prompt_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Generate response
            response_ids, _, _ = self.generate_response(prompt_ids, attention_mask)
            
            # Compute reward
            reward = self.compute_reward(prompt_ids, response_ids)
            
            # Compute KL
            kl = self.compute_kl_penalty(prompt_ids, response_ids)
            
            # Decode response
            response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
            
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


def main():
    """Main entry point"""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="RLHF Training with PPO")
    parser.add_argument("--config", type=str, required=True, help="Config YAML file")
    parser.add_argument("--prompts", type=str, required=True, help="Prompts JSON file")
    parser.add_argument("--eval-prompts", type=str, default=None, help="Evaluation prompts JSON file")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with reduced steps")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract PPO parameters with defaults
    ppo_config = config_dict.get('algorithm', {}).get('ppo', {})
    training_config = config_dict.get('training', {})
    generation_config = config_dict.get('generation', {})
    data_config = config_dict.get('data', {})
    
    # Build RLHFConfig
    config = RLHFConfig(
        # Model paths
        policy_model_path=config_dict['model']['policy']['name'],
        reward_model_path=config_dict['model']['reward']['name'],
        output_dir=training_config.get('output_dir', './outputs/rlhf'),
        
        # PPO hyperparameters
        num_epochs=ppo_config.get('num_epochs', 4),
        num_steps=training_config.get('total_steps', 10000),
        batch_size=training_config.get('per_device_batch_size', 4),
        learning_rate=training_config.get('policy_learning_rate', 1e-6),
        
        # KL penalty
        init_kl_coef=ppo_config.get('init_kl_coef', ppo_config.get('kl_coef', 0.05)),
        target_kl=ppo_config.get('target_kl', 0.1),
        
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
    
    # Optional evaluation
    if args.eval_prompts:
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
