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
    gamma: float = 0.99
    
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


class PPOBuffer:
    """
    Experience replay buffer for PPO
    
    Stores trajectories: (states, actions, rewards, values, log_probs)
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.advantages = []
        self.returns = []
    
    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        value: float,
        log_prob: float
    ):
        """Add experience to buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
    
    def compute_advantages(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        advantages = []
        gae = 0
        
        # Reverse iteration for GAE computation
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[i + 1]
            
            # TD error: reward + gamma * next_value - current_value
            delta = self.rewards[i] + gamma * next_value - self.values[i]
            
            # GAE accumulation
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)
        
        self.advantages = advantages
        
        # Compute returns: advantages + values
        self.returns = [adv + val for adv, val in zip(self.advantages, self.values)]
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get all experiences as tensors"""
        return {
            'states': torch.stack(self.states),
            'actions': torch.stack(self.actions),
            'old_log_probs': torch.tensor(self.log_probs),
            'advantages': torch.tensor(self.advantages),
            'returns': torch.tensor(self.returns),
        }
    
    def clear(self):
        """Clear buffer"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.advantages.clear()
        self.returns.clear()


class ValueHead(nn.Module):
    """
    Value function head for PPO
    Estimates state value V(s)
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.value_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.value_head.weight, mean=0.0, std=0.01)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            values: [batch_size, 1]
        """
        # Use last token hidden state
        last_hidden = hidden_states[:, -1, :]
        values = self.value_head(last_hidden)
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
            prompt_ids: Prompt token IDs
            attention_mask: Attention mask
            
        Returns:
            response_ids: Generated token IDs [batch_size, response_len]
            log_probs: Log probabilities of generated tokens [batch_size, response_len]
            values: Value estimates [batch_size]
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
        
        # Compute value estimates using value head
        # We need to get hidden states from the full sequence
        full_ids = outputs.sequences
        with torch.no_grad():
            model_outputs = self.policy_model(
                full_ids,
                output_hidden_states=True,
                return_dict=True
            )
            # Get last hidden state
            hidden_states = model_outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            values = self.value_head(hidden_states).squeeze(-1)  # [batch_size]
        
        return response_ids, log_probs, values
    
    @torch.no_grad()
    def compute_reward(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reward using reward model
        
        Args:
            prompt_ids: Prompt token IDs
            response_ids: Response token IDs
            
        Returns:
            rewards: Scalar rewards
        """
        # Combine prompt and response
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        
        # Get reward from reward model
        # Assuming reward model outputs scalar via custom head
        outputs = self.reward_model(full_ids)
        
        # Extract reward (simplified - depends on reward model architecture)
        rewards = outputs.logits[:, -1, 0]  # Last token, first logit
        
        return rewards
    
    def compute_kl_penalty(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence from reference model
        
        KL(policy || reference)
        
        Args:
            prompt_ids: Prompt token IDs
            response_ids: Response token IDs
            
        Returns:
            kl_divergence: KL penalty
        """
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        
        # Policy model logits
        policy_outputs = self.policy_model(full_ids)
        policy_logits = policy_outputs.logits
        
        # Reference model logits
        with torch.no_grad():
            ref_outputs = self.ref_model(full_ids)
            ref_logits = ref_outputs.logits
        
        # Compute KL divergence
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        
        kl_div = F.kl_div(
            policy_log_probs,
            ref_log_probs,
            reduction='batchmean',
            log_target=True
        )
        
        return kl_div
    
    def compute_log_probs_and_entropy(
        self,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities and entropy for given sequences
        
        Args:
            input_ids: Full input sequence (prompt + response) [batch_size, seq_len]
            response_ids: Response tokens [batch_size, response_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            log_probs: Log probabilities for response tokens [batch_size, response_len]
            entropy: Entropy of the distribution [batch_size]
            values: Value estimates [batch_size]
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
        
        # Compute entropy: -sum(p * log(p))
        probs = F.softmax(logits, dim=-1)  # [batch_size, response_len, vocab_size]
        entropy_per_token = -torch.sum(probs * log_probs_all, dim=-1)  # [batch_size, response_len]
        
        # Create mask for valid (non-padding) tokens in response
        pad_token_id = self.tokenizer.pad_token_id
        response_mask = (response_ids != pad_token_id).float()  # [batch_size, response_len]
        
        # Masked mean entropy
        entropy = (entropy_per_token * response_mask).sum(dim=-1) / (response_mask.sum(dim=-1) + 1e-8)  # [batch_size]
        
        # Sum log probs over response (masked)
        log_probs_sum = (log_probs * response_mask).sum(dim=-1)  # [batch_size]
        
        # Compute values from hidden states
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        values = self.value_head(hidden_states).squeeze(-1)  # [batch_size]
        
        return log_probs_sum, entropy, values
    
    def ppo_update(
        self,
        batch: Dict[str, torch.Tensor],
        num_ppo_epochs: int = 4
    ) -> Dict[str, float]:
        """
        PPO policy update with multiple epochs over the batch
        
        Args:
            batch: Batch of experiences containing:
                - input_ids: Full sequences (prompt + response) [batch_size, seq_len]
                - response_ids: Response tokens only [batch_size, response_len]
                - attention_mask: Attention mask [batch_size, seq_len]
                - old_log_probs: Log probs from rollout [batch_size]
                - old_values: Value estimates from rollout [batch_size]
                - advantages: Computed advantages [batch_size]
                - returns: Computed returns [batch_size]
            num_ppo_epochs: Number of PPO epochs per batch
            
        Returns:
            metrics: Training metrics
        """
        input_ids = batch['input_ids'].to(self.device)
        response_ids = batch['response_ids'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        old_log_probs = batch['old_log_probs'].to(self.device).detach()
        old_values = batch['old_values'].to(self.device).detach()
        advantages = batch['advantages'].to(self.device).detach()
        returns = batch['returns'].to(self.device).detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Track metrics across PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0
        
        for ppo_epoch in range(num_ppo_epochs):
            # Compute new log probs, entropy, and values
            new_log_probs, entropy, values = self.compute_log_probs_and_entropy(
                input_ids, response_ids, attention_mask
            )
            
            # Compute ratio: exp(new_log_prob - old_log_prob)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 
                1.0 - self.config.clip_range, 
                1.0 + self.config.clip_range
            ) * advantages
            
            # Policy loss (negative because we want to maximize)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss with clipping (optional but recommended)
            values_clipped = old_values + torch.clamp(
                values - old_values,
                -self.config.clip_range,
                self.config.clip_range
            )
            value_loss_unclipped = (values - returns) ** 2
            value_loss_clipped = (values_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            
            # Entropy bonus (mean over batch)
            entropy_bonus = entropy.mean()
            
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
            
            # Compute metrics
            with torch.no_grad():
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                clip_frac = ((ratio - 1.0).abs() > self.config.clip_range).float().mean()
            
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
        Main training loop
        
        Args:
            prompts_file: Path to prompts JSON file
        """
        logger.info("Starting RLHF training")
        logger.info(f"Config: num_epochs={self.config.num_epochs}, num_steps={self.config.num_steps}")
        logger.info(f"Batch size: {self.config.batch_size}, Learning rate: {self.config.learning_rate}")
        
        # Create dataset
        dataset = PromptDataset(prompts_file, self.tokenizer, self.config.max_prompt_length)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Initialize PPO buffer
        ppo_buffer = PPOBuffer()
        rollout_batch_size = 4  # Number of experiences to collect before PPO update
        
        # Training metrics
        global_step = 0
        total_episodes = 0
        running_reward = 0.0
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            for batch_idx, batch in enumerate(dataloader):
                if global_step >= self.config.num_steps:
                    break
                
                prompt_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                batch_size = prompt_ids.shape[0]
                
                # ==================== Rollout Phase ====================
                # Generate responses and collect experiences
                with torch.no_grad():
                    response_ids, log_probs, values = self.generate_response(prompt_ids, attention_mask)
                    
                    # Compute rewards from reward model
                    rewards = self.compute_reward(prompt_ids, response_ids)
                    
                    # Compute KL penalty
                    kl_penalty = self.compute_kl_penalty(prompt_ids, response_ids)
                    
                    # Adjusted reward = reward - kl_coef * kl_penalty
                    # KL penalty is per-batch, rewards is per-sample
                    adjusted_rewards = rewards - self.kl_coef * kl_penalty
                
                # Store experiences in buffer
                # For simplicity, we treat each response as one "step"
                # Sum log probs over response tokens for each sample
                pad_token_id = self.tokenizer.pad_token_id
                response_mask = (response_ids != pad_token_id).float()
                log_probs_sum = (log_probs * response_mask).sum(dim=-1)  # [batch_size]
                
                for i in range(batch_size):
                    ppo_buffer.add(
                        state=prompt_ids[i],
                        action=response_ids[i],
                        reward=adjusted_rewards[i].item() if adjusted_rewards.dim() > 0 else adjusted_rewards.item(),
                        value=values[i].item(),
                        log_prob=log_probs_sum[i].item()
                    )
                
                total_episodes += batch_size
                running_reward += rewards.sum().item()
                
                # ==================== PPO Update Phase ====================
                # Update policy when buffer has enough experiences
                if len(ppo_buffer.rewards) >= rollout_batch_size:
                    # Compute advantages using GAE
                    ppo_buffer.compute_advantages(
                        gamma=self.config.gamma,
                        gae_lambda=self.config.gae_lambda
                    )
                    
                    # Create batch for PPO update
                    # We need to reconstruct full sequences
                    buffer_size = len(ppo_buffer.states)
                    
                    # Stack states and actions
                    states_batch = torch.stack(ppo_buffer.states)  # [buffer_size, prompt_len]
                    actions_batch = torch.stack(ppo_buffer.actions)  # [buffer_size, response_len]
                    
                    # Create full input_ids (prompt + response)
                    input_ids_batch = torch.cat([states_batch, actions_batch], dim=1)
                    
                    # Create attention mask
                    attention_mask_batch = (input_ids_batch != self.tokenizer.pad_token_id).long()
                    
                    # Create PPO batch
                    ppo_batch = {
                        'input_ids': input_ids_batch,
                        'response_ids': actions_batch,
                        'attention_mask': attention_mask_batch,
                        'old_log_probs': torch.tensor(ppo_buffer.log_probs, dtype=torch.float32),
                        'old_values': torch.tensor(ppo_buffer.values, dtype=torch.float32),
                        'advantages': torch.tensor(ppo_buffer.advantages, dtype=torch.float32),
                        'returns': torch.tensor(ppo_buffer.returns, dtype=torch.float32),
                    }
                    
                    # PPO update
                    metrics = self.ppo_update(ppo_batch, num_ppo_epochs=self.config.num_epochs)
                    
                    # Clear buffer
                    ppo_buffer.clear()
                    
                    # Log metrics
                    avg_reward = running_reward / total_episodes if total_episodes > 0 else 0
                    logger.info(
                        f"Step {global_step}: "
                        f"reward={avg_reward:.3f}, "
                        f"kl={kl_penalty.item():.4f}, "
                        f"policy_loss={metrics['policy_loss']:.4f}, "
                        f"value_loss={metrics['value_loss']:.4f}, "
                        f"entropy={metrics['entropy']:.4f}, "
                        f"clip_frac={metrics['clip_frac']:.3f}, "
                        f"kl_coef={self.kl_coef:.4f}"
                    )
                    
                    # Reset running metrics
                    running_reward = 0.0
                    total_episodes = 0
                
                # ==================== Adaptive KL Coefficient ====================
                current_kl = kl_penalty.item()
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
