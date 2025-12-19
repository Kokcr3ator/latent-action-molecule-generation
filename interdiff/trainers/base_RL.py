"""Base trainer class for Reinforcement Learning (PPO) finetuning.

This module provides a trainer class for PPO-based RL training, analogous to
TrainerBase for supervised learning. It handles the training loop, checkpointing,
logging, and evaluation for PPO finetuning of language models.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import numpy as np

from interdiff.io import load_tokenizer
from interdiff.ppo import PPO
from interdiff.envs import Env, Timestep
from interdiff.utils.eval_utils import tokens_to_smiles
from interdiff.metrics import validity, uniqueness, novelty


@dataclass
class RLTrainConfig:
    """Configuration for RL/PPO training."""
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training budget
    budget: int = 1_000_000  # Total environment steps
    
    # Logging and checkpointing
    log_frequency: int = 1  # Log every N updates
    eval_frequency: int = 10  # Evaluate every N updates
    ckpt_path: str = "ckpts/ppo"
    always_save_checkpoint: bool = False
    
    # Evaluation
    n_mols_generate: int = 100
    
    # Tokenizer
    tokenizer_dir: str = ""
    pad_token_id: int = 0


@dataclass
class RLTrainState:
    """Tracks the state of RL training."""
    update: int = 0
    env_steps: int = 0
    best_reward: float = float("-inf")


class RLTrainerBase:
    """Base trainer for PPO-based reinforcement learning.
    
    This trainer manages the PPO training loop including:
    - Environment interaction and experience collection
    - Policy updates via PPO
    - Checkpointing and logging
    - Periodic evaluation
    
    Args:
        ppo_agent: Instantiated PPO agent with encoder and optimizer.
        env: RL environment (e.g., MoleculeGenerationEnv).
        logger: Optional logger for metrics (e.g., WandB logger).
        train_cfg: Training configuration.
    """
    
    def __init__(
        self,
        ppo_agent: PPO,
        env: Env,
        logger: Optional[Any] = None,
        train_cfg: Optional[RLTrainConfig] = None,
    ) -> None:
        self.ppo_agent = ppo_agent
        self.env = env
        self.logger = logger
        self.cfg = train_cfg or RLTrainConfig()
        
        self.device = torch.device(self.cfg.device)
        self.ppo_agent = self.ppo_agent.to(self.device)
        
        # Load tokenizer if path provided
        self.tokenizer = load_tokenizer(os.path.join(self.cfg.tokenizer_dir, "tokenizer.json"))
        
        # Create checkpoint directory
        self.ckpt_path = Path(self.cfg.ckpt_path)
        self.ckpt_path.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.state = RLTrainState()
        
        # Reference SMILES for novelty calculation (built from train_dataloader on first eval)
        self.reference_smiles = []
    
    def fit(self, train_dataloader=None) -> None:
        """Run the full PPO training loop.
        
        Args:
            train_dataloader: Optional training dataloader for building reference SMILES
                            for novelty calculation. If not provided, novelty will be 0.
        """
        hparams = self.ppo_agent.hparams
        total_updates = self.cfg.budget // (hparams.num_envs * hparams.num_steps)
        
        self._log_training_info(total_updates)
        
        # Initialize environment
        timestep = self._reset_environment()
        
        # Training loop
        while self.state.update < total_updates:
            # Run PPO update (collect experience + gradient updates)
            logs = self.ppo_agent.update(self.env, timestep)
            
            # Update state
            self.state.update += 1
            self.state.env_steps = self.ppo_agent.env_steps
            
            # Reset environment after each rollout to maintain valid state
            timestep = self._reset_environment()
            
            # Log metrics
            if self.state.update % self.cfg.log_frequency == 0:
                self._log_metrics(logs)
            
            # Evaluate periodically
            if self.state.update % self.cfg.eval_frequency == 0:
                eval_metrics = self.evaluate(train_dataloader=train_dataloader)
                self._log_eval_metrics(eval_metrics)
                
                # Save checkpoint if improved
                mean_reward = eval_metrics.get("eval/mean_reward", float("-inf"))
                if mean_reward > self.state.best_reward:
                    self.state.best_reward = mean_reward
                    self.save_checkpoint(
                        self.ckpt_path / f"best_reward_{self.state.best_reward:.4f}.pt"
                    )
                elif self.cfg.always_save_checkpoint:
                    self.save_checkpoint(
                        self.ckpt_path / f"update_{self.state.update}.pt"
                    )
            
            # Check if budget exhausted
            if self.ppo_agent.env_steps >= self.cfg.budget:
                print(f"Budget exhausted after {self.ppo_agent.env_steps} env steps")
                break
        
        # Final checkpoint
        self.save_checkpoint(self.ckpt_path / "finetuned.pt")
        
        if self.logger:
            self.logger.finalize()
        
        print("PPO training completed!")
    
    def _reset_environment(self) -> Timestep:
        """Reset the environment and return initial timestep."""
        seed = torch.randint(0, 2**31, (self.env.num_envs,), device=self.device).to(torch.long)
        return self.env.reset(timestep= None, seed=seed)
    
    def _log_training_info(self, total_updates: int) -> None:
        """Log training configuration at start."""
        hparams = self.ppo_agent.hparams
        print(f"Starting PPO training")
        print(f"  Total updates: {total_updates}")
        print(f"  Budget: {self.cfg.budget} env steps")
        print(f"  Batch size: {hparams.num_envs * hparams.num_steps}")
        print(f"  Num envs: {hparams.num_envs}")
        print(f"  Num steps per rollout: {hparams.num_steps}")
        print(f"  Device: {self.device}")
    
    def _log_metrics(self, logs: Dict[str, Any]) -> None:
        """Log training metrics."""
        log_msg = f"Update {self.state.update} | Env steps {self.state.env_steps}"
        for k, v in logs.items():
            if isinstance(v, float):
                log_msg += f" | {k}: {v:.4f}"
        print(log_msg)
        
        if self.logger:
            logs["step"] = self.state.env_steps
            logs["update"] = self.state.update
            self.logger.log(logs)
    
    def _log_eval_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log evaluation metrics."""
        print(f"Eval @ update {self.state.update}: {metrics}")
        if self.logger:
            metrics["step"] = self.state.env_steps
            metrics["update"] = self.state.update
            self.logger.log(metrics)
    
    @torch.no_grad()
    def evaluate(self, train_dataloader=None) -> Dict[str, float]:
        """Evaluate the agent by generating molecules and computing metrics.
        
        Generates molecules and computes:
        - Task-specific reward (based on env.reward_fn.task: qed, logp, mw, tpsa, sa)
        - Validity (percentage of valid SMILES)
        - Uniqueness (percentage of unique molecules)
        - Novelty (percentage not in training set)
        - Mean episode reward and length
        
        Args:
            train_dataloader: Optional training dataloader for building reference SMILES.
                            If not provided, novelty will be 0.
        
        Returns:
            Dictionary of evaluation metrics including task reward, validity,
            uniqueness, novelty, mean reward, and episode length.
        """
        self.ppo_agent.eval()
        
        num_episodes = self.cfg.n_mols_generate
        total_rewards = []
        episode_lengths = []
        generated_sequences = []  # Store all generated token sequences
        
        # Reset environment for evaluation
        timestep = self._reset_environment()
        episode_rewards = torch.zeros(self.env.num_envs, device=self.device)
        episode_lens = torch.zeros(self.env.num_envs, device=self.device)
        
        max_steps = self.env.context_length * (num_episodes // self.env.num_envs + 2)
        
        for t in range(max_steps):
            if len(total_rewards) >= num_episodes:
                break
            
            # Get action from policy
            dist, _ = self.ppo_agent(timestep.observation)
            action = dist.sample()
            
            # Step environment
            timestep = self.env.step(timestep, action)
            episode_rewards += timestep.reward
            episode_lens += 1
            
            # Check for terminated episodes
            terminated = timestep.step_type > 0
            if terminated.any():
                for i in range(self.env.num_envs):
                    if terminated[i] and len(total_rewards) < num_episodes:
                        total_rewards.append(episode_rewards[i].item())
                        episode_lengths.append(episode_lens[i].item())
                        # Store the generated sequence
                        generated_sequences.append(timestep.observation[i].cpu())
                        episode_rewards[i] = 0.0
                        episode_lens[i] = 0.0
        
        self.ppo_agent.train()
        
        if len(total_rewards) == 0:
            return {
                "eval/mean_reward": 0.0,
                "eval/mean_episode_length": 0.0,
                "eval/validity": 0.0,
                "eval/uniqueness": 0.0,
                "eval/novelty": 0.0,
            }
        
        # Convert generated sequences to SMILES
        generated_tokens = torch.stack(generated_sequences) if generated_sequences else torch.empty(0)
        generated_smiles = tokens_to_smiles(generated_tokens, tokenizer=self.tokenizer)
        
        # Calculate molecular metrics
        valid = validity(generated_smiles)
        unique = uniqueness(generated_smiles)
        
        # Build reference set from training data on first eval (for novelty)
        # Same approach as GPTTrainer
        if len(self.reference_smiles) == 0 and train_dataloader is not None:
            for train_batch in train_dataloader:
                batch_smiles = tokens_to_smiles(train_batch['x'], tokenizer=self.tokenizer)
                self.reference_smiles.extend(batch_smiles)
        
        # Calculate novelty if we have reference SMILES
        if len(self.reference_smiles) > 0:
            novel = novelty(generated_smiles, reference_smiles=self.reference_smiles)
        else:
            novel = 0.0
        
        # Calculate task-specific metric (same as reward function)
        task = self.env.reward_fn.task
        task_scores = [self.env.reward_fn.reward_from_smiles(smi) for smi in generated_smiles]
        task_score = np.mean(task_scores) if len(task_scores) > 0 else 0.0
        
        return {
            "eval/mean_reward": sum(total_rewards) / len(total_rewards),
            "eval/mean_episode_length": sum(episode_lengths) / len(episode_lengths),
            "eval/num_episodes": len(total_rewards),
            f"eval/{task}": task_score,
            "eval/validity": valid,
            "eval/uniqueness": unique,
            "eval/novelty": novel,
        }
    
    def save_checkpoint(self, path: Path | str) -> None:
        """Save training checkpoint.
        
        Args:
            path: Path to save the checkpoint.
        """
        path = Path(path)
        torch.save({
            "ppo_agent": self.ppo_agent.state_dict(),
            "optimizer": self.ppo_agent.optimiser.state_dict(),
            "env_steps": self.ppo_agent.env_steps,
            "update_steps": self.ppo_agent.update_steps,
            "state": {
                "update": self.state.update,
                "env_steps": self.state.env_steps,
                "best_reward": self.state.best_reward,
            },
        }, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path | str) -> None:
        """Load training checkpoint.
        
        Args:
            path: Path to the checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        if "ppo_agent" in checkpoint:
            self.ppo_agent.load_state_dict(checkpoint["ppo_agent"])
        if "optimizer" in checkpoint:
            self.ppo_agent.optimiser.load_state_dict(checkpoint["optimizer"])
        if "env_steps" in checkpoint:
            self.ppo_agent.env_steps = checkpoint["env_steps"]
        if "update_steps" in checkpoint:
            self.ppo_agent.update_steps = checkpoint["update_steps"]
        if "state" in checkpoint:
            state = checkpoint["state"]
            self.state.update = state.get("update", 0)
            self.state.env_steps = state.get("env_steps", 0)
            self.state.best_reward = state.get("best_reward", float("-inf"))
        
        print(f"Loaded checkpoint from {path}")