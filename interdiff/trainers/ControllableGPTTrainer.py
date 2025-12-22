from typing import Dict, Any, Iterable
import itertools
import os

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .base import TrainerBase

class ControllableGPTTrainer(TrainerBase):
    """Trainer for ControllableGPT model pretraining.
    
    Trains a ControllableGPT model that combines a Latent Action Model (LAM)
    and a Dynamics Model, with vector quantization loss.
    
    Args:
        model: ControllableGPT model to train.
        optimizer: Optimizer for training.
        scheduler: Optional learning rate scheduler.
        logger: Optional logger (e.g., WandB logger).
        train_cfg: Training configuration.
    """
    def __init__(self, model, optimizer, scheduler, logger, train_cfg):
        super().__init__(model, optimizer, scheduler, logger, train_cfg)
        self.reference_smiles = []
    
    def forward_loss(self, batch) -> torch.Tensor:
            """Compute combined loss for ControllableGPT.
            
            Computes cross-entropy losses for both the Latent Action Model
            and Dynamics Model, plus the vector quantization loss.
            
            Args:
                batch: Dictionary containing 'x' (input tokens) and 'y' (target tokens).
                
            Returns:
                Total loss combining LAM loss, dynamics loss, and VQ loss.
            """
            x = batch['x']
            y = batch['y']
            lam_logits, dynamics_model_logits, vq_loss_dict = self.model(x)
            lam_ce_loss = F.cross_entropy(
                lam_logits.view(-1, lam_logits.size(-1)),
                y.reshape(-1),
                ignore_index=self.pad_token_id,
                )
            
            dynamics_loss = F.cross_entropy(
                dynamics_model_logits.view(-1, dynamics_model_logits.size(-1)),
                y.reshape(-1),
                ignore_index=self.pad_token_id,
                )
            
            vq_loss = vq_loss_dict['vq_loss']
            lam_loss = lam_ce_loss + vq_loss
            total_loss = lam_loss + dynamics_loss
            return total_loss

    def forward_loss_with_components(self, batch) -> Dict[str, torch.Tensor]:
            """Compute combined loss for ControllableGPT with all individual components.
            
            Computes cross-entropy losses for both the Latent Action Model
            and Dynamics Model, plus the vector quantization loss components.
            
            Args:
                batch: Dictionary containing 'x' (input tokens) and 'y' (target tokens).
                
            Returns:
                Dictionary containing all loss components:
                    - 'total_loss': Total combined loss.
                    - 'lam_loss': LAM loss (CE + VQ).
                    - 'lam_ce_loss': LAM cross-entropy loss.
                    - 'dynamics_loss': Dynamics model cross-entropy loss.
                    - 'vq_loss': Total VQ loss.
                    - 'vq_q_loss': VQ reconstruction loss.
                    - 'vq_commit_loss': VQ commitment loss.
                    - 'vq_entropy_loss': VQ entropy regularization loss.
                    - 'vq_entropy': Raw codebook usage entropy.
            """
            x = batch['x']
            y = batch['y']
            lam_logits, dynamics_model_logits, vq_loss_dict = self.model(x)
            lam_ce_loss = F.cross_entropy(
                lam_logits.view(-1, lam_logits.size(-1)),
                y.reshape(-1),
                ignore_index=self.pad_token_id,
                )
            
            dynamics_loss = F.cross_entropy(
                dynamics_model_logits.view(-1, dynamics_model_logits.size(-1)),
                y.reshape(-1),
                ignore_index=self.pad_token_id,
                )
            
            vq_loss = vq_loss_dict['vq_loss']
            lam_loss = lam_ce_loss + vq_loss
            total_loss = lam_loss + dynamics_loss
            
            return {
                'total_loss': total_loss,
                'lam_loss': lam_loss,
                'lam_ce_loss': lam_ce_loss,
                'dynamics_loss': dynamics_loss,
                'vq_loss': vq_loss,
                'vq_q_loss': vq_loss_dict['q_loss'],
                'vq_commit_loss': vq_loss_dict['commit_loss'],
                'vq_entropy_loss': vq_loss_dict['entropy_loss'],
                'vq_entropy': vq_loss_dict['entropy'],
            }

    def fit(self, train_dataloader: Iterable, val_dataloader: Iterable):
        """Train the model with detailed loss component logging."""
        self.model.train()
        dl = itertools.cycle(train_dataloader)
        os.makedirs(self.ckpt_path, exist_ok=True)

        # Running averages for all loss components
        running_metrics = {
            'total_loss': 0.0,
            'lam_loss': 0.0,
            'lam_ce_loss': 0.0,
            'dynamics_loss': 0.0,
            'vq_loss': 0.0,
            'vq_q_loss': 0.0,
            'vq_commit_loss': 0.0,
            'vq_entropy_loss': 0.0,
            'vq_entropy': 0.0,
        }

        while self.state.step < self.max_iters:
            with torch.amp.autocast(device_type=self.device, dtype=self.mixed_dtype, enabled=self.mixed_dtype != torch.float32):
                loss_dict = self.forward_loss_with_components(next(dl))
                loss = loss_dict['total_loss'].float() / self.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            # Accumulate all metrics
            for key in running_metrics:
                running_metrics[key] += float(loss_dict[key].detach().cpu()) / self.gradient_accumulation_steps

            if (self.state.step + 1) % self.gradient_accumulation_steps == 0:
                if self.grad_clip and self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)

                if self.state.step < self.warmup_iters:
                    scale = self.get_lr_scale()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.base_lr * scale

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()

            if self.logger and self.state.step % self.log_interval == 0:
                log_dict = {"step": self.state.step}
                for key, value in running_metrics.items():
                    log_dict[f"train/{key}"] = value / max(1, self.log_interval)
                self.logger.log(log_dict)
                # Reset running metrics
                for key in running_metrics:
                    running_metrics[key] = 0.0

            elif self.state.step % self.log_interval == 0:
                avg_loss = running_metrics['total_loss'] / max(1, self.log_interval)
                print(f"step {self.state.step}: train loss {avg_loss}")
                for key in running_metrics:
                    running_metrics[key] = 0.0

            if self.state.step % self.eval_interval == 0:
                val_dict = self.evaluate(val_dataloader=val_dataloader, train_dataloader=train_dataloader)
                val = val_dict.get('val_loss', float('inf'))
                if self.logger:
                    val_dict_log = {"step": self.state.step}
                    val_dict_log.update(val_dict)
                    self.logger.log(val_dict_log)
                else:
                    print(f"step {self.state.step}: " + ", ".join([f"{k} {v}" for k, v in val_dict.items()]))
                improved = val < self.state.best_val
                self.state.best_val = min(self.state.best_val, val)
                if self.always_save_checkpoint or improved:
                    self.save_checkpoint(os.path.join(self.ckpt_path, "best.pt"))
            self.state.step += 1

    @torch.no_grad()
    def evaluate(self, val_dataloader: Iterable, **kwargs) -> Dict[str, float]:
        """Evaluate model with detailed loss component logging."""
        self.model.eval()
        
        # Accumulators for all loss components
        accumulated_metrics = {
            'total_loss': 0.0,
            'lam_loss': 0.0,
            'lam_ce_loss': 0.0,
            'dynamics_loss': 0.0,
            'vq_loss': 0.0,
            'vq_q_loss': 0.0,
            'vq_commit_loss': 0.0,
            'vq_entropy_loss': 0.0,
            'vq_entropy': 0.0,
        }
        
        for i, batch in zip(range(self.eval_iters), val_dataloader):
            with torch.amp.autocast(device_type=self.device, dtype=self.mixed_dtype, enabled=self.mixed_dtype != torch.float32):
                loss_dict = self.forward_loss_with_components(batch)
            
            for key in accumulated_metrics:
                accumulated_metrics[key] += float(loss_dict[key].detach().cpu())
        
        self.model.train()
        
        # Compute averages and format output
        num_batches = max(1, self.eval_iters)
        result = {
            'val_loss': accumulated_metrics['total_loss'] / num_batches,
            'val_lam_loss': accumulated_metrics['lam_loss'] / num_batches,
            'val_lam_ce_loss': accumulated_metrics['lam_ce_loss'] / num_batches,
            'val_dynamics_loss': accumulated_metrics['dynamics_loss'] / num_batches,
            'val_vq_loss': accumulated_metrics['vq_loss'] / num_batches,
            'val_vq_q_loss': accumulated_metrics['vq_q_loss'] / num_batches,
            'val_vq_commit_loss': accumulated_metrics['vq_commit_loss'] / num_batches,
            'val_vq_entropy_loss': accumulated_metrics['vq_entropy_loss'] / num_batches,
            'val_vq_entropy': accumulated_metrics['vq_entropy'] / num_batches,
        }
        return result