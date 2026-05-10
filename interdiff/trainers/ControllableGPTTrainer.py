from typing import Dict, Any, Iterable
import itertools
import os

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

try:
    import wandb
except ImportError:
    wandb = None

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
                'vq_norm_penalty': vq_loss_dict['vq_norm_penalty'],
                'vq_indices': vq_loss_dict['indices'],
            }

    def _codebook_stats(self, counts: torch.Tensor) -> Dict[str, float]:
        """Compute codebook health metrics from a bincount of index usage."""
        num_latents = self.model.lam.vq.num_latents
        norms = getattr(self.model, '_orig_mod', self.model).lam.vq.codebook.data.float().norm(dim=-1)
        utilization = (counts > 0).sum().item() / num_latents
        return {
            'codebook_norm_mean': norms.mean().item(),
            'codebook_norm_max': norms.max().item(),
            'codebook_norm_std': norms.std().item(),
            'codebook_utilization': utilization,
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
            'vq_norm_penalty': 0.0,
        }
        num_latents = getattr(self.model, '_orig_mod', self.model).lam.vq.num_latents
        running_counts = torch.zeros(num_latents, dtype=torch.long)

        while self.state.step < self.max_iters:
            with torch.amp.autocast(device_type=self.device, dtype=self.mixed_dtype, enabled=self.mixed_dtype != torch.float32):
                loss_dict = self.forward_loss_with_components(next(dl))
                loss = loss_dict['total_loss'].float() / self.gradient_accumulation_steps

            # Detect NaN/Inf before backward to surface the exact step and component
            nan_keys = [k for k in running_metrics if not torch.isfinite(loss_dict[k])]
            if nan_keys:
                print(f"[step {self.state.step}] WARNING: NaN/Inf in {nan_keys} — "
                      + ", ".join(f"{k}={float(loss_dict[k]):.4g}" for k in running_metrics))

            self.scaler.scale(loss).backward()

            # Accumulate all metrics
            for key in running_metrics:
                running_metrics[key] += float(loss_dict[key].detach().cpu()) / self.gradient_accumulation_steps
            running_counts += torch.bincount(loss_dict['vq_indices'].detach().cpu().view(-1), minlength=num_latents)

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
                orig = getattr(self.model, '_orig_mod', self.model)
                if orig.lam.vq.norm_mode == "step":
                    orig.lam.vq._normalize_codebook()

            log_dict = {}

            if self.state.step % self.log_interval == 0:
                for key, value in running_metrics.items():
                    log_key = "loss" if key == "total_loss" else key
                    log_dict[f"train/{log_key}"] = value / max(1, self.log_interval)
                log_dict["train/lr"] = self.optimizer.param_groups[0]["lr"]
                for key in running_metrics:
                    running_metrics[key] = 0.0

                cb_stats = self._codebook_stats(running_counts)
                for k, v in cb_stats.items():
                    log_dict[f"train/{k}"] = v
                running_counts.zero_()

                print(
                    f"step {self.state.step:6d} | "
                    + " | ".join(
                        f"{k.replace('train/', '')}={v:.4g}"
                        for k, v in log_dict.items()
                        if k.startswith("train/")
                    )
                )

            if self.state.step % self.eval_interval == 0:
                val_dict = self.evaluate(val_dataloader=val_dataloader, train_dataloader=train_dataloader)
                val = val_dict.get('val/loss', float('inf'))
                log_dict.update(val_dict)
                improved = val < self.state.best_val
                self.state.best_val = min(self.state.best_val, val)
                if self.always_save_checkpoint or improved:
                    ckpt_path = os.path.join(self.ckpt_path, "best.pt")
                    self.save_checkpoint(ckpt_path)

            if log_dict:
                if self.logger:
                    self.logger.log({"step": self.state.step, **log_dict})

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
            'vq_norm_penalty': 0.0,
        }
        
        num_latents = getattr(self.model, '_orig_mod', self.model).lam.vq.num_latents
        all_counts = torch.zeros(num_latents, dtype=torch.long)

        for i, batch in zip(range(self.eval_iters), val_dataloader):
            with torch.amp.autocast(device_type=self.device, dtype=self.mixed_dtype, enabled=self.mixed_dtype != torch.float32):
                loss_dict = self.forward_loss_with_components(batch)

            for key in accumulated_metrics:
                accumulated_metrics[key] += float(loss_dict[key].detach().cpu())
            all_counts += torch.bincount(loss_dict['vq_indices'].detach().cpu().view(-1), minlength=num_latents)

        self.model.train()

        # Compute averages and format output
        num_batches = max(1, self.eval_iters)
        result = {
            'val/loss': accumulated_metrics['total_loss'] / num_batches,
            'val/lam_loss': accumulated_metrics['lam_loss'] / num_batches,
            'val/lam_ce_loss': accumulated_metrics['lam_ce_loss'] / num_batches,
            'val/dynamics_loss': accumulated_metrics['dynamics_loss'] / num_batches,
            'val/vq_loss': accumulated_metrics['vq_loss'] / num_batches,
            'val/vq_q_loss': accumulated_metrics['vq_q_loss'] / num_batches,
            'val/vq_commit_loss': accumulated_metrics['vq_commit_loss'] / num_batches,
            'val/vq_entropy_loss': accumulated_metrics['vq_entropy_loss'] / num_batches,
            'val/vq_entropy': accumulated_metrics['vq_entropy'] / num_batches,
        }

        cb_stats = self._codebook_stats(all_counts)
        for k, v in cb_stats.items():
            result[f"val/{k}"] = v

        if self.logger is not None and wandb is not None:
            probs = all_counts.float() / all_counts.sum()
            table = wandb.Table(
                columns=["action", "probability"],
                data=[[i, p.item()] for i, p in enumerate(probs)],
            )
            result['val/action_histogram'] = wandb.plot.bar(table, "action", "probability", title="Action usage")

        return result