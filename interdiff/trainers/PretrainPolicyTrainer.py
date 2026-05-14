from typing import Dict, Iterable, List
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
from interdiff.models import ControllableGPT
from interdiff.tokenise import tokens_to_smiles

_LABEL_SMOOTHING = 0.0


class PretrainPolicyTrainer(TrainerBase):
    """Trainer for policy distillation (PolicyNetwork supervised on LAM actions)."""

    def __init__(self, model, optimizer, scheduler, logger, train_cfg, controllable_gpt_path: str):
        super().__init__(model, optimizer, scheduler, logger, train_cfg)
        controllable_gpt = ControllableGPT.load(controllable_gpt_path).to(self.device)
        self.lam = controllable_gpt.lam
        self.dm = controllable_gpt.dynamics_model

    def _generate_smiles(self) -> List[str]:
        tokens = self.model.generate(
            dynamics_model=self.dm, lam=self.lam, n_mols=self.n_mols_generate
        )
        return tokens_to_smiles(tokens, tokenizer=self.tokenizer)

    def forward_loss(self, batch) -> torch.Tensor:
        x, y = batch['x'], batch['y']
        logits, _ = self.model(x)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=self.pad_token_id,
            label_smoothing=_LABEL_SMOOTHING,
        )

    def _codebook_stats(self, counts: torch.Tensor) -> Dict[str, float]:
        num_latents = self.lam.vq.num_latents
        norms = self.lam.vq.codebook.data.float().norm(dim=-1)
        utilization = (counts > 0).sum().item() / num_latents
        return {
            'codebook_norm_mean': norms.mean().item(),
            'codebook_norm_max': norms.max().item(),
            'codebook_norm_std': norms.std().item(),
            'codebook_utilization': utilization,
        }

    def fit(self, train_dataloader: Iterable, val_dataloader: Iterable):
        self.model.train()
        dl = itertools.cycle(train_dataloader)
        os.makedirs(self.ckpt_path, exist_ok=True)

        num_latents = self.lam.vq.num_latents
        running_loss = 0.0
        running_counts = torch.zeros(num_latents, dtype=torch.long)

        while self.state.step < self.max_iters:
            batch = next(dl)
            with torch.amp.autocast(device_type=self.device, dtype=self.mixed_dtype,
                                    enabled=self.mixed_dtype != torch.float32):
                x, y = batch['x'], batch['y']
                logits, _ = self.model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=self.pad_token_id,
                    label_smoothing=_LABEL_SMOOTHING,
                ).float() / self.gradient_accumulation_steps

            self.scaler.scale(loss).backward()
            running_loss += float(loss.detach().cpu())

            pred_indices = logits.detach().argmax(-1).cpu().view(-1)
            running_counts += torch.bincount(pred_indices, minlength=num_latents)

            if (self.state.step + 1) % self.gradient_accumulation_steps == 0:
                if self.grad_clip and self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)

                if self.state.step < self.warmup_iters:
                    scale = self.get_lr_scale()
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.base_lr * scale

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()

            log_dict = {}

            if self.state.step % self.log_interval == 0:
                log_dict["train/loss"] = running_loss / max(1, self.log_interval)
                log_dict["train/lr"] = self.optimizer.param_groups[0]["lr"]
                cb_stats = self._codebook_stats(running_counts)
                for k, v in cb_stats.items():
                    log_dict[f"train/{k}"] = v
                running_loss = 0.0
                running_counts.zero_()
                print(
                    f"step {self.state.step:6d} | "
                    + " | ".join(f"{k.split('/')[-1]}={v:.4g}" for k, v in log_dict.items()
                                 if k.startswith("train/"))
                )

            if self.state.step % self.eval_interval == 0:
                val_dict = self.evaluate(val_dataloader=val_dataloader, train_dataloader=train_dataloader)
                val = val_dict.get('val/loss', float('inf'))
                log_dict.update(val_dict)
                improved = val < self.state.best_val
                self.state.best_val = min(self.state.best_val, val)
                if self.always_save_checkpoint or improved:
                    self.save_checkpoint(os.path.join(self.ckpt_path, "best.pt"))

            if log_dict and self.logger:
                self.logger.log({"step": self.state.step, **log_dict})

            self.state.step += 1

    @torch.no_grad()
    def evaluate(self, val_dataloader: Iterable, train_dataloader: Iterable) -> Dict[str, float]:
        self.model.eval()

        num_latents = self.lam.vq.num_latents
        val_losses = []
        all_counts = torch.zeros(num_latents, dtype=torch.long)

        for _, batch in zip(range(self.eval_iters), val_dataloader):
            with torch.amp.autocast(device_type=self.device, dtype=self.mixed_dtype,
                                    enabled=self.mixed_dtype != torch.float32):
                x, y = batch['x'], batch['y']
                logits, _ = self.model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=self.pad_token_id,
                    label_smoothing=_LABEL_SMOOTHING,
                )
            val_losses.append(float(loss.float().detach().cpu()))
            pred_indices = logits.detach().argmax(-1).cpu().view(-1)
            all_counts += torch.bincount(pred_indices, minlength=num_latents)

        self.model.train()

        result = {'val/loss': sum(val_losses) / max(1, len(val_losses))}
        cb_stats = self._codebook_stats(all_counts)
        for k, v in cb_stats.items():
            result[f"val/{k}"] = v

        if self.logger is not None and wandb is not None:
            probs = all_counts.float() / all_counts.sum().clamp(min=1)
            table = wandb.Table(
                columns=["action", "probability"],
                data=[[i, p.item()] for i, p in enumerate(probs)],
            )
            result['val/action_histogram'] = wandb.plot.bar(
                table, "action", "probability", title="Policy predicted action usage"
            )

        return result
