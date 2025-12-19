from __future__ import annotations
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Iterable
import itertools
from omegaconf import MISSING

import torch
from torch.nn.utils import clip_grad_norm_

from interdiff.io import load_tokenizer

@dataclass
class TrainConfig:
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    warmup_iters: int = 100
    batch_size: int = 4096
    max_iters: int = 10_000_000
    gradient_accumulation_steps: int = 1
    log_interval: int = 40
    ckpt_path: str = MISSING
    compile_model: bool = True
    always_save_checkpoint: bool = False
    eval_interval: int = 250
    eval_iters: int = 200
    mixed_dtype: str = 'auto' # float32, float16, bfloat16
    n_mols_generate: int = 1000
    pad_token_id: int = 0
    tokenizer_dir: str = MISSING

@dataclass
class TrainState:
    step: int = 1
    best_val: float = float("inf")

class TrainerBase(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        logger: Optional[Any],
        train_cfg: TrainConfig
    ) -> None:
        self.model = model.to(train_cfg.device)
        if train_cfg.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[attr-defined]
        self.optimizer = optimizer
        self.base_lr = self.optimizer.param_groups[0]["lr"]

        self.scheduler = scheduler

        self.logger = logger
        self.grad_clip = train_cfg.grad_clip
        self.device = train_cfg.device
        self.mixed_dtype = self._resolve_dtype(train_cfg.mixed_dtype)
        self.scaler = torch.amp.GradScaler(device = self.device, enabled=self.mixed_dtype != torch.float32)
        self.eval_interval = train_cfg.eval_interval
        self.eval_iters = train_cfg.eval_iters
        self.log_interval = train_cfg.log_interval
        self.max_iters = train_cfg.max_iters
        self.gradient_accumulation_steps = train_cfg.gradient_accumulation_steps
        self.always_save_checkpoint = train_cfg.always_save_checkpoint
        self.warmup_iters = train_cfg.warmup_iters
        self.ckpt_path = os.path.join('ckpts', train_cfg.ckpt_path)

        # some models may not need this parameter in case set to 0
        try:
            self.n_mols_generate = train_cfg.n_mols_generate
            self.pad_token_id = train_cfg.pad_token_id
        except AttributeError:
            self.n_mols_generate = 0
            self.pad_token_id = 0

        self.tokenizer = load_tokenizer(os.path.join(train_cfg.tokenizer_dir, "tokenizer.json"))
        self.state = TrainState()

    def _resolve_dtype(self, d: str):
        if d == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[d]

    def get_lr_scale(self) -> float:
        """Linear warmup scaling factor for the learning rate."""
        if self.state.step < self.warmup_iters:
            return float(self.state.step + 1) / float(max(1, self.warmup_iters))
        return 1.0

    @abstractmethod
    def forward_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return scalar loss tensor given a batch."""
        ...

    def fit(self, train_dataloader: Iterable, val_dataloader: Iterable):
        self.model.train()
        dl = itertools.cycle(train_dataloader)
        os.makedirs(self.ckpt_path, exist_ok= True)

        running = 0.0
        while self.state.step < self.max_iters:
            with torch.amp.autocast(device_type = self.device, dtype=self.mixed_dtype, enabled=self.mixed_dtype != torch.float32):
                loss = self.forward_loss(next(dl)).float() / self.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
            running += float(loss.detach().cpu())

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
                self.logger.log({"train/loss": running / max(1, self.log_interval), "step": self.state.step})
                running = 0.0

            elif self.state.step % self.log_interval == 0:
                print(f"step {self.state.step}: train loss {running / max(1, self.log_interval)}")
                running = 0.0

            if self.state.step % self.eval_interval == 0:
                # train_dataloader is also passed for building reference set for novelty metric if needed
                val_dict = self.evaluate(val_dataloader = val_dataloader, train_dataloader =train_dataloader)
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
        self.model.eval()
        losses = []
        for i, batch in zip(range(self.eval_iters), val_dataloader):
            with torch.amp.autocast(device_type = self.device, dtype=self.mixed_dtype, enabled=self.mixed_dtype != torch.float32):
                loss = self.forward_loss(batch).float()
            losses.append(float(loss.detach().cpu()))
        self.model.train()
        return {'val_loss': sum(losses) / max(1, len(losses))}

    def save_checkpoint(self, path: str):
        self.model.save(path)

    def load_checkpoint(self, path: str):
        self.model.load(path)