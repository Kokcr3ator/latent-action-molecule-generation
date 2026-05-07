from typing import Dict, Iterable

import torch
import torch.nn.functional as F

from .base import TrainerBase


class GPTTrainer(TrainerBase):
    """Trainer class for base GPT model pretraining."""

    def __init__(self, model, optimizer, scheduler, logger, train_cfg):
        super().__init__(model, optimizer, scheduler, logger, train_cfg)

    def forward_loss(self, batch):
        x = batch['x']
        y = batch['y']
        logits, _ = self.model(x)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=self.pad_token_id
        )
