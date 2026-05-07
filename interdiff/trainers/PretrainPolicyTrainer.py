from typing import Dict, Iterable

import torch
import torch.nn.functional as F

from .base import TrainerBase
from interdiff.models import ControllableGPT


class PretrainPolicyTrainer(TrainerBase):

    def __init__(self, model, optimizer, scheduler, logger, train_cfg, controllable_gpt_path: str):
        super().__init__(model, optimizer, scheduler, logger, train_cfg)

        self.controllable_gpt_path = controllable_gpt_path
        controllable_gpt = ControllableGPT.load(controllable_gpt_path).to(self.device)
        self.lam = controllable_gpt.lam
        self.dm = controllable_gpt.dynamics_model

    def forward_loss(self, batch):
        x = batch['x']
        y = batch['y']
        logits, _ = self.model(x)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=self.pad_token_id
        )
