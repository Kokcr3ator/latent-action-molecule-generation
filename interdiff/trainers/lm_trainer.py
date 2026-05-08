"""Shared base for autoregressive language-model trainers (GPT and PolicyNetwork)."""
from __future__ import annotations

from abc import abstractmethod
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F

from .base import TrainerBase
from interdiff.tokenise import tokens_to_smiles
from interdiff.metrics import (all_property_satisfaction_rates,
                                validity, uniqueness, novelty, vun)


class LanguageModelTrainer(TrainerBase):
    """TrainerBase subclass for token-prediction models.

    Subclasses only need to implement `_generate_smiles()`.
    """

    def __init__(self, model, optimizer, scheduler, logger, train_cfg):
        super().__init__(model, optimizer, scheduler, logger, train_cfg)
        self.reference_smiles: List[str] = []

    @abstractmethod
    def _generate_smiles(self) -> List[str]:
        """Generate a batch of SMILES strings from the model."""
        ...

    def forward_loss(self, batch) -> torch.Tensor:
        x = batch['x']
        y = batch['y']
        logits, _ = self.model(x)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=self.pad_token_id
        )

    @torch.no_grad()
    def evaluate(self, val_dataloader: Iterable, train_dataloader: Iterable) -> Dict[str, float]:
        self.model.eval()
        val_losses = []
        for _, batch in zip(range(self.eval_iters), val_dataloader):
            with torch.amp.autocast(device_type=self.device, dtype=self.mixed_dtype,
                                    enabled=self.mixed_dtype != torch.float32):
                val_losses.append(float(self.forward_loss(batch).float().detach().cpu()))

        generated_smiles = self._generate_smiles()

        if len(self.reference_smiles) == 0:
            for train_batch in train_dataloader:
                self.reference_smiles.extend(
                    tokens_to_smiles(train_batch['x'], tokenizer=self.tokenizer)
                )

        self.model.train()

        pct_metrics = all_property_satisfaction_rates(generated_smiles)
        return {
            'val/loss': sum(val_losses) / max(1, len(val_losses)),
            **{f'eval/{k}': v for k, v in pct_metrics.items()},
            'eval/validity': validity(generated_smiles),
            'eval/uniqueness': uniqueness(generated_smiles),
            'eval/novelty': novelty(generated_smiles, reference_smiles=self.reference_smiles),
            'eval/vun': vun(generated_smiles, reference_smiles=self.reference_smiles),
        }
