from typing import Dict, Iterable

import torch
import torch.nn.functional as F

from .base import TrainerBase
from interdiff.models import ControllableGPT
from interdiff.utils.eval_utils import tokens_to_smiles
from interdiff.metrics import (all_property_satisfaction_rates,
                                validity, uniqueness, novelty, vun)


class PretrainPolicyTrainer(TrainerBase):

    def __init__(self, model, optimizer, scheduler, logger, train_cfg, controllable_gpt_path: str):
        super().__init__(model, optimizer, scheduler, logger, train_cfg)

        self.controllable_gpt_path = controllable_gpt_path
        controllable_gpt = ControllableGPT.load(controllable_gpt_path).to(self.device)
        self.lam = controllable_gpt.lam
        self.dm = controllable_gpt.dynamics_model
        self.reference_smiles = []

    @torch.no_grad()
    def evaluate(self, val_dataloader: Iterable, train_dataloader: Iterable) -> Dict[str, float]:
        self.model.eval()
        val_losses = []
        for i, batch in zip(range(self.eval_iters), val_dataloader):
            with torch.amp.autocast(device_type=self.device, dtype=self.mixed_dtype, enabled=self.mixed_dtype != torch.float32):
                loss = self.forward_loss(batch).float()
            val_losses.append(float(loss.detach().cpu()))

        generated_tokens = self.model.generate(
            dynamics_model=self.dm, lam=self.lam, n_mols=self.n_mols_generate
        )
        generated_smiles = tokens_to_smiles(generated_tokens, tokenizer=self.tokenizer)

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

    def forward_loss(self, batch):
        x = batch['x']
        y = batch['y']
        logits, _ = self.model(x)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=self.pad_token_id
        )
