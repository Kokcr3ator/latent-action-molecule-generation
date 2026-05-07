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
        """
        Evaluate the model on the validation set and calculate molecular metrics.

        Args:
            val_dataloader (Iterable): DataLoader for the validation set.
            train_dataloader (Iterable): DataLoader for the training set (used to build reference set for novelty).
        
        Returns:
            Dict[str, float]: Dictionary containing validation loss, 
            property satisfaction rates, validity, uniqueness, novelty, and VUN score.
        """
        self.model.eval()
        val_losses = []
        for i, batch in zip(range(self.eval_iters), val_dataloader):
            with torch.amp.autocast(device_type = self.device, dtype=self.mixed_dtype, enabled=self.mixed_dtype != torch.float32):
                loss = self.forward_loss(batch).float()
                generated_tokens = self.model.generate(dynamics_model = self.dm,
                                                        lam = self.lam, 
                                                        n_mols = self.n_mols_generate)
                generated_smiles = tokens_to_smiles(generated_tokens, tokenizer=self.tokenizer)

                pct_metrics = all_property_satisfaction_rates(generated_smiles)

                valid = validity(generated_smiles)
                unique = uniqueness(generated_smiles)
                # novelty requires a reference set and we build it from training data on first eval
                if len(self.reference_smiles) == 0:
                    for train_batch in train_dataloader:
                        batch_smiles = tokens_to_smiles(train_batch['x'], tokenizer=self.tokenizer)
                        self.reference_smiles.extend(batch_smiles)
                novel = novelty(generated_smiles, reference_smiles=self.reference_smiles)
                vun = vun(generated_smiles, reference_smiles=self.reference_smiles)

            val_losses.append(float(loss.detach().cpu()))
        self.model.train()
        
        # Log generated SMILES to wandb table if logger supports it
        if self.logger and hasattr(self.logger, 'log_table'):
            smiles_data = [[smi] for smi in generated_smiles]
            self.logger.log_table(
                table_name="generated_molecules",
                columns=["SMILES"],
                data=smiles_data,
                step=self.state.step if hasattr(self.state, 'step') else None
            )
        
        return {'val/loss': sum(val_losses) / max(1, len(val_losses)),
                **{f'val/{k}': v for k, v in pct_metrics.items()},
                'val/validity': valid,
                'val/uniqueness': unique,
                'val/novelty': novel,
                'val/vun': vun,
                }
    
    def forward_loss(self, batch):
            x = batch['x']
            y = batch['y']
            logits, _ = self.model(x)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=self.pad_token_id
            )
            
            return loss


        