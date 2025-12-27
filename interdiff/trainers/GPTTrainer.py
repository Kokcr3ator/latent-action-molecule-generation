from typing import Dict, Iterable

import torch
import torch.nn.functional as F
import numpy as np

from .base import TrainerBase
from interdiff.utils.eval_utils import tokens_to_smiles
from interdiff.metrics import ( qed, synthetic_accessibility,\
                                logp, molecular_weight, tpsa,\
                                validity, uniqueness, novelty)

class GPTTrainer(TrainerBase):
    """
    Trainer class for base GPT model pretraining.
    """
    def __init__(self, model, optimizer, scheduler, logger, train_cfg):
        super().__init__(model, optimizer, scheduler, logger, train_cfg)
        self.reference_smiles = []
    
    @torch.no_grad()
    def evaluate(self, val_dataloader: Iterable, train_dataloader: Iterable) -> Dict[str, float]:
        self.model.eval()
        val_losses = []
        for i, batch in zip(range(self.eval_iters), val_dataloader):
            with torch.amp.autocast(device_type = self.device, dtype=self.mixed_dtype, enabled=self.mixed_dtype != torch.float32):
                loss = self.forward_loss(batch).float()
                generated_tokens = self.model.generate(n_mols = self.n_mols_generate)
                generated_smiles = tokens_to_smiles(generated_tokens, tokenizer=self.tokenizer)

                qed_scores = [qed(smi) for smi in generated_smiles]
                qed_score = np.mean(qed_scores) if len(qed_scores) > 0 else 0.0

                sa_scores = [synthetic_accessibility(smi) for smi in generated_smiles]
                sa = np.mean(sa_scores) if len(sa_scores) > 0 else 0

                logp_scores_list = [logp(smi) for smi in generated_smiles]
                logp_scores = np.mean(logp_scores_list) if len(logp_scores_list) > 0 else 0

                mw_list = [molecular_weight(smi) for smi in generated_smiles]
                mw = np.mean(mw_list) if len(mw_list) > 0 else 0

                tpsa_list = [tpsa(smi) for smi in generated_smiles]
                tpsa_scores = np.mean(tpsa_list) if len(tpsa_list) > 0 else 0

                valid = validity(generated_smiles)
                unique = uniqueness(generated_smiles)
                # novelty requires a reference set and we build it from training data on first eval
                if len(self.reference_smiles) == 0:
                    for train_batch in train_dataloader:
                        batch_smiles = tokens_to_smiles(train_batch['x'], tokenizer=self.tokenizer)
                        self.reference_smiles.extend(batch_smiles)
                novel = novelty(generated_smiles, reference_smiles=self.reference_smiles)

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
        
        return {'val_loss': sum(val_losses) / max(1, len(val_losses)), 
                'qed': qed_score,
                'sa': sa,
                'logp': logp_scores,
                'mw': mw,
                'tpsa': tpsa_scores,
                'validity': valid,
                'uniqueness': unique,
                'novelty': novel}
    
    def forward_loss(self, batch):
            x = batch['x']
            y = batch['y']
            logits, _ = self.model(x)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=self.pad_token_id
            )
            
            return loss