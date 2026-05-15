from typing import Dict

import torch
import torch.nn.functional as F

from .ControllableGPTTrainer import ControllableGPTTrainer


class LAMTrainer(ControllableGPTTrainer):
    """Stage-1 trainer: trains the LAM (encoder + VQ + decoder) only.

    The dynamics model exists in the ControllableGPT graph but receives no
    gradients — its parameters are never updated here.  Saves a full
    ControllableGPT checkpoint so stage-2 (DynamicsTrainer) can load it.
    """

    def forward_loss_with_components(self, batch) -> Dict[str, torch.Tensor]:
        x, y = batch['x'], batch['y']
        lam_logits, _, vq_loss_dict = self.model.lam(x)

        lam_ce_loss = F.cross_entropy(
            lam_logits.view(-1, lam_logits.size(-1)),
            y.reshape(-1),
            ignore_index=self.pad_token_id,
        )
        vq_loss = vq_loss_dict['vq_loss']
        lam_loss = lam_ce_loss + vq_loss

        zero = torch.zeros(1, device=x.device)
        return {
            'total_loss': lam_loss,
            'lam_loss': lam_loss,
            'lam_ce_loss': lam_ce_loss,
            'dynamics_loss': zero,
            'vq_loss': vq_loss,
            'vq_q_loss': vq_loss_dict['q_loss'],
            'vq_commit_loss': vq_loss_dict['commit_loss'],
            'vq_entropy_loss': vq_loss_dict['entropy_loss'],
            'vq_entropy': vq_loss_dict['entropy'],
            'vq_norm_penalty': vq_loss_dict['vq_norm_penalty'],
            'vq_indices': vq_loss_dict['indices'],
        }
