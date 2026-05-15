from typing import Dict

import torch
import torch.nn.functional as F

from interdiff.models import ControllableGPT
from .ControllableGPTTrainer import ControllableGPTTrainer


class DynamicsTrainer(ControllableGPTTrainer):
    """Stage-2 trainer: loads a frozen LAM from a stage-1 checkpoint, then
    trains only the dynamics model.

    The LAM weights (encoder, VQ, codebook, decoder) are copied from
    lam_checkpoint_path and frozen for the entire run.  The optimizer
    therefore contains only dynamics_model parameters.
    """

    def __init__(self, model, optimizer, scheduler, logger, train_cfg,
                 lam_checkpoint_path: str):
        super().__init__(model, optimizer, scheduler, logger, train_cfg)

        lam_ckpt = ControllableGPT.load(lam_checkpoint_path, device=self.device)
        # Copy LAM weights into the model and freeze them
        self.model.lam.load_state_dict(lam_ckpt.lam.state_dict())
        self.model.lam.requires_grad_(False)

    def forward_loss_with_components(self, batch) -> Dict[str, torch.Tensor]:
        x, y = batch['x'], batch['y']

        with torch.no_grad():
            _, actions, vq_loss_dict = self.model.lam(x)

        dynamics_logits = self.model.dynamics_model(x[..., :-1], actions)
        dynamics_loss = F.cross_entropy(
            dynamics_logits.view(-1, dynamics_logits.size(-1)),
            y.reshape(-1),
            ignore_index=self.pad_token_id,
        )

        zero = torch.zeros(1, device=x.device)
        return {
            'total_loss': dynamics_loss,
            'lam_loss': zero,
            'lam_ce_loss': zero,
            'dynamics_loss': dynamics_loss,
            'vq_loss': zero,
            'vq_q_loss': zero,
            'vq_commit_loss': zero,
            'vq_entropy_loss': zero,
            'vq_entropy': vq_loss_dict['entropy'],
            'vq_norm_penalty': zero,
            'vq_indices': vq_loss_dict['indices'],
        }
