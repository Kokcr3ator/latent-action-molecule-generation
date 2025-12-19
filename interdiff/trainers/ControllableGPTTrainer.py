import torch.nn.functional as F

from .base import TrainerBase

class ControllableGPTTrainer(TrainerBase):
    """Trainer for ControllableGPT model pretraining.
    
    Trains a ControllableGPT model that combines a Latent Action Model (LAM)
    and a Dynamics Model, with vector quantization loss.
    
    Args:
        model: ControllableGPT model to train.
        optimizer: Optimizer for training.
        scheduler: Optional learning rate scheduler.
        logger: Optional logger (e.g., WandB logger).
        train_cfg: Training configuration.
    """
    def __init__(self, model, optimizer, scheduler, logger, train_cfg):
        super().__init__(model, optimizer, scheduler, logger, train_cfg)
        self.reference_smiles = []
    
    def forward_loss(self, batch):
            """Compute combined loss for ControllableGPT.
            
            Computes cross-entropy losses for both the Latent Action Model
            and Dynamics Model, plus the vector quantization loss.
            
            Args:
                batch: Dictionary containing 'x' (input tokens) and 'y' (target tokens).
                
            Returns:
                Total loss combining LAM loss, dynamics loss, and VQ loss.
            """
            x = batch['x']
            y = batch['y']
            lam_logits, dynamics_model_logits, vq_loss = self.model(x)
            lam_loss = F.cross_entropy(
                lam_logits.view(-1, lam_logits.size(-1)),
                y.reshape(-1),
                ignore_index=self.pad_token_id,
                )
            
            dynamics_loss = F.cross_entropy(
                dynamics_model_logits.view(-1, dynamics_model_logits.size(-1)),
                y.reshape(-1),
                ignore_index=self.pad_token_id,
                )
            return lam_loss + dynamics_loss + vq_loss