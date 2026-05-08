from typing import List

from .lm_trainer import LanguageModelTrainer
from interdiff.models import ControllableGPT
from interdiff.tokenise import tokens_to_smiles


class PretrainPolicyTrainer(LanguageModelTrainer):
    """Trainer for policy distillation (PolicyNetwork supervised on LAM actions)."""

    def __init__(self, model, optimizer, scheduler, logger, train_cfg, controllable_gpt_path: str):
        super().__init__(model, optimizer, scheduler, logger, train_cfg)
        controllable_gpt = ControllableGPT.load(controllable_gpt_path).to(self.device)
        self.lam = controllable_gpt.lam
        self.dm = controllable_gpt.dynamics_model

    def _generate_smiles(self) -> List[str]:
        tokens = self.model.generate(
            dynamics_model=self.dm, lam=self.lam, n_mols=self.n_mols_generate
        )
        return tokens_to_smiles(tokens, tokenizer=self.tokenizer)
