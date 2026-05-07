from typing import List

from .lm_trainer import LanguageModelTrainer
from interdiff.utils.eval_utils import tokens_to_smiles


class GPTTrainer(LanguageModelTrainer):
    """Trainer for base GPT pretraining."""

    def _generate_smiles(self) -> List[str]:
        tokens = self.model.generate(n_mols=self.n_mols_generate)
        return tokens_to_smiles(tokens, tokenizer=self.tokenizer)
