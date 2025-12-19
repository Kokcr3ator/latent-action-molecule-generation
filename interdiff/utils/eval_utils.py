from typing import List

import torch
import torch.nn.functional as F

def tokens_to_smiles(token_ids: torch.Tensor, tokenizer) -> List[str]:
    """Convert a batch of token IDs to their corresponding SMILES strings.
    Args:
        token_ids (torch.Tensor): A tensor of shape (batch_size, seq_len) containing token IDs.
        tokenizer: The tokenizer used to decode the token IDs.
    Returns:
        List[str]: A list of decoded SMILES strings for each sequence in the batch.
    """
    if not isinstance(token_ids, torch.Tensor):
        raise ValueError("token_ids must be a torch.Tensor")

    # add batch dimension if missing
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)
    
    smiles = []
    for ids in token_ids:
        smile = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
        # Remove spaces that have been added during tokenizer decoding
        smiles.append(smile.replace(" ", ""))
    return smiles

def sample_from_logits(tensor_logits: torch.Tensor, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
    if temperature <= 0:
        # greedy
        filtered = tensor_logits
    else:
        filtered = tensor_logits / temperature

    if top_k is not None:
        k = min(top_k, filtered.size(-1))
        v, _ = torch.topk(filtered, k, dim=-1)
        cutoff = v[:, [-1]]
        filtered = torch.where(filtered < cutoff, torch.full_like(filtered, float('-inf')), filtered)

    if temperature <= 0:
        pred = torch.argmax(filtered, dim=-1, keepdim=True)            # (B,1)
    else:
        probs = F.softmax(filtered, dim=-1)
        pred = torch.multinomial(probs, num_samples=1)                 # (B,1)
    
    return pred