# this script is used for pretraining a policy from the actions that are generated from the latent action model
# in particular tokenise_dataset.py script is used to tokenise the dataset which will repretsent the input of the policy model.
# the policy model is trained to predict the action generated from the latent action model given the tokenised input

import logging

import torch
from tqdm import tqdm

from interdiff.models import ControllableGPT
from interdiff.io import _load_tensor_from_safetensors


def run_action_generation(controllable_gpt_path: str, dataset_path: str, batch_size: int, pad_token_id: int) -> torch.Tensor:
    """
    Generate actions using the latent action model inside ControllableGPT and return them as a tensor.

    Args:
        controllable_gpt_path (str): Path to the pretrained ControllableGPT model.
        dataset_path (str): Path to the tokenized dataset.
        batch_size (int): Batch size for processing the dataset.
        pad_token_id (int): Padding token ID used in the model.

    Returns:
        Tensor of latent action indices, shape (N, T-1), dtype torch.long.
    """
    log = logging.getLogger("generate_actions")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Loading ControllableGPT model from {controllable_gpt_path}")
    controllable_gpt = ControllableGPT.load(controllable_gpt_path).to(device)
    log.info(f"Model has {controllable_gpt.num_latents} latent actions and vocab size {controllable_gpt.vocab_size}")

    log.info(f"Loading tokenized dataset from {dataset_path}")
    tokenized_dataset = _load_tensor_from_safetensors(dataset_path)

    lam = controllable_gpt.lam

    log.info("Generating actions...")
    action_chunks = []
    with torch.no_grad():
        for i in tqdm(range(0, len(tokenized_dataset), batch_size)):
            batch = tokenized_dataset[i : i + batch_size].to(torch.long).to(device)
            pad_mask = (batch[..., :-1] == pad_token_id)
            _, _, action_idxs = lam.vq_encode(batch)
            action_idxs[pad_mask] = pad_token_id
            action_chunks.append(action_idxs.cpu())

    return torch.cat(action_chunks, dim=0).to(torch.long)
