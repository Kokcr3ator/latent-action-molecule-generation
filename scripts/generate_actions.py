# this script is used for pretraining a policy from the actions that are generated from the latent action model
# in particular tokenise_dataset.py script is used to tokenise the dataset which will repretsent the input of the policy model.
# the policy model is trained to predict the action generated from the latent action model given the tokenised input

import os
import logging

import torch
from tqdm import tqdm
import safetensors.torch

from interdiff.models import ControllableGPT
from interdiff.utils import _load_tensor_from_safetensors


def run_action_generation(controllable_gpt_path: str, dataset_path: str, batch_size: int, pad_token_id: int, out_dir: str) -> str:
    """
    Generate action using the latent action model inside ControllableGPT model and save them to out_dir.

    Args:
        controllable_gpt_path (str): Path to the pretrained ControllableGPT model.
        pad_token_id (int): Padding token ID used in the model.
        out_dir (str): Directory to save the generated actions.
        dataset_path (str): Path to the tokenized dataset.
        batch_size (int): Batch size for processing the dataset.
    
    Returns:
        actions_path (str): Path to the saved actions dataset.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("generate_actions")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Loading ControllableGPT model from {controllable_gpt_path}")

    controllable_gpt = ControllableGPT.load(controllable_gpt_path).to(device)
    n_latent_actions = controllable_gpt.num_latents
    vocab_size = controllable_gpt.vocab_size
    log.info(f"Model has {n_latent_actions} latent actions and vocab size {vocab_size}")

    log.info(f"Loading tokenized dataset from {dataset_path}")
    tokenized_dataset = _load_tensor_from_safetensors(dataset_path)

    lam = controllable_gpt.lam
    latent_action_dim = lam.latent_action_dim

    log.info("Generating actions...")

    action_dataset = []
    with torch.no_grad():
        for i in tqdm(range(0, len(tokenized_dataset), batch_size)):
            batch = tokenized_dataset[i : i + batch_size].to(torch.long).to(device)
            pad_mask = (batch[...,:-1] == pad_token_id)
            _, _, action_idxs = lam.vq_encode(batch)
            action_idxs[pad_mask] = pad_token_id
            action_dataset.append(action_idxs)

    if latent_action_dim <= 256:
        dtype = torch.uint8
    elif latent_action_dim <= 32767:
        dtype = torch.int16
    else:
        dtype = torch.int32

    action_dataset = torch.cat(action_dataset, dim=0).to(dtype)
    os.makedirs(out_dir, exist_ok=True)
    actions_path = os.path.join(out_dir, f"actions_dataset_num_latent_actions_{n_latent_actions}_vocab_size_{vocab_size}.safetensors")
    safetensors.torch.save_file({"data": action_dataset},
        actions_path,
    )

    return actions_path 



