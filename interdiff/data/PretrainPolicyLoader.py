from __future__ import annotations
from typing import Tuple
import logging

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from interdiff.io import _load_tensor_from_safetensors
from interdiff.dataset_entropy import empirical_conditional_entropy
from scripts.generate_actions import run_action_generation

log = logging.getLogger(__name__)


class PolicyPretrainDataset(Dataset):
    """Dataset for policy pretraining with token sequences and latent actions."""

    def __init__(self,
                 controllable_gpt_path: str,
                 dataset_path: str,
                 batch_size: int,
                 pad_token_id: int):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.x = _load_tensor_from_safetensors(dataset_path).to(torch.long)
        self.y = run_action_generation(
            controllable_gpt_path=controllable_gpt_path,
            dataset_path=dataset_path,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
        )

        if self.x.size(0) != self.y.size(0):
            raise ValueError(f"Number of samples in x ({self.x.size(0)}) and y ({self.y.size(0)}) do not match.")

        log.info("Computing empirical conditional entropy H(A | context)...")
        num_latents = int(self.y.max().item()) + 1
        self.conditional_entropy = empirical_conditional_entropy(
            tokens=self.x,
            actions=self.y,
            num_latents=num_latents,
            pad_token_id=pad_token_id,
        )
        log.info(f"H(A | context) = {self.conditional_entropy:.4f} nats  "
                 f"(max possible: {torch.tensor(num_latents).float().log().item():.4f})")

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        x = self.x[idx][..., :-1]
        y = self.y[idx]
        return {"x": x.to(device=self.device), "y": y.to(device=self.device)}


def build_dataloaders(
    controllable_gpt_path: str,
    dataset_path: str,
    pad_token_id: int,
    seed: int,
    val_ratio: float = 0.1,
    batch_size: int = 64,
    shuffle_train: bool = True,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, float]:
    """Build training and validation DataLoaders for policy pretraining.

    Returns:
        (train_loader, val_loader, conditional_entropy)
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1).")

    dataset = PolicyPretrainDataset(
        controllable_gpt_path=controllable_gpt_path,
        dataset_path=dataset_path,
        batch_size=batch_size,
        pad_token_id=pad_token_id,
    )
    conditional_entropy = dataset.conditional_entropy

    n_total = len(dataset)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = n_total - n_val
    if n_train < 1:
        raise ValueError("val_ratio too large; no rows left for training.")

    g = torch.Generator()
    g.manual_seed(int(seed))

    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, conditional_entropy


class PretrainPolicyLoader:
    """Data loader wrapper for policy distillation training."""

    def __init__(self, controllable_gpt_path: str, dataset_path: str, pad_token_id: int, batch_size: int, seed: int, val_ratio: float = 0.1):
        self.train_loader, self.val_loader, self.conditional_entropy = build_dataloaders(
            controllable_gpt_path=controllable_gpt_path,
            dataset_path=dataset_path,
            pad_token_id=pad_token_id,
            seed=seed,
            val_ratio=val_ratio,
            batch_size=batch_size,
            shuffle_train=True,
            drop_last=True,
            pin_memory=False,
        )
