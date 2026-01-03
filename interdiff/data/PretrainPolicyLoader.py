from __future__ import annotations
from typing import Tuple
import logging
import os

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from interdiff.utils.torch_utils import _load_tensor_from_safetensors
from scripts.generate_actions import run_action_generation

class PolicyPretrainDataset(Dataset):
    """Dataset for policy pretraining with token sequences and latent actions.
    
    Loads tokenized sequences and generates corresponding latent actions from
    a trained ControllableGPT model. Used for training policy models that
    predict actions from sequences.
    
    Args:
        controllable_gpt_path: Path to trained ControllableGPT model checkpoint.
        dataset_path: Path to tokenized dataset SafeTensors file.
        batch_size: Batch size for action generation.
        pad_token_id: Token ID used for padding.
        action_dataset_out_dir: Directory to save/load generated actions.
        
    Raises:
        ValueError: If the number of samples in tokens and actions don't match.
    """
    def __init__(self, 
                 controllable_gpt_path: str, 
                 dataset_path: str, 
                 batch_size: int, 
                 pad_token_id: int, 
                 action_dataset_out_dir: str):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.x = _load_tensor_from_safetensors(dataset_path).to(torch.long)
        self.y = self._ingest_actions(
            controllable_gpt_path=controllable_gpt_path,
            dataset_path=dataset_path,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
            action_dataset_out_dir=action_dataset_out_dir,
        ).to(torch.long)
        
        # check that x and y have compatible shapes
        if self.x.size(0) != self.y.size(0):
            raise ValueError(f"Number of samples in x ({self.x.size(0)}) and y ({self.y.size(0)}) do not match.")

    def __len__(self) -> int:
        """Return the number of sequences in the dataset.
        
        Returns:
            Number of sequences.
        """
        return self.x.size(0)

    def __getitem__(self, idx: int):
        """Get a single training example.
        
        Args:
            idx: Index of the sequence to retrieve.
            
        Returns:
            Dictionary containing:
                - x: Input token sequence (excluding last token)
                - y: Corresponding latent action sequence
        """
        x = self.x[idx][..., :-1]  # exclude last token
        y = self.y[idx]
        return {"x": x.to(device=self.device), "y": y.to(device=self.device)}

    def _ingest_actions(self,
                        controllable_gpt_path: str, 
                        dataset_path: str, 
                        batch_size: int, 
                        pad_token_id: int, 
                        action_dataset_out_dir: str) -> torch.Tensor:
        """Generate latent actions from tokenized sequences.
        
        Generates action dataset using the ControllableGPT model
        
        Args:
            controllable_gpt_path: Path to trained ControllableGPT model.
            dataset_path: Path to tokenized sequences.
            batch_size: Batch size for action generation.
            pad_token_id: Token ID for padding.
            action_dataset_out_dir: Directory to save/load actions.
            
        Returns:
            Tensor of latent action indices.
        """

        actions_path = run_action_generation(
                controllable_gpt_path=controllable_gpt_path,
                dataset_path=dataset_path,
                batch_size=batch_size,
                pad_token_id=pad_token_id,
                out_dir=action_dataset_out_dir,
            )

        return _load_tensor_from_safetensors(actions_path)

def build_dataloaders(
    controllable_gpt_path: str,
    dataset_path: str,
    action_dataset_out_dir: str,
    pad_token_id: int,
    seed: int,
    val_ratio: float = 0.1,
    batch_size: int = 64,
    shuffle_train: bool = True,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Build training and validation DataLoaders for policy pretraining.
    
    Creates train/val split from tokenized sequences and their corresponding
    latent actions extracted from a ControllableGPT model.

    Args:
        controllable_gpt_path: Path to trained ControllableGPT model checkpoint.
        dataset_path: Path to tokenized dataset SafeTensors file.
        action_dataset_out_dir: Directory to save/load generated action dataset.
        pad_token_id: Token ID used for padding.
        seed: RNG seed for deterministic train/val split and shuffling.
        val_ratio: Fraction of data for validation (0 < val_ratio < 1).
        batch_size: Number of samples per batch.
        shuffle_train: Whether to shuffle training batches each epoch.
        drop_last: Whether to drop the last incomplete batch.
        pin_memory: Set True for faster data transfer to CUDA.

    Returns:
        Tuple of (train_loader, val_loader).
        
    Raises:
        ValueError: If val_ratio is not in (0, 1) or if it leaves no training data.
    """

    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1).")

    dataset = PolicyPretrainDataset(
        controllable_gpt_path=controllable_gpt_path,
        dataset_path=dataset_path,
        batch_size=batch_size,
        pad_token_id=pad_token_id,
        action_dataset_out_dir=action_dataset_out_dir,
    )

    n_total = len(dataset)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = n_total - n_val
    if n_train < 1:
        raise ValueError("val_ratio too large; no rows left for training.")

    g = torch.Generator()
    g.manual_seed(int(seed))

    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)

    # For deterministic train shuffling across epochs, pass a generator to DataLoader
    # and re-seed it externally each epoch if you need epoch-dependent determinism.
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

    return train_loader, val_loader

class PretrainPolicyLoader:
    """Data loader wrapper for policy distillation training.
    
    Provides convenient access to training and validation data loaders
    for training policy models that predict latent actions from token sequences.
    
    Args:
        controllable_gpt_path: Path to trained ControllableGPT model checkpoint.
        dataset_path: Path to tokenized dataset file.
        action_dataset_out_dir: Directory to save/load generated actions.
        pad_token_id: Token ID used for padding.
        batch_size: Number of samples per batch.
        seed: Random seed for reproducible data splitting.
        val_ratio: Fraction of data to use for validation (default: 0.1).
        
    Attributes:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
    """
    def __init__(self, controllable_gpt_path: str, dataset_path: str, action_dataset_out_dir: str, pad_token_id: int, batch_size: int, seed: int, val_ratio: float = 0.1):
        self.train_loader, self.val_loader = build_dataloaders(
            controllable_gpt_path=controllable_gpt_path,
            dataset_path=dataset_path,
            action_dataset_out_dir=action_dataset_out_dir,
            pad_token_id=pad_token_id,
            seed=seed,
            val_ratio=val_ratio,
            batch_size=batch_size,
            shuffle_train=True,
            drop_last=True,
            pin_memory=False,
        )