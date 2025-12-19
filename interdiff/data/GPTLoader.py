from __future__ import annotations
from typing import Tuple
import logging

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from interdiff.utils import _load_tensor_from_safetensors

class NextTokenDataset(Dataset):
    """Dataset for next-token prediction training.
    
    Provides token sequences for autoregressive language modeling where
    each target is the input shifted left by one position.
    
    Args:
        tokens_2d: 2D tensor of shape [n_rows, context_len] containing token IDs.
        pad_token_id: Token ID used for padding the last position of targets.
        
    Returns:
        Dictionary with keys:
            - x: LongTensor [context_len] - input sequence
            - y: LongTensor [context_len] - target sequence (x shifted left, 
                 last position filled with pad_token_id)
    """
    def __init__(self, tokens_2d: torch.Tensor, pad_token_id: int):
        if tokens_2d.dtype != torch.long:
            tokens_2d = tokens_2d.to(torch.long)
        self.x = tokens_2d
        self.pad_id = int(pad_token_id)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
                - x: Input token sequence
                - y: Target token sequence (shifted and padded)
        """
        row = self.x[idx]
        # Build y by shifting left and padding last position
        y = torch.empty_like(row)
        y[:-1] = row[1:]
        y[-1] = self.pad_id
        return {"x": row.to(device=self.device), "y": y.to(device=self.device)}

def build_dataloaders(
    path: str,
    pad_token_id: int,
    seed: int,
    val_ratio: float = 0.1,
    batch_size: int = 64,
    shuffle_train: bool = True,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Build training and validation DataLoaders for GPT training.
    
    Creates train/val split from a SafeTensors file containing tokenized
    sequences and returns DataLoaders for both splits.

    Args:
        path: Path to .safetensors file containing [n_rows, context_len] tensor.
        pad_token_id: Token ID used as padding for the last position in targets.
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

    Note:
        - The split is deterministic given the seed.
        - Dataset keeps data in memory; large tensors may require significant RAM.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1).")

    full = _load_tensor_from_safetensors(path)
    dataset = NextTokenDataset(full, pad_token_id)

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

class GPTLoader:
    """Data loader wrapper for standard GPT training.
    
    Provides convenient access to training and validation data loaders
    for autoregressive language model training.
    
    Args:
        dataset_path: Path to the tokenized dataset file.
        pad_token_id: Token ID used for padding.
        batch_size: Number of samples per batch.
        seed: Random seed for reproducible data splitting.
        val_ratio: Fraction of data to use for validation (default: 0.1).
        
    Attributes:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
    """
    def __init__(self, dataset_path: str, pad_token_id: int, batch_size: int, seed: int, val_ratio: float = 0.1):
        self.train_loader, self.val_loader = build_dataloaders(
            path=dataset_path,
            pad_token_id=pad_token_id,
            seed=seed,
            val_ratio=val_ratio,
            batch_size=batch_size,
            shuffle_train=True,
            drop_last=True,
            pin_memory=False,
        )