from __future__ import annotations
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from safetensors import safe_open

class ControllableGPTDataset(Dataset):
    """
    Expects a 2D LongTensor [n_rows, context_len].
    __getitem__ returns:
      {
        "x": LongTensor [context_len],
        "y": LongTensor [context_len] (x shifted left) with length context_len -1
      }
    """
    def __init__(self, tokens_2d: torch.Tensor):
        if tokens_2d.dtype != torch.long:
            tokens_2d = tokens_2d.to(torch.long)
        self.x = tokens_2d
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        row = self.x[idx]
        # Build y by shifting left
        y = torch.empty_like(row)
        y[:-1] = row[1:]
        y = row[1:]
        return {"x": row.to(device=self.device), "y": y.to(device=self.device)}


def _load_tensor_from_safetensors(path: str) -> torch.Tensor:
    """
    Loads a single tensor from a .safetensors file as a torch tensor.
    """
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        return f.get_tensor(keys[0])


def build_dataloaders(
    path: str,
    seed: int,
    val_ratio: float = 0.1,
    batch_size: int = 64,
    shuffle_train: bool = True,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader).

    Args:
      path: path to .safetensors file containing a single [n_rows, context_len] tensor
      pad_token_id: int used as the last value in y.
      seed: RNG seed for the train/val split and per-epoch train shuffling.
      val_ratio: fraction of rows for validation. 0 < val_ratio < 1.
      batch_size: DataLoader batch size.
      num_workers: DataLoader workers.
      shuffle_train: shuffle training batches each epoch.
      drop_last: whether to drop the last incomplete batch.
      pin_memory: set True if loading to CUDA.

    Notes:
      - The split is deterministic given `seed`.
      - Dataset keeps the data in memory; very large tensors may require more RAM.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1).")

    full = _load_tensor_from_safetensors(path)
    dataset = ControllableGPTDataset(full)

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

class ControllableGPTLoader:
    def __init__(self, dataset_path: str, batch_size: int, seed: int, val_ratio: float = 0.1):
        self.train_loader, self.val_loader = build_dataloaders(
            path=dataset_path,
            seed=seed,
            val_ratio=val_ratio,
            batch_size=batch_size,
            shuffle_train=True,
            drop_last=True,
            pin_memory=False)