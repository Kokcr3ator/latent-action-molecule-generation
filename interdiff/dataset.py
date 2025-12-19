from typing import Tuple

from tokenizers import Tokenizer
import torch


class Sampler:
    def __init__(
        self,
        data: torch.Tensor,
        tokeniser: Tokenizer,
        batch_size: int,
        device: str = "cuda",
    ):
        self.data = data
        self.indices = torch.randperm(len(data))
        self.batch_size = batch_size
        self.device = device
        self.current_idx = 0
        self.tokeniser = tokeniser
        self.smiles = tokeniser.decode_batch(data)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.Tensor(self.data[self.indices[idx]], device=self.device)
        y = torch.roll(x, shifts=-1)  # Shift x to create y
        return x, y

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        i = self.current_idx
        self.current_idx += self.batch_size
        if self.current_idx >= len(self.indices):
            self.current_idx = 0
        if i + self.batch_size > len(self.indices):
            i = len(self.indices) - self.batch_size
        return self[i : i + self.batch_size]

    def sample_x(self) -> torch.Tensor:
        i = self.current_idx
        self.current_idx += self.batch_size
        if self.current_idx >= len(self.indices):
            self.current_idx = 0
        if i + self.batch_size > len(self.indices):
            i = len(self.indices) - self.batch_size
        return torch.Tensor(self.data[self.indices[i]], device=self.device)
