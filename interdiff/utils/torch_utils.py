import os
import random
import numpy as np
import torch
from safetensors import safe_open

def seed_all(seed: int = 42, deterministic: bool = False, disable_cudnn_benchmark: bool = True) -> None:
    """
    Seed all common random number generators for reproducibility.

    Args:
        seed (int): The base random seed to use across libraries.
        deterministic (bool): If True, force deterministic behavior in CUDA operations
            (can slow down training).
        disable_cudnn_benchmark (bool): If True, disables CuDNN auto-tuner to prevent
            nondeterministic algorithm selection.

    Notes:
        - Sets the PYTHONHASHSEED environment variable.
        - Seeds Python's built-in random, NumPy, and torch (CPU + CUDA).
        - Optionally configures CuDNN for deterministic results.
    """
    # ---- Python & environment ----
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # ---- NumPy ----
    np.random.seed(seed)

    # ---- PyTorch ----
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure reproducibility for CuDNN
    if deterministic:
        torch.backends.cudnn.deterministic = True
        if disable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = not disable_cudnn_benchmark

    print(f"[SeedAll] Global seed set to {seed} (deterministic={deterministic})")

def _load_tensor_from_safetensors(path: str) -> torch.Tensor:
    """
    Loads a single tensor from a .safetensors file as a torch tensor.
    """
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        return f.get_tensor(keys[0])
