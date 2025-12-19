import json
import os
import logging
from typing import Any, Dict, List

import tokenizers
import torch
import safetensors.torch

from .utils import with_bounds


def load_vocab(vocab_path: str) -> Dict[str, Any]:
    """Load vocabulary from a JSON file.
    
    Args:
        vocab_path: Path to the vocabulary JSON file.
        
    Returns:
        Dictionary containing the vocabulary.
        
    Raises:
        FileNotFoundError: If the vocabulary file doesn't exist.
    """
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")

    logging.info(f"Loading vocabulary from {vocab_path}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab


def load_smiles(path: str) -> List[str]:
    """Read SMILES strings from a text file.
    
    Reads one SMILES string per line, wraps each with BOS/EOS tokens,
    and drops empty lines.
    
    Args:
        path: Path to the SMILES text file.
        
    Returns:
        List of SMILES strings with BOS/EOS bounds.
        
    Raises:
        FileNotFoundError: If the SMILES file doesn't exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"SMILES file not found at {path}")

    logging.info(f"Loading SMILES from {path}")
    with open(path, "r") as f:
        return [with_bounds(line.strip()) for line in f if line.strip()]


def load_tokenised_dataset(path: str) -> Dict[str, Any]:
    """Load a tokenized dataset from a SafeTensors file.
    
    Args:
        path: Path to the tokenized dataset file.
        
    Returns:
        Dictionary containing 'dataset' and 'tokeniser' keys.
        
    Raises:
        FileNotFoundError: If the dataset file doesn't exist.
        KeyError: If required keys are missing from the file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tokenized dataset file not found at {path}")

    logging.info(f"Loading tokenized dataset from {path}")
    data = safetensors.torch.load_file(path)
    if "dataset" not in data:
        raise KeyError("The loaded file does not contain 'dataset' key.")
    if "tokeniser" not in data:
        raise KeyError("The loaded file does not contain 'tokeniser' key.")

    return data


def save_tokenised_dataset(
    tokenized_data: List[tokenizers.Encoding],
    tokeniser: tokenizers.Tokenizer,
    output_path: str,
    dtype: torch.dtype,
):
    """Save the tokenized dataset to a SafeTensors file.
    
    Args:
        tokenized_data: List of tokenized sequences.
        tokeniser: The tokenizer used for tokenization.
        output_path: Path where the dataset will be saved.
        dtype: PyTorch dtype for the tensor data.
    """
    logging.info(f"Saving tokenized dataset to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_tensor = torch.tensor(tokenized_data, dtype=dtype)
    data = {"dataset": data_tensor}
    metadata = {"tokeniser": tokeniser.to_str(), "vocab_size": str(tokeniser.get_vocab_size())}
    
    safetensors.torch.save_file(data,
        output_path,
        metadata=metadata
    )
    logging.info("Tokenized dataset saved successfully.")


def load_tokenizer(tokenizer_path: str) -> tokenizers.Tokenizer:
    """Load a tokenizer from a JSON file.
    
    Args:
        tokenizer_path: Path to the tokenizer JSON file.
        
    Returns:
        Loaded tokenizer instance.
        
    Raises:
        FileNotFoundError: If the tokenizer file doesn't exist.
    """
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")

    logging.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
    return tokenizer
