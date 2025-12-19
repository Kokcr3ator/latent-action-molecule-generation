from collections import OrderedDict
import logging
from typing import List

from tokenizers.models import WordLevel, BPE
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit
from tokenizers.trainers import BpeTrainer

from .molecules import smiles_to_selfies
from .utils import with_bounds

def train_smiles_tokeniser(
    smiles: List[str],
    vocab_size: int,
    special_tokens: List[str] = ["[PAD]", "[UNK]", "[MASK]", "[BOS]", "[EOS]"],
) -> Tokenizer:
    """Train a tokenizer on a list of SMILES strings.
    Args:
        smiles (List[str]): List of SMILES strings.
    Returns:
        Tokenizer: A trained tokenizer.
    """

    logging.info("Training BPE tokenizer on SMILES strings...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()  # type: ignore

    # Build vocab with special tokens first
    logging.info("Building vocabulary with special tokens...")
    trainer = BpeTrainer(
        vocab_size=vocab_size,  # type: ignore
        special_tokens=special_tokens,  # type: ignore
    )

    # Train the tokenizer on the SMILES strings
    logging.info("Training tokenizer on SMILES strings...")
    smiles_list = map(with_bounds, smiles)
    tokenizer.train_from_iterator(smiles_list, trainer=trainer)
    return tokenizer


def train_selfies_tokeniser(
    smiles: List[str],
    special_tokens: List[str] = ["[PAD]", "[UNK]", "[MASK]", "[BOS]", "[EOS]"],
) -> Tokenizer:
    logging.info("Training Word-level tokenizer on SELFIES token strings...")

    # Convert SMILES -> SELFIES token strings
    logging.info("Converting SMILES to SELFIES token strings...")
    token_strs = map(lambda x: with_bounds(smiles_to_selfies(x)), smiles)

    # Extract all unique tokens
    logging.info("Counting tokens in SELFIES token strings...")
    token_counts = {}
    for line in token_strs:
        for token in line.split():
            if token not in special_tokens:
                token_counts[token] = token_counts.get(token, 0) + 1

    # Build vocab with special tokens first
    logging.info("Building vocabulary with special tokens...")
    vocab = OrderedDict()
    for i, tok in enumerate(special_tokens):
        vocab[tok] = i

    for i, tok in enumerate(sorted(token_counts), start=len(special_tokens)):
        vocab[tok] = i
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.add_special_tokens(special_tokens)  # type: ignore
    tokenizer.pre_tokenizer = WhitespaceSplit()  # type: ignore
    return tokenizer
