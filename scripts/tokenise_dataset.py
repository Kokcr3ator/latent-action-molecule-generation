# scripts/tokenize.py
from dataclasses import dataclass
from typing import List, Optional
import os
import logging

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, MISSING
import torch

from interdiff.io import load_smiles, save_tokenised_dataset
from interdiff.tokenise import train_smiles_tokeniser, train_selfies_tokeniser

@dataclass
class TokenizerCfg:
    dataset_path: str = MISSING
    output_dir: str = "data/processed/zinc_tok_seqlen_{seq_len}_vocabsize_{vocab_size}"
    use_selfies: bool = False
    vocab_size: int = 500             
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    mask_token: str = "[MASK]"
    bos_token: str = "[BOS]"
    eos_token: str = "[EOS]"
    seq_length: int = 128

    def __init__(self, dataset_path: str, output_dir: str,
                 use_selfies: bool, vocab_size: int,
                 pad_token: str, unk_token: str,
                 mask_token: str, bos_token: str,
                 eos_token: str, seq_length: int, **kwargs) -> None:
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.use_selfies = use_selfies
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.seq_length = seq_length

def run_tokenisation(cfg: DictConfig) -> str:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("tokenize")

    seq_length: int = int(cfg.tokenizer.seq_length)
    tcfg = TokenizerCfg(**cfg.tokenizer)

    dataset_path = to_absolute_path(tcfg.dataset_path)
    out_dir = to_absolute_path(tcfg.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    log.info(f"Seq length (context.seq_len): {seq_length}")
    log.info(f"Loading SMILES from {dataset_path}")
    smiles_list = load_smiles(dataset_path)
    log.info(f"Loaded {len(smiles_list)} SMILES")

    special_tokens = [
        tcfg.pad_token,
        tcfg.unk_token,
        tcfg.mask_token,
        tcfg.bos_token,
        tcfg.eos_token,
    ]

    log.info(f"Training {'SELFIES' if tcfg.use_selfies else 'SMILES'} tokenizer")
    if tcfg.use_selfies:
        tokenizer = train_selfies_tokeniser(smiles_list, special_tokens=special_tokens)
    else:
        tokenizer = train_smiles_tokeniser(
            smiles_list,
            vocab_size=tcfg.vocab_size,
            special_tokens=special_tokens,
        )

    tokenizer.enable_truncation(max_length=seq_length)
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id(tcfg.pad_token),
        pad_token=tcfg.pad_token,
        length=seq_length,
    )
    log.info("Tokenizer training complete.")

    tok_path = os.path.join(out_dir, "tokenizer.json")
    log.info(f"Saving tokenizer to {tok_path}")
    tokenizer.save(tok_path)

    log.info("Tokenizing dataset...")
    encodings = tokenizer.encode_batch(smiles_list)
    tokenised_dataset = [e.ids for e in encodings]

    vocab_size = len(tokenizer.get_vocab())
    if vocab_size <= 256:
        dtype = torch.uint8
    elif vocab_size <= 32767:
        dtype = torch.int16
    else:
        dtype = torch.int32

    save_path = os.path.join(out_dir, "dataset.safetensors")

    save_tokenised_dataset(
        tokenized_data=tokenised_dataset,
        tokeniser=tokenizer,
        output_path=save_path,
        dtype=dtype,
    )
    log.info(f"Tokenized dataset saved to {save_path}")
    return save_path

@hydra.main(version_base=None, config_path="../interdiff/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    save_path = run_tokenisation(cfg)
    print(save_path)

if __name__ == "__main__":
    main()



