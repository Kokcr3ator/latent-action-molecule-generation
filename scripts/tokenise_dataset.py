from dataclasses import dataclass
import os
import logging

import torch
import tyro

from interdiff.config import to_absolute_path
from interdiff.io import load_smiles, save_tokenised_dataset
from interdiff.tokenise import train_smiles_tokeniser, train_selfies_tokeniser


_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[MASK]", "[BOS]", "[EOS]"]


def run_tokenisation(
    data_smiles: str,
    vocab_size: int,
    context_length: int,
    use_selfies: bool = False,
) -> tuple[str, str]:
    """Tokenise a SMILES dataset and return (tokenizer_dir, dataset_path)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("tokenize")

    out_dir = to_absolute_path(
        f"interdiff/data/processed/zinc_tok_seqlen_{context_length}_vocabsize_{vocab_size}"
    )
    dataset_path = to_absolute_path(data_smiles)
    os.makedirs(out_dir, exist_ok=True)

    log.info(f"Loading SMILES from {dataset_path}")
    smiles_list = load_smiles(dataset_path)
    log.info(f"Loaded {len(smiles_list)} SMILES")

    log.info(f"Training {'SELFIES' if use_selfies else 'SMILES'} tokenizer")
    if use_selfies:
        tokenizer = train_selfies_tokeniser(smiles_list, special_tokens=_SPECIAL_TOKENS)
    else:
        tokenizer = train_smiles_tokeniser(
            smiles_list,
            vocab_size=vocab_size,
            special_tokens=_SPECIAL_TOKENS,
        )

    pad_token = _SPECIAL_TOKENS[0]
    tokenizer.enable_truncation(max_length=context_length)
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id(pad_token),
        pad_token=pad_token,
        length=context_length,
    )
    log.info("Tokenizer training complete.")
    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))

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
    return out_dir, save_path


@dataclass
class TokeniseCfg:
    data_smiles: str
    vocab_size: int = 2_048
    context_length: int = 128
    use_selfies: bool = False


def main() -> None:
    cfg = tyro.cli(TokeniseCfg)
    tok_dir, save_path = run_tokenisation(
        data_smiles=cfg.data_smiles,
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        use_selfies=cfg.use_selfies,
    )
    print(save_path)


if __name__ == "__main__":
    main()
