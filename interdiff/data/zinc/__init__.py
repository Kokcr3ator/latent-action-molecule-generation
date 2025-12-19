import os
import json


def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab


EOS_tok = None
BOS_tok = None
PAD_tok = None
tokenizer_path = os.path.join(os.path.dirname(__file__), "zinc_tokenizer.json")
if not os.path.exists(tokenizer_path):
    from ....scripts import tokenise_dataset

if os.path.exists(tokenizer_path):
    vocab = load_vocab(tokenizer_path)
    special_tokens = set()

    for special_token in vocab["added_tokens"]:
        special_tokens.add(special_token["id"])
        if special_token["content"] == "[EOS]":
            EOS_tok = special_token["id"]
        if special_token["content"] == "[BOS]":
            BOS_tok = special_token["id"]
        if special_token["content"] == "[PAD]":
            PAD_tok = special_token["id"]
else:
    print("Tokenizer file not found. Please run the preparation script to generate it.")
