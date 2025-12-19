import networkx as nx
import numpy as np
from rdkit import Chem
import torch
from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import selfies as sf

from .io import load_vocab


def smiles_to_graph(smiles):
    """
    Convert a SMILES string to a graph representation.

    Args:
        smiles (str): The SMILES string to convert.

    Returns:
        tuple: A tuple containing the graph representation of the molecule.
    """
    # Convert SMILES to RDKit molecule object
    mol = MolFromSmiles(smiles)

    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom.GetAtomicNum())

    # Get bond features
    bond_features = []
    for bond in mol.GetBonds():
        bond_features.append(bond.GetBondTypeAsDouble())

    # Create adjacency matrix
    adj_matrix = GetAdjacencyMatrix(mol)

    return np.array(atom_features), np.array(bond_features), adj_matrix


def tokens_to_smiles(tokens: torch.Tensor, vocab_path: str) -> str:
    vocab = load_vocab(vocab_path)
    special_tokens = set()
    EOS_tok = None

    for special_token in vocab["added_tokens"]:
        special_tokens.add(special_token["id"])
        if special_token["content"] == "[EOS]":
            EOS_tok = special_token["id"]

    vocab = {v: k for k, v in vocab["model"]["vocab"].items()}
    out = []
    for seq in tokens:
        tok = []
        for idx in seq.view(-1).tolist():
            if idx == EOS_tok:
                break
            if idx in special_tokens:
                continue
            token = vocab.get(idx)
            tok.append(token)
        out.append("".join(tok))

    out = "".join(out)
    out = sf.decoder(out)
    return out  # type: ignore


def smiles_to_selfies(smiles: str) -> str:
    """
    Convert a SMILES string to SELFIES and return a space‑separated
    string of SELFIES symbols, ready for Whitespace tokenization.
    """
    selfies_str = sf.encoder(smiles)
    symbols = sf.split_selfies(selfies_str)  # list -> ['[C]', '[=C]', ...]
    return " ".join(symbols)
