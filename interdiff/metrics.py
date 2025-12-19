from __future__ import annotations
from typing import List

import rdkit.Chem.rdMolDescriptors
import rdkit.Chem.rdmolfiles
from rdkit.Chem import QED, Crippen, Descriptors
import sascorer  # type: ignore


def get_mol(smiles):
    """Convert a SMILES string to an RDKit molecule object.
    
    Args:
        smiles: SMILES string representation of a molecule.
        
    Returns:
        RDKit Mol object, or None if conversion fails.
    """
    return rdkit.Chem.rdmolfiles.MolFromSmiles(smiles)


def is_valid_smiles(smiles: str) -> bool:
    """
    Check if a SMILES string is valid.

    Args:
        smiles (str): The SMILES string to check.

    Returns:
        bool: True if the SMILES string is valid, False otherwise.
    """
    mol = rdkit.Chem.rdmolfiles.MolFromSmiles(smiles)
    return mol is not None


def filter_valid_smiles(smiles_list: List[str]) -> List[str]:
    """
    Filter a list of SMILES strings to only include valid ones.

    Args:
        smiles_list (List[str]): The list of SMILES strings to filter.

    Returns:
        List[str]: A list of valid SMILES strings.
    """
    return [smiles for smiles in smiles_list if is_valid_smiles(smiles)]


def qed(smiles: str, threshold: float = 0.7, as_reward: bool = False) -> float:
    """Calculate the QED (Quantitative Estimation of Drug-likeness) score.
    
    Args:
        smiles: SMILES string representation of a molecule.
        threshold: Threshold for binary reward computation.
        as_reward: If True, returns 1.0 if QED >= threshold, otherwise returns raw QED.
        
    Returns:
        QED score (0-1 range) or binary reward. Returns 0.0 for invalid molecules.
    """
    mol = get_mol(smiles)
    if mol is None:
        return 0.0
    if as_reward:
        out = 1.0 if QED.qed(mol) >= threshold else QED.qed(mol)
    else:
        out = QED.qed(mol)
    return out



def logp(smiles: str, min_val: float = 1.0, max_val: float = 3.5, as_reward: bool = False) -> float:
    """Calculate the LogP (partition coefficient) for a molecule.
    
    LogP measures lipophilicity - the tendency to dissolve in fats vs water.
    
    Args:
        smiles: SMILES string representation of a molecule.
        min_val: Minimum LogP value for binary reward.
        max_val: Maximum LogP value for binary reward.
        as_reward: If True, returns 1.0 if LogP in range, otherwise returns raw LogP.
        
    Returns:
        LogP value or binary reward. Returns 0.0 for invalid molecules.
    """
    mol = get_mol(smiles)
    if mol is None:
        return 0.0
    logp = Crippen.MolLogP(mol)
    if as_reward:
        out = 1.0 if min_val <= logp <= max_val else 0.0
    else:
        out = logp
    return out


def molecular_weight(smiles: str, min_val: int = 200, max_val: int = 450, as_reward: bool = False) -> float:
    """Calculate the molecular weight of a molecule.
    
    Args:
        smiles: SMILES string representation of a molecule.
        min_val: Minimum weight for binary reward.
        max_val: Maximum weight for binary reward.
        as_reward: If True, returns 1.0 if weight in range, otherwise returns raw weight.
        
    Returns:
        Molecular weight in Daltons or binary reward. Returns 0.0 for invalid molecules.
    """
    mol = get_mol(smiles)
    if mol is None:
        return 0.0
    mw = Descriptors.MolWt(mol)
    if as_reward:
        out = 1.0 if min_val <= mw <= max_val else 0.0
    else:
        out = mw
    return out


def tpsa(smiles: str, max_val: float = 90.0, as_reward: bool = False) -> float:
    """Calculate the Topological Polar Surface Area (TPSA).
    
    TPSA is useful for predicting drug transport properties.
    
    Args:
        smiles: SMILES string representation of a molecule.
        max_val: Maximum TPSA value for binary reward.
        as_reward: If True, returns 1.0 if TPSA <= max_val, otherwise returns raw TPSA.
        
    Returns:
        TPSA value in Ų or binary reward. Returns 0.0 for invalid molecules.
    """
    mol = get_mol(smiles)
    if mol is None:
        return 0.0
    tpsa = rdkit.Chem.rdMolDescriptors.CalcTPSA(mol)
    if as_reward:
        out = 1.0 if tpsa <= max_val else 0.0
    else:
        out = tpsa

    return out


def synthetic_accessibility(smiles: str, max_val: float = 4.0, as_reward: bool = False) -> float:
    """Calculate the Synthetic Accessibility (SA) score.
    
    SA score estimates how easy it is to synthesize a molecule (1=easy, 10=hard).
    
    Args:
        smiles: SMILES string representation of a molecule.
        max_val: Maximum SA score for binary reward.
        as_reward: If True, returns 1.0 if SA <= max_val, otherwise returns raw SA.
        
    Returns:
        SA score (1-10 scale) or binary reward. Returns 0.0 for invalid molecules.
    """
    mol = get_mol(smiles)
    if mol is None:
        return 0.0
    sa = sascorer.calculateScore(mol)
    if sa is None:
        sa = 0.0
    if as_reward:
        out = 1.0 if sa <= max_val else 0.0
    else:
        out = sa

    return out


def validity(smiles_list: List[str]) -> float:
    """
    Calculate the validity of a list of SMILES strings.

    Args:
        smiles_list (List[str]): The list of SMILES strings to evaluate.

    Returns:
        float: The percentage of valid SMILES strings in the list.
    """
    valid_smiles = filter_valid_smiles(smiles_list)
    return len(valid_smiles) / len(smiles_list) if len(smiles_list) > 0 else 0.0


def uniqueness(smiles_list: List[str]) -> float:
    """
    Calculate the uniqueness of a list of SMILES strings.

    Args:
        smiles_list (List[str]): The list of SMILES strings to evaluate.

    Returns:
        float: The percentage of unique SMILES strings in the list.
    """
    smiles_list = filter_valid_smiles(smiles_list)
    unique_smiles = set(smiles_list)
    return len(unique_smiles) / len(smiles_list) if len(smiles_list) > 0 else 0.0


def novelty(smiles_list: List[str], reference_smiles: List[str]) -> float:
    """
    Calculate the novelty of a list of SMILES strings compared to a reference set.

    Args:
        smiles_list (List[str]): The list of SMILES strings to evaluate.
        reference_smiles (List[str]): The reference set of SMILES strings.

    Returns:
        float: The percentage of novel SMILES strings in the list compared to the reference set.
    """
    smiles_list = filter_valid_smiles(smiles_list)
    reference_set = set(reference_smiles)
    novel_smiles = set(smiles_list) - reference_set
    return len(novel_smiles) / len(smiles_list) if len(smiles_list) > 0 else 0.0
