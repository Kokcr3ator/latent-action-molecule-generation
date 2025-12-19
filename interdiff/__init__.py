import sys
import os

# This file is part of RDKit.
from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))

# Disable RDKit logging
# This is necessary to prevent RDKit from printing debug information to the console.
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from .metrics import (
    get_mol,
    qed,
    logp,
    molecular_weight,
    tpsa,
    synthetic_accessibility,
)
from .molecules import (
    tokens_to_smiles,
    smiles_to_graph,
    smiles_to_selfies,
)
from .io import load_vocab
from .dataset import Sampler
