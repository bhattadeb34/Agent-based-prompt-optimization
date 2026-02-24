"""SMILES validation and canonicalization utilities for polymer structures."""
from __future__ import annotations

import re
from typing import Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


# Polymer backbone placeholders (Cu = start, Au = end of chain)
POLYMER_START = "[Cu]"
POLYMER_END = "[Au]"


def validate_polymer_smiles(smiles: str) -> Tuple[bool, str]:
    """
    Validate a polymer SMILES string.

    Returns:
        (is_valid, reason) — reason is empty string if valid.
    """
    if not smiles or not smiles.strip():
        return False, "Empty SMILES"

    if POLYMER_START not in smiles:
        return False, f"Missing polymer start marker {POLYMER_START}"
    if POLYMER_END not in smiles:
        return False, f"Missing polymer end marker {POLYMER_END}"

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "RDKit could not parse SMILES"
    except Exception as e:
        return False, f"RDKit error: {e}"

    return True, ""


def canonicalize(smiles: str) -> Optional[str]:
    """Return canonical SMILES, or None if invalid."""
    try:
        return Chem.CanonSmiles(smiles)
    except Exception:
        return None


def tanimoto_similarity(smiles1: str, smiles2: str,
                        radius: int = 2, n_bits: int = 2048) -> float:
    """
    Morgan fingerprint Tanimoto similarity between two SMILES.
    Returns 0.0 if either is invalid.
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=n_bits)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=n_bits)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return 0.0


def tanimoto_on_repeat_unit(smiles1: str, smiles2: str) -> float:
    """
    Compute Tanimoto on the polymer repeat unit
    (strips [Cu] and [Au] placeholders before fingerprinting).
    """
    def strip_placeholders(s: str) -> str:
        return s.replace("[Cu]", "").replace("[Au]", "")

    return tanimoto_similarity(strip_placeholders(smiles1), strip_placeholders(smiles2))


def is_novel(smiles: str, reference_set: set) -> bool:
    """Check if a SMILES is not in the reference set (after canonicalisation)."""
    canonical = canonicalize(smiles)
    if canonical is None:
        return False
    return canonical not in reference_set
