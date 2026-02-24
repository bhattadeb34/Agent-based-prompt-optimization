"""SMILES validation, canonicalization, and similarity utilities.

All functions are domain-agnostic. Marker requirements (e.g. [Cu]/[Au] for polymers)
are passed explicitly — empty list means a plain RDKit validity check only.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_smiles(
    smiles: str,
    required_markers: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    Validate a SMILES string.

    Args:
        smiles: The SMILES string to validate.
        required_markers: Optional list of substrings that MUST appear in the SMILES.
            E.g. ["[Cu]", "[Au]"] for polymer repeat units.
            Empty list / None = only RDKit validity is checked.

    Returns:
        (is_valid, reason) — reason is "" if valid.
    """
    if not smiles or not smiles.strip():
        return False, "Empty SMILES"

    # Check required markers
    if required_markers:
        for marker in required_markers:
            if marker not in smiles:
                return False, f"Missing required marker: {marker}"

    # RDKit parse
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "RDKit could not parse SMILES"
    except Exception as e:
        return False, f"RDKit error: {e}"

    return True, ""


# Keep backward-compatible alias (polymer-specific usage in existing surrogates)
def validate_polymer_smiles(smiles: str) -> Tuple[bool, str]:
    """Backward-compatible wrapper: requires [Cu] and [Au] markers."""
    return validate_smiles(smiles, required_markers=["[Cu]", "[Au]"])


# ──────────────────────────────────────────────────────────────────────────────
# Canonicalization
# ──────────────────────────────────────────────────────────────────────────────

def canonicalize(smiles: str) -> Optional[str]:
    """Return canonical SMILES, or None if invalid."""
    try:
        return Chem.CanonSmiles(smiles)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Similarity
# ──────────────────────────────────────────────────────────────────────────────

def tanimoto_similarity(
    smiles1: str,
    smiles2: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> float:
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


def tanimoto_similarity_stripped(
    smiles1: str,
    smiles2: str,
    strip_tokens: Optional[List[str]] = None,
) -> float:
    """
    Tanimoto similarity after stripping specified tokens from both SMILES.
    Useful for polymer repeat-unit comparison where backbone markers inflate similarity.

    Args:
        strip_tokens: Tokens to remove before fingerprinting. E.g. ["[Cu]", "[Au]"].
    """
    if strip_tokens:
        for tok in strip_tokens:
            smiles1 = smiles1.replace(tok, "")
            smiles2 = smiles2.replace(tok, "")
    return tanimoto_similarity(smiles1, smiles2)


def compute_similarity(
    smiles1: str,
    smiles2: str,
    similarity_on_repeat_unit: bool = False,
    marker_strip_tokens: Optional[List[str]] = None,
) -> float:
    """
    Unified similarity function — use this throughout the framework.

    Dispatches to `tanimoto_similarity_stripped` if `similarity_on_repeat_unit` is True.
    """
    if similarity_on_repeat_unit and marker_strip_tokens:
        return tanimoto_similarity_stripped(smiles1, smiles2,
                                            strip_tokens=marker_strip_tokens)
    return tanimoto_similarity(smiles1, smiles2)


def is_novel(smiles: str, reference_set: set) -> bool:
    """Check if a SMILES is not in the reference set (after canonicalization)."""
    canonical = canonicalize(smiles)
    if canonical is None:
        return False
    return canonical not in reference_set
