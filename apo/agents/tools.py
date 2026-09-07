"""
Chemistry-specific tools for agentic workflow.

Tools provide agents with capabilities beyond just calling LLMs:
- SMILES validation and repair
- Similarity calculation
- Property prediction
- Chemistry knowledge lookup
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Observation, Tool
from ..task_context import TaskContext
from ..utils.smiles_utils import compute_similarity, validate_smiles


class SMILESValidatorTool(Tool):
    """Validate SMILES strings using RDKit before sending to predictor."""

    def __init__(self, task_context: Optional[TaskContext] = None):
        self.ctx = task_context

    @property
    def name(self) -> str:
        return "validate_smiles"

    @property
    def description(self) -> str:
        return (
            "Validate one or more SMILES strings using RDKit. "
            "Returns validity status and detailed error messages if invalid."
        )

    @property
    def parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "smiles_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of SMILES strings to validate",
                },
            },
            "required": ["smiles_list"],
        }

    def execute(self, smiles_list: List[str]) -> Observation:
        """Validate SMILES and return detailed results."""
        try:
            from rdkit import Chem
        except ImportError:
            return Observation(
                success=False,
                result=None,
                error="RDKit not available",
            )

        required_markers = self.ctx.smiles_markers if self.ctx else []
        results = []
        for smi in smiles_list:
            ok, reason = validate_smiles(smi, required_markers=required_markers)
            if not ok:
                results.append({
                    "smiles": smi,
                    "valid": False,
                    "error": reason,
                })
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                results.append({
                    "smiles": smi,
                    "valid": False,
                    "error": "RDKit parsing failed",
                })
            else:
                # Check for common issues
                try:
                    Chem.SanitizeMol(mol)
                    canonical = Chem.MolToSmiles(mol)
                    results.append({
                        "smiles": smi,
                        "valid": True,
                        "canonical": canonical,
                        "num_atoms": mol.GetNumAtoms(),
                    })
                except Exception as e:
                    results.append({
                        "smiles": smi,
                        "valid": False,
                        "error": f"Sanitization failed: {str(e)}",
                    })

        n_valid = sum(1 for r in results if r["valid"])
        return Observation(
            success=True,
            result=results,
            metadata={
                "n_total": len(smiles_list),
                "n_valid": n_valid,
                "validity_rate": n_valid / len(smiles_list) if smiles_list else 0,
            },
        )


class SMILESRepairTool(Tool):
    """Attempt to repair invalid SMILES by fixing common errors."""

    @property
    def name(self) -> str:
        return "repair_smiles"

    @property
    def description(self) -> str:
        return (
            "Attempt to repair invalid SMILES by fixing common syntax errors. "
            "Useful when generator produces almost-valid SMILES."
        )

    @property
    def parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "Invalid SMILES string to repair",
                },
                "error_hint": {
                    "type": "string",
                    "description": "Error message from validation (optional)",
                },
            },
            "required": ["smiles"],
        }

    def execute(self, smiles: str, error_hint: str = "") -> Observation:
        """Try common repairs."""
        from rdkit import Chem

        repairs = [
            smiles.replace("()", ""),  # Remove empty parentheses
            smiles.replace("N()", "N"),  # Fix N() → N
            smiles.replace("C(=O)()", "C(=O)"),  # Fix C(=O)() → C(=O)
            smiles.replace("F(F)", "F"),  # Fix valence errors
            smiles.replace("Cl(Cl)", "Cl"),
        ]

        for repaired in repairs:
            if repaired != smiles:
                mol = Chem.MolFromSmiles(repaired)
                if mol is not None:
                    try:
                        Chem.SanitizeMol(mol)
                        canonical = Chem.MolToSmiles(mol)
                        return Observation(
                            success=True,
                            result={
                                "original": smiles,
                                "repaired": repaired,
                                "canonical": canonical,
                            },
                            metadata={"repair_type": "syntax_fix"},
                        )
                    except:
                        continue

        return Observation(
            success=False,
            result=None,
            error="Could not repair SMILES",
        )


class SimilarityCalculatorTool(Tool):
    """Calculate structural similarity between molecules."""

    def __init__(self, task_context: Optional[TaskContext] = None):
        self.ctx = task_context

    @property
    def name(self) -> str:
        return "calculate_similarity"

    @property
    def description(self) -> str:
        return (
            "Calculate Tanimoto similarity between two SMILES strings. "
            "Returns similarity score (0-1)."
        )

    @property
    def parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "smiles1": {"type": "string", "description": "First SMILES"},
                "smiles2": {"type": "string", "description": "Second SMILES"},
            },
            "required": ["smiles1", "smiles2"],
        }

    def execute(self, smiles1: str, smiles2: str) -> Observation:
        """Calculate Tanimoto similarity."""
        ctx = self.ctx or TaskContext()
        similarity = compute_similarity(
            smiles1,
            smiles2,
            similarity_on_repeat_unit=ctx.similarity_on_repeat_unit,
            marker_strip_tokens=ctx.marker_strip_tokens,
        )
        if similarity == 0.0:
            valid1, _ = validate_smiles(smiles1, required_markers=ctx.smiles_markers)
            valid2, _ = validate_smiles(smiles2, required_markers=ctx.smiles_markers)
            if not (valid1 and valid2):
                return Observation(
                    success=False,
                    result=None,
                    error="One or both SMILES are invalid",
                )

        return Observation(
            success=True,
            result={"similarity": similarity},
            metadata={"smiles1": smiles1, "smiles2": smiles2},
        )


def _predict_single(surrogate, smiles: str) -> Optional[float]:
    """Call the surrogate single-item API without passing bare strings to predict()."""
    if hasattr(surrogate, "predict_single"):
        return surrogate.predict_single(smiles)
    preds = surrogate.predict([smiles])
    return preds[0] if preds else None


def _predict_batch(surrogate, smiles_list: List[str]) -> List[Optional[float]]:
    preds = surrogate.predict(smiles_list)
    if len(preds) != len(smiles_list):
        raise ValueError(
            f"surrogate returned {len(preds)} predictions for {len(smiles_list)} SMILES"
        )
    return preds


class PropertyPredictorTool(Tool):
    """Predict molecular property using surrogate model."""

    def __init__(self, surrogate, property_name: str = "property"):
        self.surrogate = surrogate
        self.property_name = property_name

    @property
    def name(self) -> str:
        return "predict_property"

    @property
    def description(self) -> str:
        return f"Predict {self.property_name} for a SMILES string using the surrogate model."

    @property
    def parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "SMILES string to evaluate"},
            },
            "required": ["smiles"],
        }

    def execute(self, smiles: str) -> Observation:
        """Predict property value."""
        try:
            value = _predict_single(self.surrogate, smiles)
            if value is None:
                return Observation(
                    success=False,
                    result=None,
                    error="Prediction returned None (possibly invalid SMILES)",
                )
            return Observation(
                success=True,
                result={self.property_name: value, "smiles": smiles},
                metadata={"predictor": "surrogate_model"},
            )
        except Exception as e:
            return Observation(
                success=False,
                result=None,
                error=f"Prediction failed: {str(e)}",
            )


class ChemistryKnowledgeTool(Tool):
    """Query chemistry knowledge (functional groups, substructures, etc.)."""

    @property
    def name(self) -> str:
        return "query_chemistry_knowledge"

    @property
    def description(self) -> str:
        return (
            "Query chemistry knowledge about functional groups, substructures, or properties. "
            "Examples: 'What functional groups contain ether oxygen?', 'List common high-Tg motifs'"
        )

    @property
    def parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Chemistry question"},
                "domain": {
                    "type": "string",
                    "description": "Domain: 'functional_groups', 'polymers', 'conductivity', 'general'",
                    "default": "general",
                },
            },
            "required": ["query"],
        }

    def execute(self, query: str, domain: str = "general") -> Observation:
        """Query hardcoded chemistry knowledge base."""
        # Simple knowledge base (could be expanded with vector DB, web search, etc.)
        knowledge = {
            "high_tg_motifs": [
                "Aromatic rings (benzene, naphthalene)",
                "Imide groups (polyimides)",
                "Amide linkages",
                "Rigid cyclic structures",
                "Bulky pendant groups",
            ],
            "ether_groups": [
                "Ethylene oxide (EO): -CH2CH2O-",
                "Propylene oxide (PO): -CH(CH3)CH2O-",
                "Diethylene glycol: -OCH2CH2OCH2CH2O-",
            ],
            "conductivity_enhancers": [
                "Ether oxygen density",
                "Flexible backbone",
                "Polar pendant groups (nitrile, carbonate)",
                "Low glass transition temperature",
            ],
        }

        # Simple keyword matching
        query_lower = query.lower()
        results = []

        if "tg" in query_lower or "glass transition" in query_lower:
            results.extend(knowledge["high_tg_motifs"])
        if "ether" in query_lower or "oxygen" in query_lower:
            results.extend(knowledge["ether_groups"])
        if "conductivity" in query_lower or "ion" in query_lower:
            results.extend(knowledge["conductivity_enhancers"])

        if not results:
            return Observation(
                success=False,
                result=None,
                error=f"No knowledge found for query: {query}",
            )

        return Observation(
            success=True,
            result={"query": query, "knowledge": results},
            metadata={"source": "hardcoded_kb"},
        )


class BatchPropertyPredictorTool(Tool):
    """Predict properties for multiple SMILES in batch (more efficient)."""

    def __init__(self, surrogate, property_name: str = "property"):
        self.surrogate = surrogate
        self.property_name = property_name

    @property
    def name(self) -> str:
        return "batch_predict_property"

    @property
    def description(self) -> str:
        return f"Predict {self.property_name} for multiple SMILES strings efficiently in batch."

    @property
    def parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "smiles_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of SMILES strings",
                },
            },
            "required": ["smiles_list"],
        }

    def execute(self, smiles_list: List[str]) -> Observation:
        """Batch prediction."""
        results = []
        try:
            values = _predict_batch(self.surrogate, smiles_list)
        except Exception as e:
            values = [None] * len(smiles_list)
            errors = [str(e)] * len(smiles_list)
        else:
            errors = [""] * len(smiles_list)

        for smi, value, error in zip(smiles_list, values, errors):
            result = {
                "smiles": smi,
                "property": value,
                "valid": value is not None,
            }
            if error:
                result["error"] = error
            results.append(result)

        n_valid = sum(1 for r in results if r["valid"])
        return Observation(
            success=True,
            result=results,
            metadata={
                "n_total": len(smiles_list),
                "n_valid": n_valid,
                "validity_rate": n_valid / len(smiles_list) if smiles_list else 0,
            },
        )
