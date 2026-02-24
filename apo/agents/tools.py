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


class SMILESValidatorTool(Tool):
    """Validate SMILES strings using RDKit before sending to predictor."""

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

        results = []
        for smi in smiles_list:
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
        try:
            from rdkit import Chem, DataStructs
            from rdkit.Chem import AllChem
        except ImportError:
            return Observation(success=False, result=None, error="RDKit not available")

        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return Observation(
                success=False,
                result=None,
                error="One or both SMILES are invalid",
            )

        fp1 = AllChem.GetMorganFingerprint(mol1, 2)
        fp2 = AllChem.GetMorganFingerprint(mol2, 2)
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

        return Observation(
            success=True,
            result={"similarity": similarity},
            metadata={"smiles1": smiles1, "smiles2": smiles2},
        )


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
            value = self.surrogate.predict(smiles)
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
        for smi in smiles_list:
            try:
                value = self.surrogate.predict(smi)
                results.append({
                    "smiles": smi,
                    "property": value,
                    "valid": value is not None,
                })
            except Exception as e:
                results.append({
                    "smiles": smi,
                    "property": None,
                    "valid": False,
                    "error": str(e),
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
