"""
TaskContext — carries all domain-specific knowledge for a property optimization task.

This is the single source of truth for everything that varies between tasks:
- What kind of molecules (polymers, drugs, organics, materials)
- What structural markers/constraints to enforce
- What domain knowledge to inject into LLM prompts
- How to compute structural similarity

By constructing TaskContext from the YAML config, NO other code needs to know
what domain it is operating in. The same optimizer loops work for conductivity,
glass transition temperature, HOMO/LUMO, solubility, viscosity, or anything else.

Example YAML block (under `task:`):
    molecule_type: "polymer electrolyte"
    domain_context: >
      [Cu] and [Au] mark backbone connection points.
      Every SMILES must contain exactly one [Cu] and one [Au].
    smiles_markers: ["[Cu]", "[Au]"]
    similarity_on_repeat_unit: true
    marker_strip_tokens: ["[Cu]", "[Au]"]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TaskContext:
    """
    Domain context for a property optimization task.
    All fields come from the YAML config `task:` block.
    """

    # ── Identity ────────────────────────────────────────────────────────────
    property_name: str = "Property"
    property_units: str = ""
    maximize: bool = True

    # ── Molecule type ────────────────────────────────────────────────────────
    molecule_type: str = "molecule"
    """Human-readable label, e.g. 'polymer electrolyte', 'drug-like molecule', 'organic compound'."""

    # ── Prompt injection ─────────────────────────────────────────────────────
    domain_context: str = ""
    """
    Free-text injected verbatim into all LLM system prompts (worker, critic, meta, extractor).
    Use this to explain domain-specific SMILES notation, constraints, or chemical intuitions.

    Example for polymers:
        [Cu] and [Au] are backbone connection points, not actual metals.
        Every SMILES must contain exactly one [Cu] and one [Au].
    """

    # ── SMILES validation ────────────────────────────────────────────────────
    smiles_markers: List[str] = field(default_factory=list)
    """
    List of substrings that MUST appear in every generated SMILES.
    Empty list = no marker requirement (pure RDKit validity check only).

    Example: ["[Cu]", "[Au]"] for polymer repeat-unit notation.
    """

    # ── Similarity calculation ───────────────────────────────────────────────
    similarity_on_repeat_unit: bool = False
    """
    If True, strip `marker_strip_tokens` before computing Tanimoto fingerprints.
    Useful when backbone markers would inflate similarity between different structures.
    """
    marker_strip_tokens: List[str] = field(default_factory=list)
    """Tokens to strip when computing similarity. Defaults to smiles_markers if not set."""

    # ── Seed strategy ────────────────────────────────────────────────────────
    seed_strategy: str = ""
    """
    Starting strategy prompt for the first epoch.
    Should be domain-specific but not mention model names or specific molecules.
    """

    # ── Example SMILES for prompt context ────────────────────────────────────
    example_smiles: List[str] = field(default_factory=list)
    """
    Optional example valid SMILES to include in the worker LLM system prompt.
    Helps LLM understand the expected format for this domain.
    """

    def __post_init__(self):
        # Default marker_strip_tokens to smiles_markers if not explicitly set
        if not self.marker_strip_tokens and self.smiles_markers:
            self.marker_strip_tokens = list(self.smiles_markers)

        # Default seed strategy
        if not self.seed_strategy:
            direction = "maximise" if self.maximize else "minimise"
            self.seed_strategy = (
                f"Generate {self.molecule_type} SMILES that {direction} "
                f"{self.property_name} ({self.property_units}) while maintaining "
                f"structural similarity to the parent molecules."
            )

    @classmethod
    def from_config(cls, cfg: dict, surrogate=None) -> "TaskContext":
        """
        Build TaskContext from the parsed YAML `task:` dict.
        Optionally accepts a SurrogatePredictor instance to auto-fill property_name etc.
        """
        task = cfg.get("task", {})

        # Start with surrogate metadata if available
        prop_name = getattr(surrogate, "property_name", task.get("property_name", "Property"))
        prop_units = getattr(surrogate, "property_units", task.get("property_units", ""))
        maximize = getattr(surrogate, "maximize", task.get("maximize", True))

        return cls(
            property_name=prop_name,
            property_units=prop_units,
            maximize=maximize,
            molecule_type=task.get("molecule_type", "molecule"),
            domain_context=task.get("domain_context", "").strip(),
            smiles_markers=task.get("smiles_markers", []),
            similarity_on_repeat_unit=task.get("similarity_on_repeat_unit", False),
            marker_strip_tokens=task.get("marker_strip_tokens",
                                         task.get("smiles_markers", [])),
            seed_strategy=cfg.get("optimization", {}).get("seed_strategy", "").strip(),
            example_smiles=task.get("example_smiles", []),
        )

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "property_name": self.property_name,
            "property_units": self.property_units,
            "maximize": self.maximize,
            "molecule_type": self.molecule_type,
            "smiles_markers": self.smiles_markers,
            "similarity_on_repeat_unit": self.similarity_on_repeat_unit,
        }

    @property
    def direction_word(self) -> str:
        return "maximise" if self.maximize else "minimise"

    @property
    def direction_word_us(self) -> str:
        return "maximize" if self.maximize else "minimize"
