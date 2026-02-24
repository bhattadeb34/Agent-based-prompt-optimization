"""Abstract base class for surrogate property predictors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class SurrogatePredictor(ABC):
    """
    Abstract base for any surrogate property predictor.

    Concrete implementations must populate the class attributes and implement predict().
    These can be GNNs, random forests, deep learning models, or external API calls.
    """
    property_name: str = "Property"   # human-readable name, e.g. "Conductivity"
    property_units: str = ""          # display units, e.g. "mS/cm"
    maximize: bool = True             # True = higher is better (most properties)

    @abstractmethod
    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        """
        Predict the target property for a list of SMILES.

        Args:
            smiles_list: List of polymer SMILES strings.

        Returns:
            List of predicted float values (in property_units).
            None entries indicate failed predictions (invalid SMILES, etc.).
        """
        ...

    def predict_single(self, smiles: str) -> Optional[float]:
        """Convenience wrapper for a single SMILES."""
        results = self.predict([smiles])
        return results[0] if results else None

    def config_dict(self) -> dict:
        """Return metadata dict for logging."""
        return {
            "class": self.__class__.__name__,
            "property_name": self.property_name,
            "property_units": self.property_units,
            "maximize": self.maximize,
        }
