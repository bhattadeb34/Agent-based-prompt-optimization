"""Surrogate predictor registry — maps config names to classes."""
from __future__ import annotations

from typing import Any, Dict, Optional, Type

from .base import SurrogatePredictor


_REGISTRY: Dict[str, Type[SurrogatePredictor]] = {}


def register(name: str):
    """Decorator to register a surrogate class under a given name."""
    def decorator(cls: Type[SurrogatePredictor]):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_surrogate(name: str, **kwargs: Any) -> SurrogatePredictor:
    """
    Instantiate a surrogate by registry name.

    Example:
        pred = get_surrogate("gnn_conductivity", model_base_path="/path/to/models")
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(
            f"Unknown surrogate '{name}'. Available surrogates: {available}\n"
            "Register a new surrogate with the @register('name') decorator."
        )
    return _REGISTRY[name](**kwargs)


def list_surrogates() -> Dict[str, Type[SurrogatePredictor]]:
    """Return a copy of the surrogate registry."""
    return dict(_REGISTRY)


# Import concrete implementations so they self-register
def _lazy_import_all():
    """Import all surrogate modules so their @register decorators fire."""
    from . import gnn_predictor  # noqa: F401
    from . import tg_predictor   # noqa: F401

_lazy_import_all()
