"""Reward functions for evaluating a batch of generated candidates."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Pareto helpers
# ──────────────────────────────────────────────────────────────────────────────

def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-efficient points (maximising both objectives)."""
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            dominates = np.all(costs >= c, axis=1) & np.any(costs > c, axis=1)
            is_efficient[i] = not dominates.any()
    return is_efficient


def calculate_hypervolume(pareto_points: np.ndarray,
                          reference_point: Optional[np.ndarray] = None) -> float:
    """
    2-D hypervolume indicator (maximisation).
    reference_point defaults to (0, 0) in improvement × similarity space.
    """
    if len(pareto_points) == 0:
        return 0.0
    if reference_point is None:
        reference_point = np.array([0.0, 0.0])
    sorted_pts = pareto_points[np.argsort(pareto_points[:, 0])]
    hv = 0.0
    prev_x = reference_point[0]
    for pt in sorted_pts:
        width = pt[0] - prev_x
        height = max(pt[1] - reference_point[1], 0.0)
        hv += width * height
        prev_x = pt[0]
    return hv


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class RewardFunction(ABC):
    """
    Evaluate a batch of generated candidates and return a scalar reward.

    Each candidate dict is expected to have at minimum:
        - improvement_factor  (child_property / parent_property)
        - similarity          (Tanimoto 0..1)
        - child_property      (raw predicted value)
        - parent_property     (raw parent value)
    """

    @abstractmethod
    def compute(self, candidates: List[Dict]) -> float:
        """Return scalar reward for the epoch."""
        ...

    def pareto_data(self, candidates: List[Dict]) -> Dict:
        """Return Pareto front metadata (shared logic, used by logger)."""
        if not candidates:
            return {"pareto_front": [], "hypervolume": 0.0, "n_evaluated": 0}
        improvements = np.array([c.get("improvement_factor", 0.0) for c in candidates], dtype=float)
        similarities = np.array([c.get("similarity", 0.0) for c in candidates], dtype=float)
        valid = np.isfinite(improvements) & np.isfinite(similarities)
        if valid.sum() < 1:
            return {"pareto_front": [], "hypervolume": 0.0, "n_evaluated": 0}
        pts = np.stack([improvements[valid], similarities[valid]], axis=1)
        mask = is_pareto_efficient(pts)
        front = pts[mask].tolist()
        hv = calculate_hypervolume(pts[mask])
        return {"pareto_front": front, "hypervolume": round(hv, 6), "n_evaluated": int(valid.sum())}


# ──────────────────────────────────────────────────────────────────────────────
# Concrete implementations
# ──────────────────────────────────────────────────────────────────────────────

class ParetoHypervolume(RewardFunction):
    """
    Multi-objective reward: hypervolume of the (improvement, similarity) Pareto front.
    Higher hypervolume = better coverage of the objective space.
    """

    def compute(self, candidates: List[Dict]) -> float:
        data = self.pareto_data(candidates)
        return data["hypervolume"]


class WeightedSum(RewardFunction):
    """
    Weighted combination: alpha * avg_improvement + (1-alpha) * avg_similarity.
    Good for single-objective guidance with a similarity regulariser.
    """

    def __init__(self, alpha: float = 0.7):
        assert 0 <= alpha <= 1, "alpha must be in [0, 1]"
        self.alpha = alpha

    def compute(self, candidates: List[Dict]) -> float:
        if not candidates:
            return 0.0
        improvements = [c.get("improvement_factor", 0.0) for c in candidates]
        similarities = [c.get("similarity", 0.0) for c in candidates]
        avg_imp = float(np.nanmean(improvements))
        avg_sim = float(np.nanmean(similarities))
        return self.alpha * avg_imp + (1 - self.alpha) * avg_sim


class PropertyOnly(RewardFunction):
    """
    Simple average improvement factor (ignores similarity).
    Useful for exploratory runs or as a baseline.
    """

    def compute(self, candidates: List[Dict]) -> float:
        if not candidates:
            return 0.0
        improvements = [c.get("improvement_factor", 0.0) for c in candidates]
        return float(np.nanmean(improvements))


class MaxPropertyOnly(RewardFunction):
    """Tracks the best individual improvement (not the average)."""

    def compute(self, candidates: List[Dict]) -> float:
        if not candidates:
            return 0.0
        return float(max(c.get("improvement_factor", 0.0) for c in candidates))


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

REWARD_REGISTRY: Dict[str, type] = {
    "pareto_hypervolume": ParetoHypervolume,
    "weighted_sum": WeightedSum,
    "property_only": PropertyOnly,
    "max_property_only": MaxPropertyOnly,
}


def get_reward_function(name: str, **kwargs) -> RewardFunction:
    """Instantiate a reward function by name, passing kwargs to constructor."""
    if name not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward function '{name}'. "
            f"Available: {list(REWARD_REGISTRY.keys())}"
        )
    return REWARD_REGISTRY[name](**kwargs)
