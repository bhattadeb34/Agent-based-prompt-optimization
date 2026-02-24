"""Tests for reward functions."""
import numpy as np
import pytest
from apo.core.reward import (
    ParetoHypervolume,
    WeightedSum,
    PropertyOnly,
    MaxPropertyOnly,
    get_reward_function,
)


def make_candidate(improvement, similarity, valid=True):
    return {
        "improvement_factor": improvement,
        "similarity": similarity,
        "valid": valid,
    }


class TestParetoHypervolume:
    def test_empty(self):
        fn = ParetoHypervolume()
        assert fn.compute([]) == 0.0

    def test_single_candidate(self):
        fn = ParetoHypervolume()
        c = [make_candidate(2.0, 0.8)]
        score = fn.compute(c)
        assert score > 0.0

    def test_dominated_candidates_less_hv(self):
        fn = ParetoHypervolume()
        good = [make_candidate(2.0, 0.8), make_candidate(1.5, 0.9)]
        bad = [make_candidate(0.5, 0.2), make_candidate(0.3, 0.1)]
        assert fn.compute(good) > fn.compute(bad)

    def test_pareto_data_has_hypervolume(self):
        fn = ParetoHypervolume()
        candidates = [make_candidate(2.0, 0.8), make_candidate(1.2, 0.95)]
        data = fn.pareto_data(candidates)
        assert "hypervolume" in data
        assert data["hypervolume"] >= 0.0
        assert "pareto_front" in data
        assert "n_evaluated" in data


class TestWeightedSum:
    def test_basic(self):
        fn = WeightedSum(alpha=0.5)
        candidates = [make_candidate(2.0, 0.8)]
        score = fn.compute(candidates)
        assert abs(score - (0.5 * 2.0 + 0.5 * 0.8)) < 1e-6

    def test_alpha_bounds(self):
        with pytest.raises(AssertionError):
            WeightedSum(alpha=1.5)

    def test_empty(self):
        fn = WeightedSum()
        assert fn.compute([]) == 0.0


class TestPropertyOnly:
    def test_average(self):
        fn = PropertyOnly()
        candidates = [make_candidate(2.0, 0.5), make_candidate(4.0, 0.9)]
        assert abs(fn.compute(candidates) - 3.0) < 1e-6

    def test_empty(self):
        fn = PropertyOnly()
        assert fn.compute([]) == 0.0


class TestMaxPropertyOnly:
    def test_max(self):
        fn = MaxPropertyOnly()
        candidates = [make_candidate(1.0, 0.5), make_candidate(5.0, 0.2)]
        assert abs(fn.compute(candidates) - 5.0) < 1e-6


class TestRegistry:
    def test_get_pareto(self):
        fn = get_reward_function("pareto_hypervolume")
        assert isinstance(fn, ParetoHypervolume)

    def test_get_weighted_sum_with_kwargs(self):
        fn = get_reward_function("weighted_sum", alpha=0.6)
        assert isinstance(fn, WeightedSum)
        assert abs(fn.alpha - 0.6) < 1e-6

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_reward_function("nonexistent_reward")
