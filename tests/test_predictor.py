"""Tests for surrogate predictor interface and mock implementations."""
import pytest
from typing import List, Optional
from apo.surrogates.base import SurrogatePredictor
from apo.surrogates.registry import get_surrogate, list_surrogates, register


class MockSurrogate(SurrogatePredictor):
    """Mock surrogate that always returns 1.0."""
    property_name = "Mock Property"
    property_units = "units"
    maximize = True

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        return [1.0] * len(smiles_list)


class TestSurrogateBase:
    def test_predict_single(self):
        s = MockSurrogate()
        val = s.predict_single("C")
        assert val == 1.0

    def test_predict_empty(self):
        s = MockSurrogate()
        result = s.predict([])
        assert result == []

    def test_config_dict(self):
        s = MockSurrogate()
        cfg = s.config_dict()
        assert "property_name" in cfg
        assert "maximize" in cfg
        assert cfg["property_name"] == "Mock Property"


class TestRegistry:
    def test_list_surrogates_contains_gnn(self):
        surrogates = list_surrogates()
        assert "gnn_conductivity" in surrogates
        assert "gnn_li_diffusivity" in surrogates

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_surrogate("nonexistent_surrogate")

    def test_custom_register(self):
        @register("test_mock_surrogate")
        class _TestSurrogate(SurrogatePredictor):
            property_name = "Test"
            property_units = ""
            maximize = True
            def predict(self, smiles_list):
                return [0.5] * len(smiles_list)

        s = get_surrogate("test_mock_surrogate")
        assert s.predict(["C"]) == [0.5]


class TestGNNPredictor:
    def test_instantiation(self):
        """GNNPredictor should instantiate without errors (model file may be absent)."""
        from apo.surrogates.gnn_predictor import GNNPredictor
        pred = GNNPredictor("conductivity", model_base_path="/tmp/nonexistent")
        assert pred.property_name == "Conductivity"
        assert pred.maximize is True

    def test_predict_empty(self):
        from apo.surrogates.gnn_predictor import GNNPredictor
        pred = GNNPredictor("conductivity", model_base_path="/tmp/nonexistent")
        result = pred.predict([])
        assert result == []

    def test_predict_missing_model_returns_none(self):
        """When model file is absent, all predictions should be None."""
        from apo.surrogates.gnn_predictor import GNNPredictor
        pred = GNNPredictor("conductivity", model_base_path="/tmp/nonexistent")
        result = pred.predict(["CC(CO[Cu])CSCCOC(=O)[Au]"])
        assert result == [None]
