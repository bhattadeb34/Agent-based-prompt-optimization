"""Regression tests for agentic surrogate prediction calls."""
from typing import List, Optional

from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


class StrictListSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise AssertionError("predict() requires a list of SMILES, not a string")
        self.calls.append(list(smiles_list))
        return [float(len(smiles)) for smiles in smiles_list]


def test_property_predictor_tool_uses_single_prediction_helper():
    surrogate = StrictListSurrogate()
    tool = PropertyPredictorTool(surrogate, "TestProp")

    obs = tool.execute("CC")

    assert obs.success is True
    assert obs.result == {"TestProp": 2.0, "smiles": "CC"}
    assert surrogate.calls == [["CC"]]


def test_batch_property_predictor_passes_a_smiles_list():
    surrogate = StrictListSurrogate()
    tool = BatchPropertyPredictorTool(surrogate, "TestProp")

    obs = tool.execute(["CC", "CCC"])

    assert obs.success is True
    assert obs.result == [
        {"smiles": "CC", "property": 2.0, "valid": True},
        {"smiles": "CCC", "property": 3.0, "valid": True},
    ]
    assert surrogate.calls == [["CC", "CCC"]]


def test_worker_validation_stores_scalar_parent_and_child_properties():
    surrogate = StrictListSurrogate()
    ctx = TaskContext(
        property_name="TestProp",
        property_units="units",
        maximize=True,
        molecule_type="organic compound",
    )
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=ctx,
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": "CC",
            "child_smiles": "CCC",
            "explanation": "extend the chain",
        }
    ])

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["valid"] is True
    assert candidate["parent_property"] == 2.0
    assert candidate["child_property"] == 3.0
    assert candidate["improvement_factor"] == 1.5
    assert surrogate.calls == [["CC"], ["CCC"]]
