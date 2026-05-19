"""Regression tests for agentic surrogate predictor calls."""
from typing import List, Optional

from apo.agents.base import Observation, Tool
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


class StrictSurrogate(SurrogatePredictor):
    property_name = "StrictProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise TypeError("predict expects a list of SMILES, not a string")

        self.calls.append(list(smiles_list))
        values = {
            "CC": 2.0,
            "CCO": 3.0,
            "CCC": None,
        }
        return [values.get(smi, 1.0) for smi in smiles_list]


class FakeValidator(Tool):
    @property
    def name(self) -> str:
        return "validate_smiles"

    @property
    def description(self) -> str:
        return "fake validator"

    @property
    def parameters(self) -> dict:
        return {}

    def execute(self, smiles_list):
        return Observation(
            success=True,
            result=[{"smiles": smi, "valid": True} for smi in smiles_list],
        )


class FakeSimilarity(Tool):
    @property
    def name(self) -> str:
        return "calculate_similarity"

    @property
    def description(self) -> str:
        return "fake similarity"

    @property
    def parameters(self) -> dict:
        return {}

    def execute(self, smiles1, smiles2):
        return Observation(success=True, result={"similarity": 0.75})


def test_property_predictor_tool_uses_single_prediction_wrapper():
    surrogate = StrictSurrogate()
    tool = PropertyPredictorTool(surrogate, "StrictProp")

    obs = tool.execute("CCO")

    assert obs.success is True
    assert obs.result == {"StrictProp": 3.0, "smiles": "CCO"}
    assert surrogate.calls == [["CCO"]]


def test_batch_property_predictor_tool_calls_predict_with_full_list():
    surrogate = StrictSurrogate()
    tool = BatchPropertyPredictorTool(surrogate, "StrictProp")

    obs = tool.execute(["CC", "CCO", "CCC"])

    assert obs.success is True
    assert obs.result == [
        {"smiles": "CC", "property": 2.0, "valid": True},
        {"smiles": "CCO", "property": 3.0, "valid": True},
        {"smiles": "CCC", "property": None, "valid": False},
    ]
    assert surrogate.calls == [["CC", "CCO", "CCC"]]


def test_worker_validation_uses_scalar_prediction_wrapper():
    surrogate = StrictSurrogate()
    ctx = TaskContext(
        property_name="StrictProp",
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
    worker.tools = [FakeValidator(), FakeSimilarity()]

    candidates = [{
        "parent_smiles": "CC",
        "child_smiles": "CCO",
        "explanation": "add oxygen",
    }]

    validated = worker._validate_candidates(candidates)

    assert validated[0]["valid"] is True
    assert validated[0]["parent_property"] == 2.0
    assert validated[0]["child_property"] == 3.0
    assert validated[0]["improvement_factor"] == 1.5
    assert validated[0]["similarity"] == 0.75
    assert surrogate.calls == [["CC"], ["CCO"]]
