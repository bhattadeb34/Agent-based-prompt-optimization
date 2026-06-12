"""Regression tests for agentic surrogate predictor API usage."""
from typing import List, Optional

from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


class StrictListSurrogate(SurrogatePredictor):
    property_name = "Length"
    property_units = "chars"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise TypeError("predict() expects a list of SMILES, not a string")
        self.calls.append(list(smiles_list))
        return [float(len(smiles)) for smiles in smiles_list]


GENERIC_CTX = TaskContext(
    property_name="Length",
    property_units="chars",
    maximize=True,
    molecule_type="organic compound",
)


def test_worker_validation_uses_predict_single_for_scalar_properties():
    surrogate = StrictListSurrogate()
    worker = WorkerAgent(
        model="mock/model",
        api_keys={},
        task_context=GENERIC_CTX,
        surrogate=surrogate,
        parent_cache={},
    )

    validated = worker._validate_candidates([
        {
            "parent_smiles": "CC",
            "child_smiles": "CCO",
            "explanation": "Add an oxygen.",
        }
    ])

    assert validated[0]["valid"] is True
    assert validated[0]["parent_property"] == 2.0
    assert validated[0]["child_property"] == 3.0
    assert validated[0]["improvement_factor"] == 1.5
    assert surrogate.calls == [["CC"], ["CCO"]]


def test_property_tools_respect_scalar_and_batch_predictor_contracts():
    surrogate = StrictListSurrogate()

    scalar_obs = PropertyPredictorTool(surrogate, "Length").execute("CCO")
    assert scalar_obs.success is True
    assert scalar_obs.result["Length"] == 3.0
    assert surrogate.calls == [["CCO"]]

    batch_obs = BatchPropertyPredictorTool(surrogate, "Length").execute(["CC", "CCO"])
    assert batch_obs.success is True
    assert [item["property"] for item in batch_obs.result] == [2.0, 3.0]
    assert surrogate.calls == [["CCO"], ["CC", "CCO"]]
