"""Regression tests for agentic surrogate prediction calls."""
from typing import List, Optional

from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


class StrictSurrogate(SurrogatePredictor):
    """Surrogate that rejects accidental string-as-batch calls."""

    property_name = "Mock Property"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise AssertionError("predict() must receive a list, not a single SMILES string")
        self.calls.append(list(smiles_list))
        values = {"CC": 2.0, "CCC": 4.0, "CCO": 3.0}
        return [values.get(smi) for smi in smiles_list]


GENERIC_CTX = TaskContext(
    property_name="Mock Property",
    property_units="units",
    maximize=True,
    molecule_type="organic compound",
    smiles_markers=[],
)


def test_worker_validation_uses_scalar_predict_helper_for_properties():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=GENERIC_CTX,
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": "CC",
            "child_smiles": "CCC",
            "explanation": "extend chain",
        }
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 2.0
    assert candidates[0]["child_property"] == 4.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert surrogate.calls == [["CC"], ["CCC"]]


def test_property_predictor_tools_use_scalar_predict_helper():
    surrogate = StrictSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "Mock Property").execute("CC")
    batch_obs = BatchPropertyPredictorTool(surrogate, "Mock Property").execute(["CC", "CCO"])

    assert single_obs.success is True
    assert single_obs.result["Mock Property"] == 2.0
    assert batch_obs.success is True
    assert [item["property"] for item in batch_obs.result] == [2.0, 3.0]
    assert surrogate.calls == [["CC"], ["CC"], ["CCO"]]
