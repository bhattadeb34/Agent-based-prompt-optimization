import json
from typing import List, Optional
from unittest.mock import patch

from apo.agentic_engine import _merge_usage_summaries
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


class StrictSurrogate(SurrogatePredictor):
    property_name = "Strict Property"
    property_units = "units"
    maximize = True

    def __init__(self, values):
        self.values = values
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise AssertionError("predict() received a string instead of a list")
        self.calls.append(list(smiles_list))
        return [self.values.get(smiles) for smiles in smiles_list]


POLYMER_CTX = TaskContext(
    property_name="Strict Property",
    property_units="units",
    maximize=True,
    molecule_type="polymer electrolyte",
    smiles_markers=["[Cu]", "[Au]"],
    similarity_on_repeat_unit=True,
)


MINIMIZE_CTX = TaskContext(
    property_name="Strict Property",
    property_units="units",
    maximize=False,
    molecule_type="organic compound",
    smiles_markers=[],
)


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


def test_agentic_property_tools_use_surrogate_list_api():
    surrogate = StrictSurrogate({"CC": 1.0, "CCC": 2.0})

    single = PropertyPredictorTool(surrogate, "Strict Property").execute("CC")
    batch = BatchPropertyPredictorTool(surrogate, "Strict Property").execute(["CC", "CCC"])

    assert single.success is True
    assert single.result["Strict Property"] == 1.0
    assert batch.success is True
    assert [row["property"] for row in batch.result] == [1.0, 2.0]
    assert surrogate.calls == [["CC"], ["CC", "CCC"]]


def test_worker_validation_uses_predict_single_and_task_marker_rules():
    surrogate = StrictSurrogate({VALID_PARENT: 1.0, VALID_CHILD: 2.0})
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "explanation": "valid edit",
        },
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": "CCO",
            "explanation": "missing polymer markers",
        },
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 1.0
    assert candidates[0]["child_property"] == 2.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert candidates[1]["valid"] is False
    assert "Missing required marker" in candidates[1]["invalid_reason"]
    assert all(not isinstance(call, str) for call in surrogate.calls)


def test_worker_improvement_respects_minimization_direction():
    surrogate = StrictSurrogate({"CC": 10.0, "C": 5.0})
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=MINIMIZE_CTX,
        surrogate=surrogate,
        parent_cache={},
    )

    [candidate] = worker._validate_candidates([
        {"parent_smiles": "CC", "child_smiles": "C", "explanation": "lower is better"}
    ])

    assert candidate["valid"] is True
    assert candidate["improvement_factor"] == 2.0


def test_worker_parses_fenced_generated_molecules_mapping():
    surrogate = StrictSurrogate({})
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        surrogate=surrogate,
        parent_cache={},
    )
    payload = {
        "generated_molecules": {
            VALID_PARENT: {
                "smiles": [VALID_CHILD],
                "reasoning": ["added ether oxygen"],
            }
        }
    }
    usage = LLMUsage("test-model", 1, 1, 0.1)

    with patch("apo.agents.worker.call_llm", return_value=(f"```json\n{json.dumps(payload)}\n```", usage)):
        candidates = worker._call_llm_for_generation()

    assert candidates == [{
        "parent_smiles": VALID_PARENT,
        "child_smiles": VALID_CHILD,
        "explanation": "added ether oxygen",
    }]


def test_agentic_usage_summaries_merge_without_llmusage_objects():
    left = {
        "total_calls": 1,
        "total_prompt_tokens": 2,
        "total_completion_tokens": 3,
        "total_tokens": 5,
        "total_latency_s": 0.5,
        "by_model": {"worker": {"calls": 1, "tokens": 5}},
    }
    right = {
        "total_calls": 2,
        "total_prompt_tokens": 7,
        "total_completion_tokens": 11,
        "total_tokens": 18,
        "total_latency_s": 1.25,
        "by_model": {"critic": {"calls": 2, "tokens": 18}},
    }

    merged = _merge_usage_summaries(left, right)

    assert merged["total_calls"] == 3
    assert merged["total_prompt_tokens"] == 9
    assert merged["total_completion_tokens"] == 14
    assert merged["total_tokens"] == 23
    assert merged["total_latency_s"] == 1.75
    assert merged["by_model"] == {
        "worker": {"calls": 1, "tokens": 5},
        "critic": {"calls": 2, "tokens": 18},
    }
