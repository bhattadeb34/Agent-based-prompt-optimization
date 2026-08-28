from typing import List, Optional

from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage, aggregate_usage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext
from apo.utils.smiles_utils import canonicalize


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictListSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise TypeError("predict expects a list, not a string")
        self.calls.append(list(smiles_list))
        values = {
            VALID_PARENT: 1.0,
            canonicalize(VALID_PARENT): 1.0,
            VALID_CHILD: 2.0,
            canonicalize(VALID_CHILD): 2.0,
        }
        return [values.get(smiles, 1.5) for smiles in smiles_list]


def polymer_context() -> TaskContext:
    return TaskContext(
        property_name="TestProp",
        property_units="units",
        maximize=True,
        molecule_type="polymer",
        domain_context="[Cu] and [Au] are backbone markers.",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_agentic_property_tools_use_surrogate_list_api():
    surrogate = StrictListSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "TestProp").execute(VALID_PARENT)
    batch_obs = BatchPropertyPredictorTool(surrogate, "TestProp").execute([VALID_PARENT, VALID_CHILD])

    assert single_obs.success
    assert single_obs.result["TestProp"] == 1.0
    assert batch_obs.success
    assert [row["property"] for row in batch_obs.result] == [1.0, 2.0]
    assert surrogate.calls == [[VALID_PARENT], [VALID_PARENT, VALID_CHILD]]


def test_worker_validation_uses_list_api_and_enforces_task_markers():
    surrogate = StrictListSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {"parent_smiles": VALID_PARENT, "child_smiles": VALID_CHILD, "explanation": "valid"},
        {"parent_smiles": VALID_PARENT, "child_smiles": "CCO", "explanation": "missing markers"},
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["child_property"] == 2.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert candidates[1]["valid"] is False
    assert "Missing required marker" in candidates[1]["invalid_reason"]
    assert all(not isinstance(call, str) for call in surrogate.calls)


def test_worker_parses_generated_molecules_mapping_payload():
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
        surrogate=StrictListSurrogate(),
        parent_cache={},
    )

    parsed = worker._parse_candidate_payload({
        "generated_molecules": {
            VALID_PARENT: {
                "smiles": [VALID_CHILD],
                "reasoning": ["added ether"],
            }
        }
    })

    assert parsed == [{
        "parent_smiles": VALID_PARENT,
        "child_smiles": VALID_CHILD,
        "explanation": "added ether",
    }]


def test_aggregate_usage_accepts_agent_usage_summaries():
    raw = LLMUsage("worker-model", prompt_tokens=10, completion_tokens=5, latency_s=0.25)
    critic_summary = {
        "total_calls": 2,
        "total_prompt_tokens": 20,
        "total_completion_tokens": 8,
        "total_tokens": 28,
        "total_latency_s": 0.5,
        "by_model": {"critic-model": {"calls": 2, "tokens": 28}},
    }

    result = aggregate_usage([raw, critic_summary])

    assert result["total_calls"] == 3
    assert result["total_tokens"] == 43
    assert result["by_model"]["worker-model"] == {"calls": 1, "tokens": 15}
    assert result["by_model"]["critic-model"] == {"calls": 2, "tokens": 28}


def test_meta_recent_strategy_format_uses_history_api():
    history = PromptStateHistory()
    history.add(PromptState.seed("seed strategy"))
    history.add(PromptState(strategy_text="next strategy", version=1, rationale="test"))

    meta = MetaAgent(model="test-model", api_keys={}, task_context=polymer_context())
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v0: seed strategy" in formatted
    assert "v1: next strategy" in formatted
