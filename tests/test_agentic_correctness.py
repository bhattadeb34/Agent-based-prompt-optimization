from typing import List, Optional

from apo.agentic_engine import _merge_usage_summary
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage, aggregate_usage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictListSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise TypeError("predict expects a list of SMILES")
        self.calls.append(list(smiles_list))
        return [float(len(smi)) for smi in smiles_list]


def polymer_context(maximize: bool = True) -> TaskContext:
    return TaskContext(
        property_name="TestProp",
        property_units="units",
        maximize=maximize,
        molecule_type="polymer",
        domain_context="[Cu] and [Au] are backbone markers.",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_agentic_worker_uses_single_item_surrogate_api_for_validation():
    surrogate = StrictListSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
        surrogate=surrogate,
        parent_cache={},
    )

    [candidate] = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "explanation": "add ether oxygen",
        }
    ])

    assert candidate["valid"] is True
    assert isinstance(candidate["parent_property"], float)
    assert isinstance(candidate["child_property"], float)
    assert candidate["improvement_factor"] > 0
    assert all(isinstance(call, list) for call in surrogate.calls)
    assert all(len(call) == 1 for call in surrogate.calls)


def test_agentic_worker_rejects_missing_required_markers():
    surrogate = StrictListSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
        surrogate=surrogate,
        parent_cache={},
    )

    [candidate] = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": "CCO",
            "explanation": "invalid for polymer task",
        }
    ])

    assert candidate["valid"] is False
    assert "Missing required marker" in candidate["invalid_reason"]


def test_property_tools_respect_list_surrogate_contract():
    surrogate = StrictListSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "TestProp").execute(VALID_CHILD)
    batch_obs = BatchPropertyPredictorTool(surrogate, "TestProp").execute([VALID_PARENT, VALID_CHILD])

    assert single_obs.success is True
    assert batch_obs.success is True
    assert [len(call) for call in surrogate.calls] == [1, 2]


def test_meta_formats_recent_strategies_with_history_api():
    history = PromptStateHistory()
    for version in range(4):
        history.add(PromptState(strategy_text=f"strategy {version}", version=version))

    meta = MetaAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
    )
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted


def test_agentic_usage_summaries_merge_without_polluting_llm_usage_list():
    worker_usage = LLMUsage("worker-model", 10, 5, 0.25)
    base = aggregate_usage([worker_usage])
    critic_summary = {
        "total_calls": 2,
        "total_prompt_tokens": 20,
        "total_completion_tokens": 10,
        "total_tokens": 30,
        "total_latency_s": 0.75,
        "by_model": {"critic-model": {"calls": 2, "tokens": 30}},
    }

    merged = _merge_usage_summary(base, critic_summary)

    assert merged["total_calls"] == 3
    assert merged["total_tokens"] == 45
    assert merged["by_model"]["worker-model"]["calls"] == 1
    assert merged["by_model"]["critic-model"]["calls"] == 2
