from typing import List, Optional

from apo.agentic_engine import _merge_usage_summaries
from apo.agents.meta import MetaAgent
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage, aggregate_usage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext
from apo.utils.smiles_utils import canonicalize


PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictBatchSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []
        self.values = {
            PARENT: 10.0,
            canonicalize(PARENT) or PARENT: 10.0,
            CHILD: 20.0,
            canonicalize(CHILD) or CHILD: 20.0,
        }

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        assert isinstance(smiles_list, list), "predict must receive a list"
        self.calls.append(list(smiles_list))
        return [self.values.get(s) for s in smiles_list]


def _ctx(maximize=True):
    return TaskContext(
        property_name="TestProp",
        property_units="units",
        maximize=maximize,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_worker_parses_legacy_generated_molecules_mapping():
    text = """
    ```json
    {
      "generated_molecules": {
        "CC(CO[Cu])CSCCOC(=O)[Au]": {
          "smiles": ["CC(CO[Cu])COCCOC(=O)[Au]"],
          "reasoning": ["add ether oxygen"]
        }
      }
    }
    ```
    """

    parsed = WorkerAgent._parse_generation_response(text)

    assert parsed == [{
        "parent_smiles": PARENT,
        "child_smiles": CHILD,
        "explanation": "add ether oxygen",
    }]


def test_worker_validation_uses_batch_surrogate_and_task_context():
    surrogate = StrictBatchSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=_ctx(),
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([{
        "parent_smiles": PARENT,
        "child_smiles": CHILD,
        "explanation": "add ether oxygen",
    }])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 10.0
    assert candidates[0]["child_property"] == 20.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert len(surrogate.calls) == 2
    assert surrogate.calls[0] == [PARENT]
    assert surrogate.calls[1] == [canonicalize(CHILD)]


def test_worker_rejects_candidates_missing_required_markers():
    surrogate = StrictBatchSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=_ctx(),
        surrogate=surrogate,
        parent_cache={canonicalize(PARENT) or PARENT: 10.0},
    )

    candidates = worker._validate_candidates([{
        "parent_smiles": PARENT,
        "child_smiles": "CCO",
        "explanation": "drops markers",
    }])

    assert candidates[0]["valid"] is False
    assert "Missing required marker" in candidates[0]["invalid_reason"]
    assert all("CCO" not in call for call in surrogate.calls)


def test_merge_usage_summaries_accepts_aggregated_dicts():
    worker_usage = aggregate_usage([LLMUsage("worker", 3, 4, 0.5)])
    critic_usage = aggregate_usage([LLMUsage("critic", 5, 6, 0.7)])

    merged = _merge_usage_summaries(worker_usage, critic_usage)

    assert merged["total_calls"] == 2
    assert merged["total_tokens"] == 18
    assert merged["by_model"]["worker"]["calls"] == 1
    assert merged["by_model"]["critic"]["tokens"] == 11


def test_meta_agent_formats_recent_history_with_prompt_state_api():
    history = PromptStateHistory()
    for i in range(4):
        history.add(PromptState(strategy_text=f"strategy {i}", version=i))

    meta = MetaAgent(
        model="test-model",
        api_keys={},
        task_context=_ctx(),
    )
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted
    assert "v0: strategy 0" not in formatted
