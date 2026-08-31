import json
from typing import List, Optional
from unittest.mock import patch

from apo.agentic_engine import _merge_usage_summaries
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage, aggregate_usage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"
CANONICAL_PARENT = "CC(C[O][Cu])CSCCO[C](=O)[Au]"
CANONICAL_CHILD = "CC(C[O][Cu])COCCO[C](=O)[Au]"


class StrictSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []
        self.values = {
            VALID_PARENT: 2.0,
            VALID_CHILD: 4.0,
            CANONICAL_PARENT: 2.0,
            CANONICAL_CHILD: 4.0,
        }

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        assert isinstance(smiles_list, list), "predict() requires a list of SMILES"
        self.calls.append(list(smiles_list))
        return [self.values.get(smiles, 1.0) for smiles in smiles_list]


def polymer_context(maximize=True):
    return TaskContext(
        property_name="TestProp",
        property_units="units",
        maximize=maximize,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_agentic_property_tools_use_surrogate_list_contract():
    surrogate = StrictSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "TestProp").execute(VALID_PARENT)
    batch_obs = BatchPropertyPredictorTool(surrogate, "TestProp").execute([VALID_PARENT, VALID_CHILD])

    assert single_obs.success
    assert batch_obs.success
    assert surrogate.calls == [[VALID_PARENT], [VALID_PARENT, VALID_CHILD]]


def test_worker_validates_and_scores_with_batch_predictions():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "explanation": "replace sulfur with oxygen",
        }
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 2.0
    assert candidates[0]["child_property"] == 4.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_worker_rejects_candidates_missing_required_markers():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
        surrogate=surrogate,
        parent_cache={VALID_PARENT: 2.0},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": "CCO",
            "explanation": "drops polymer markers",
        }
    ])

    assert candidates[0]["valid"] is False
    assert "Missing required marker" in candidates[0]["invalid_reason"]


def test_worker_parses_fenced_generated_molecules_mapping():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
        surrogate=surrogate,
        parent_cache={},
    )
    response = "```json\n" + json.dumps({
        "generated_molecules": {
            VALID_PARENT: {
                "smiles": [VALID_CHILD],
                "reasoning": ["added ether oxygen"],
            }
        }
    }) + "\n```"

    with patch("apo.agents.worker.call_llm", return_value=(response, LLMUsage("m", 1, 1, 0.1))):
        candidates = worker._call_llm_for_generation()

    assert candidates == [{
        "parent_smiles": VALID_PARENT,
        "child_smiles": VALID_CHILD,
        "explanation": "added ether oxygen",
    }]


def test_meta_agent_formats_recent_history_without_missing_all_method():
    history = PromptStateHistory()
    history.add(PromptState.seed("seed"))
    history.add(PromptState(strategy_text="next", version=1, rationale="r"))

    meta = MetaAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
    )
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v0:" in formatted
    assert "v1:" in formatted


def test_agentic_engine_merges_raw_and_aggregate_usage_summaries():
    raw = aggregate_usage([LLMUsage("worker-model", 10, 5, 0.5)])
    critic = {
        "total_calls": 2,
        "total_prompt_tokens": 20,
        "total_completion_tokens": 8,
        "total_tokens": 28,
        "total_latency_s": 1.0,
        "by_model": {"critic-model": {"calls": 2, "tokens": 28}},
    }

    merged = _merge_usage_summaries(raw, critic)

    assert merged["total_calls"] == 3
    assert merged["total_tokens"] == 43
    assert merged["by_model"]["worker-model"]["calls"] == 1
    assert merged["by_model"]["critic-model"]["tokens"] == 28
