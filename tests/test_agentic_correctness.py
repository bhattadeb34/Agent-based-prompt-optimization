"""Regression tests for critical agentic workflow correctness paths."""
from typing import List, Optional

from apo.agentic_engine import _merge_usage_summary
from apo.agents.critic import CriticAgent
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictListSurrogate(SurrogatePredictor):
    property_name = "StrictProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise TypeError("predict expects a list, not a scalar string")
        self.calls.append(list(smiles_list))
        values = []
        for smiles in smiles_list:
            values.append(5.0 if "COCC" in smiles else 10.0)
        return values


def polymer_ctx(maximize=True):
    return TaskContext(
        property_name="StrictProp",
        property_units="units",
        maximize=maximize,
        molecule_type="polymer",
        domain_context="[Cu] and [Au] are required repeat-unit markers.",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_agentic_property_tools_use_surrogate_list_contract_once():
    surrogate = StrictListSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "StrictProp").execute(VALID_PARENT)
    batch_obs = BatchPropertyPredictorTool(surrogate, "StrictProp").execute([VALID_PARENT, VALID_CHILD])

    assert single_obs.success is True
    assert batch_obs.success is True
    assert surrogate.calls == [[VALID_PARENT], [VALID_PARENT, VALID_CHILD]]


def test_worker_validation_rejects_missing_required_markers_without_scalar_predict():
    surrogate = StrictListSurrogate()
    worker = WorkerAgent(
        model="mock-model",
        api_keys={},
        task_context=polymer_ctx(),
        surrogate=surrogate,
        parent_cache={},
    )

    [candidate] = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": "CCO",
            "explanation": "drops polymer markers",
        }
    ])

    assert candidate["valid"] is False
    assert "Missing required marker" in candidate["invalid_reason"]
    assert all(not isinstance(call, str) for call in surrogate.calls)


def test_worker_validation_scores_minimization_with_canonical_child():
    surrogate = StrictListSurrogate()
    worker = WorkerAgent(
        model="mock-model",
        api_keys={},
        task_context=polymer_ctx(maximize=False),
        surrogate=surrogate,
        parent_cache={},
    )

    [candidate] = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "explanation": "lower is better",
        }
    ])

    assert candidate["valid"] is True
    assert candidate["parent_property"] == 10.0
    assert candidate["child_property"] == 5.0
    assert candidate["improvement_factor"] == 2.0
    assert candidate["similarity"] > 0


def test_worker_parses_generated_molecules_mapping_schema():
    entries = WorkerAgent._generated_molecules_to_entries({
        VALID_PARENT: {
            "smiles": [VALID_CHILD],
            "reasoning": ["adds ether oxygen"],
        }
    })

    assert entries == [
        {
            "parent": VALID_PARENT,
            "candidates": [{"smiles": VALID_CHILD, "explanation": "adds ether oxygen"}],
        }
    ]


def test_critic_scores_evaluated_current_state(monkeypatch):
    critic = CriticAgent(
        model="mock-model",
        api_keys={},
        task_context=polymer_ctx(),
        reward_fn=ParetoHypervolume(),
    )
    current = PromptState.seed("seed")
    history = PromptStateHistory()
    history.add(current)

    def fake_run(self, initial_state):
        self.new_state = PromptState(
            strategy_text="next",
            version=current.version + 1,
            parent_version=current.version,
        )
        return (self.new_state, []), []

    monkeypatch.setattr(CriticAgent, "run", fake_run)
    monkeypatch.setattr(CriticAgent, "_save_trace_to_disk", lambda self: {})

    candidates = [{
        "valid": True,
        "improvement_factor": 2.0,
        "similarity": 0.5,
        "child_smiles": VALID_CHILD,
    }]
    new_state, _, _ = critic.refine(candidates, current, history)

    assert new_state.version == 1
    assert current.score == 1.0


def test_meta_formats_recent_strategies_without_missing_history_method():
    history = PromptStateHistory()
    for i in range(4):
        history.add(PromptState(strategy_text=f"strategy {i}", version=i))

    meta = MetaAgent(model="mock-model", api_keys={}, task_context=polymer_ctx())
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted
    assert "v0: strategy 0" not in formatted


def test_merge_usage_summary_preserves_model_breakdown():
    base = {
        "total_calls": 1,
        "total_tokens": 10,
        "by_model": {"worker": {"calls": 1, "tokens": 10}},
    }
    extra = {
        "total_calls": 2,
        "total_tokens": 25,
        "by_model": {"critic": {"calls": 2, "tokens": 25}},
    }

    merged = _merge_usage_summary(base, extra)

    assert merged["total_calls"] == 3
    assert merged["total_tokens"] == 35
    assert merged["by_model"]["worker"] == {"calls": 1, "tokens": 10}
    assert merged["by_model"]["critic"] == {"calls": 2, "tokens": 25}
