"""Regression tests for high-impact agentic workflow correctness."""
from typing import List, Optional

from apo.agentic_engine import _merge_usage_summaries
from apo.agents.critic import CriticAgent
from apo.agents.meta import MetaAgent
from apo.agents.worker import WorkerAgent
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


class StrictSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self, values=None):
        self.values = values or {}
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        assert isinstance(smiles_list, list), "surrogate.predict must receive a list"
        self.calls.append(list(smiles_list))
        return [self.values.get(smi, 1.0) for smi in smiles_list]


def test_worker_uses_scalar_predictor_wrapper_and_scores_valid_candidates():
    ctx = TaskContext(
        property_name="TestProp",
        property_units="units",
        maximize=True,
        molecule_type="organic compound",
    )
    surrogate = StrictSurrogate({"CC": 2.0, "CCC": 3.0})
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=ctx,
        surrogate=surrogate,
        parent_cache={},
    )

    [candidate] = worker._validate_candidates([
        {"parent_smiles": "CC", "child_smiles": "CCC", "explanation": "extend chain"}
    ])

    assert candidate["valid"] is True
    assert candidate["parent_property"] == 2.0
    assert candidate["child_property"] == 3.0
    assert candidate["improvement_factor"] == 1.5
    assert ["CC"] in surrogate.calls
    assert ["CCC"] in surrogate.calls


def test_worker_enforces_required_markers_after_rdkit_validation():
    ctx = TaskContext(
        property_name="TestProp",
        property_units="units",
        maximize=True,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=ctx,
        surrogate=StrictSurrogate(),
        parent_cache={},
    )

    [candidate] = worker._validate_candidates([
        {
            "parent_smiles": "CC(CO[Cu])CSCCOC(=O)[Au]",
            "child_smiles": "CCO",
            "explanation": "missing polymer markers",
        }
    ])

    assert candidate["valid"] is False
    assert candidate["invalid_reason"] == "Missing required marker: [Cu]"


def test_critic_sets_reward_on_evaluated_state(monkeypatch):
    ctx = TaskContext(property_name="TestProp", property_units="units")
    critic = CriticAgent(
        model="test-model",
        api_keys={},
        task_context=ctx,
        reward_fn=ParetoHypervolume(),
    )
    current = PromptState.seed("seed strategy")
    history = PromptStateHistory()
    history.add(current)

    def fake_run(initial_state):
        critic.new_state = PromptState(
            strategy_text="next strategy",
            version=current.version + 1,
            parent_version=current.version,
        )
        return (critic.new_state, critic.analysis), []

    monkeypatch.setattr(critic, "run", fake_run)
    new_state, _, usage = critic.refine(
        candidates=[{"valid": True, "improvement_factor": 2.0, "similarity": 0.5}],
        current_state=current,
        history=history,
    )

    assert current.score == 1.0
    assert new_state.score is None
    assert usage["total_calls"] == 0


def test_meta_formats_recent_strategies_without_history_api_crash():
    ctx = TaskContext(property_name="TestProp", property_units="units")
    meta = MetaAgent(model="test-model", api_keys={}, task_context=ctx)
    history = PromptStateHistory()
    history.add(PromptState.seed("seed"))
    history.add(PromptState(strategy_text="strategy one", version=1))
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v0: seed" in formatted
    assert "v1: strategy one" in formatted


def test_agentic_usage_summary_merges_dicts_without_raw_usage_objects():
    merged = _merge_usage_summaries(
        {
            "total_calls": 1,
            "total_prompt_tokens": 10,
            "total_completion_tokens": 5,
            "total_tokens": 15,
            "total_latency_s": 0.2,
            "by_model": {"worker": {"calls": 1, "tokens": 15}},
        },
        {
            "total_calls": 2,
            "total_prompt_tokens": 7,
            "total_completion_tokens": 3,
            "total_tokens": 10,
            "total_latency_s": 0.4,
            "by_model": {"critic": {"calls": 2, "tokens": 10}},
        },
    )

    assert merged["total_calls"] == 3
    assert merged["total_tokens"] == 25
    assert merged["by_model"]["worker"]["calls"] == 1
    assert merged["by_model"]["critic"]["tokens"] == 10
