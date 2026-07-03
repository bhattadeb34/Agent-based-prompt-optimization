from typing import List, Optional

from apo.agentic_engine import _empty_usage_summary, _merge_usage_summaries
from apo.agents.critic import CriticAgent
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage, aggregate_usage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictListSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self, values=None):
        self.values = values or {}
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise TypeError("predict expects a list of SMILES")
        self.calls.append(list(smiles_list))
        return [self.values.get(smiles, 1.0) for smiles in smiles_list]


def polymer_ctx(maximize=True):
    return TaskContext(
        property_name="TestProp",
        property_units="units",
        maximize=maximize,
        molecule_type="polymer",
        domain_context="[Cu] and [Au] are backbone markers.",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_property_tools_use_surrogate_list_api():
    surrogate = StrictListSurrogate({VALID_PARENT: 2.0, VALID_CHILD: 3.0})

    single_obs = PropertyPredictorTool(surrogate, "TestProp").execute(VALID_PARENT)
    assert single_obs.success
    assert single_obs.result["TestProp"] == 2.0

    batch_obs = BatchPropertyPredictorTool(surrogate, "TestProp").execute([VALID_PARENT, VALID_CHILD])
    assert batch_obs.success
    assert [r["property"] for r in batch_obs.result] == [2.0, 3.0]
    assert surrogate.calls == [[VALID_PARENT], [VALID_PARENT, VALID_CHILD]]


def test_worker_validation_uses_predict_single_and_task_markers():
    surrogate = StrictListSurrogate({VALID_PARENT: 2.0, VALID_CHILD: 3.0})
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(),
        surrogate=surrogate,
        parent_cache={},
    )

    valid, missing_marker = worker._validate_candidates([
        {"parent_smiles": VALID_PARENT, "child_smiles": VALID_CHILD, "explanation": "valid"},
        {"parent_smiles": VALID_PARENT, "child_smiles": "CCO", "explanation": "missing markers"},
    ])

    assert valid["valid"] is True
    assert valid["parent_property"] == 2.0
    assert valid["child_property"] == 3.0
    assert valid["improvement_factor"] == 1.5
    assert missing_marker["valid"] is False
    assert "Missing required marker" in missing_marker["invalid_reason"]
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_worker_improvement_factor_respects_minimization_direction():
    surrogate = StrictListSurrogate({VALID_PARENT: 10.0, VALID_CHILD: 5.0})
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(maximize=False),
        surrogate=surrogate,
        parent_cache={},
    )

    [candidate] = worker._validate_candidates([
        {"parent_smiles": VALID_PARENT, "child_smiles": VALID_CHILD, "explanation": "lower is better"},
    ])

    assert candidate["valid"] is True
    assert candidate["improvement_factor"] == 2.0


def test_critic_scores_current_state_from_evaluated_candidates(monkeypatch):
    current = PromptState.seed("seed strategy")
    history = PromptStateHistory()
    history.add(current)
    critic = CriticAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(),
        reward_fn=ParetoHypervolume(),
    )
    candidates = [
        {"valid": True, "improvement_factor": 2.0, "similarity": 0.8},
        {"valid": False, "improvement_factor": 100.0, "similarity": 1.0},
    ]

    def fake_run(initial_state):
        critic.new_state = PromptState(
            strategy_text="next strategy",
            version=1,
            parent_version=current.version,
        )
        return (critic.new_state, []), []

    monkeypatch.setattr(critic, "run", fake_run)

    critic.refine(candidates, current, history)

    assert current.score == ParetoHypervolume().compute([candidates[0]])


def test_meta_formats_recent_strategies_from_history():
    history = PromptStateHistory()
    for version in range(4):
        history.add(PromptState(strategy_text=f"strategy {version}", version=version))
    meta = MetaAgent(model="test-model", api_keys={}, task_context=polymer_ctx())
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted
    assert "v0: strategy 0" not in formatted


def test_usage_summary_merging_handles_aggregate_dicts():
    worker_usage = aggregate_usage([LLMUsage("worker-model", 10, 5, 0.2)])
    critic_usage = {
        "total_calls": 2,
        "total_prompt_tokens": 20,
        "total_completion_tokens": 10,
        "total_tokens": 30,
        "total_latency_s": 0.4,
        "by_model": {"critic-model": {"calls": 2, "tokens": 30}},
    }

    merged = _merge_usage_summaries(_empty_usage_summary(), worker_usage, critic_usage)

    assert merged["total_calls"] == 3
    assert merged["total_tokens"] == 45
    assert merged["by_model"]["worker-model"] == {"calls": 1, "tokens": 15}
    assert merged["by_model"]["critic-model"] == {"calls": 2, "tokens": 30}
