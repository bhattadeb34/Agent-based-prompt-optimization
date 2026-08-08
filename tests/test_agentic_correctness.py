import json
from typing import List, Optional

from apo.agentic_engine import _merge_usage_dict
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


class StrictSurrogate(SurrogatePredictor):
    property_name = "Strict"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.predict_calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise TypeError("predict expects a list, not a string")
        self.predict_calls.append(list(smiles_list))
        return [float(len(s)) for s in smiles_list]


def polymer_ctx(maximize=True):
    return TaskContext(
        property_name="Strict",
        property_units="units",
        maximize=maximize,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def usage(model="test-model"):
    return LLMUsage(model=model, prompt_tokens=3, completion_tokens=4, latency_s=0.1)


def test_agentic_property_tools_use_surrogate_list_api_once_for_batches():
    surrogate = StrictSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "Strict").execute(VALID_PARENT)
    assert single_obs.success is True
    assert surrogate.predict_calls[-1] == [VALID_PARENT]

    batch_obs = BatchPropertyPredictorTool(surrogate, "Strict").execute([VALID_PARENT, VALID_CHILD])
    assert batch_obs.success is True
    assert surrogate.predict_calls[-1] == [VALID_PARENT, VALID_CHILD]


def test_worker_rejects_markerless_smiles_and_scores_valid_candidate_with_strict_surrogate():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(),
        surrogate=surrogate,
        parent_cache={VALID_PARENT: 10.0},
    )

    candidates = worker._validate_candidates([
        {"parent_smiles": VALID_PARENT, "child_smiles": "CCO", "explanation": "missing markers"},
        {"parent_smiles": VALID_PARENT, "child_smiles": VALID_CHILD, "explanation": "valid polymer"},
    ])

    assert candidates[0]["valid"] is False
    assert "Missing required marker" in candidates[0]["invalid_reason"]
    assert candidates[1]["valid"] is True
    assert candidates[1]["child_property"] is not None
    assert candidates[1]["improvement_factor"] > 0


def test_worker_parses_generated_molecules_mapping_schema(monkeypatch):
    response = json.dumps({
        "generated_molecules": {
            VALID_PARENT: {
                "smiles": [VALID_CHILD],
                "reasoning": ["add oxygen"],
            }
        }
    })
    monkeypatch.setattr("apo.agents.worker.call_llm", lambda **kwargs: (response, usage()))

    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(),
        surrogate=StrictSurrogate(),
        parent_cache={},
    )
    worker.parent_smiles_list = [VALID_PARENT]

    candidates = worker._call_llm_for_generation()

    assert candidates == [{
        "parent_smiles": VALID_PARENT,
        "child_smiles": VALID_CHILD,
        "explanation": "add oxygen",
    }]


def test_worker_uses_minimization_direction_for_improvement():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(maximize=False),
        surrogate=surrogate,
        parent_cache={VALID_PARENT: 20.0},
    )

    [candidate] = worker._validate_candidates([
        {"parent_smiles": VALID_PARENT, "child_smiles": VALID_CHILD, "explanation": "valid polymer"},
    ])

    assert candidate["valid"] is True
    assert candidate["improvement_factor"] == candidate["parent_property"] / candidate["child_property"]


def test_critic_scores_evaluated_current_state_before_returning_new_strategy(monkeypatch):
    current = PromptState.seed("seed")
    history = PromptStateHistory()
    history.add(current)
    critic = CriticAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(),
        reward_fn=ParetoHypervolume(),
    )

    def fake_run(initial_state):
        critic.new_state = PromptState(
            strategy_text="strategy A",
            version=current.version + 1,
            rationale="test",
            parent_version=current.version,
            model_used=critic.model,
        )
        critic.analysis = {"pareto_insights": ["good"]}
        critic.all_usages = [usage()]
        return (critic.new_state, critic.analysis), []

    monkeypatch.setattr(critic, "run", fake_run)

    new_state, _, critic_usage = critic.refine(
        candidates=[{
            "valid": True,
            "improvement_factor": 2.0,
            "similarity": 0.5,
            "child_smiles": VALID_CHILD,
            "child_property": 2.0,
            "parent_property": 1.0,
        }],
        current_state=current,
        history=history,
    )

    assert current.score == 1.0
    assert new_state.version == 1
    assert critic_usage["total_calls"] == 1


def test_meta_formats_recent_strategies_from_prompt_history():
    history = PromptStateHistory()
    for i in range(4):
        history.add(PromptState(strategy_text=f"strategy {i}", version=i))

    meta = MetaAgent(model="test-model", api_keys={}, task_context=polymer_ctx())
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted


def test_agentic_usage_dicts_merge_without_llmusage_objects():
    total = {"total_calls": 0, "total_tokens": 0, "total_latency_s": 0.0, "by_model": {}}

    _merge_usage_dict(total, {
        "total_calls": 2,
        "total_prompt_tokens": 6,
        "total_completion_tokens": 8,
        "total_tokens": 14,
        "total_latency_s": 0.2,
        "by_model": {"test-model": {"calls": 2, "tokens": 14}},
    })
    _merge_usage_dict(total, {
        "total_calls": 1,
        "total_prompt_tokens": 3,
        "total_completion_tokens": 4,
        "total_tokens": 7,
        "total_latency_s": 0.1,
        "by_model": {"test-model": {"calls": 1, "tokens": 7}},
    })

    assert total["total_calls"] == 3
    assert total["total_tokens"] == 21
    assert total["by_model"]["test-model"] == {"calls": 3, "tokens": 21}
