from typing import List, Optional

from apo.agentic_engine import _merge_usage_summaries
from apo.agents.critic import CriticAgent
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import PropertyOnly
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictSurrogate(SurrogatePredictor):
    property_name = "StrictProp"
    property_units = "units"
    maximize = True

    def __init__(self, values=None):
        self.values = values or {PARENT: 2.0, CHILD: 4.0, "CCO": 1.0}
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        assert isinstance(smiles_list, list), "predict() must receive a list"
        self.calls.append(list(smiles_list))
        return [self.values.get(smiles) for smiles in smiles_list]


def polymer_context(maximize=True):
    return TaskContext(
        property_name="StrictProp",
        property_units="units",
        maximize=maximize,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_worker_validation_uses_scalar_predictor_api_and_task_context():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {"parent_smiles": PARENT, "child_smiles": CHILD, "explanation": "valid"},
        {"parent_smiles": PARENT, "child_smiles": "CCO", "explanation": "missing markers"},
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 2.0
    assert candidates[0]["child_property"] == 4.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert candidates[0]["similarity"] > 0.0
    assert candidates[1]["valid"] is False
    assert "Missing required marker" in candidates[1]["invalid_reason"]
    assert surrogate.calls == [[PARENT], [CHILD]]


def test_worker_validation_respects_minimization_direction():
    surrogate = StrictSurrogate(values={PARENT: 4.0, CHILD: 2.0})
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(maximize=False),
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {"parent_smiles": PARENT, "child_smiles": CHILD, "explanation": "lower is better"},
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["improvement_factor"] == 2.0


def test_property_tools_use_declared_surrogate_api():
    surrogate = StrictSurrogate()

    single = PropertyPredictorTool(surrogate, "StrictProp").execute(CHILD)
    batch = BatchPropertyPredictorTool(surrogate, "StrictProp").execute([PARENT, CHILD])

    assert single.success is True
    assert single.result["StrictProp"] == 4.0
    assert batch.success is True
    assert [r["property"] for r in batch.result] == [2.0, 4.0]
    assert surrogate.calls == [[CHILD], [PARENT, CHILD]]


def test_critic_scores_evaluated_current_state(monkeypatch):
    critic = CriticAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
        reward_fn=PropertyOnly(),
    )
    current = PromptState.seed("initial")
    history = PromptStateHistory()
    history.add(current)
    next_state = PromptState(strategy_text="next", version=1, rationale="test")

    def fake_run(initial_state):
        critic.new_state = next_state
        critic.analysis = {}
        return (next_state, {}), []

    monkeypatch.setattr(critic, "run", fake_run)
    monkeypatch.setattr(critic, "_save_trace_to_disk", lambda: None)

    new_state, _, usage = critic.refine(
        candidates=[{"valid": True, "improvement_factor": 3.0, "similarity": 0.5}],
        current_state=current,
        history=history,
    )

    assert new_state is next_state
    assert current.score == 3.0
    assert usage["total_calls"] == 0


def test_meta_formats_recent_strategies_with_history_accessor():
    meta = MetaAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
    )
    history = PromptStateHistory()
    for i in range(5):
        history.add(PromptState(strategy_text=f"strategy {i}", version=i, rationale=""))
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v2: strategy 2" in formatted
    assert "v4: strategy 4" in formatted
    assert "v1: strategy 1" not in formatted


def test_usage_summary_merge_preserves_model_breakdown():
    first = {
        "total_calls": 1,
        "total_prompt_tokens": 10,
        "total_completion_tokens": 5,
        "total_tokens": 15,
        "total_latency_s": 0.2,
        "by_model": {"worker": {"calls": 1, "tokens": 15}},
    }
    second = {
        "total_calls": 2,
        "total_prompt_tokens": 20,
        "total_completion_tokens": 10,
        "total_tokens": 30,
        "total_latency_s": 0.4,
        "by_model": {"critic": {"calls": 2, "tokens": 30}},
    }
    raw = LLMUsage("worker", prompt_tokens=3, completion_tokens=2, latency_s=0.1)

    merged = _merge_usage_summaries(first, second, {
        "total_calls": 1,
        "total_prompt_tokens": raw.prompt_tokens,
        "total_completion_tokens": raw.completion_tokens,
        "total_tokens": raw.total_tokens,
        "total_latency_s": raw.latency_s,
        "by_model": {raw.model: {"calls": 1, "tokens": raw.total_tokens}},
    })

    assert merged["total_calls"] == 4
    assert merged["total_tokens"] == 50
    assert merged["by_model"]["worker"] == {"calls": 2, "tokens": 20}
    assert merged["by_model"]["critic"] == {"calls": 2, "tokens": 30}
