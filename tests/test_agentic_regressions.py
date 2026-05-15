"""Regression tests for high-impact agentic workflow bugs."""
from typing import Dict, List, Optional

from apo.agents.base import Observation
from apo.agents.critic import CriticAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage, aggregate_usage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


class StrictSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self, values: Dict[str, float]):
        self.values = values
        self.calls: List[List[str]] = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise TypeError("predict expects a list of SMILES")
        self.calls.append(smiles_list)
        return [self.values.get(smiles) for smiles in smiles_list]


class FakeValidator:
    name = "validate_smiles"

    def execute(self, smiles_list):
        return Observation(
            success=True,
            result=[{"smiles": smiles, "valid": True} for smiles in smiles_list],
        )


class FakeSimilarity:
    name = "calculate_similarity"

    def execute(self, smiles1, smiles2):
        return Observation(success=True, result={"similarity": 0.75})


def _ctx() -> TaskContext:
    return TaskContext(
        property_name="TestProp",
        property_units="units",
        maximize=True,
        molecule_type="molecule",
    )


def test_agentic_worker_uses_single_prediction_api_for_scalar_smiles():
    surrogate = StrictSurrogate({"CC": 1.0, "CCO": 2.0})
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=_ctx(),
        surrogate=surrogate,
        parent_cache={},
    )
    worker.tools = [FakeValidator(), FakeSimilarity()]

    candidates = worker._validate_candidates([
        {"parent_smiles": "CC", "child_smiles": "CCO", "explanation": "add oxygen"}
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 1.0
    assert candidates[0]["child_property"] == 2.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert surrogate.calls == [["CC"], ["CCO"]]


def test_agentic_property_tools_use_surrogate_batch_contract():
    surrogate = StrictSurrogate({"CC": 1.0, "CCO": 2.0})

    single = PropertyPredictorTool(surrogate, property_name="TestProp").execute("CC")
    batch = BatchPropertyPredictorTool(surrogate, property_name="TestProp").execute(["CC", "CCO"])

    assert single.success is True
    assert single.result["TestProp"] == 1.0
    assert batch.success is True
    assert [row["property"] for row in batch.result] == [1.0, 2.0]
    assert surrogate.calls == [["CC"], ["CC", "CCO"]]


def test_aggregate_usage_accepts_nested_agent_summaries():
    usage = LLMUsage("worker-model", prompt_tokens=10, completion_tokens=5, latency_s=0.2)
    critic_summary = {
        "total_calls": 2,
        "total_prompt_tokens": 7,
        "total_completion_tokens": 3,
        "total_tokens": 10,
        "total_latency_s": 0.4,
        "by_model": {"critic-model": {"calls": 2, "tokens": 10}},
    }

    total = aggregate_usage([usage, critic_summary])

    assert total["total_calls"] == 3
    assert total["total_prompt_tokens"] == 17
    assert total["total_completion_tokens"] == 8
    assert total["total_tokens"] == 25
    assert total["by_model"]["worker-model"] == {"calls": 1, "tokens": 15}
    assert total["by_model"]["critic-model"] == {"calls": 2, "tokens": 10}


def test_critic_refine_scores_current_strategy(monkeypatch):
    critic = CriticAgent(
        model="critic-model",
        api_keys={},
        task_context=_ctx(),
        reward_fn=ParetoHypervolume(),
    )
    current = PromptState.seed("current strategy")
    history = PromptStateHistory()
    history.add(current)

    def fake_run(initial_state):
        critic.new_state = PromptState(
            strategy_text="next strategy",
            version=current.version + 1,
            parent_version=current.version,
        )
        critic.analysis = {"ok": True}
        critic.all_usages.append(LLMUsage("critic-model", 1, 1, 0.1))
        return critic.new_state, []

    monkeypatch.setattr(critic, "run", fake_run)

    new_state, analysis, usage = critic.refine(
        candidates=[{"valid": True, "improvement_factor": 2.0, "similarity": 0.5}],
        current_state=current,
        history=history,
    )

    assert current.score == 1.0
    assert new_state.version == 1
    assert analysis == {"ok": True}
    assert usage["total_calls"] == 1
