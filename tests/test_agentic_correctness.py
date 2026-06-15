from typing import List, Optional
from unittest.mock import patch

import pytest

from apo.agentic_engine import _merge_usage_summary
from apo.agents.critic import CriticAgent
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

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise TypeError("predict expects a list of SMILES")
        self.calls.append(smiles_list)
        return [float(len(smi)) for smi in smiles_list]


def polymer_context() -> TaskContext:
    return TaskContext(
        property_name="TestProp",
        property_units="units",
        maximize=True,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_property_tools_use_surrogate_api_shape():
    surrogate = StrictListSurrogate()

    single = PropertyPredictorTool(surrogate, "TestProp").execute("CCO")
    batch = BatchPropertyPredictorTool(surrogate, "TestProp").execute(["CC", "CCC"])

    assert single.success
    assert single.result["TestProp"] == 3.0
    assert batch.success
    assert [item["property"] for item in batch.result] == [2.0, 3.0]
    assert surrogate.calls == [["CCO"], ["CC", "CCC"]]


def test_worker_validation_uses_predict_single_and_task_markers():
    surrogate = StrictListSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {"parent_smiles": VALID_PARENT, "child_smiles": VALID_CHILD, "explanation": "valid polymer"},
        {"parent_smiles": VALID_PARENT, "child_smiles": "CCO", "explanation": "missing markers"},
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["child_property"] == float(len(VALID_CHILD))
    assert candidates[0]["parent_property"] == float(len(VALID_PARENT))
    assert candidates[1]["valid"] is False
    assert "Missing required marker" in candidates[1]["invalid_reason"]
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_critic_scores_evaluated_current_state():
    ctx = polymer_context()
    critic = CriticAgent(
        model="test-model",
        api_keys={},
        task_context=ctx,
        reward_fn=ParetoHypervolume(),
    )
    current = PromptState.seed("current strategy")
    history = PromptStateHistory()
    history.add(current)

    def fake_run(self, initial_state):
        self.new_state = PromptState(
            strategy_text="next strategy",
            version=self.current_state.version + 1,
            rationale="test",
            parent_version=self.current_state.version,
        )
        self.analysis = {"pareto_insights": ["ok"]}
        return (self.new_state, self.analysis), []

    candidates = [{
        "valid": True,
        "improvement_factor": 2.0,
        "similarity": 0.5,
        "child_property": 2.0,
        "parent_property": 1.0,
    }]

    with patch.object(CriticAgent, "run", fake_run):
        new_state, _, usage = critic.refine(candidates, current, history)

    assert current.score == pytest.approx(1.0)
    assert new_state.score is None
    assert new_state.metadata["reward"] == pytest.approx(1.0)
    assert usage["total_calls"] == 0


def test_usage_summary_merge_keeps_dicts_out_of_llm_usage_lists():
    worker_usage = aggregate_usage([LLMUsage("worker-model", 10, 5, 0.25)])
    critic_usage = {
        "total_calls": 2,
        "total_prompt_tokens": 20,
        "total_completion_tokens": 10,
        "total_tokens": 30,
        "total_latency_s": 0.75,
        "by_model": {"critic-model": {"calls": 2, "tokens": 30}},
    }

    merged = _merge_usage_summary(worker_usage, critic_usage)

    assert merged["total_calls"] == 3
    assert merged["total_tokens"] == 45
    assert merged["by_model"]["worker-model"] == {"calls": 1, "tokens": 15}
    assert merged["by_model"]["critic-model"] == {"calls": 2, "tokens": 30}
