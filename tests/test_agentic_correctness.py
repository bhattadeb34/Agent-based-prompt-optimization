from typing import List, Optional
from unittest.mock import patch

from apo.agentic_engine import _merge_usage_summaries
from apo.agents.critic import CriticAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import WeightedSum
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictBatchSurrogate(SurrogatePredictor):
    property_name = "StrictProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise AssertionError("predict() must receive a list, not a string")
        self.calls.append(list(smiles_list))
        return [float(len(smiles)) for smiles in smiles_list]


def polymer_ctx() -> TaskContext:
    return TaskContext(
        property_name="StrictProp",
        property_units="units",
        maximize=True,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_agentic_worker_uses_scalar_surrogate_api_and_task_validation():
    surrogate = StrictBatchSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(),
        surrogate=surrogate,
        parent_cache={},
    )

    validated = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "explanation": "keeps markers",
        },
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": "CCO",
            "explanation": "drops polymer markers",
        },
    ])

    assert validated[0]["valid"] is True
    assert isinstance(validated[0]["parent_property"], float)
    assert isinstance(validated[0]["child_property"], float)
    assert validated[0]["similarity"] >= 0.0
    assert validated[1]["valid"] is False
    assert "Missing required marker" in validated[1]["invalid_reason"]
    assert [VALID_PARENT] in surrogate.calls
    assert [VALID_CHILD] in surrogate.calls


def test_agentic_property_tools_respect_surrogate_batch_contract():
    surrogate = StrictBatchSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "StrictProp").execute(VALID_CHILD)
    batch_obs = BatchPropertyPredictorTool(surrogate, "StrictProp").execute([
        VALID_PARENT,
        VALID_CHILD,
    ])

    assert single_obs.success is True
    assert single_obs.result["StrictProp"] == float(len(VALID_CHILD))
    assert batch_obs.success is True
    assert [VALID_PARENT, VALID_CHILD] in surrogate.calls


def test_critic_scores_evaluated_state_not_new_strategy():
    ctx = polymer_ctx()
    current = PromptState.seed("current strategy")
    history = PromptStateHistory()
    history.add(current)
    critic = CriticAgent(
        model="critic-model",
        api_keys={},
        task_context=ctx,
        reward_fn=WeightedSum(alpha=0.5),
    )
    candidates = [{
        "valid": True,
        "improvement_factor": 2.0,
        "similarity": 0.8,
    }]

    def fake_run(agent, initial_state):
        agent.new_state = PromptState(
            strategy_text="next strategy",
            version=current.version + 1,
            rationale="test",
            parent_version=current.version,
        )
        return (agent.new_state, agent.analysis), []

    with patch.object(CriticAgent, "run", fake_run):
        new_state, _, usage = critic.refine(candidates, current, history)

    assert current.score == 1.4
    assert new_state.score is None
    assert new_state.metadata["parent_reward"] == 1.4
    assert usage["total_calls"] == 0


def test_agentic_usage_summary_merges_dicts_without_llmusage_crash():
    worker_usage = LLMUsage("worker-model", 10, 5, 0.25)
    worker_summary = {
        "total_calls": 1,
        "total_prompt_tokens": worker_usage.prompt_tokens,
        "total_completion_tokens": worker_usage.completion_tokens,
        "total_tokens": worker_usage.total_tokens,
        "total_latency_s": worker_usage.latency_s,
        "by_model": {"worker-model": {"calls": 1, "tokens": 15}},
    }
    critic_summary = {
        "total_calls": 2,
        "total_prompt_tokens": 7,
        "total_completion_tokens": 3,
        "total_tokens": 10,
        "total_latency_s": 0.5,
        "by_model": {"critic-model": {"calls": 2, "tokens": 10}},
    }

    merged = _merge_usage_summaries(worker_summary, critic_summary)

    assert merged["total_calls"] == 3
    assert merged["total_tokens"] == 25
    assert merged["by_model"]["worker-model"] == {"calls": 1, "tokens": 15}
    assert merged["by_model"]["critic-model"] == {"calls": 2, "tokens": 10}
