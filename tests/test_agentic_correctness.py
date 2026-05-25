import json
from typing import List, Optional

from apo.agentic_engine import _merge_usage_summary
from apo.agents.critic import CriticAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


class StrictBatchSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise AssertionError("predict() must receive a list of SMILES")
        self.calls.append(list(smiles_list))
        return [float(len(smiles)) for smiles in smiles_list]


GENERIC_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="organic compound",
    smiles_markers=[],
)


def test_agentic_property_tools_respect_surrogate_batch_contract():
    surrogate = StrictBatchSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "TestProp").execute("CCO")
    assert single_obs.success is True
    assert single_obs.result["TestProp"] == 3.0

    batch_obs = BatchPropertyPredictorTool(surrogate, "TestProp").execute(["CC", "CCCC"])
    assert batch_obs.success is True
    assert [row["property"] for row in batch_obs.result] == [2.0, 4.0]
    assert surrogate.calls == [["CCO"], ["CC", "CCCC"]]


def test_worker_validation_uses_single_prediction_wrapper():
    surrogate = StrictBatchSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=GENERIC_CTX,
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": "CC",
            "child_smiles": "CCCC",
            "explanation": "extend chain",
        }
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 2.0
    assert candidates[0]["child_property"] == 4.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert surrogate.calls == [["CC"], ["CCCC"]]


def test_critic_refine_scores_current_and_new_state(monkeypatch):
    responses = iter([
        json.dumps({
            "pareto_insights": ["larger molecule helped"],
            "failure_patterns": [],
            "unexplored_space": ["rings"],
            "tradeoffs": "none",
            "confidence": 0.9,
        }),
        json.dumps({
            "alternative_1": {
                "name": "Exploit",
                "strategy": "keep extending chains",
                "rationale": "observed improvement",
            },
            "alternative_2": {
                "name": "Explore",
                "strategy": "try rings",
                "rationale": "unexplored",
            },
        }),
        json.dumps({
            "consensus": "A",
            "consensus_rationale": "best evidence",
            "confidence": 0.8,
        }),
    ])

    def fake_call_llm(*args, **kwargs):
        return next(responses), LLMUsage("test-model", 1, 1, 0.01)

    monkeypatch.setattr("apo.agents.critic.call_llm", fake_call_llm)

    current = PromptState.seed("initial")
    history = PromptStateHistory()
    history.add(current)
    critic = CriticAgent(
        model="test-model",
        api_keys={},
        task_context=GENERIC_CTX,
        reward_fn=ParetoHypervolume(),
    )

    new_state, _, usage = critic.refine(
        candidates=[{
            "valid": True,
            "improvement_factor": 2.0,
            "similarity": 0.5,
            "child_smiles": "CCCC",
        }],
        current_state=current,
        history=history,
    )

    assert current.score == 1.0
    assert new_state.score == 1.0
    assert new_state.metadata["reward"] == 1.0
    assert usage["total_calls"] == 3


def test_usage_summary_merge_preserves_aggregate_dicts():
    total = {
        "total_calls": 1,
        "total_prompt_tokens": 2,
        "total_completion_tokens": 3,
        "total_tokens": 5,
        "total_latency_s": 0.1,
        "by_model": {"worker": {"calls": 1, "tokens": 5}},
    }
    usage = {
        "total_calls": 2,
        "total_prompt_tokens": 7,
        "total_completion_tokens": 11,
        "total_tokens": 18,
        "total_latency_s": 0.2,
        "by_model": {"critic": {"calls": 2, "tokens": 18}},
    }

    merged = _merge_usage_summary(total, usage)

    assert merged["total_calls"] == 3
    assert merged["total_tokens"] == 23
    assert merged["by_model"]["worker"] == {"calls": 1, "tokens": 5}
    assert merged["by_model"]["critic"] == {"calls": 2, "tokens": 18}
