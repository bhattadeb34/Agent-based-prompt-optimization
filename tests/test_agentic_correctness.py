from typing import List, Optional

from apo.agentic_engine import _merge_usage_summaries
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


class StrictBatchSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        assert isinstance(smiles_list, list), "predict() requires a list of SMILES"
        self.calls.append(smiles_list)
        return [2.0 if smiles == VALID_CHILD else 1.0 for smiles in smiles_list]


def polymer_context() -> TaskContext:
    return TaskContext(
        property_name="TestProp",
        property_units="units",
        maximize=True,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_worker_validation_uses_predict_single_and_task_markers():
    surrogate = StrictBatchSurrogate()
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
            "explanation": "valid polymer",
        },
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": "CCO",
            "explanation": "missing required polymer markers",
        },
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 1.0
    assert candidates[0]["child_property"] == 2.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert candidates[1]["valid"] is False
    assert "Missing required marker" in candidates[1]["invalid_reason"]
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_property_tools_respect_scalar_and_batch_predictor_contracts():
    surrogate = StrictBatchSurrogate()

    scalar_obs = PropertyPredictorTool(surrogate, "TestProp").execute(VALID_CHILD)
    assert scalar_obs.success is True
    assert scalar_obs.result["TestProp"] == 2.0

    batch_obs = BatchPropertyPredictorTool(surrogate, "TestProp").execute([
        VALID_PARENT,
        VALID_CHILD,
    ])
    assert batch_obs.success is True
    assert [item["property"] for item in batch_obs.result] == [1.0, 2.0]
    assert surrogate.calls[-1] == [VALID_PARENT, VALID_CHILD]


def test_critic_scores_evaluated_current_state(monkeypatch):
    current = PromptState.seed("seed strategy")
    history = PromptStateHistory()
    history.add(current)
    critic = CriticAgent(
        model="critic-model",
        api_keys={},
        task_context=polymer_context(),
        reward_fn=ParetoHypervolume(),
    )

    def fake_run(self, initial_state):
        self.new_state = PromptState(
            strategy_text="next strategy",
            version=self.current_state.version + 1,
            rationale="test",
            parent_version=self.current_state.version,
            model_used=self.model,
        )
        return (self.new_state, self.analysis), []

    monkeypatch.setattr(CriticAgent, "run", fake_run)

    new_state, _, _ = critic.refine(
        candidates=[{
            "valid": True,
            "improvement_factor": 2.0,
            "similarity": 0.8,
        }],
        current_state=current,
        history=history,
    )

    assert current.score == 1.6
    assert new_state.version == 1


def test_meta_formats_recent_strategies_from_history_api():
    history = PromptStateHistory()
    for version in range(4):
        history.add(PromptState(strategy_text=f"strategy {version}", version=version))

    meta = MetaAgent(
        model="meta-model",
        api_keys={},
        task_context=polymer_context(),
    )
    meta.history = history

    formatted = meta._format_recent_strategies()
    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted


def test_agentic_usage_summary_merges_aggregate_dicts():
    worker_usage = aggregate_usage([
        LLMUsage("worker", 10, 5, 0.1),
    ])
    critic_usage = aggregate_usage([
        LLMUsage("critic", 20, 10, 0.2),
    ])
    meta_usage = aggregate_usage([
        LLMUsage("meta", 4, 6, 0.3),
    ])

    merged = _merge_usage_summaries(worker_usage, critic_usage, meta_usage)

    assert merged["total_calls"] == 3
    assert merged["total_tokens"] == 55
    assert merged["by_model"]["worker"]["calls"] == 1
    assert merged["by_model"]["critic"]["tokens"] == 30
    assert merged["by_model"]["meta"]["tokens"] == 10
