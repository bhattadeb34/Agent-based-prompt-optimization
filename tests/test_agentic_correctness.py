from typing import List, Optional

from apo.agentic_engine import _merge_usage_summary
from apo.agents.critic import CriticAgent
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictBatchSurrogate(SurrogatePredictor):
    property_name = "TestProp"

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise TypeError("predict expects a list of SMILES, not a string")
        self.calls.append(list(smiles_list))
        values = {
            VALID_PARENT: 1.0,
            VALID_CHILD: 2.0,
            "CC": 1.0,
            "CCC": 3.0,
        }
        return [values.get(smiles) for smiles in smiles_list]


def polymer_context() -> TaskContext:
    return TaskContext(
        property_name="TestProp",
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_agentic_worker_uses_predict_single_and_enforces_task_markers():
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
            "explanation": "valid marker-preserving edit",
        },
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": "CCO",
            "explanation": "missing polymer markers",
        },
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 1.0
    assert candidates[0]["child_property"] == 2.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert candidates[1]["valid"] is False
    assert "Missing required marker" in candidates[1]["invalid_reason"]
    assert surrogate.calls == [[VALID_PARENT], [VALID_CHILD]]


def test_agentic_property_tools_follow_batch_predictor_contract():
    surrogate = StrictBatchSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "TestProp").execute("CC")
    assert single_obs.success is True
    assert single_obs.result["TestProp"] == 1.0

    batch_obs = BatchPropertyPredictorTool(surrogate, "TestProp").execute(["CC", "CCC"])
    assert batch_obs.success is True
    assert [row["property"] for row in batch_obs.result] == [1.0, 3.0]
    assert surrogate.calls == [["CC"], ["CC", "CCC"]]


def test_critic_scores_evaluated_current_state(monkeypatch):
    ctx = polymer_context()
    critic = CriticAgent(
        model="test-model",
        api_keys={},
        task_context=ctx,
        reward_fn=ParetoHypervolume(),
    )
    current = PromptState.seed("seed strategy")
    history = PromptStateHistory()
    history.add(current)

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

    new_state, _, usage = critic.refine(
        candidates=[{
            "valid": True,
            "improvement_factor": 2.0,
            "similarity": 0.5,
        }],
        current_state=current,
        history=history,
    )

    assert current.score == 1.0
    assert new_state.score is None
    assert new_state.metadata["reward"] == 1.0
    assert usage["total_calls"] == 0


def test_meta_formats_recent_history_without_missing_all_method():
    history = PromptStateHistory()
    for i in range(4):
        history.add(PromptState(strategy_text=f"strategy {i}", version=i, rationale=""))

    meta = MetaAgent(model="test-model", api_keys={}, task_context=polymer_context())

    meta.history = history
    rendered = meta._format_recent_strategies()

    assert "v1: strategy 1" in rendered
    assert "v3: strategy 3" in rendered


def test_agentic_engine_merges_aggregated_usage_summaries():
    total = {"total_calls": 1, "total_tokens": 10, "by_model": {"worker": {"calls": 1, "tokens": 10}}}
    _merge_usage_summary(
        total,
        {"total_calls": 2, "total_tokens": 20, "by_model": {"critic": {"calls": 2, "tokens": 20}}},
    )

    assert total["total_calls"] == 3
    assert total["total_tokens"] == 30
    assert total["by_model"]["worker"] == {"calls": 1, "tokens": 10}
    assert total["by_model"]["critic"] == {"calls": 2, "tokens": 20}
