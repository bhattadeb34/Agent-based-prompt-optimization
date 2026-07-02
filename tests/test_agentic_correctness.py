from typing import List, Optional

from apo.agentic_engine import _merge_usage_summaries
from apo.agents.critic import CriticAgent
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import WeightedSum
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


POLYMER_CTX = TaskContext(
    property_name="TestProp",
    maximize=True,
    molecule_type="polymer",
    smiles_markers=["[Cu]", "[Au]"],
    similarity_on_repeat_unit=True,
)
PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictListSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise TypeError("predict requires a list of SMILES")
        self.calls.append(list(smiles_list))
        return [2.0 if smiles == CHILD else 1.0 for smiles in smiles_list]


def test_property_tools_respect_surrogate_list_api():
    surrogate = StrictListSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "TestProp").execute(CHILD)
    batch_obs = BatchPropertyPredictorTool(surrogate, "TestProp").execute([PARENT, CHILD])

    assert single_obs.success is True
    assert single_obs.result["TestProp"] == 2.0
    assert batch_obs.success is True
    assert [row["property"] for row in batch_obs.result] == [1.0, 2.0]
    assert surrogate.calls == [[CHILD], [PARENT, CHILD]]


def test_worker_validation_rejects_markerless_smiles_and_uses_predict_single():
    surrogate = StrictListSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        surrogate=surrogate,
        parent_cache={PARENT: 1.0},
    )

    candidates = worker._validate_candidates([
        {"parent_smiles": PARENT, "child_smiles": "CCO", "explanation": "missing markers"},
        {"parent_smiles": PARENT, "child_smiles": CHILD, "explanation": "valid polymer"},
    ])

    assert candidates[0]["valid"] is False
    assert "Missing required marker" in candidates[0]["invalid_reason"]
    assert candidates[1]["valid"] is True
    assert candidates[1]["child_property"] == 2.0
    assert candidates[1]["improvement_factor"] == 2.0
    assert surrogate.calls == [[CHILD]]


class NoLLMCritic(CriticAgent):
    def run(self, initial_state: str = ""):
        self.new_state = PromptState(
            strategy_text="next",
            version=self.current_state.version + 1,
            parent_version=self.current_state.version,
        )
        return None, []


def test_critic_scores_evaluated_current_state_before_refining():
    current = PromptState.seed("current")
    critic = NoLLMCritic(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        reward_fn=WeightedSum(alpha=0.5),
    )

    new_state, _, usage = critic.refine(
        candidates=[
            {
                "valid": True,
                "improvement_factor": 2.0,
                "similarity": 0.5,
                "child_property": 2.0,
                "parent_property": 1.0,
            }
        ],
        current_state=current,
        history=PromptStateHistory(),
    )

    assert current.score == 1.25
    assert new_state.score is None
    assert usage["total_calls"] == 0


def test_meta_formats_recent_strategies_from_history_api():
    history = PromptStateHistory()
    for idx in range(4):
        history.add(PromptState(strategy_text=f"strategy {idx}", version=idx))

    meta = MetaAgent("test-model", {}, POLYMER_CTX)
    meta.history = history

    formatted = meta._format_recent_strategies()
    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted
    assert "v0: strategy 0" not in formatted


def test_merge_usage_summaries_combines_nested_model_stats():
    merged = _merge_usage_summaries(
        {
            "total_calls": 1,
            "total_prompt_tokens": 3,
            "total_completion_tokens": 4,
            "total_tokens": 7,
            "total_latency_s": 0.5,
            "by_model": {"worker": {"calls": 1, "tokens": 7}},
        },
        {
            "total_calls": 2,
            "total_prompt_tokens": 5,
            "total_completion_tokens": 6,
            "total_tokens": 11,
            "total_latency_s": 1.0,
            "by_model": {"worker": {"calls": 1, "tokens": 5}, "critic": {"calls": 1, "tokens": 6}},
        },
    )

    assert merged["total_calls"] == 3
    assert merged["total_tokens"] == 18
    assert merged["by_model"]["worker"] == {"calls": 2, "tokens": 12}
    assert merged["by_model"]["critic"] == {"calls": 1, "tokens": 6}
