from types import MethodType
from typing import List, Optional

from apo.agentic_engine import _merge_usage_summaries
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
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        assert isinstance(smiles_list, list), "surrogate.predict expects a list of SMILES"
        self.calls.append(list(smiles_list))
        return [float(len(s)) for s in smiles_list]


POLYMER_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="polymer",
    domain_context="[Cu] and [Au] are backbone markers.",
    smiles_markers=["[Cu]", "[Au]"],
    similarity_on_repeat_unit=True,
)


def test_worker_uses_predict_single_and_enforces_task_markers():
    surrogate = StrictBatchSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "explanation": "valid polymer edit",
        },
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": "CCO",
            "explanation": "missing polymer markers",
        },
    ])

    valid, invalid = candidates
    assert valid["valid"] is True
    assert isinstance(valid["parent_property"], float)
    assert isinstance(valid["child_property"], float)
    assert invalid["valid"] is False
    assert "Missing required marker" in invalid["invalid_reason"]
    assert surrogate.calls == [[VALID_PARENT], [VALID_CHILD]]


def test_property_tools_preserve_batch_predictor_contract():
    surrogate = StrictBatchSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "TestProp").execute("CC")
    assert single_obs.success is True
    assert single_obs.result["TestProp"] == 2.0

    batch_obs = BatchPropertyPredictorTool(surrogate, "TestProp").execute(["CC", "CCC"])
    assert batch_obs.success is True
    assert [r["property"] for r in batch_obs.result] == [2.0, 3.0]
    assert surrogate.calls == [["CC"], ["CC", "CCC"]]


def test_critic_scores_evaluated_state_before_refining(monkeypatch):
    critic = CriticAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        reward_fn=ParetoHypervolume(),
    )
    current = PromptState.seed("seed strategy")
    history = PromptStateHistory()
    history.add(current)
    candidates = [
        {
            "valid": True,
            "improvement_factor": 2.0,
            "similarity": 0.5,
            "child_smiles": VALID_CHILD,
        }
    ]

    def fake_run(self, initial_state):
        self.new_state = PromptState(
            strategy_text="next strategy",
            version=self.current_state.version + 1,
            parent_version=self.current_state.version,
        )
        return self.new_state, []

    monkeypatch.setattr(critic, "run", MethodType(fake_run, critic))

    new_state, _, _ = critic.refine(candidates, current, history)

    assert current.score == ParetoHypervolume().compute(candidates)
    assert new_state.version == 1


def test_meta_recent_strategy_format_uses_history_api():
    history = PromptStateHistory()
    for i in range(4):
        history.add(PromptState(strategy_text=f"strategy {i}", version=i))

    meta = MetaAgent(model="test-model", api_keys={}, task_context=POLYMER_CTX)
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v0:" not in formatted
    assert "v1:" in formatted
    assert "v2:" in formatted
    assert "v3:" in formatted


def test_agentic_usage_summary_merge_handles_aggregated_dicts():
    merged = _merge_usage_summaries(
        {
            "total_calls": 1,
            "total_tokens": 10,
            "total_latency_s": 0.5,
            "by_model": {"worker": {"calls": 1, "tokens": 10}},
        },
        {
            "total_calls": 2,
            "total_tokens": 30,
            "total_latency_s": 1.0,
            "by_model": {"critic": {"calls": 2, "tokens": 30}},
        },
    )

    assert merged["total_calls"] == 3
    assert merged["total_tokens"] == 40
    assert merged["by_model"]["worker"] == {"calls": 1, "tokens": 10}
    assert merged["by_model"]["critic"] == {"calls": 2, "tokens": 30}
