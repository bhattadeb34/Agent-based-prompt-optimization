from types import MethodType
from typing import List, Optional

from apo.agentic_engine import _merge_usage_summaries
from apo.agents.critic import CriticAgent
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictSurrogate(SurrogatePredictor):
    property_name = "StrictProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise TypeError("predict expects a list of SMILES")
        self.calls.append(list(smiles_list))
        return [2.0 if smi == CHILD else 1.0 for smi in smiles_list]


def polymer_ctx(maximize: bool = True) -> TaskContext:
    return TaskContext(
        property_name="StrictProp",
        property_units="units",
        maximize=maximize,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_worker_validation_uses_single_predict_and_task_markers():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(),
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {"parent_smiles": PARENT, "child_smiles": CHILD, "explanation": "valid"},
        {"parent_smiles": PARENT, "child_smiles": "CCO", "explanation": "missing markers"},
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["child_property"] == 2.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert candidates[1]["valid"] is False
    assert "Missing required marker" in candidates[1]["invalid_reason"]
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_property_tools_use_surrogate_list_contract():
    surrogate = StrictSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "StrictProp").execute(CHILD)
    batch_obs = BatchPropertyPredictorTool(surrogate, "StrictProp").execute([PARENT, CHILD])

    assert single_obs.success is True
    assert single_obs.result["StrictProp"] == 2.0
    assert batch_obs.success is True
    assert [row["property"] for row in batch_obs.result] == [1.0, 2.0]
    assert surrogate.calls == [[CHILD], [PARENT, CHILD]]


def test_critic_scores_evaluated_current_state():
    critic = CriticAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(),
        reward_fn=ParetoHypervolume(),
    )
    current = PromptState.seed("current strategy")
    history = PromptStateHistory()
    history.add(current)

    def fake_run(self, initial_state=""):
        self.new_state = PromptState(
            strategy_text="next strategy",
            version=self.current_state.version + 1,
            parent_version=self.current_state.version,
        )
        self.all_usages = [LLMUsage("test-model", 1, 2, 0.1)]
        return (self.new_state, self.analysis), []

    critic.run = MethodType(fake_run, critic)
    new_state, _, usage = critic.refine(
        candidates=[{"valid": True, "improvement_factor": 2.0, "similarity": 0.5}],
        current_state=current,
        history=history,
    )

    assert current.score == 1.0
    assert new_state.score is None
    assert usage["total_calls"] == 1


def test_meta_formats_recent_strategies_with_history_api():
    history = PromptStateHistory()
    for version in range(4):
        history.add(PromptState(strategy_text=f"strategy {version}", version=version))

    meta = MetaAgent(model="test-model", api_keys={}, task_context=polymer_ctx())
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted


def test_usage_summary_merge_handles_dicts_without_llmusage_crash():
    base = {
        "total_calls": 1,
        "total_prompt_tokens": 2,
        "total_completion_tokens": 3,
        "total_tokens": 5,
        "total_latency_s": 0.5,
        "by_model": {"worker": {"calls": 1, "tokens": 5}},
    }
    extra = {
        "total_calls": 2,
        "total_prompt_tokens": 7,
        "total_completion_tokens": 11,
        "total_tokens": 18,
        "total_latency_s": 1.0,
        "by_model": {"critic": {"calls": 2, "tokens": 18}},
    }

    merged = _merge_usage_summaries(base, extra)

    assert merged["total_calls"] == 3
    assert merged["total_tokens"] == 23
    assert merged["by_model"]["worker"]["calls"] == 1
    assert merged["by_model"]["critic"]["tokens"] == 18
