from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

from apo.agentic_engine import _merge_usage_summaries
from apo.agents.critic import CriticAgent
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.engine import _run_agent_mode
from apo.logging.run_logger import RunLogger
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


class StrictListSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self, values=None):
        self.values = values or {}
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        assert isinstance(smiles_list, list), "predict must receive a list"
        self.calls.append(list(smiles_list))
        return [self.values.get(smi, 1.0) for smi in smiles_list]


def test_worker_uses_task_context_and_scalar_predictor_api():
    ctx = TaskContext(
        property_name="TestProp",
        property_units="units",
        maximize=True,
        smiles_markers=["*"],
        similarity_on_repeat_unit=True,
        marker_strip_tokens=["*"],
    )
    surrogate = StrictListSurrogate({"*CC*": 2.0, "*CCO*": 3.0})
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=ctx,
        surrogate=surrogate,
        parent_cache={},
    )

    results = worker._validate_candidates([
        {"parent_smiles": "*CC*", "child_smiles": "CCO", "explanation": "missing marker"},
        {"parent_smiles": "*CC*", "child_smiles": "*CCO*", "explanation": "valid"},
    ])

    assert results[0]["valid"] is False
    assert "Missing required marker" in results[0]["invalid_reason"]
    assert results[1]["valid"] is True
    assert results[1]["parent_property"] == 2.0
    assert results[1]["child_property"] == 3.0
    assert results[1]["improvement_factor"] == 1.5
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_worker_respects_minimization_direction():
    ctx = TaskContext(
        property_name="Energy",
        property_units="eV",
        maximize=False,
        molecule_type="organic compound",
    )
    surrogate = StrictListSurrogate({"CC": 10.0, "CO": 5.0})
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=ctx,
        surrogate=surrogate,
        parent_cache={},
    )

    result = worker._validate_candidates([
        {"parent_smiles": "CC", "child_smiles": "CO", "explanation": "lower is better"},
    ])[0]

    assert result["valid"] is True
    assert result["improvement_factor"] == 2.0


def test_property_tools_use_surrogate_list_api():
    surrogate = StrictListSurrogate({"CC": 2.0, "CO": 3.0})

    single = PropertyPredictorTool(surrogate, "TestProp").execute("CC")
    batch = BatchPropertyPredictorTool(surrogate, "TestProp").execute(["CC", "CO"])

    assert single.success is True
    assert single.result["TestProp"] == 2.0
    assert batch.success is True
    assert [row["property"] for row in batch.result] == [2.0, 3.0]
    assert surrogate.calls == [["CC"], ["CC", "CO"]]


def test_critic_scores_evaluated_current_state_without_llm(monkeypatch):
    ctx = TaskContext(property_name="TestProp", property_units="units")
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
        candidates=[{"valid": True, "improvement_factor": 2.0, "similarity": 0.5}],
        current_state=current,
        history=history,
    )

    assert current.score == 1.0
    assert new_state.score is None
    assert new_state.metadata["reward"] == 1.0
    assert usage["total_calls"] == 0


def test_meta_agent_formats_recent_history_without_all_accessor():
    ctx = TaskContext(property_name="TestProp", property_units="units")
    meta = MetaAgent(model="test-model", api_keys={}, task_context=ctx)
    history = PromptStateHistory()
    for version in range(4):
        history.add(PromptState(strategy_text=f"strategy {version}", version=version))
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted


def test_agentic_usage_summary_merges_nested_dicts():
    merged = _merge_usage_summaries(
        {
            "total_calls": 1,
            "total_prompt_tokens": 2,
            "total_completion_tokens": 3,
            "total_tokens": 5,
            "total_latency_s": 0.5,
            "by_model": {"a": {"calls": 1, "tokens": 5}},
        },
        {
            "total_calls": 2,
            "total_prompt_tokens": 4,
            "total_completion_tokens": 6,
            "total_tokens": 10,
            "total_latency_s": 1.0,
            "by_model": {"a": {"calls": 1, "tokens": 4}, "b": {"calls": 1, "tokens": 6}},
        },
    )

    assert merged["total_calls"] == 3
    assert merged["total_tokens"] == 15
    assert merged["by_model"]["a"] == {"calls": 2, "tokens": 9}
    assert merged["by_model"]["b"] == {"calls": 1, "tokens": 6}


def test_agent_mode_saves_prompt_history(tmp_path):
    ctx = TaskContext(property_name="TestProp", property_units="units")
    logger = RunLogger(str(tmp_path))
    history = PromptStateHistory()
    current = PromptState.seed("seed")
    history.add(current)

    class FakeAgent:
        def __init__(self, **kwargs):
            self.history = kwargs["history"]
            self.logger = kwargs["logger"]
            self._current_state = None

        def run(self):
            self.history.add(PromptState(strategy_text="next", version=1))
            return str(self.logger.run_dir)

        def total_usage(self):
            return {"total_calls": 0, "total_tokens": 0, "by_model": {}}

    with patch("apo.agent.OrchestratorAgent", FakeAgent):
        _run_agent_mode(
            cfg={},
            ctx=ctx,
            inner=None,
            outer=None,
            meta=None,
            logger=logger,
            history=history,
            current_state=current,
            reward_fn=ParetoHypervolume(),
            all_smiles=[],
            model_cfg={"meta": "test-model"},
            api_keys={},
            opt_cfg={"tool_budget": 1},
        )

    assert Path(logger.run_dir, "prompt_history.json").exists()
