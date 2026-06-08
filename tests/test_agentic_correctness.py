"""Regression tests for critical agentic-mode correctness paths."""
from typing import List, Optional

from apo.agentic_engine import run_agentic_mode
from apo.agents.base import Action, Observation
from apo.agents.critic import CriticAgent
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage, aggregate_usage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.logging.run_logger import RunLogger
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"
MISSING_MARKER_CHILD = "CCO"


class StrictSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise AssertionError("predict() must receive a list, not a scalar string")
        self.calls.append(list(smiles_list))
        values = []
        for smiles in smiles_list:
            if "COCCOC" in smiles:
                values.append(2.0)
            elif smiles == MISSING_MARKER_CHILD:
                values.append(3.0)
            else:
                values.append(1.0)
        return values


POLYMER_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="polymer",
    smiles_markers=["[Cu]", "[Au]"],
    similarity_on_repeat_unit=True,
)


def test_worker_validation_uses_predict_single_and_enforces_task_markers():
    surrogate = StrictSurrogate()
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
            "explanation": "valid polymer",
        },
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": MISSING_MARKER_CHILD,
            "explanation": "missing markers",
        },
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 1.0
    assert candidates[0]["child_property"] == 2.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert candidates[1]["valid"] is False
    assert "Missing required marker" in candidates[1]["invalid_reason"]
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_property_tools_follow_surrogate_batch_contract():
    surrogate = StrictSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "TestProp").execute(VALID_CHILD)
    batch_obs = BatchPropertyPredictorTool(surrogate, "TestProp").execute([
        VALID_PARENT,
        VALID_CHILD,
    ])

    assert single_obs.success is True
    assert single_obs.result["TestProp"] == 2.0
    assert batch_obs.success is True
    assert [row["property"] for row in batch_obs.result] == [1.0, 2.0]
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_meta_formats_recent_strategies_without_history_api_crash():
    history = PromptStateHistory()
    history.add(PromptState.seed("seed strategy"))
    history.add(PromptState(strategy_text="first refinement", version=1))

    meta = MetaAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
    )
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v0: seed strategy" in formatted
    assert "v1: first refinement" in formatted


def test_critic_refine_scores_evaluated_current_state(monkeypatch):
    responses = iter([
        (
            '{"pareto_insights": ["good"], "failure_patterns": [], '
            '"unexplored_space": [], "tradeoffs": "none", "confidence": 1.0}'
        ),
        (
            '{"alternative_1": {"name": "Exploit", "strategy": "next", "rationale": "r"}, '
            '"alternative_2": {"name": "Explore", "strategy": "other", "rationale": "r2"}}'
        ),
        '{"consensus": "A", "consensus_rationale": "best", "confidence": 0.9}',
    ])

    def fake_call_llm(*args, **kwargs):
        return next(responses), LLMUsage("test-model", 1, 1, 0.0)

    monkeypatch.setattr("apo.agents.critic.call_llm", fake_call_llm)
    monkeypatch.setattr(
        CriticAgent,
        "_select_action",
        lambda self, thought: Action("noop", {}, ""),
    )
    monkeypatch.setattr(
        CriticAgent,
        "_execute_action",
        lambda self, action: Observation(success=True, result=None),
    )

    current = PromptState.seed("current")
    history = PromptStateHistory()
    history.add(current)
    critic = CriticAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        reward_fn=ParetoHypervolume(),
    )

    new_state, _, usage = critic.refine(
        candidates=[{"valid": True, "improvement_factor": 2.0, "similarity": 0.5}],
        current_state=current,
        history=history,
    )

    assert current.score == 1.0
    assert new_state.version == 1
    assert new_state.metadata["reward"] == 1.0
    assert usage["total_calls"] == 3


def test_agentic_engine_logs_evaluated_reward_with_usage_dicts(monkeypatch, tmp_path):
    class FakeWorker:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def generate(self, strategy, parent_smiles, n_per_molecule):
            return [
                {
                    "parent_smiles": VALID_PARENT,
                    "child_smiles": VALID_CHILD,
                    "valid": True,
                    "improvement_factor": 2.0,
                    "similarity": 0.5,
                    "parent_property": 1.0,
                    "child_property": 2.0,
                }
            ], [LLMUsage("worker-model", 3, 4, 0.1)]

    class FakeCritic:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def refine(self, candidates, current_state, history, meta_advice):
            assert current_state.score == 2.0
            new_state = PromptState(
                strategy_text="next",
                version=current_state.version + 1,
                parent_version=current_state.version,
            )
            return new_state, {"ok": True}, aggregate_usage([LLMUsage("critic-model", 5, 6, 0.2)])

    class FakeMeta:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def get_advice(self, history, reward_history):
            return "", aggregate_usage([LLMUsage("meta-model", 7, 8, 0.3)])

    monkeypatch.setattr("apo.agentic_engine.get_surrogate", lambda *args, **kwargs: StrictSurrogate())
    monkeypatch.setattr("apo.agentic_engine.WorkerAgent", FakeWorker)
    monkeypatch.setattr("apo.agentic_engine.CriticAgent", FakeCritic)
    monkeypatch.setattr("apo.agentic_engine.MetaAgent", FakeMeta)

    logger = RunLogger(str(tmp_path / "runs"))
    cfg = {
        "task": {"surrogate": "strict"},
        "models": {"worker": "worker-model", "critic": "critic-model", "meta": "meta-model"},
        "optimization": {
            "n_outer_epochs": 1,
            "n_per_molecule": 1,
            "batch_size": 1,
            "reward_function": "weighted_sum",
            "reward_function_kwargs": {"alpha": 1.0},
        },
    }

    run_agentic_mode(cfg, POLYMER_CTX, [VALID_PARENT], logger, {})

    records = logger.load_existing_epochs()
    assert len(records) == 1
    assert records[0]["reward"] == 2.0
    assert records[0]["prompt_state"]["version"] == 0
    assert records[0]["llm_usage"]["total_calls"] == 2
