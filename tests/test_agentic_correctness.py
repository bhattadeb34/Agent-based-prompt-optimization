from pathlib import Path
from typing import List, Optional

from apo.agentic_engine import run_agentic_mode
from apo.agents.critic import CriticAgent
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import WeightedSum
from apo.logging.run_logger import RunLogger
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise TypeError("predict expects a list of SMILES")
        return [1.0 if "CS" in smiles else 2.0 for smiles in smiles_list]


POLYMER_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="polymer",
    smiles_markers=["[Cu]", "[Au]"],
    similarity_on_repeat_unit=True,
)


GENERIC_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="organic compound",
)


def test_agentic_predictor_tools_use_single_smiles_wrapper():
    surrogate = StrictSurrogate()

    single = PropertyPredictorTool(surrogate, "score").execute("CCO")
    batch = BatchPropertyPredictorTool(surrogate, "score").execute(["CCO", "CCN"])

    assert single.success is True
    assert single.result["score"] == 2.0
    assert batch.success is True
    assert [row["property"] for row in batch.result] == [2.0, 2.0]


def test_worker_validation_uses_scalar_predictions_and_task_markers():
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        surrogate=StrictSurrogate(),
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "explanation": "valid polymer child",
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


def test_critic_persists_reward_and_returns_usage_objects(monkeypatch):
    critic = CriticAgent(
        model="critic-model",
        api_keys={},
        task_context=GENERIC_CTX,
        reward_fn=WeightedSum(alpha=0.5),
    )
    current = PromptState.seed("seed strategy")
    history = PromptStateHistory()
    history.add(current)
    usage = LLMUsage("critic-model", 3, 4, 0.1)

    def fake_run(initial_state=""):
        critic.new_state = PromptState(
            strategy_text="next strategy",
            version=current.version + 1,
            parent_version=current.version,
        )
        critic.all_usages.append(usage)
        return (critic.new_state, []), []

    monkeypatch.setattr(critic, "run", fake_run)
    monkeypatch.setattr(critic, "_save_trace_to_disk", lambda: None)

    new_state, _analysis, usages = critic.refine(
        candidates=[{"valid": True, "improvement_factor": 2.0, "similarity": 0.8}],
        current_state=current,
        history=history,
    )

    assert current.score == 1.4
    assert new_state.score == 1.4
    assert usages == [usage]


def test_meta_formats_recent_strategies_from_history():
    meta = MetaAgent(model="meta-model", api_keys={}, task_context=GENERIC_CTX)
    history = PromptStateHistory()
    history.add(PromptState.seed("first"))
    history.add(PromptState(strategy_text="second", version=1))
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v0: first" in formatted
    assert "v1: second" in formatted


def test_agentic_engine_aggregates_usage_lists_and_logs_real_reward(monkeypatch, tmp_path):
    usage = LLMUsage("test-model", 10, 5, 0.2)

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            self._interpretability_trace = {}

        def generate(self, strategy, parent_smiles, n_per_molecule):
            return [
                {
                    "parent_smiles": VALID_PARENT,
                    "child_smiles": VALID_CHILD,
                    "valid": True,
                    "improvement_factor": 2.0,
                    "similarity": 0.8,
                }
            ], [usage]

    class FakeCritic:
        def __init__(self, *args, **kwargs):
            self._interpretability_trace = {}

        def refine(self, candidates, current_state, history, meta_advice=""):
            current_state.score = 1.4
            return (
                PromptState(
                    strategy_text="next",
                    version=current_state.version + 1,
                    score=1.4,
                    parent_version=current_state.version,
                ),
                {},
                [usage],
            )

    class FakeMeta:
        def __init__(self, *args, **kwargs):
            self._interpretability_trace = {}

        def get_advice(self, history, reward_history):
            return "", [usage]

    monkeypatch.setattr("apo.agentic_engine.WorkerAgent", FakeWorker)
    monkeypatch.setattr("apo.agentic_engine.CriticAgent", FakeCritic)
    monkeypatch.setattr("apo.agentic_engine.MetaAgent", FakeMeta)
    monkeypatch.setattr("apo.agentic_engine.get_surrogate", lambda *args, **kwargs: StrictSurrogate())

    logger = RunLogger(str(tmp_path / "runs"))
    cfg = {
        "models": {"worker": "worker-model", "critic": "critic-model", "meta": "meta-model"},
        "task": {"surrogate": "strict"},
        "optimization": {
            "reward_function": "weighted_sum",
            "n_outer_epochs": 1,
            "n_per_molecule": 1,
            "batch_size": 1,
            "meta_interval": 1,
        },
    }

    run_dir = run_agentic_mode(cfg, GENERIC_CTX, [VALID_PARENT], logger, api_keys={})

    records = logger.load_existing_epochs()
    assert Path(run_dir).exists()
    assert records[0]["reward"] == 1.4
    assert records[0]["llm_usage"]["total_calls"] == 2
