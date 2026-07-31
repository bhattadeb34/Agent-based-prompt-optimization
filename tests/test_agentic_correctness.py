"""Regression tests for critical agentic workflow correctness paths."""
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

from apo.agentic_engine import run_agentic_mode
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage, aggregate_usage
from apo.core.prompt_state import PromptState
from apo.logging.run_logger import RunLogger
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictListSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise TypeError("predict expects a list of SMILES, not a scalar string")
        self.calls.append(list(smiles_list))
        return [2.0 if "COCC" in smiles else 1.0 for smiles in smiles_list]


def polymer_context(maximize: bool = True) -> TaskContext:
    return TaskContext(
        property_name="TestProp",
        property_units="units",
        maximize=maximize,
        molecule_type="polymer",
        domain_context="[Cu] and [Au] are backbone markers.",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_worker_validation_uses_scalar_safe_prediction_and_task_markers():
    surrogate = StrictListSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {"parent_smiles": PARENT, "child_smiles": CHILD, "explanation": "valid"},
        {"parent_smiles": PARENT, "child_smiles": "CCO", "explanation": "missing markers"},
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 1.0
    assert candidates[0]["child_property"] == 2.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert candidates[1]["valid"] is False
    assert "Missing required marker" in candidates[1]["invalid_reason"]
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_property_tools_honor_list_predictor_contract():
    surrogate = StrictListSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "TestProp").execute(CHILD)
    assert single_obs.success is True
    assert single_obs.result["TestProp"] == 2.0

    batch_obs = BatchPropertyPredictorTool(surrogate, "TestProp").execute([PARENT, CHILD])
    assert batch_obs.success is True
    assert [row["property"] for row in batch_obs.result] == [1.0, 2.0]
    assert surrogate.calls[-1] == [PARENT, CHILD]


def test_meta_agent_formats_recent_strategies_without_missing_history_method():
    class BareMeta(MetaAgent):
        def _init_tools(self):
            return []

    meta = BareMeta(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
    )
    history = type("History", (), {"get_recent": lambda self, n: [
        PromptState(strategy_text="strategy", version=1),
    ]})()
    meta.history = history

    assert "v1: strategy" in meta._format_recent_strategies()


class FakeWorker:
    def __init__(self, *args, **kwargs):
        self._interpretability_trace = {}

    def generate(self, *args, **kwargs):
        return ([{
            "parent_smiles": PARENT,
            "child_smiles": CHILD,
            "parent_property": 1.0,
            "child_property": 2.0,
            "improvement_factor": 2.0,
            "similarity": 0.5,
            "valid": True,
            "explanation": "valid",
        }], [LLMUsage("worker-model", 10, 5, 0.1)])


class FakeCritic:
    def __init__(self, *args, **kwargs):
        self._interpretability_trace = {}

    def refine(self, candidates, current_state, history, meta_advice=""):
        assert current_state.score == 1.0
        usage = aggregate_usage([LLMUsage("critic-model", 20, 10, 0.2)])
        return (
            PromptState(
                strategy_text="next strategy",
                version=current_state.version + 1,
                parent_version=current_state.version,
            ),
            {"analysis": "ok"},
            usage,
        )


class FakeMeta:
    def __init__(self, *args, **kwargs):
        self._interpretability_trace = {}

    def get_advice(self, history, reward_history):
        usage = aggregate_usage([LLMUsage("meta-model", 5, 5, 0.1)])
        return "keep going", usage


def test_agentic_engine_scores_current_state_and_merges_aggregate_usage(tmp_path):
    cfg = {
        "task": {"surrogate": "strict"},
        "models": {"worker": "worker-model", "critic": "critic-model", "meta": "meta-model"},
        "optimization": {
            "n_outer_epochs": 1,
            "n_per_molecule": 1,
            "batch_size": 1,
            "meta_interval": 1,
            "reward_function": "pareto_hypervolume",
        },
    }
    logger = RunLogger(str(tmp_path / "runs"))

    with patch("apo.agentic_engine.get_surrogate", return_value=StrictListSurrogate()), \
         patch("apo.agentic_engine.WorkerAgent", FakeWorker), \
         patch("apo.agentic_engine.CriticAgent", FakeCritic), \
         patch("apo.agentic_engine.MetaAgent", FakeMeta):
        run_agentic_mode(cfg, polymer_context(), [PARENT], logger, api_keys={})

    records = logger.load_existing_epochs()
    assert len(records) == 1
    assert records[0]["reward"] == 1.0
    assert records[0]["prompt_state"]["version"] == 0
    assert (Path(logger.run_dir) / "prompt_history.json").exists()
