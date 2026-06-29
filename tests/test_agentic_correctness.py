from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

from apo.agentic_engine import run_agentic_mode
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState
from apo.logging.run_logger import RunLogger
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictSurrogate(SurrogatePredictor):
    property_name = "StrictProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls: List[List[str]] = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise TypeError("predict expects a list of SMILES")
        self.calls.append(list(smiles_list))
        values = []
        for smiles in smiles_list:
            if smiles == VALID_PARENT:
                values.append(1.0)
            elif "[Cu]" in smiles and "[Au]" in smiles:
                values.append(2.0)
            elif smiles == "CCO":
                values.append(1.5)
            elif smiles == "CCN":
                values.append(1.7)
            else:
                values.append(1.0)
        return values


def polymer_context() -> TaskContext:
    return TaskContext(
        property_name="StrictProp",
        property_units="units",
        maximize=True,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_worker_agent_uses_list_safe_predictions_and_enforces_markers():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
        surrogate=surrogate,
        parent_cache={},
    )

    validated = worker._validate_candidates([
        {"parent_smiles": VALID_PARENT, "child_smiles": VALID_CHILD, "explanation": "valid"},
        {"parent_smiles": VALID_PARENT, "child_smiles": "CCO", "explanation": "missing markers"},
    ])

    assert validated[0]["valid"] is True
    assert validated[0]["parent_property"] == 1.0
    assert validated[0]["child_property"] == 2.0
    assert validated[0]["improvement_factor"] == 2.0
    assert "Missing required marker" in validated[1]["invalid_reason"]
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_property_tools_use_single_and_batch_surrogate_apis():
    surrogate = StrictSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "StrictProp").execute("CCO")
    batch_obs = BatchPropertyPredictorTool(surrogate, "StrictProp").execute(["CCO", "CCN"])

    assert single_obs.success is True
    assert single_obs.result["StrictProp"] == 1.5
    assert batch_obs.success is True
    assert [r["property"] for r in batch_obs.result] == [1.5, 1.7]
    assert surrogate.calls == [["CCO"], ["CCO", "CCN"]]


class FakeWorker:
    def __init__(self, *args, **kwargs):
        self._interpretability_trace = {"worker": "trace"}

    def generate(self, strategy, parent_smiles, n_per_molecule):
        return [
            {
                "parent_smiles": VALID_PARENT,
                "child_smiles": VALID_CHILD,
                "valid": True,
                "parent_property": 1.0,
                "child_property": 2.0,
                "improvement_factor": 2.0,
                "similarity": 0.5,
            }
        ], [LLMUsage("worker-model", 10, 5, 0.1)]


class FakeCritic:
    def __init__(self, *args, **kwargs):
        self._interpretability_trace = {"critic": "trace"}

    def refine(self, candidates, current_state, history, meta_advice=""):
        return (
            PromptState(
                strategy_text="next strategy",
                version=current_state.version + 1,
                parent_version=current_state.version,
                rationale="test",
                model_used="critic-model",
            ),
            {"analysis": "ok"},
            {
                "total_calls": 1,
                "total_prompt_tokens": 20,
                "total_completion_tokens": 10,
                "total_tokens": 30,
                "total_latency_s": 0.2,
                "by_model": {"critic-model": {"calls": 1, "tokens": 30}},
            },
        )


class FakeMeta:
    def __init__(self, *args, **kwargs):
        self._interpretability_trace = {"meta": "trace"}

    def get_advice(self, history, reward_history):
        return "", {
            "total_calls": 1,
            "total_prompt_tokens": 5,
            "total_completion_tokens": 5,
            "total_tokens": 10,
            "total_latency_s": 0.1,
            "by_model": {"meta-model": {"calls": 1, "tokens": 10}},
        }


def test_run_agentic_mode_logs_evaluated_reward_and_merges_usage(tmp_path):
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

    with patch("apo.agentic_engine.get_surrogate", return_value=StrictSurrogate()), \
         patch("apo.agentic_engine.WorkerAgent", FakeWorker), \
         patch("apo.agentic_engine.CriticAgent", FakeCritic), \
         patch("apo.agentic_engine.MetaAgent", FakeMeta):
        run_agentic_mode(cfg, polymer_context(), [VALID_PARENT], logger, {})

    records = logger.load_existing_epochs()
    assert records[0]["reward"] == 1.0
    assert records[0]["prompt_state"]["version"] == 0
    assert records[0]["prompt_state"]["score"] == 1.0
    assert records[0]["llm_usage"]["total_calls"] == 2
    assert (Path(logger.run_dir) / "prompt_history.json").exists()


def test_meta_agent_formats_recent_strategies_without_history_all_method():
    history_type = type("History", (), {"get_recent": lambda self, n: [PromptState("strategy", version=1)]})
    meta = MetaAgent("meta-model", {}, polymer_context())
    meta.history = history_type()

    assert "v1: strategy" in meta._format_recent_strategies()
