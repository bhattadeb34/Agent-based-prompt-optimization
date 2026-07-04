from typing import Dict, List, Optional

import pytest

from apo import agentic_engine
from apo.agentic_engine import run_agentic_mode
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState
from apo.logging.run_logger import RunLogger
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


class StrictSurrogate(SurrogatePredictor):
    property_name = "StrictProp"
    property_units = "units"
    maximize = True

    def __init__(self, values: Optional[Dict[str, float]] = None):
        self.values = values or {}
        self.calls: List[List[str]] = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        assert isinstance(smiles_list, list), "predict expects a list of SMILES"
        assert not isinstance(smiles_list, str), "strings must use predict_single"
        self.calls.append(list(smiles_list))
        return [self.values.get(smiles, 1.0) for smiles in smiles_list]


def test_property_tools_use_surrogate_list_contract():
    surrogate = StrictSurrogate({"CC": 1.0, "CCO": 2.0})

    single_obs = PropertyPredictorTool(surrogate, "StrictProp").execute("CC")
    batch_obs = BatchPropertyPredictorTool(surrogate, "StrictProp").execute(["CC", "CCO"])

    assert single_obs.success is True
    assert single_obs.result["StrictProp"] == 1.0
    assert batch_obs.success is True
    assert [r["property"] for r in batch_obs.result] == [1.0, 2.0]
    assert surrogate.calls == [["CC"], ["CC", "CCO"]]


def test_worker_validation_is_task_aware_and_minimize_safe():
    ctx = TaskContext(
        property_name="Loss",
        property_units="",
        maximize=False,
        molecule_type="organic compound",
        smiles_markers=[],
    )
    surrogate = StrictSurrogate({"CC": 10.0, "CCO": 5.0})
    worker = WorkerAgent(
        model="mock",
        api_keys={},
        task_context=ctx,
        surrogate=surrogate,
        parent_cache={},
    )

    [candidate] = worker._validate_candidates([
        {"parent_smiles": "CC", "child_smiles": "CCO", "explanation": "lower property"}
    ])

    assert candidate["valid"] is True
    assert candidate["parent_property"] == 10.0
    assert candidate["child_property"] == 5.0
    assert candidate["improvement_factor"] == pytest.approx(2.0)
    assert surrogate.calls == [["CC"], ["CCO"]]


def test_worker_rejects_missing_required_markers_after_rdkit_validation():
    ctx = TaskContext(
        property_name="Conductivity",
        property_units="",
        maximize=True,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
    )
    worker = WorkerAgent(
        model="mock",
        api_keys={},
        task_context=ctx,
        surrogate=StrictSurrogate({"CC": 1.0}),
        parent_cache={},
    )

    [candidate] = worker._validate_candidates([
        {"parent_smiles": "CC", "child_smiles": "CCO", "explanation": "missing markers"}
    ])

    assert candidate["valid"] is False
    assert candidate["invalid_reason"] == "Missing required marker: [Cu]"


def test_agentic_engine_logs_reward_on_evaluated_current_state(monkeypatch, tmp_path):
    class FakeWorker:
        _interpretability_trace = {}

        def __init__(self, *args, **kwargs):
            pass

        def generate(self, strategy, parent_smiles, n_per_molecule):
            return [
                {
                    "parent_smiles": "CC",
                    "child_smiles": "CCO",
                    "valid": True,
                    "parent_property": 1.0,
                    "child_property": 2.0,
                    "improvement_factor": 2.0,
                    "similarity": 0.5,
                    "explanation": "improves property",
                }
            ], [LLMUsage("worker-model", 1, 1, 0.1)]

    class FakeCritic:
        _interpretability_trace = {}

        def __init__(self, *args, **kwargs):
            pass

        def refine(self, candidates, current_state, history, meta_advice=""):
            assert current_state.score == pytest.approx(1.0)
            return (
                PromptState(
                    strategy_text="next strategy",
                    version=current_state.version + 1,
                    rationale="refined",
                    parent_version=current_state.version,
                    model_used="critic-model",
                ),
                {"pareto_insights": ["ok"]},
                {
                    "total_calls": 1,
                    "total_tokens": 2,
                    "by_model": {"critic-model": {"calls": 1, "tokens": 2}},
                },
            )

    class FakeMeta:
        _interpretability_trace = {}

        def __init__(self, *args, **kwargs):
            pass

        def get_advice(self, history, reward_history):
            return "", {
                "total_calls": 1,
                "total_tokens": 2,
                "by_model": {"meta-model": {"calls": 1, "tokens": 2}},
            }

    monkeypatch.setattr(agentic_engine, "get_surrogate", lambda *args, **kwargs: StrictSurrogate())
    monkeypatch.setattr(agentic_engine, "WorkerAgent", FakeWorker)
    monkeypatch.setattr(agentic_engine, "CriticAgent", FakeCritic)
    monkeypatch.setattr(agentic_engine, "MetaAgent", FakeMeta)

    logger = RunLogger(str(tmp_path), run_id="agentic")
    cfg = {
        "task": {"surrogate": "strict"},
        "models": {"worker": "worker-model", "critic": "critic-model", "meta": "meta-model"},
        "optimization": {
            "n_outer_epochs": 1,
            "n_per_molecule": 1,
            "batch_size": 1,
            "meta_interval": 1,
            "reward_function": "pareto_hypervolume",
            "seed_strategy": "seed",
        },
    }
    ctx = TaskContext(property_name="StrictProp", maximize=True, seed_strategy="seed")

    run_agentic_mode(cfg, ctx, ["CC"], logger, api_keys={})

    [record] = logger.load_existing_epochs()
    assert record["reward"] == pytest.approx(1.0)
    assert record["prompt_state"]["version"] == 0
    assert record["prompt_state"]["score"] == pytest.approx(1.0)
    assert record["llm_usage"]["total_calls"] == 2
