"""Regression tests for the agentic optimization engine."""
import json
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

from apo.agentic_engine import run_agentic_mode
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState
from apo.logging.run_logger import RunLogger
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class MockSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        return [1.0] * len(smiles_list)


POLYMER_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="polymer",
    domain_context="[Cu] and [Au] are backbone markers.",
    smiles_markers=["[Cu]", "[Au]"],
    similarity_on_repeat_unit=True,
)


class FakeWorkerAgent:
    def __init__(self, **kwargs):
        self._interpretability_trace = {"agent": "worker"}

    def generate(self, strategy, parent_smiles, n_per_molecule):
        return (
            [{
                "parent_smiles": VALID_PARENT,
                "child_smiles": VALID_CHILD,
                "improvement_factor": 1.5,
                "similarity": 0.8,
                "valid": True,
                "child_property": 1.5,
                "parent_property": 1.0,
                "explanation": "test",
                "invalid_reason": "",
            }],
            [LLMUsage("worker-model", 10, 5, 0.1)],
        )


class FakeCriticAgent:
    def __init__(self, **kwargs):
        self._interpretability_trace = {"agent": "critic"}

    def refine(self, candidates, current_state, history, meta_advice=""):
        new_state = PromptState(
            strategy_text="refined strategy",
            version=current_state.version + 1,
            score=None,
            rationale="test",
            parent_version=current_state.version,
        )
        usage = {
            "total_calls": 1,
            "total_prompt_tokens": 20,
            "total_completion_tokens": 10,
            "total_tokens": 30,
            "total_latency_s": 0.2,
            "avg_latency_s": 0.2,
            "by_model": {"critic-model": {"calls": 1, "tokens": 30}},
        }
        return new_state, {"analysis": "ok"}, usage


class FakeMetaAgent:
    def __init__(self, **kwargs):
        self._interpretability_trace = {"agent": "meta"}

    def get_advice(self, history, reward_history):
        usage = {
            "total_calls": 1,
            "total_prompt_tokens": 6,
            "total_completion_tokens": 4,
            "total_tokens": 10,
            "total_latency_s": 0.1,
            "avg_latency_s": 0.1,
            "by_model": {"meta-model": {"calls": 1, "tokens": 10}},
        }
        return "", usage


def test_agentic_mode_merges_aggregated_usage_and_records_computed_reward(tmp_path):
    cfg = {
        "models": {"worker": "worker-model", "critic": "critic-model", "meta": "meta-model"},
        "optimization": {
            "n_outer_epochs": 1,
            "n_per_molecule": 1,
            "batch_size": 1,
            "meta_interval": 1,
            "reward_function": "pareto_hypervolume",
        },
        "task": {"surrogate": "mock"},
    }
    logger = RunLogger(str(tmp_path / "runs"))

    with patch("apo.agentic_engine.get_surrogate", return_value=MockSurrogate()), \
         patch("apo.agentic_engine.WorkerAgent", FakeWorkerAgent), \
         patch("apo.agentic_engine.CriticAgent", FakeCriticAgent), \
         patch("apo.agentic_engine.MetaAgent", FakeMetaAgent):
        run_agentic_mode(cfg, POLYMER_CTX, [VALID_PARENT], logger, api_keys={})

    records = logger.load_existing_epochs()
    assert len(records) == 1
    assert records[0]["llm_usage"]["total_calls"] == 2
    assert records[0]["llm_usage"]["total_tokens"] == 45
    assert records[0]["reward"] == 1.2
    assert records[0]["prompt_state"]["score"] == 1.2

    prompt_history = json.loads((Path(logger.run_dir) / "prompt_history.json").read_text())
    assert prompt_history[-1]["score"] == 1.2
