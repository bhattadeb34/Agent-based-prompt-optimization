"""Regression tests for critical agentic-mode correctness paths."""
from typing import List, Optional
from unittest.mock import patch

from apo.agentic_engine import run_agentic_mode
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage, aggregate_usage
from apo.core.prompt_state import PromptState
from apo.logging.run_logger import RunLogger
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


class StrictSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        assert isinstance(smiles_list, list), "predict() must receive a list, not a bare string"
        self.calls.append(list(smiles_list))
        values = {"CC": 1.0, "CCO": 2.0, "CCC": 3.0}
        return [values.get(smi, 1.0) for smi in smiles_list]


GENERIC_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="organic compound",
)


def test_worker_validation_uses_single_smiles_predictor_contract():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=GENERIC_CTX,
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([{
        "parent_smiles": "CC",
        "child_smiles": "CCO",
        "explanation": "add oxygen",
    }])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 1.0
    assert candidates[0]["child_property"] == 2.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert surrogate.calls == [["CC"], ["CCO"]]


def test_property_tools_preserve_scalar_and_batch_predict_contracts():
    surrogate = StrictSurrogate()

    single = PropertyPredictorTool(surrogate, "TestProp").execute("CC")
    batch = BatchPropertyPredictorTool(surrogate, "TestProp").execute(["CC", "CCC"])

    assert single.success is True
    assert single.result["TestProp"] == 1.0
    assert batch.success is True
    assert [row["property"] for row in batch.result] == [1.0, 3.0]
    assert surrogate.calls == [["CC"], ["CC", "CCC"]]


def test_agentic_mode_scores_logged_strategy_and_merges_usage(tmp_path):
    worker_usage = LLMUsage("worker-model", 10, 5, 0.25)
    critic_usage = aggregate_usage([LLMUsage("critic-model", 7, 3, 0.5)])
    meta_usage = aggregate_usage([LLMUsage("meta-model", 4, 2, 0.1)])

    class FakeWorker:
        def __init__(self, **kwargs):
            self._interpretability_trace = {"agent": "worker"}

        def generate(self, strategy, parent_smiles, n_per_molecule):
            return ([{
                "parent_smiles": "CC",
                "child_smiles": "CCO",
                "valid": True,
                "parent_property": 1.0,
                "child_property": 2.0,
                "improvement_factor": 2.0,
                "similarity": 0.5,
                "explanation": "add oxygen",
            }], [worker_usage])

    class FakeCritic:
        def __init__(self, **kwargs):
            self._interpretability_trace = {"agent": "critic"}

        def refine(self, candidates, current_state, history, meta_advice=""):
            assert current_state.score == 1.0
            return (
                PromptState(
                    strategy_text="next strategy",
                    version=current_state.version + 1,
                    rationale="test",
                    parent_version=current_state.version,
                ),
                {"analysis": "ok"},
                critic_usage,
            )

    class FakeMeta:
        def __init__(self, **kwargs):
            self._interpretability_trace = {"agent": "meta"}

        def get_advice(self, history, reward_history):
            return "keep going", meta_usage

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
        run_agentic_mode(cfg, GENERIC_CTX, ["CC"], logger, api_keys={})

    records = logger.load_existing_epochs()
    assert len(records) == 1
    assert records[0]["reward"] == 1.0
    assert records[0]["prompt_state"]["version"] == 0
    assert records[0]["llm_usage"]["total_calls"] == 2
    assert logger.reward_history == [1.0]
