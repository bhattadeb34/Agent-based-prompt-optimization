import json
from typing import List, Optional

from apo.agentic_engine import run_agentic_mode
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.logging.run_logger import RunLogger
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


class StrictSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self, values):
        self.values = values
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise TypeError("predict expects a list of SMILES")
        self.calls.append(list(smiles_list))
        return [self.values.get(smiles) for smiles in smiles_list]


class FakeValidator:
    name = "validate_smiles"

    def execute(self, smiles_list):
        return type(
            "Observation",
            (),
            {
                "success": True,
                "result": [{"smiles": smiles, "valid": True} for smiles in smiles_list],
            },
        )()


class FakeSimilarity:
    name = "calculate_similarity"

    def execute(self, smiles1, smiles2):
        return type(
            "Observation",
            (),
            {"success": True, "result": {"similarity": 0.75}},
        )()


def test_agentic_property_tools_call_single_smiles_wrapper():
    surrogate = StrictSurrogate({"CC": 1.0, "CCO": 2.0})

    single = PropertyPredictorTool(surrogate, "TestProp").execute("CCO")
    batch = BatchPropertyPredictorTool(surrogate, "TestProp").execute(["CC", "CCO"])

    assert single.success is True
    assert single.result["TestProp"] == 2.0
    assert batch.success is True
    assert [row["property"] for row in batch.result] == [1.0, 2.0]
    assert surrogate.calls == [["CCO"], ["CC"], ["CCO"]]


def test_worker_validation_scores_candidates_with_scalar_predictions():
    surrogate = StrictSurrogate({"CC": 1.0, "CCO": 2.0})
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=TaskContext(property_name="TestProp", property_units="units"),
        surrogate=surrogate,
        parent_cache={},
        max_retries_per_batch=1,
    )
    worker.tools = [FakeValidator(), FakeSimilarity()]

    candidates = worker._validate_candidates(
        [{"parent_smiles": "CC", "child_smiles": "CCO", "explanation": "add oxygen"}]
    )

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 1.0
    assert candidates[0]["child_property"] == 2.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert candidates[0]["similarity"] == 0.75
    assert surrogate.calls == [["CC"], ["CCO"]]


def test_meta_formats_recent_strategies_from_prompt_state_history():
    history = PromptStateHistory()
    for i in range(4):
        history.add(PromptState(strategy_text=f"strategy {i}", version=i))

    meta = MetaAgent(
        model="test-model",
        api_keys={},
        task_context=TaskContext(property_name="TestProp", property_units="units"),
    )
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v1: strategy 1" in formatted
    assert "v2: strategy 2" in formatted
    assert "v3: strategy 3" in formatted
    assert "v0: strategy 0" not in formatted


def test_agentic_mode_scores_new_state_and_keeps_aggregate_usage_dicts_out_of_raw_usage(monkeypatch, tmp_path):
    worker_usage = LLMUsage("worker-model", 10, 5, 0.1)
    critic_usage = {
        "total_calls": 2,
        "total_tokens": 30,
        "by_model": {"critic-model": {"calls": 2, "tokens": 30}},
    }
    meta_usage = {
        "total_calls": 1,
        "total_tokens": 7,
        "by_model": {"meta-model": {"calls": 1, "tokens": 7}},
    }

    class FakeWorker:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def generate(self, strategy, parent_smiles, n_per_molecule):
            return (
                [
                    {
                        "parent_smiles": "CC",
                        "child_smiles": "CCO",
                        "valid": True,
                        "improvement_factor": 2.0,
                        "similarity": 0.75,
                        "child_property": 2.0,
                        "parent_property": 1.0,
                    }
                ],
                [worker_usage],
            )

    class FakeCritic:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def refine(self, candidates, current_state, history, meta_advice):
            return (
                PromptState(
                    strategy_text="next strategy",
                    version=current_state.version + 1,
                    parent_version=current_state.version,
                ),
                {"analysis": "ok"},
                critic_usage,
            )

    class FakeMeta:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def get_advice(self, history, reward_history):
            return "", meta_usage

    monkeypatch.setattr("apo.agentic_engine.get_surrogate", lambda *args, **kwargs: StrictSurrogate({}))
    monkeypatch.setattr("apo.agentic_engine.WorkerAgent", FakeWorker)
    monkeypatch.setattr("apo.agentic_engine.CriticAgent", FakeCritic)
    monkeypatch.setattr("apo.agentic_engine.MetaAgent", FakeMeta)

    logger = RunLogger(str(tmp_path), run_id="agentic")
    cfg = {
        "task": {"surrogate": "mock"},
        "models": {"worker": "worker-model", "critic": "critic-model", "meta": "meta-model"},
        "optimization": {
            "n_outer_epochs": 1,
            "n_per_molecule": 1,
            "batch_size": 1,
            "meta_interval": 1,
            "reward_function": "pareto_hypervolume",
        },
    }

    run_agentic_mode(
        cfg=cfg,
        ctx=TaskContext(property_name="TestProp", property_units="units"),
        all_smiles=["CC"],
        logger=logger,
        api_keys={},
    )

    records = logger.load_existing_epochs()
    assert records[0]["reward"] == 1.5
    assert records[0]["prompt_state"]["score"] == 1.5
    assert records[0]["llm_usage"]["total_calls"] == 3
    assert records[0]["llm_usage"]["total_tokens"] == 45

    with open(tmp_path / "agentic" / "prompt_history.json") as f:
        prompt_history = json.load(f)
    assert prompt_history[-1]["score"] == 1.5
