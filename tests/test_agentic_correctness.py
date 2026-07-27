"""Regression tests for critical agentic workflow correctness."""
from typing import List, Optional

from apo.agentic_engine import run_agentic_mode
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.logging.run_logger import RunLogger
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


class StrictSurrogate(SurrogatePredictor):
    property_name = "StrictProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise TypeError("predict expects a list of SMILES, not a scalar string")
        self.calls.append(list(smiles_list))
        return [float(len(smi)) for smi in smiles_list]


def test_agentic_worker_uses_list_safe_surrogate_api():
    surrogate = StrictSurrogate()
    ctx = TaskContext(property_name="StrictProp", molecule_type="organic", smiles_markers=[])
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=ctx,
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {"parent_smiles": "CCO", "child_smiles": "CCN", "explanation": "replace O with N"}
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 3.0
    assert candidates[0]["child_property"] == 3.0
    assert surrogate.calls == [["CCO"], ["CCN"]]


def test_agentic_worker_enforces_task_markers():
    surrogate = StrictSurrogate()
    ctx = TaskContext(
        property_name="StrictProp",
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
    )
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=ctx,
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": "CC(CO[Cu])CSCCOC(=O)[Au]",
            "child_smiles": "CCO",
            "explanation": "drops polymer markers",
        }
    ])

    assert candidates[0]["valid"] is False
    assert "Missing required marker" in candidates[0]["invalid_reason"]
    assert surrogate.calls == [["CC(CO[Cu])CSCCOC(=O)[Au]"]]


def test_agentic_property_tools_do_not_pass_scalar_strings_to_predict():
    surrogate = StrictSurrogate()

    single = PropertyPredictorTool(surrogate, "StrictProp").execute("CCO")
    batch = BatchPropertyPredictorTool(surrogate, "StrictProp").execute(["CCO", "CCCC"])

    assert single.success is True
    assert single.result["StrictProp"] == 3.0
    assert batch.success is True
    assert [row["property"] for row in batch.result] == [3.0, 4.0]
    assert surrogate.calls == [["CCO"], ["CCO", "CCCC"]]


def test_meta_agent_formats_recent_history_without_missing_all_method():
    ctx = TaskContext(property_name="StrictProp", molecule_type="organic")
    history = PromptStateHistory()
    for i in range(4):
        history.add(PromptState(strategy_text=f"strategy {i}", version=i))

    meta = MetaAgent(model="test-model", api_keys={}, task_context=ctx)
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted


def test_agentic_engine_logs_current_reward_and_merges_dict_usage(monkeypatch, tmp_path):
    class FakeWorker:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def generate(self, strategy, parent_smiles, n_per_molecule):
            return [
                {
                    "parent_smiles": "CCO",
                    "child_smiles": "CCN",
                    "valid": True,
                    "parent_property": 1.0,
                    "child_property": 2.0,
                    "improvement_factor": 2.0,
                    "similarity": 0.5,
                }
            ], [LLMUsage("worker-model", 10, 5, 0.1)]

    class FakeCritic:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def refine(self, candidates, current_state, history, meta_advice=""):
            current_state.score = ParetoHypervolume().compute(candidates)
            return (
                PromptState(
                    strategy_text="next",
                    version=current_state.version + 1,
                    parent_version=current_state.version,
                ),
                {"analysis": "ok"},
                {"total_calls": 1, "total_tokens": 7, "by_model": {"critic-model": {"calls": 1, "tokens": 7}}},
            )

    class FakeMeta:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def get_advice(self, history, reward_history):
            return "", {"total_calls": 1, "total_tokens": 3, "by_model": {"meta-model": {"calls": 1, "tokens": 3}}}

    monkeypatch.setattr("apo.agentic_engine.get_surrogate", lambda *args, **kwargs: StrictSurrogate())
    monkeypatch.setattr("apo.agentic_engine.WorkerAgent", FakeWorker)
    monkeypatch.setattr("apo.agentic_engine.CriticAgent", FakeCritic)
    monkeypatch.setattr("apo.agentic_engine.MetaAgent", FakeMeta)

    cfg = {
        "task": {"surrogate": "strict"},
        "models": {"worker": "worker-model", "critic": "critic-model", "meta": "meta-model"},
        "optimization": {"n_outer_epochs": 1, "n_per_molecule": 1, "batch_size": 1, "meta_interval": 1},
    }
    ctx = TaskContext(property_name="StrictProp", molecule_type="organic", seed_strategy="seed")
    logger = RunLogger(str(tmp_path), run_id="agentic")

    run_agentic_mode(cfg=cfg, ctx=ctx, all_smiles=["CCO"], logger=logger, api_keys={})

    [record] = logger.load_existing_epochs()
    assert record["reward"] == 1.0
    assert record["prompt_state"]["version"] == 0
    assert record["llm_usage"]["total_calls"] == 2
    assert record["llm_usage"]["total_tokens"] == 22
