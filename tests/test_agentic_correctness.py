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


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictListSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        assert isinstance(smiles_list, list), "surrogate.predict requires a list"
        self.calls.append(list(smiles_list))
        values = []
        for smiles in smiles_list:
            values.append(2.0 if smiles == VALID_CHILD else 1.0)
        return values


POLYMER_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="polymer",
    domain_context="[Cu] and [Au] are required polymer markers.",
    smiles_markers=["[Cu]", "[Au]"],
    similarity_on_repeat_unit=True,
)


def test_worker_uses_scalar_predict_wrapper_and_task_marker_validation():
    surrogate = StrictListSurrogate()
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
            "explanation": "valid polymer repeat unit",
        },
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": "CCO",
            "explanation": "RDKit-valid but missing task markers",
        },
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 1.0
    assert candidates[0]["child_property"] == 2.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert isinstance(candidates[0]["similarity"], float)

    assert candidates[1]["valid"] is False
    assert "Missing required marker" in candidates[1]["invalid_reason"]
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_property_tools_use_surrogate_batch_contract():
    surrogate = StrictListSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "TestProp").execute(VALID_CHILD)
    assert single_obs.success is True
    assert single_obs.result["TestProp"] == 2.0

    batch_obs = BatchPropertyPredictorTool(surrogate, "TestProp").execute([VALID_PARENT, VALID_CHILD])
    assert batch_obs.success is True
    assert [row["property"] for row in batch_obs.result] == [1.0, 2.0]
    assert surrogate.calls[-1] == [VALID_PARENT, VALID_CHILD]


def test_meta_formats_recent_strategies_from_history():
    history = PromptStateHistory()
    for i in range(4):
        history.add(PromptState(strategy_text=f"strategy {i}", version=i))

    meta = MetaAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
    )
    meta.history = history

    formatted = meta._format_recent_strategies()
    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted


def test_agentic_engine_logs_evaluated_reward_and_merges_usage(tmp_path, monkeypatch):
    class DummySurrogate(StrictListSurrogate):
        pass

    class DummyWorker:
        def __init__(self, **kwargs):
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

    class DummyCritic:
        def __init__(self, **kwargs):
            self._interpretability_trace = {"critic": "trace"}

        def refine(self, candidates, current_state, history, meta_advice):
            return (
                PromptState(
                    strategy_text="next strategy",
                    version=current_state.version + 1,
                    parent_version=current_state.version,
                ),
                {"analysis": "ok"},
                {
                    "total_calls": 1,
                    "total_prompt_tokens": 7,
                    "total_completion_tokens": 3,
                    "total_tokens": 10,
                    "total_latency_s": 0.2,
                    "by_model": {"critic-model": {"calls": 1, "tokens": 10}},
                },
            )

    class DummyMeta:
        def __init__(self, **kwargs):
            self._interpretability_trace = {"meta": "trace"}

        def get_advice(self, history, reward_history):
            return "", {
                "total_calls": 1,
                "total_tokens": 4,
                "total_latency_s": 0.3,
                "by_model": {"meta-model": {"calls": 1, "tokens": 4}},
            }

    monkeypatch.setattr("apo.agentic_engine.get_surrogate", lambda *args, **kwargs: DummySurrogate())
    monkeypatch.setattr("apo.agentic_engine.WorkerAgent", DummyWorker)
    monkeypatch.setattr("apo.agentic_engine.CriticAgent", DummyCritic)
    monkeypatch.setattr("apo.agentic_engine.MetaAgent", DummyMeta)

    cfg = {
        "task": {"surrogate": "dummy", "model_base_path": ""},
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

    run_agentic_mode(
        cfg=cfg,
        ctx=POLYMER_CTX,
        all_smiles=[VALID_PARENT],
        logger=logger,
        api_keys={},
    )

    records = logger.load_existing_epochs()
    assert len(records) == 1
    assert records[0]["reward"] == 1.0
    assert records[0]["prompt_state"]["version"] == 0
    assert records[0]["prompt_state"]["score"] == 1.0
    assert records[0]["llm_usage"]["total_calls"] == 2
    assert records[0]["llm_usage"]["total_tokens"] == 25
