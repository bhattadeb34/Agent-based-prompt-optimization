"""Regression tests for high-impact agentic workflow correctness."""
from typing import List, Optional

from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage, aggregate_usage
from apo.core.prompt_state import PromptState
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
        if not isinstance(smiles_list, list):
            raise TypeError("predict expects a list of SMILES")
        self.calls.append(list(smiles_list))
        return [2.0 if smi == VALID_PARENT else 4.0 for smi in smiles_list]


POLYMER_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="polymer",
    domain_context="[Cu] and [Au] are backbone markers.",
    smiles_markers=["[Cu]", "[Au]"],
    similarity_on_repeat_unit=True,
)


def test_property_tools_preserve_list_surrogate_contract():
    surrogate = StrictListSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "TestProp").execute(VALID_PARENT)
    assert single_obs.success
    assert single_obs.result["TestProp"] == 2.0
    assert surrogate.calls[-1] == [VALID_PARENT]

    batch_obs = BatchPropertyPredictorTool(surrogate, "TestProp").execute(
        [VALID_PARENT, VALID_CHILD]
    )
    assert batch_obs.success
    assert [row["property"] for row in batch_obs.result] == [2.0, 4.0]
    assert surrogate.calls[-1] == [VALID_PARENT, VALID_CHILD]


def test_worker_validation_uses_predict_single_and_task_markers():
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
            "explanation": "valid marker-preserving edit",
        },
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": "CCO",
            "explanation": "missing polymer markers",
        },
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 2.0
    assert candidates[0]["child_property"] == 4.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert candidates[1]["valid"] is False
    assert "Missing required marker" in candidates[1]["invalid_reason"]


def test_agentic_mode_logs_evaluated_state_and_merges_dict_usage(monkeypatch, tmp_path):
    import apo.agentic_engine as agentic_engine

    class DummyWorker:
        _interpretability_trace = {}

        def __init__(self, **kwargs):
            pass

        def generate(self, strategy, parent_smiles, n_per_molecule):
            return (
                [{
                    "parent_smiles": VALID_PARENT,
                    "child_smiles": VALID_CHILD,
                    "parent_property": 2.0,
                    "child_property": 4.0,
                    "improvement_factor": 2.0,
                    "similarity": 0.5,
                    "valid": True,
                }],
                [LLMUsage("worker-model", 10, 5, 0.1)],
            )

    class DummyCritic:
        _interpretability_trace = {}

        def __init__(self, **kwargs):
            pass

        def refine(self, candidates, current_state, history, meta_advice=""):
            assert current_state.score == 1.0
            return (
                PromptState(
                    strategy_text="next strategy",
                    version=current_state.version + 1,
                    rationale="test",
                ),
                {"analysis": "ok"},
                aggregate_usage([LLMUsage("critic-model", 20, 10, 0.2)]),
            )

    class DummyMeta:
        _interpretability_trace = {}

        def __init__(self, **kwargs):
            pass

        def get_advice(self, history, reward_history):
            return "", aggregate_usage([LLMUsage("meta-model", 5, 5, 0.1)])

    monkeypatch.setattr(agentic_engine, "WorkerAgent", DummyWorker)
    monkeypatch.setattr(agentic_engine, "CriticAgent", DummyCritic)
    monkeypatch.setattr(agentic_engine, "MetaAgent", DummyMeta)
    monkeypatch.setattr(agentic_engine, "get_surrogate", lambda *args, **kwargs: object())

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
    logger = RunLogger(str(tmp_path / "runs"))

    agentic_engine.run_agentic_mode(
        cfg=cfg,
        ctx=POLYMER_CTX,
        all_smiles=[VALID_PARENT],
        logger=logger,
        api_keys={},
    )

    records = logger.load_existing_epochs()
    assert records[0]["prompt_state"]["version"] == 0
    assert records[0]["reward"] == 1.0
    assert records[0]["llm_usage"]["total_calls"] == 2
    assert records[0]["llm_usage"]["by_model"]["worker-model"]["calls"] == 1
    assert records[0]["llm_usage"]["by_model"]["critic-model"]["calls"] == 1
