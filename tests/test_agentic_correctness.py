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
    property_name = "StrictProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise TypeError("predict expects a list of SMILES")
        self.calls.append(list(smiles_list))
        return [float(len(smi)) for smi in smiles_list]


def polymer_context() -> TaskContext:
    return TaskContext(
        property_name="StrictProp",
        property_units="units",
        maximize=True,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_property_tools_use_surrogate_list_api():
    surrogate = StrictListSurrogate()

    single = PropertyPredictorTool(surrogate, "StrictProp").execute("CC")
    assert single.success is True
    assert single.result["StrictProp"] == 2.0

    batch = BatchPropertyPredictorTool(surrogate, "StrictProp").execute(["CC", "CCC"])
    assert batch.success is True
    assert [r["property"] for r in batch.result] == [2.0, 3.0]
    assert surrogate.calls == [["CC"], ["CC", "CCC"]]


def test_worker_validation_uses_predict_single_and_requires_task_markers():
    surrogate = StrictListSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
        surrogate=surrogate,
        parent_cache={},
    )

    valid, missing_marker = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "explanation": "valid polymer edit",
        },
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": "CCO",
            "explanation": "plain molecule missing polymer markers",
        },
    ])

    assert valid["valid"] is True
    assert isinstance(valid["parent_property"], float)
    assert isinstance(valid["child_property"], float)
    assert valid["improvement_factor"] > 0

    assert missing_marker["valid"] is False
    assert "Missing required marker" in missing_marker["invalid_reason"]


def test_meta_agent_formats_recent_strategies_with_history_api():
    history = PromptStateHistory()
    history.add(PromptState.seed("seed strategy"))
    history.add(PromptState(strategy_text="next strategy", version=1))

    meta = MetaAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
    )
    meta.history = history

    formatted = meta._format_recent_strategies()
    assert "v0: seed strategy" in formatted
    assert "v1: next strategy" in formatted


def test_agentic_engine_scores_current_state_and_aggregates_dict_usage(monkeypatch, tmp_path):
    class FakeWorker:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def generate(self, strategy, parent_smiles, n_per_molecule):
            return [
                {
                    "valid": True,
                    "improvement_factor": 1.5,
                    "similarity": 0.5,
                    "parent_smiles": VALID_PARENT,
                    "child_smiles": VALID_CHILD,
                }
            ], [LLMUsage("worker-model", 10, 5, 0.1)]

    class FakeCritic:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def refine(self, candidates, current_state, history, meta_advice=""):
            assert current_state.score == 0.75
            return (
                PromptState(
                    strategy_text="refined",
                    version=current_state.version + 1,
                    parent_version=current_state.version,
                ),
                {"ok": True},
                {
                    "total_calls": 1,
                    "total_prompt_tokens": 7,
                    "total_completion_tokens": 3,
                    "total_tokens": 10,
                    "total_latency_s": 0.2,
                    "by_model": {"critic-model": {"calls": 1, "tokens": 10}},
                },
            )

    class FakeMeta:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def get_advice(self, history, reward_history):
            assert reward_history == [0.75]
            return "", {
                "total_calls": 1,
                "total_prompt_tokens": 4,
                "total_completion_tokens": 2,
                "total_tokens": 6,
                "total_latency_s": 0.1,
                "by_model": {"meta-model": {"calls": 1, "tokens": 6}},
            }

    monkeypatch.setattr("apo.agentic_engine.WorkerAgent", FakeWorker)
    monkeypatch.setattr("apo.agentic_engine.CriticAgent", FakeCritic)
    monkeypatch.setattr("apo.agentic_engine.MetaAgent", FakeMeta)
    monkeypatch.setattr("apo.agentic_engine.get_surrogate", lambda *args, **kwargs: StrictListSurrogate())

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
        "temperatures": {},
    }
    logger = RunLogger(str(tmp_path / "runs"))

    run_dir = run_agentic_mode(
        cfg=cfg,
        ctx=polymer_context(),
        all_smiles=[VALID_PARENT],
        logger=logger,
        api_keys={},
    )

    assert run_dir == str(logger.run_dir)
    assert logger.reward_history == [0.75]
