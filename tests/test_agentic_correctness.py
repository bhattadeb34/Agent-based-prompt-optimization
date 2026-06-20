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
from apo.utils.smiles_utils import compute_similarity


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        assert isinstance(smiles_list, list), "predict must receive a list"
        self.calls.append(list(smiles_list))
        values = {
            VALID_PARENT: 1.0,
            VALID_CHILD: 2.0,
        }
        return [values.get(smiles, 1.0) for smiles in smiles_list]


def polymer_ctx() -> TaskContext:
    return TaskContext(
        property_name="TestProp",
        property_units="units",
        maximize=True,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_worker_validation_uses_list_surrogate_api_and_task_constraints():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(),
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "explanation": "valid polymer",
        },
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": "CCO",
            "explanation": "missing markers",
        },
    ])

    valid = candidates[0]
    assert valid["valid"] is True
    assert valid["parent_property"] == 1.0
    assert valid["child_property"] == 2.0
    assert valid["improvement_factor"] == 2.0
    assert valid["similarity"] == compute_similarity(
        VALID_PARENT,
        VALID_CHILD,
        similarity_on_repeat_unit=True,
        marker_strip_tokens=["[Cu]", "[Au]"],
    )
    assert surrogate.calls == [[VALID_PARENT], [VALID_CHILD]]

    invalid = candidates[1]
    assert invalid["valid"] is False
    assert invalid["invalid_reason"] == "Missing required marker: [Cu]"


def test_property_tools_use_surrogate_batch_contract():
    surrogate = StrictSurrogate()

    single = PropertyPredictorTool(surrogate, property_name="TestProp")
    obs = single.execute(VALID_CHILD)
    assert obs.success is True
    assert obs.result["TestProp"] == 2.0

    batch = BatchPropertyPredictorTool(surrogate, property_name="TestProp")
    obs = batch.execute([VALID_PARENT, VALID_CHILD])
    assert obs.success is True
    assert [r["property"] for r in obs.result] == [1.0, 2.0]
    assert surrogate.calls == [[VALID_CHILD], [VALID_PARENT, VALID_CHILD]]


def test_meta_agent_formats_recent_history_without_missing_all_method():
    history = PromptStateHistory()
    for version in range(4):
        history.add(PromptState(strategy_text=f"strategy {version}", version=version))

    meta = MetaAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(),
    )
    meta.history = history

    formatted = meta._format_recent_strategies()
    assert "v1: strategy 1" in formatted
    assert "v2: strategy 2" in formatted
    assert "v3: strategy 3" in formatted
    assert "v0: strategy 0" not in formatted


def test_agentic_engine_persists_computed_reward_and_handles_usage_dicts(monkeypatch, tmp_path):
    class DummyWorker:
        def __init__(self, *args, **kwargs):
            self._interpretability_trace = {}

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
            ], [LLMUsage("worker-model", 2, 3, 0.1)]

    class DummyCritic:
        def __init__(self, *args, **kwargs):
            self._interpretability_trace = {}

        def refine(self, candidates, current_state, history, meta_advice=""):
            new_state = PromptState(
                strategy_text="next strategy",
                version=current_state.version + 1,
                parent_version=current_state.version,
            )
            return new_state, {"analysis": "ok"}, {
                "total_calls": 1,
                "total_tokens": 7,
                "total_latency_s": 0.2,
                "by_model": {"critic-model": {"calls": 1, "tokens": 7}},
            }

    class DummyMeta:
        def __init__(self, *args, **kwargs):
            self._interpretability_trace = {}

        def get_advice(self, history, reward_history):
            return "", {
                "total_calls": 1,
                "total_tokens": 5,
                "total_latency_s": 0.1,
                "by_model": {"meta-model": {"calls": 1, "tokens": 5}},
            }

    monkeypatch.setattr("apo.agentic_engine.WorkerAgent", DummyWorker)
    monkeypatch.setattr("apo.agentic_engine.CriticAgent", DummyCritic)
    monkeypatch.setattr("apo.agentic_engine.MetaAgent", DummyMeta)
    monkeypatch.setattr("apo.agentic_engine.get_surrogate", lambda *args, **kwargs: StrictSurrogate())

    cfg = {
        "task": {"surrogate": "strict"},
        "models": {
            "worker": "worker-model",
            "critic": "critic-model",
            "meta": "meta-model",
        },
        "optimization": {
            "n_outer_epochs": 1,
            "n_per_molecule": 1,
            "batch_size": 1,
            "meta_interval": 1,
            "reward_function": "pareto_hypervolume",
        },
    }
    logger = RunLogger(str(tmp_path), run_id="agentic-test")

    run_agentic_mode(
        cfg=cfg,
        ctx=polymer_ctx(),
        all_smiles=[VALID_PARENT],
        logger=logger,
        api_keys={},
    )

    [record] = logger.load_existing_epochs()
    assert record["reward"] == 1.0
    assert record["prompt_state"]["score"] == 1.0
    assert record["llm_usage"]["total_calls"] == 2
    assert record["llm_usage"]["total_tokens"] == 12
