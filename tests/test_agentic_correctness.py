from typing import List, Optional

from apo.agentic_engine import run_agentic_mode
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState
from apo.logging.run_logger import RunLogger
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext
from apo.utils.smiles_utils import canonicalize


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise TypeError("predict expects a list, not a string")
        canonical_child = canonicalize(VALID_CHILD)
        return [2.0 if smi == canonical_child else 1.0 for smi in smiles_list]


def polymer_ctx() -> TaskContext:
    return TaskContext(
        property_name="TestProp",
        property_units="units",
        maximize=True,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_worker_uses_single_prediction_and_enforces_markers():
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(),
        surrogate=StrictSurrogate(),
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {"parent_smiles": VALID_PARENT, "child_smiles": VALID_CHILD, "explanation": "valid"},
        {"parent_smiles": VALID_PARENT, "child_smiles": "CCO", "explanation": "missing markers"},
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 1.0
    assert candidates[0]["child_property"] == 2.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert candidates[1]["valid"] is False
    assert "Missing required marker" in candidates[1]["invalid_reason"]


def test_property_tools_do_not_pass_strings_to_batch_predictor():
    surrogate = StrictSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "TestProp").execute(VALID_CHILD)
    assert single_obs.success is True
    assert single_obs.result["TestProp"] == 2.0

    batch_obs = BatchPropertyPredictorTool(surrogate, "TestProp").execute([VALID_PARENT, VALID_CHILD])
    assert batch_obs.success is True
    assert [row["property"] for row in batch_obs.result] == [1.0, 2.0]


def test_agentic_mode_logs_evaluated_reward_and_aggregates_usage(tmp_path, monkeypatch):
    class DummyWorker:
        _interpretability_trace = {}

        def __init__(self, **kwargs):
            pass

        def generate(self, strategy, parent_smiles, n_per_molecule):
            return [
                {
                    "parent_smiles": VALID_PARENT,
                    "child_smiles": VALID_CHILD,
                    "parent_property": 1.0,
                    "child_property": 2.0,
                    "improvement_factor": 2.0,
                    "similarity": 0.5,
                    "valid": True,
                }
            ], [LLMUsage("worker-model", 10, 5, 0.1)]

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
                    parent_version=current_state.version,
                ),
                {},
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
        _interpretability_trace = {}

        def __init__(self, **kwargs):
            pass

        def get_advice(self, history, reward_history):
            return "", {
                "total_calls": 1,
                "total_prompt_tokens": 4,
                "total_completion_tokens": 2,
                "total_tokens": 6,
                "total_latency_s": 0.1,
                "by_model": {"meta-model": {"calls": 1, "tokens": 6}},
            }

    monkeypatch.setattr("apo.agentic_engine.get_surrogate", lambda *args, **kwargs: StrictSurrogate())
    monkeypatch.setattr("apo.agentic_engine.WorkerAgent", DummyWorker)
    monkeypatch.setattr("apo.agentic_engine.CriticAgent", DummyCritic)
    monkeypatch.setattr("apo.agentic_engine.MetaAgent", DummyMeta)

    cfg = {
        "models": {"worker": "worker-model", "critic": "critic-model", "meta": "meta-model"},
        "task": {"surrogate": "strict"},
        "optimization": {
            "n_outer_epochs": 1,
            "n_per_molecule": 1,
            "batch_size": 1,
            "meta_interval": 1,
            "reward_function": "pareto_hypervolume",
        },
    }
    logger = RunLogger(str(tmp_path / "runs"))

    run_agentic_mode(cfg, polymer_ctx(), [VALID_PARENT], logger, api_keys={})

    records = logger.load_existing_epochs()
    assert len(records) == 1
    assert records[0]["reward"] == 1.0
    assert records[0]["prompt_state"]["version"] == 0
    assert records[0]["prompt_state"]["score"] == 1.0
