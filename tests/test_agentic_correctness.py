from typing import List, Optional

from apo.agentic_engine import _merge_usage_summary, run_agentic_mode
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState
from apo.logging.run_logger import RunLogger
from apo.task_context import TaskContext


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"
MISSING_MARKER_CHILD = "CC(CO[Cu])COCCOC(=O)"


class StrictBatchSurrogate:
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        assert isinstance(smiles_list, list)
        self.calls.append(list(smiles_list))
        return [float(len(s)) for s in smiles_list]

    def predict_single(self, smiles: str) -> Optional[float]:
        values = self.predict([smiles])
        return values[0] if values else None


def polymer_context() -> TaskContext:
    return TaskContext(
        property_name="TestProp",
        property_units="units",
        maximize=True,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_agentic_property_tools_use_surrogate_list_api():
    surrogate = StrictBatchSurrogate()

    single = PropertyPredictorTool(surrogate, "TestProp").execute(VALID_CHILD)
    assert single.success is True
    assert surrogate.calls[-1] == [VALID_CHILD]

    batch = BatchPropertyPredictorTool(surrogate, "TestProp").execute([VALID_PARENT, VALID_CHILD])
    assert batch.success is True
    assert [row["smiles"] for row in batch.result] == [VALID_PARENT, VALID_CHILD]
    assert surrogate.calls[-1] == [VALID_PARENT, VALID_CHILD]


def test_worker_validation_enforces_markers_and_strict_surrogate_api():
    surrogate = StrictBatchSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": MISSING_MARKER_CHILD,
            "explanation": "missing one required marker",
        },
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "explanation": "valid marker-preserving edit",
        },
    ])

    assert candidates[0]["valid"] is False
    assert candidates[0]["invalid_reason"] == "Missing required marker: [Au]"
    assert candidates[0]["child_property"] is None

    assert candidates[1]["valid"] is True
    assert candidates[1]["parent_property"] is not None
    assert candidates[1]["child_property"] is not None
    assert candidates[1]["similarity"] > 0.0
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_agentic_engine_logs_evaluated_state_and_merges_dict_usage(tmp_path, monkeypatch):
    class FakeWorker:
        def __init__(self, *args, **kwargs):
            self._interpretability_trace = {}

        def generate(self, strategy, parent_smiles, n_per_molecule):
            return [
                {
                    "valid": True,
                    "parent_smiles": VALID_PARENT,
                    "child_smiles": VALID_CHILD,
                    "improvement_factor": 2.0,
                    "similarity": 0.5,
                    "parent_property": 1.0,
                    "child_property": 2.0,
                }
            ], [LLMUsage("worker-model", 10, 5, 0.1)]

    class FakeCritic:
        def __init__(self, *args, **kwargs):
            self._interpretability_trace = {}

        def refine(self, candidates, current_state, history, meta_advice=""):
            assert current_state.score == 1.0
            return (
                PromptState(
                    strategy_text="next strategy",
                    version=current_state.version + 1,
                    parent_version=current_state.version,
                ),
                {"ok": True},
                {
                    "total_calls": 1,
                    "total_prompt_tokens": 3,
                    "total_completion_tokens": 2,
                    "total_tokens": 5,
                    "total_latency_s": 0.2,
                    "by_model": {"critic-model": {"calls": 1, "tokens": 5}},
                },
            )

    class FakeMeta:
        def __init__(self, *args, **kwargs):
            self._interpretability_trace = {}

        def get_advice(self, history, reward_history):
            return "", {
                "total_calls": 1,
                "total_prompt_tokens": 4,
                "total_completion_tokens": 1,
                "total_tokens": 5,
                "total_latency_s": 0.3,
                "by_model": {"meta-model": {"calls": 1, "tokens": 5}},
            }

    monkeypatch.setattr("apo.agentic_engine.WorkerAgent", FakeWorker)
    monkeypatch.setattr("apo.agentic_engine.CriticAgent", FakeCritic)
    monkeypatch.setattr("apo.agentic_engine.MetaAgent", FakeMeta)
    monkeypatch.setattr("apo.agentic_engine.get_surrogate", lambda *args, **kwargs: StrictBatchSurrogate())

    cfg = {
        "task": {"surrogate": "fake"},
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
    logger = RunLogger(str(tmp_path))

    run_agentic_mode(cfg, polymer_context(), [VALID_PARENT], logger, {})

    records = logger.load_existing_epochs()
    assert records[0]["prompt_state"]["version"] == 0
    assert records[0]["reward"] == 1.0
    assert records[0]["prompt_state"]["score"] == 1.0


def test_merge_usage_summary_handles_aggregate_dicts():
    target = {
        "total_calls": 1,
        "total_prompt_tokens": 10,
        "total_completion_tokens": 5,
        "total_tokens": 15,
        "total_latency_s": 0.1,
        "by_model": {"a": {"calls": 1, "tokens": 15}},
    }
    _merge_usage_summary(target, {
        "total_calls": 2,
        "total_prompt_tokens": 4,
        "total_completion_tokens": 6,
        "total_tokens": 10,
        "total_latency_s": 0.2,
        "by_model": {"a": {"calls": 1, "tokens": 3}, "b": {"calls": 1, "tokens": 7}},
    })

    assert target["total_calls"] == 3
    assert target["total_tokens"] == 25
    assert target["by_model"]["a"] == {"calls": 2, "tokens": 18}
    assert target["by_model"]["b"] == {"calls": 1, "tokens": 7}
