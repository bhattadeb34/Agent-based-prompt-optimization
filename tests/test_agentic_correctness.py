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


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise TypeError("predict expects a list, not a string")
        self.calls.append(list(smiles_list))
        return [2.0 if "[Au]" in smi else 1.0 for smi in smiles_list]


POLYMER_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="polymer",
    domain_context="[Cu] and [Au] are required backbone markers.",
    smiles_markers=["[Cu]", "[Au]"],
    similarity_on_repeat_unit=True,
)


def test_worker_validation_uses_predict_single_and_required_markers():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {"parent_smiles": VALID_PARENT, "child_smiles": VALID_CHILD, "explanation": "valid"},
        {"parent_smiles": VALID_PARENT, "child_smiles": "CCO", "explanation": "missing markers"},
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["child_property"] == 2.0
    assert candidates[0]["parent_property"] == 2.0
    assert candidates[0]["improvement_factor"] == 1.0
    assert candidates[1]["valid"] is False
    assert "Missing required marker" in candidates[1]["invalid_reason"]
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_property_tools_use_surrogate_batch_api():
    surrogate = StrictSurrogate()

    single = PropertyPredictorTool(surrogate, "TestProp").execute(VALID_CHILD)
    batch = BatchPropertyPredictorTool(surrogate, "TestProp").execute([VALID_PARENT, VALID_CHILD])

    assert single.success is True
    assert single.result["TestProp"] == 2.0
    assert batch.success is True
    assert [row["property"] for row in batch.result] == [2.0, 2.0]
    assert surrogate.calls[-1] == [VALID_PARENT, VALID_CHILD]


def test_meta_recent_strategy_format_does_not_call_missing_history_all():
    history = PromptStateHistory()
    history.add(PromptState.seed("seed strategy"))
    history.add(PromptState(strategy_text="second strategy", version=1))

    meta = MetaAgent(model="test-model", api_keys={}, task_context=POLYMER_CTX)
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v0: seed strategy" in formatted
    assert "v1: second strategy" in formatted


def test_agentic_engine_logs_current_reward_and_merges_dict_usage(tmp_path, monkeypatch):
    class FakeWorker:
        def __init__(self, **kwargs):
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
                    "explanation": "improved",
                }
            ], [LLMUsage("worker-model", 10, 5, 0.1)]

    class FakeCritic:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def refine(self, candidates, current_state, history, meta_advice=""):
            new_state = PromptState(
                strategy_text="next strategy",
                version=current_state.version + 1,
                rationale="test",
                parent_version=current_state.version,
                model_used="critic-model",
            )
            usage = {
                "total_calls": 1,
                "total_tokens": 7,
                "total_prompt_tokens": 4,
                "total_completion_tokens": 3,
                "total_latency_s": 0.2,
                "by_model": {"critic-model": {"calls": 1, "tokens": 7}},
            }
            return new_state, {"analysis": "ok"}, usage

    class FakeMeta:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def get_advice(self, history, reward_history):
            assert reward_history == [1.0]
            usage = {
                "total_calls": 1,
                "total_tokens": 3,
                "total_prompt_tokens": 2,
                "total_completion_tokens": 1,
                "total_latency_s": 0.05,
                "by_model": {"meta-model": {"calls": 1, "tokens": 3}},
            }
            return "", usage

    monkeypatch.setattr("apo.agentic_engine.get_surrogate", lambda *args, **kwargs: StrictSurrogate())
    monkeypatch.setattr("apo.agentic_engine.WorkerAgent", FakeWorker)
    monkeypatch.setattr("apo.agentic_engine.CriticAgent", FakeCritic)
    monkeypatch.setattr("apo.agentic_engine.MetaAgent", FakeMeta)

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
    logger = RunLogger(str(tmp_path), run_id="agentic")

    run_agentic_mode(cfg, POLYMER_CTX, [VALID_PARENT], logger, api_keys={})

    record = json.loads(logger.log_path.read_text().strip())
    assert record["reward"] == 1.0
    assert record["prompt_state"]["version"] == 0
    assert record["prompt_state"]["score"] == 1.0
    assert record["llm_usage"]["total_calls"] == 2
    assert record["llm_usage"]["by_model"]["critic-model"]["tokens"] == 7
