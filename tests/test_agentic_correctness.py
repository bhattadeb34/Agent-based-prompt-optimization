from typing import List, Optional

from apo.agentic_engine import run_agentic_mode
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.logging.run_logger import RunLogger
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictListSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise TypeError("predict expects a list, not a string")
        self.calls.append(list(smiles_list))
        return [2.0 if smi == CHILD else 1.0 for smi in smiles_list]


def polymer_ctx() -> TaskContext:
    return TaskContext(
        property_name="TestProp",
        property_units="units",
        maximize=True,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_worker_validation_uses_list_predictor_and_task_markers():
    surrogate = StrictListSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(),
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {"parent_smiles": PARENT, "child_smiles": CHILD, "explanation": "valid"},
        {"parent_smiles": PARENT, "child_smiles": "CCO", "explanation": "missing markers"},
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 1.0
    assert candidates[0]["child_property"] == 2.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert candidates[1]["valid"] is False
    assert "Missing required marker" in candidates[1]["invalid_reason"]
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_batch_predictor_calls_surrogate_once_with_full_list():
    surrogate = StrictListSurrogate()
    tool = BatchPropertyPredictorTool(surrogate, "TestProp")

    obs = tool.execute([PARENT, CHILD])

    assert obs.success is True
    assert surrogate.calls == [[PARENT, CHILD]]
    assert [row["property"] for row in obs.result] == [1.0, 2.0]


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

    text = meta._format_recent_strategies()

    assert "v1: strategy 1" in text
    assert "v3: strategy 3" in text


def test_agentic_engine_logs_evaluated_state_and_merges_usage(monkeypatch, tmp_path):
    ctx = polymer_ctx()
    surrogate = StrictListSurrogate()
    logger = RunLogger(str(tmp_path))
    usage = LLMUsage("test-model", prompt_tokens=10, completion_tokens=5, latency_s=0.1)

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

    def fake_generate(self, strategy, parent_smiles, n_per_molecule):
        return ([{
            "parent_smiles": PARENT,
            "child_smiles": CHILD,
            "parent_property": 1.0,
            "child_property": 2.0,
            "improvement_factor": 2.0,
            "similarity": 0.5,
            "valid": True,
        }], [usage])

    def fake_refine(self, candidates, current_state, history, meta_advice=""):
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

    monkeypatch.setattr("apo.agentic_engine.get_surrogate", lambda *args, **kwargs: surrogate)
    monkeypatch.setattr("apo.agents.worker.WorkerAgent.generate", fake_generate)
    monkeypatch.setattr("apo.agents.critic.CriticAgent.refine", fake_refine)
    monkeypatch.setattr("apo.agents.meta.MetaAgent.get_advice", lambda *args, **kwargs: ("", None))

    run_agentic_mode(cfg, ctx, [PARENT], logger, api_keys={})

    records = logger.load_existing_epochs()
    assert len(records) == 1
    assert records[0]["prompt_state"]["version"] == 0
    assert records[0]["prompt_state"]["score"] == 1.0
    assert records[0]["reward"] == 1.0
    assert records[0]["llm_usage"]["total_calls"] == 2
    assert records[0]["llm_usage"]["by_model"]["critic-model"]["tokens"] == 5
