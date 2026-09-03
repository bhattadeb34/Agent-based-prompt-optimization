"""Regression tests for critical agentic-mode correctness paths."""
from typing import List, Optional

from apo.agents.meta import MetaAgent
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.logging.run_logger import RunLogger
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


class StrictSurrogate(SurrogatePredictor):
    property_name = "MockProperty"
    property_units = "units"
    maximize = True

    def __init__(self, values=None):
        self.values = values or {}
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise TypeError(f"expected list, got {type(smiles_list).__name__}")
        self.calls.append(list(smiles_list))
        return [self.values.get(smi, 1.0) for smi in smiles_list]


def test_worker_validation_uses_predict_single_for_scalar_scores():
    surrogate = StrictSurrogate({"CC": 1.0, "CCO": 2.0})
    ctx = TaskContext(property_name="MockProperty", molecule_type="organic")
    worker = WorkerAgent("mock", {}, ctx, surrogate, {}, max_retries_per_batch=0)

    candidates = worker._validate_candidates([
        {"parent_smiles": "CC", "child_smiles": "CCO", "explanation": "add oxygen"}
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 1.0
    assert candidates[0]["child_property"] == 2.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert surrogate.calls == [["CC"], ["CCO"]]


def test_worker_validation_enforces_task_markers():
    ctx = TaskContext(
        property_name="MockProperty",
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
    )
    worker = WorkerAgent("mock", {}, ctx, StrictSurrogate({"CCO": 2.0}), {}, max_retries_per_batch=0)

    candidates = worker._validate_candidates([
        {"parent_smiles": "CC", "child_smiles": "CCO", "explanation": "missing markers"}
    ])

    assert candidates[0]["valid"] is False
    assert "Missing required marker" in candidates[0]["invalid_reason"]


def test_worker_validation_handles_minimization_direction():
    surrogate = StrictSurrogate({"CC": 10.0, "CCO": 5.0})
    ctx = TaskContext(property_name="MockProperty", molecule_type="organic", maximize=False)
    worker = WorkerAgent("mock", {}, ctx, surrogate, {}, max_retries_per_batch=0)

    candidates = worker._validate_candidates([
        {"parent_smiles": "CC", "child_smiles": "CCO", "explanation": "lower property"}
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["improvement_factor"] == 2.0


def test_worker_parses_existing_generated_molecules_schema(monkeypatch):
    parent = "CC"
    child = "CCO"
    response = (
        "```json\n"
        f'{{"generated_molecules": {{"{parent}": {{"smiles": ["{child}"], "reasoning": ["reason"]}}}}}}\n'
        "```"
    )
    monkeypatch.setattr(
        "apo.agents.worker.call_llm",
        lambda *args, **kwargs: (response, LLMUsage("mock", 1, 1, 0.1)),
    )
    ctx = TaskContext(property_name="MockProperty", molecule_type="organic")
    worker = WorkerAgent("mock", {}, ctx, StrictSurrogate(), {}, max_retries_per_batch=0)

    candidates = worker._call_llm_for_generation()

    assert candidates == [
        {"parent_smiles": parent, "child_smiles": child, "explanation": "reason"}
    ]


def test_worker_parses_generated_molecules_list_schema(monkeypatch):
    parent = "CC"
    child = "CCO"
    response = (
        "```json\n"
        f'{{"generated_molecules": {{"{parent}": [{{"smiles": "{child}", "explanation": "reason"}}]}}}}\n'
        "```"
    )
    monkeypatch.setattr(
        "apo.agents.worker.call_llm",
        lambda *args, **kwargs: (response, LLMUsage("mock", 1, 1, 0.1)),
    )
    ctx = TaskContext(property_name="MockProperty", molecule_type="organic")
    worker = WorkerAgent("mock", {}, ctx, StrictSurrogate(), {}, max_retries_per_batch=0)

    candidates = worker._call_llm_for_generation()

    assert candidates == [
        {"parent_smiles": parent, "child_smiles": child, "explanation": "reason"}
    ]


def test_meta_agent_formats_recent_history_without_crashing():
    history = PromptStateHistory()
    for version in range(4):
        history.add(PromptState(strategy_text=f"strategy {version}", version=version))
    ctx = TaskContext(property_name="MockProperty", molecule_type="organic")
    meta = MetaAgent("mock", {}, ctx)
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted


def test_agentic_engine_scores_evaluated_state_and_merges_usage(monkeypatch, tmp_path):
    from apo import agentic_engine

    parent = "CC"
    child = "CCO"
    ctx = TaskContext(property_name="MockProperty", molecule_type="organic")
    logger = RunLogger(str(tmp_path / "runs"))

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            self._interpretability_trace = {}

        def generate(self, *args, **kwargs):
            return [
                {
                    "parent_smiles": parent,
                    "child_smiles": child,
                    "parent_property": 1.0,
                    "child_property": 2.0,
                    "improvement_factor": 2.0,
                    "similarity": 0.5,
                    "valid": True,
                }
            ], [LLMUsage("worker", 10, 5, 0.1)]

    class FakeCritic:
        def __init__(self, *args, **kwargs):
            self._interpretability_trace = {}

        def refine(self, candidates, current_state, history, meta_advice=""):
            return (
                PromptState(
                    strategy_text="next",
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
                    "by_model": {"critic": {"calls": 1, "tokens": 10}},
                },
            )

    class FakeMeta:
        def __init__(self, *args, **kwargs):
            self._interpretability_trace = {}

        def get_advice(self, history, reward_history):
            return "", {
                "total_calls": 1,
                "total_prompt_tokens": 4,
                "total_completion_tokens": 2,
                "total_tokens": 6,
                "total_latency_s": 0.1,
                "by_model": {"meta": {"calls": 1, "tokens": 6}},
            }

    monkeypatch.setattr(agentic_engine, "get_surrogate", lambda *args, **kwargs: StrictSurrogate())
    monkeypatch.setattr(agentic_engine, "WorkerAgent", FakeWorker)
    monkeypatch.setattr(agentic_engine, "CriticAgent", FakeCritic)
    monkeypatch.setattr(agentic_engine, "MetaAgent", FakeMeta)

    cfg = {
        "task": {"surrogate": "mock"},
        "models": {"worker": "worker", "critic": "critic", "meta": "meta"},
        "optimization": {
            "n_outer_epochs": 1,
            "n_per_molecule": 1,
            "batch_size": 1,
            "meta_interval": 1,
            "reward_function": "weighted_sum",
            "reward_function_kwargs": {"alpha": 0.25},
        },
    }

    agentic_engine.run_agentic_mode(cfg, ctx, [parent], logger, {})

    records = logger.load_existing_epochs()
    assert records[0]["reward"] == 0.875
    assert records[0]["prompt_state"]["version"] == 0
    assert records[0]["prompt_state"]["score"] == 0.875
