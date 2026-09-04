import json
from typing import List, Optional

from apo.agentic_engine import run_agentic_mode
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


class StrictListSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        assert isinstance(smiles_list, list)
        self.calls.append(list(smiles_list))
        return [float(len(smiles)) for smiles in smiles_list]


class FakeLogger:
    def __init__(self):
        self.run_dir = "fake-run"
        self.reward_history = []
        self.epochs = []
        self.prompt_history = None
        self.traces = {}

    def log_epoch(self, **kwargs):
        self.epochs.append(kwargs)
        self.reward_history.append(kwargs["reward"])

    def save_agent_trace(self, name, trace):
        self.traces[name] = trace

    def save_prompt_history(self, history):
        self.prompt_history = history


class FakeWorker:
    _interpretability_trace = {}

    def __init__(self, **kwargs):
        pass

    def generate(self, strategy, parent_smiles, n_per_molecule):
        return [
            {
                "parent_smiles": "CC",
                "child_smiles": "CCC",
                "parent_property": 2.0,
                "child_property": 4.0,
                "improvement_factor": 2.0,
                "similarity": 0.5,
                "valid": True,
            }
        ], [LLMUsage("worker-model", 1, 2, 0.1)]


class FakeCritic:
    _interpretability_trace = {}

    def __init__(self, **kwargs):
        pass

    def refine(self, candidates, current_state, history, meta_advice=""):
        return (
            PromptState(
                strategy_text="next",
                version=current_state.version + 1,
                rationale="test",
                parent_version=current_state.version,
                model_used="critic-model",
            ),
            {"analysis": "ok"},
            {
                "total_calls": 1,
                "total_tokens": 7,
                "by_model": {"critic-model": {"calls": 1, "tokens": 7}},
            },
        )


class FakeMeta:
    _interpretability_trace = {}

    def __init__(self, **kwargs):
        pass

    def get_advice(self, history, reward_history):
        return "", {
            "total_calls": 1,
            "total_tokens": 3,
            "by_model": {"meta-model": {"calls": 1, "tokens": 3}},
        }


def test_agentic_property_tools_use_surrogate_contract():
    surrogate = StrictListSurrogate()

    scalar = PropertyPredictorTool(surrogate, "TestProp").execute("CC")
    assert scalar.success
    assert scalar.result["TestProp"] == 2.0

    batch = BatchPropertyPredictorTool(surrogate, "TestProp").execute(["CC", "CCC"])
    assert batch.success
    assert [r["property"] for r in batch.result] == [2.0, 3.0]
    assert surrogate.calls == [["CC"], ["CC", "CCC"]]


def test_worker_accepts_generated_molecules_mapping_and_predicts_scalars():
    surrogate = StrictListSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=TaskContext(property_name="TestProp", property_units="units"),
        surrogate=surrogate,
        parent_cache={},
    )

    data = worker._parse_generation_output(json.dumps({
        "generated_molecules": {
            "CC": {
                "smiles": ["CCC"],
                "reasoning": ["extend chain"],
            }
        }
    }))
    candidates = [
        {
            "parent_smiles": parent,
            "child_smiles": child,
            "explanation": explanation,
        }
        for parent, child, explanation in worker._iter_generated_candidates(data)
    ]

    validated = worker._validate_candidates(candidates)

    assert len(validated) == 1
    assert validated[0]["valid"] is True
    assert validated[0]["parent_property"] == 2.0
    assert validated[0]["child_property"] == 3.0
    assert surrogate.calls == [["CC"], ["CCC"]]


def test_run_agentic_mode_logs_rewards_and_merges_usage_dicts(monkeypatch):
    import apo.agentic_engine as agentic_engine

    monkeypatch.setattr(agentic_engine, "get_surrogate", lambda *args, **kwargs: StrictListSurrogate())
    monkeypatch.setattr(agentic_engine, "WorkerAgent", FakeWorker)
    monkeypatch.setattr(agentic_engine, "CriticAgent", FakeCritic)
    monkeypatch.setattr(agentic_engine, "MetaAgent", FakeMeta)

    logger = FakeLogger()
    cfg = {
        "task": {"surrogate": "strict"},
        "models": {"worker": "worker-model", "critic": "critic-model", "meta": "meta-model"},
        "optimization": {
            "n_outer_epochs": 1,
            "n_per_molecule": 1,
            "batch_size": 1,
            "meta_interval": 1,
        },
    }

    run_dir = run_agentic_mode(
        cfg=cfg,
        ctx=TaskContext(seed_strategy="seed"),
        all_smiles=["CC"],
        logger=logger,
        api_keys={},
    )

    assert run_dir == "fake-run"
    assert logger.epochs[0]["reward"] > 0
    assert logger.prompt_history[0]["score"] == logger.epochs[0]["reward"]


def test_meta_agent_formats_recent_strategy_history_without_crashing(monkeypatch):
    history = PromptStateHistory()
    history.add(PromptState.seed("seed strategy"))
    history.add(PromptState(strategy_text="first refinement", version=1, rationale="r"))
    history.add(PromptState(strategy_text="second refinement", version=2, rationale="r"))

    agent = MetaAgent(
        model="meta-model",
        api_keys={},
        task_context=TaskContext(property_name="TestProp"),
    )
    agent.history = history
    agent.reward_history = [1.0, 1.0, 1.0]
    agent.recent_analysis = {"pattern": "plateau", "confidence": 0.9}
    agent.should_intervene = True

    def fake_call_llm(**kwargs):
        assert "second refinement" in kwargs["messages"][1]["content"]
        return (
            json.dumps({
                "advice": "try a new region",
                "rationale": "plateau",
                "expected_outcome": "improvement",
                "confidence": 0.8,
            }),
            LLMUsage("meta-model", 1, 1, 0.1),
        )

    monkeypatch.setattr("apo.agents.meta.call_llm", fake_call_llm)

    thought = agent._generate_advice_text()

    assert thought.content == "try a new region"
