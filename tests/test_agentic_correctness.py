"""Regression tests for high-impact agentic workflow correctness."""
from typing import List, Optional
from unittest.mock import patch

from apo.agentic_engine import run_agentic_mode
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"
MISSING_MARKER_CHILD = "CC(CO[Cu])CO"


class StrictSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise TypeError("predict expects a list of SMILES")
        return [2.0 if smiles == VALID_CHILD else 1.0 for smiles in smiles_list]


POLYMER_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="polymer",
    domain_context="[Cu] and [Au] are required repeat-unit markers.",
    smiles_markers=["[Cu]", "[Au]"],
    similarity_on_repeat_unit=True,
)


def test_worker_uses_single_prediction_api_and_scores_valid_candidate():
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        surrogate=StrictSurrogate(),
        parent_cache={},
    )

    [candidate] = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "explanation": "swap sulfur for oxygen",
        }
    ])

    assert candidate["valid"] is True
    assert candidate["parent_property"] == 1.0
    assert candidate["child_property"] == 2.0
    assert candidate["improvement_factor"] == 2.0


def test_worker_rejects_candidates_missing_required_markers():
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        surrogate=StrictSurrogate(),
        parent_cache={},
    )

    [candidate] = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": MISSING_MARKER_CHILD,
            "explanation": "dropped the terminator marker",
        }
    ])

    assert candidate["valid"] is False
    assert candidate["invalid_reason"] == "Missing required marker: [Au]"
    assert candidate["child_property"] is None


def test_property_tools_do_not_pass_strings_to_batch_predictor():
    surrogate = StrictSurrogate()

    single = PropertyPredictorTool(surrogate, "TestProp").execute(VALID_CHILD)
    batch = BatchPropertyPredictorTool(surrogate, "TestProp").execute([VALID_PARENT, VALID_CHILD])

    assert single.success is True
    assert single.result["TestProp"] == 2.0
    assert batch.success is True
    assert [row["property"] for row in batch.result] == [1.0, 2.0]


def test_meta_agent_formats_recent_strategies_from_history():
    history = PromptStateHistory()
    history.add(PromptState.seed("seed strategy"))
    history.add(PromptState(strategy_text="next strategy", version=1, rationale="r"))

    meta = MetaAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
    )
    meta.history = history

    rendered = meta._format_recent_strategies()

    assert "v0: seed strategy" in rendered
    assert "v1: next strategy" in rendered


class DummyLogger:
    def __init__(self):
        self.run_dir = "/tmp/agentic-test"
        self.records = []
        self.traces = {}
        self.prompt_history = None

    @property
    def reward_history(self):
        return [record["reward"] for record in self.records]

    def log_epoch(self, **kwargs):
        self.records.append(kwargs)

    def save_agent_trace(self, name, trace):
        self.traces[name] = trace

    def save_prompt_history(self, history):
        self.prompt_history = history


class DummyWorker:
    _interpretability_trace = {"worker": True}

    def __init__(self, **kwargs):
        pass

    def generate(self, strategy, parent_smiles, n_per_molecule):
        return [
            {
                "parent_smiles": parent_smiles[0],
                "child_smiles": VALID_CHILD,
                "valid": True,
                "parent_property": 1.0,
                "child_property": 2.0,
                "improvement_factor": 2.0,
                "similarity": 0.5,
            }
        ], []


class DummyCritic:
    _interpretability_trace = {"critic": True}

    def __init__(self, **kwargs):
        pass

    def refine(self, candidates, current_state, history, meta_advice=""):
        return (
            PromptState(
                strategy_text="refined strategy",
                version=current_state.version + 1,
                rationale="r",
                parent_version=current_state.version,
            ),
            {"analysis": "ok"},
            {"total_calls": 1, "total_tokens": 7},
        )


class DummyMeta:
    _interpretability_trace = {"meta": True}

    def __init__(self, **kwargs):
        pass

    def get_advice(self, history, reward_history):
        return "", {"total_calls": 1, "total_tokens": 3}


def test_agentic_engine_logs_actual_reward_and_handles_aggregated_usage():
    logger = DummyLogger()
    cfg = {
        "task": {"surrogate": "dummy"},
        "models": {"worker": "worker", "critic": "critic", "meta": "meta"},
        "optimization": {
            "n_outer_epochs": 1,
            "n_per_molecule": 1,
            "batch_size": 1,
            "meta_interval": 10,
            "reward_function": "pareto_hypervolume",
        },
        "temperatures": {},
    }

    with patch("apo.agentic_engine.get_surrogate", return_value=StrictSurrogate()), \
         patch("apo.agentic_engine.get_reward_function", return_value=ParetoHypervolume()), \
         patch("apo.agentic_engine.WorkerAgent", DummyWorker), \
         patch("apo.agentic_engine.CriticAgent", DummyCritic), \
         patch("apo.agentic_engine.MetaAgent", DummyMeta):
        run_agentic_mode(cfg, POLYMER_CTX, [VALID_PARENT], logger, api_keys={})

    assert len(logger.records) == 1
    [record] = logger.records
    assert record["reward"] == 1.0
    assert record["prompt_state_dict"]["version"] == 0
    assert record["prompt_state_dict"]["score"] == 1.0
    assert record["llm_usage"]["total_calls"] == 1
    assert record["llm_usage"]["total_tokens"] == 7
    assert logger.prompt_history[-1]["version"] == 1
