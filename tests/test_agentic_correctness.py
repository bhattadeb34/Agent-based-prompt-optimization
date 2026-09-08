import json
from typing import List, Optional
from unittest.mock import patch

from apo.agentic_engine import run_agentic_mode
from apo.agents.meta import MetaAgent
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.logging.run_logger import RunLogger
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext
from apo.utils.smiles_utils import canonicalize


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"
NO_MARKER_CHILD = "CCO"
MOCK_USAGE = LLMUsage("test-model", 10, 5, 0.1)


class StrictSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = False

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise TypeError("predict expects a list of SMILES")
        values = {
            VALID_PARENT: 10.0,
            VALID_CHILD: 5.0,
            canonicalize(VALID_PARENT): 10.0,
            canonicalize(VALID_CHILD): 5.0,
        }
        return [values.get(smiles) for smiles in smiles_list]


def make_ctx(maximize=False):
    return TaskContext(
        property_name="TestProp",
        property_units="units",
        maximize=maximize,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_agentic_engine_scores_evaluated_state_and_summarizes_usage(tmp_path):
    ctx = make_ctx(maximize=True)
    logger = RunLogger(str(tmp_path / "runs"))
    new_state = PromptState(
        strategy_text="next strategy",
        version=1,
        rationale="test",
        parent_version=0,
    )
    candidates = [{
        "parent_smiles": VALID_PARENT,
        "child_smiles": VALID_CHILD,
        "parent_property": 1.0,
        "child_property": 2.0,
        "improvement_factor": 2.0,
        "similarity": 0.8,
        "valid": True,
    }]

    cfg = {
        "task": {"surrogate": "mock", "model_base_path": ""},
        "models": {"worker": "worker", "critic": "critic", "meta": "meta"},
        "optimization": {
            "n_outer_epochs": 1,
            "n_per_molecule": 1,
            "batch_size": 1,
            "meta_interval": 1,
            "reward_function": "pareto_hypervolume",
        },
        "temperatures": {},
    }

    def fake_generate(self, strategy, parent_smiles, n_per_molecule):
        self._interpretability_trace = {"worker": "trace"}
        return candidates, [MOCK_USAGE]

    def fake_refine(self, candidates, current_state, history, meta_advice=""):
        self._interpretability_trace = {"critic": "trace"}
        self.all_usages = [MOCK_USAGE]
        return new_state, {"analysis": "ok"}, {"total_calls": 1, "total_tokens": 15}

    def fake_meta_advice(self, history, reward_history):
        self._interpretability_trace = {"meta": "trace"}
        self.all_usages = [MOCK_USAGE]
        return "", {"total_calls": 1, "total_tokens": 15}

    with patch("apo.agentic_engine.get_surrogate", return_value=StrictSurrogate()), \
         patch("apo.agentic_engine.WorkerAgent.generate", fake_generate), \
         patch("apo.agentic_engine.CriticAgent.refine", fake_refine), \
         patch("apo.agentic_engine.MetaAgent.get_advice", fake_meta_advice), \
         patch.object(logger, "save_agent_trace", return_value=None):
        run_agentic_mode(cfg, ctx, [VALID_PARENT], logger, api_keys={})

    records = logger.load_existing_epochs()
    assert len(records) == 1
    assert records[0]["reward"] == 1.6
    assert records[0]["prompt_state"]["version"] == 0
    assert records[0]["prompt_state"]["score"] == 1.6


def test_meta_formats_recent_history_without_missing_all_method():
    history = PromptStateHistory()
    for i in range(4):
        history.add(PromptState(strategy_text=f"strategy {i}", version=i))

    meta = MetaAgent(
        model="meta",
        api_keys={},
        task_context=make_ctx(),
    )
    meta.history = history

    formatted = meta._format_recent_strategies()
    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted


def test_worker_uses_single_predict_and_minimize_direction_with_marker_validation():
    worker = WorkerAgent(
        model="worker",
        api_keys={},
        task_context=make_ctx(maximize=False),
        surrogate=StrictSurrogate(),
        parent_cache={},
        max_retries_per_batch=1,
    )

    candidates = worker._validate_candidates([
        {"parent_smiles": VALID_PARENT, "child_smiles": VALID_CHILD, "explanation": "lower is better"},
        {"parent_smiles": VALID_PARENT, "child_smiles": NO_MARKER_CHILD, "explanation": "missing markers"},
    ])

    valid = [c for c in candidates if c["valid"]]
    invalid = [c for c in candidates if not c["valid"]]
    assert len(valid) == 1
    assert valid[0]["improvement_factor"] == 2.0
    assert invalid[0]["invalid_reason"] == "Missing required marker: [Cu]"


def test_worker_parses_fenced_generated_molecules_mapping():
    worker = WorkerAgent(
        model="worker",
        api_keys={},
        task_context=make_ctx(),
        surrogate=StrictSurrogate(),
        parent_cache={},
        max_retries_per_batch=1,
    )
    payload = {
        "generated_molecules": {
            VALID_PARENT: {
                "smiles": [VALID_CHILD],
                "reasoning": ["swap sulfur for oxygen"],
            }
        }
    }

    with patch(
        "apo.agents.worker.call_llm",
        return_value=(f"```json\n{json.dumps(payload)}\n```", MOCK_USAGE),
    ):
        candidates = worker._call_llm_for_generation()

    assert candidates == [{
        "parent_smiles": VALID_PARENT,
        "child_smiles": VALID_CHILD,
        "explanation": "swap sulfur for oxygen",
    }]


def test_worker_deduplicates_candidates_across_retries():
    worker = WorkerAgent(
        model="worker",
        api_keys={},
        task_context=make_ctx(),
        surrogate=StrictSurrogate(),
        parent_cache={},
        max_retries_per_batch=1,
    )
    candidate = {
        "parent_smiles": VALID_PARENT,
        "child_smiles": VALID_CHILD,
        "valid": True,
    }

    worker._merge_candidates([candidate])
    worker._merge_candidates([dict(candidate)])

    assert len(worker.generated_candidates) == 1
