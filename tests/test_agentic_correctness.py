from typing import List, Optional
from unittest.mock import patch

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


class StrictBatchSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise AssertionError("predict() must be called with a list of SMILES")
        self.calls.append(list(smiles_list))
        return [2.0 if smiles == VALID_CHILD else 1.0 for smiles in smiles_list]


POLYMER_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="polymer",
    smiles_markers=["[Cu]", "[Au]"],
    similarity_on_repeat_unit=True,
)


def test_worker_validation_uses_single_predictor_wrapper_and_preserves_valid_candidate():
    surrogate = StrictBatchSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        surrogate=surrogate,
        parent_cache={},
    )

    [candidate] = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "explanation": "replace sulfur with oxygen",
        }
    ])

    assert candidate["valid"] is True
    assert candidate["parent_property"] == 1.0
    assert candidate["child_property"] == 2.0
    assert candidate["improvement_factor"] == 2.0
    assert surrogate.calls == [[VALID_PARENT], [VALID_CHILD]]


def test_worker_validation_rejects_polymer_candidate_missing_required_markers():
    surrogate = StrictBatchSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        surrogate=surrogate,
        parent_cache={VALID_PARENT: 1.0},
    )

    [candidate] = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": "CCO",
            "explanation": "plain molecule without polymer endpoints",
        }
    ])

    assert candidate["valid"] is False
    assert "Missing required marker" in candidate["invalid_reason"]
    assert candidate["child_property"] is None


def test_property_tools_respect_batch_predictor_contract():
    surrogate = StrictBatchSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "TestProp").execute(VALID_CHILD)
    batch_obs = BatchPropertyPredictorTool(surrogate, "TestProp").execute([VALID_PARENT, VALID_CHILD])

    assert single_obs.success is True
    assert single_obs.result["TestProp"] == 2.0
    assert batch_obs.success is True
    assert [r["property"] for r in batch_obs.result] == [1.0, 2.0]
    assert surrogate.calls == [[VALID_CHILD], [VALID_PARENT, VALID_CHILD]]


def test_meta_agent_formats_recent_history_without_missing_method_crash():
    history = PromptStateHistory()
    for version in range(4):
        history.add(PromptState(strategy_text=f"strategy {version}", version=version))

    meta = MetaAgent(model="test-model", api_keys={}, task_context=POLYMER_CTX)
    meta.history = history

    recent = meta._format_recent_strategies()
    assert "v1: strategy 1" in recent
    assert "v3: strategy 3" in recent


def test_agentic_mode_logs_current_state_reward_and_merges_usage_dicts(tmp_path):
    cfg = {
        "task": {"surrogate": "mock"},
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
    candidate = {
        "parent_smiles": VALID_PARENT,
        "child_smiles": VALID_CHILD,
        "valid": True,
        "parent_property": 1.0,
        "child_property": 2.0,
        "improvement_factor": 2.0,
        "similarity": 0.5,
    }

    def fake_generate(self, strategy, parent_smiles, n_per_molecule=1):
        self._interpretability_trace = {"worker": "ok"}
        return [candidate], [LLMUsage("worker-model", 4, 6, 0.2)]

    def fake_refine(self, candidates, current_state, history, meta_advice=""):
        assert current_state.score == 1.0
        self._interpretability_trace = {"critic": "ok"}
        return (
            PromptState(
                strategy_text="next strategy",
                version=current_state.version + 1,
                parent_version=current_state.version,
            ),
            {"analysis": "ok"},
            {
                "total_calls": 1,
                "total_prompt_tokens": 2,
                "total_completion_tokens": 3,
                "total_tokens": 5,
                "total_latency_s": 0.1,
                "by_model": {"critic-model": {"calls": 1, "tokens": 5}},
            },
        )

    with patch("apo.agentic_engine.get_surrogate", return_value=StrictBatchSurrogate()), \
         patch("apo.agentic_engine.WorkerAgent.generate", new=fake_generate), \
         patch("apo.agentic_engine.CriticAgent.refine", new=fake_refine), \
         patch("apo.agentic_engine.MetaAgent.get_advice", return_value=("", {
             "total_calls": 1,
             "total_prompt_tokens": 1,
             "total_completion_tokens": 1,
             "total_tokens": 2,
             "total_latency_s": 0.05,
             "by_model": {"meta-model": {"calls": 1, "tokens": 2}},
         })):
        run_agentic_mode(cfg, POLYMER_CTX, [VALID_PARENT], logger, api_keys={})

    [record] = logger.load_existing_epochs()
    assert record["prompt_state"]["version"] == 0
    assert record["reward"] == 1.0
    assert record["prompt_state"]["score"] == 1.0
    assert record["llm_usage"]["total_calls"] == 2
    assert record["llm_usage"]["total_tokens"] == 15
