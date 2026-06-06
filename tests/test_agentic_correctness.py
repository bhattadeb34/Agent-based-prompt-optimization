import json
from pathlib import Path
from unittest.mock import patch

from apo.agentic_engine import run_agentic_mode
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState
from apo.logging.run_logger import RunLogger
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictListSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list):
        assert isinstance(smiles_list, list), "surrogate.predict must receive a list"
        self.calls.append(list(smiles_list))
        return [2.0 for _ in smiles_list]


POLYMER_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="polymer",
    domain_context="[Cu] and [Au] are backbone markers.",
    smiles_markers=["[Cu]", "[Au]"],
    similarity_on_repeat_unit=True,
)


def test_worker_validates_markers_and_uses_single_prediction_api():
    surrogate = StrictListSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        surrogate=surrogate,
        parent_cache={},
    )

    valid, missing_marker = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "explanation": "keeps required markers",
        },
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": "CCO",
            "explanation": "drops polymer markers",
        },
    ])

    assert valid["valid"] is True
    assert valid["child_property"] == 2.0
    assert valid["parent_property"] == 2.0
    assert valid["improvement_factor"] == 1.0
    assert isinstance(valid["similarity"], float)
    assert missing_marker["valid"] is False
    assert missing_marker["invalid_reason"] == "Missing required marker: [Cu]"
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_property_tools_respect_surrogate_batch_contract():
    surrogate = StrictListSurrogate()

    single = PropertyPredictorTool(surrogate, "TestProp").execute(VALID_CHILD)
    batch = BatchPropertyPredictorTool(surrogate, "TestProp").execute([VALID_PARENT, VALID_CHILD])

    assert single.success is True
    assert single.result["TestProp"] == 2.0
    assert batch.success is True
    assert [r["property"] for r in batch.result] == [2.0, 2.0]
    assert surrogate.calls[-1] == [VALID_PARENT, VALID_CHILD]


def test_agentic_mode_logs_computed_reward_and_aggregates_dict_usage(tmp_path):
    logger = RunLogger(str(tmp_path), run_id="agentic_regression")
    cfg = {
        "task": {
            "surrogate": "mock",
            "property_name": "TestProp",
            "property_units": "units",
            "maximize": True,
        },
        "models": {
            "worker": "worker-model",
            "critic": "critic-model",
            "meta": "meta-model",
        },
        "optimization": {
            "mode": "agentic",
            "n_outer_epochs": 1,
            "n_per_molecule": 1,
            "batch_size": 1,
            "meta_interval": 1,
            "reward_function": "weighted_sum",
            "reward_function_kwargs": {"alpha": 0.25},
        },
    }
    candidates = [{
        "parent_smiles": VALID_PARENT,
        "child_smiles": VALID_CHILD,
        "parent_property": 1.0,
        "child_property": 2.0,
        "improvement_factor": 2.0,
        "similarity": 0.5,
        "valid": True,
        "explanation": "improves property",
    }]

    def fake_generate(self, strategy, parent_smiles, n_per_molecule):
        self._interpretability_trace = {"agent": "worker"}
        return candidates, [LLMUsage("worker-model", 5, 5, 0.1)]

    def fake_refine(self, candidates, current_state, history, meta_advice):
        self._interpretability_trace = {"agent": "critic"}
        return (
            PromptState(
                strategy_text="next strategy",
                version=current_state.version + 1,
                parent_version=current_state.version,
                rationale="test",
                model_used="critic-model",
            ),
            {"analysis": "ok"},
            {
                "total_calls": 2,
                "total_prompt_tokens": 7,
                "total_completion_tokens": 3,
                "total_tokens": 10,
                "total_latency_s": 0.2,
                "by_model": {"critic-model": {"calls": 2, "tokens": 10}},
            },
        )

    with patch("apo.agentic_engine.get_surrogate", return_value=StrictListSurrogate()), \
         patch("apo.agentic_engine.WorkerAgent.generate", fake_generate), \
         patch("apo.agentic_engine.CriticAgent.refine", fake_refine), \
         patch("apo.agentic_engine.MetaAgent.get_advice", return_value=(
             "",
             {
                 "total_calls": 1,
                 "total_prompt_tokens": 4,
                 "total_completion_tokens": 1,
                 "total_tokens": 5,
                 "total_latency_s": 0.1,
                 "by_model": {"meta-model": {"calls": 1, "tokens": 5}},
             },
         )):
        run_dir = run_agentic_mode(cfg, POLYMER_CTX, [VALID_PARENT], logger, api_keys={})

    assert Path(run_dir).exists()
    records = logger.load_existing_epochs()
    assert len(records) == 1
    assert records[0]["reward"] == 0.875
    assert records[0]["prompt_state"]["version"] == 0
    assert records[0]["prompt_state"]["score"] == 0.875
    assert records[0]["llm_usage"]["total_calls"] == 3
    assert records[0]["llm_usage"]["total_tokens"] == 20

    history = json.loads((Path(run_dir) / "prompt_history.json").read_text())
    assert history[0]["score"] == 0.875
