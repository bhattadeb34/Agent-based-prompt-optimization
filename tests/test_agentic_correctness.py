import json
from typing import List, Optional
from unittest.mock import patch

from apo.agentic_engine import _merge_usage_summaries
from apo.agents.critic import CriticAgent
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage, aggregate_usage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext
from apo.utils.smiles_utils import compute_similarity


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"
MISSING_MARKER_CHILD = "CCO"


class StrictListSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        assert isinstance(smiles_list, list)
        self.calls.append(list(smiles_list))
        return [float(len(s)) for s in smiles_list]


POLYMER_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="polymer",
    smiles_markers=["[Cu]", "[Au]"],
    similarity_on_repeat_unit=True,
)


def test_worker_validation_uses_task_constraints_and_list_safe_predictor_api():
    surrogate = StrictListSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "explanation": "keeps required markers",
        },
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": MISSING_MARKER_CHILD,
            "explanation": "missing polymer markers",
        },
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] is not None
    assert candidates[0]["child_property"] is not None
    assert candidates[0]["similarity"] == compute_similarity(
        candidates[0]["child_smiles"],
        candidates[0]["parent_smiles"],
        similarity_on_repeat_unit=True,
        marker_strip_tokens=["[Cu]", "[Au]"],
    )
    assert candidates[1]["valid"] is False
    assert "Missing required marker" in candidates[1]["invalid_reason"]
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_worker_parses_markdown_fenced_inner_loop_json_shape():
    payload = {
        "generated_molecules": {
            VALID_PARENT: {
                "smiles": [VALID_CHILD],
                "reasoning": ["reason"],
            }
        }
    }
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        surrogate=StrictListSurrogate(),
        parent_cache={},
    )

    with patch("apo.agents.worker.call_llm", return_value=(fenced, LLMUsage("m", 1, 1, 0.1))):
        parsed = worker._call_llm_for_generation()

    assert parsed == [{
        "parent_smiles": VALID_PARENT,
        "child_smiles": VALID_CHILD,
        "explanation": "reason",
    }]


def test_property_tools_use_surrogate_batch_contract():
    surrogate = StrictListSurrogate()

    single = PropertyPredictorTool(surrogate, "TestProp").execute("CCO")
    batch = BatchPropertyPredictorTool(surrogate, "TestProp").execute(["CCO", "CCN"])

    assert single.success is True
    assert batch.success is True
    assert surrogate.calls == [["CCO"], ["CCO", "CCN"]]


def test_critic_scores_evaluated_state_before_refining_strategy():
    critic = CriticAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        reward_fn=ParetoHypervolume(),
    )
    current = PromptState.seed("seed strategy")
    history = PromptStateHistory()
    history.add(current)
    candidates = [{
        "valid": True,
        "improvement_factor": 2.0,
        "similarity": 0.5,
        "child_smiles": VALID_CHILD,
        "parent_smiles": VALID_PARENT,
    }]

    def fake_run(self, initial_state):
        self.new_state = PromptState(
            strategy_text="next strategy",
            version=current.version + 1,
            parent_version=current.version,
        )
        return (self.new_state, self.analysis), []

    with patch.object(CriticAgent, "run", fake_run):
        new_state, _, usage = critic.refine(candidates, current, history)

    assert current.score == 1.0
    assert new_state.version == 1
    assert usage["total_calls"] == 0


def test_meta_recent_strategies_uses_history_api_without_crashing():
    history = PromptStateHistory()
    for i in range(4):
        history.add(PromptState(strategy_text=f"strategy {i}", version=i))
    meta = MetaAgent(model="test-model", api_keys={}, task_context=POLYMER_CTX)
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted


def test_usage_summaries_merge_dicts_without_llmusage_type_errors():
    worker_usage = aggregate_usage([LLMUsage("worker", 10, 5, 0.1)])
    critic_usage = aggregate_usage([LLMUsage("critic", 7, 3, 0.2)])

    merged = _merge_usage_summaries(worker_usage, critic_usage)

    assert merged["total_calls"] == 2
    assert merged["total_tokens"] == 25
    assert merged["by_model"]["worker"]["calls"] == 1
    assert merged["by_model"]["critic"]["tokens"] == 10
