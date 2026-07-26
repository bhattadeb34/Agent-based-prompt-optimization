from typing import List, Optional

import pytest

from apo.agentic_engine import _merge_usage_dicts
from apo.agents.critic import CriticAgent
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage, aggregate_usage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import PropertyOnly
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"
MISSING_MARKER_CHILD = "CCO"


class StrictSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"

    def __init__(self, values):
        self.values = values
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise TypeError("predict expects a list of SMILES")
        self.calls.append(list(smiles_list))
        return [self.values.get(smi) for smi in smiles_list]


POLYMER_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="polymer",
    smiles_markers=["[Cu]", "[Au]"],
    similarity_on_repeat_unit=True,
)


def test_worker_uses_list_safe_predictions_and_required_markers():
    surrogate = StrictSurrogate({VALID_PARENT: 2.0, VALID_CHILD: 4.0})
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
            "explanation": "valid child",
        },
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": MISSING_MARKER_CHILD,
            "explanation": "missing markers",
        },
    ])

    valid, invalid = candidates
    assert valid["valid"] is True
    assert valid["parent_property"] == 2.0
    assert valid["child_property"] == 4.0
    assert valid["improvement_factor"] == 2.0
    assert 0.0 <= valid["similarity"] <= 1.0
    assert invalid["valid"] is False
    assert "Missing required marker" in invalid["invalid_reason"]
    assert surrogate.calls == [[VALID_PARENT], [VALID_CHILD]]


def test_worker_improvement_honors_minimization_direction():
    ctx = TaskContext(
        property_name="TestProp",
        property_units="units",
        maximize=False,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
    )
    surrogate = StrictSurrogate({VALID_PARENT: 10.0, VALID_CHILD: 5.0})
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=ctx,
        surrogate=surrogate,
        parent_cache={},
    )

    candidate = worker._validate_candidates([{
        "parent_smiles": VALID_PARENT,
        "child_smiles": VALID_CHILD,
        "explanation": "lower is better",
    }])[0]

    assert candidate["valid"] is True
    assert candidate["improvement_factor"] == 2.0


def test_property_tools_use_surrogate_list_api_correctly():
    surrogate = StrictSurrogate({"CC": 1.0, "CO": 2.0})

    single_obs = PropertyPredictorTool(surrogate, "TestProp").execute("CC")
    batch_obs = BatchPropertyPredictorTool(surrogate, "TestProp").execute(["CC", "CO"])

    assert single_obs.success is True
    assert single_obs.result["TestProp"] == 1.0
    assert batch_obs.success is True
    assert [r["property"] for r in batch_obs.result] == [1.0, 2.0]
    assert surrogate.calls == [["CC"], ["CC", "CO"]]


def test_critic_scores_current_state_before_refinement(monkeypatch):
    critic = CriticAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        reward_fn=PropertyOnly(),
    )
    current = PromptState.seed("seed")
    history = PromptStateHistory()
    history.add(current)
    monkeypatch.setattr(critic, "run", lambda initial_state: (None, []))

    critic.refine(
        candidates=[{"valid": True, "improvement_factor": 3.5, "similarity": 0.8}],
        current_state=current,
        history=history,
    )

    assert current.score == 3.5


def test_meta_formats_recent_strategies_without_missing_all_method():
    meta = MetaAgent(model="test-model", api_keys={}, task_context=POLYMER_CTX)
    history = PromptStateHistory()
    for version in range(4):
        history.add(PromptState(strategy_text=f"strategy {version}", version=version))
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted
    assert "v0: strategy 0" not in formatted


def test_agentic_usage_merge_handles_aggregated_dicts():
    worker_usage = aggregate_usage([LLMUsage("worker-model", 10, 5, 0.1)])
    critic_usage = aggregate_usage([LLMUsage("critic-model", 20, 10, 0.2)])

    merged = _merge_usage_dicts(worker_usage, critic_usage)

    assert merged["total_calls"] == 2
    assert merged["total_tokens"] == 45
    assert merged["by_model"]["worker-model"]["calls"] == 1
    assert merged["by_model"]["critic-model"]["tokens"] == 30
