"""Regression tests for critical agentic workflow correctness bugs."""
from typing import List, Optional

from apo.agents.critic import CriticAgent
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictSurrogate(SurrogatePredictor):
    property_name = "Strict"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise AssertionError("predict() must receive a list, not a string")
        self.calls.append(list(smiles_list))
        return [2.0 if "OCCO" in smiles or "COCC" in smiles else 1.0 for smiles in smiles_list]


def polymer_context() -> TaskContext:
    return TaskContext(
        property_name="Strict",
        property_units="units",
        maximize=True,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_worker_validation_uses_predict_single_and_marker_constraints():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "explanation": "valid marker-preserving edit",
        },
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": "CCO",
            "explanation": "RDKit-valid but missing polymer markers",
        },
    ])

    valid, invalid = candidates
    assert valid["valid"] is True
    assert isinstance(valid["parent_property"], float)
    assert isinstance(valid["child_property"], float)
    assert valid["improvement_factor"] > 1.0

    assert invalid["valid"] is False
    assert "Missing required marker" in invalid["invalid_reason"]
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_property_tools_use_surrogate_batch_contract():
    surrogate = StrictSurrogate()

    single = PropertyPredictorTool(surrogate, "Strict").execute(VALID_CHILD)
    batch = BatchPropertyPredictorTool(surrogate, "Strict").execute([VALID_PARENT, VALID_CHILD])

    assert single.success is True
    assert single.result["Strict"] == 2.0
    assert batch.success is True
    assert [row["valid"] for row in batch.result] == [True, True]
    assert surrogate.calls == [[VALID_CHILD], [VALID_PARENT, VALID_CHILD]]


def test_critic_refine_assigns_reward_and_returns_usage_summary(monkeypatch):
    ctx = polymer_context()
    critic = CriticAgent(
        model="test-model",
        api_keys={},
        task_context=ctx,
        reward_fn=ParetoHypervolume(),
    )
    current = PromptState.seed("seed strategy")
    history = PromptStateHistory()
    history.add(current)

    def fake_run(initial_state):
        critic.new_state = PromptState(
            strategy_text="next strategy",
            version=current.version + 1,
            rationale="test",
            parent_version=current.version,
            model_used=critic.model,
        )
        critic.analysis = {"pareto_insights": ["test"]}
        critic.all_usages = [LLMUsage("test-model", 10, 5, 0.1)]
        return critic.new_state, []

    monkeypatch.setattr(critic, "run", fake_run)
    new_state, analysis, usage = critic.refine(
        candidates=[{"valid": True, "improvement_factor": 2.0, "similarity": 0.5}],
        current_state=current,
        history=history,
    )

    assert new_state.score == 1.0
    assert analysis == {"pareto_insights": ["test"]}
    assert usage["total_calls"] == 1
    assert critic.all_usages[0].model == "test-model"


def test_meta_formats_recent_strategies_from_history():
    meta = MetaAgent(model="test-model", api_keys={}, task_context=polymer_context())
    history = PromptStateHistory()
    for i in range(4):
        history.add(PromptState(strategy_text=f"strategy {i}", version=i))

    meta.history = history

    formatted = meta._format_recent_strategies()
    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted
