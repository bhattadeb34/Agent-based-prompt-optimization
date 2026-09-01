from typing import List, Optional

from apo.agentic_engine import _merge_usage_summary
from apo.agents.critic import CriticAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage, aggregate_usage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


class StrictSurrogate(SurrogatePredictor):
    property_name = "StrictProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise TypeError("predict expects a list of SMILES")
        self.calls.append(list(smiles_list))
        return [float(len(s)) for s in smiles_list]


GENERIC_CTX = TaskContext(
    property_name="StrictProp",
    property_units="units",
    maximize=True,
    molecule_type="organic compound",
)

POLYMER_CTX = TaskContext(
    property_name="StrictProp",
    property_units="units",
    maximize=True,
    molecule_type="polymer",
    smiles_markers=["[Cu]", "[Au]"],
)


def test_agentic_tools_use_list_safe_surrogate_calls():
    surrogate = StrictSurrogate()

    single = PropertyPredictorTool(surrogate, "StrictProp").execute("CCO")
    assert single.success is True
    assert single.result["StrictProp"] == 3.0

    batch = BatchPropertyPredictorTool(surrogate, "StrictProp").execute(["CC", "CCO"])
    assert batch.success is True
    assert [r["property"] for r in batch.result] == [2.0, 3.0]
    assert surrogate.calls == [["CCO"], ["CC", "CCO"]]


def test_worker_parses_generated_molecules_mapping_schema():
    data = {
        "generated_molecules": {
            "CC": {
                "smiles": ["CCO", "CCC"],
                "reasoning": ["add oxygen", "extend chain"],
            }
        }
    }

    candidates = WorkerAgent._extract_candidates(data)

    assert candidates == [
        {"parent_smiles": "CC", "child_smiles": "CCO", "explanation": "add oxygen"},
        {"parent_smiles": "CC", "child_smiles": "CCC", "explanation": "extend chain"},
    ]


def test_worker_validation_uses_predict_single_and_scores_candidate():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=GENERIC_CTX,
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {"parent_smiles": "CC", "child_smiles": "CCO", "explanation": "add oxygen"}
    ])

    assert len(candidates) == 1
    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 2.0
    assert candidates[0]["child_property"] == 3.0
    assert candidates[0]["improvement_factor"] == 1.5
    assert surrogate.calls == [["CC"], ["CCO"]]


def test_worker_validation_enforces_task_markers():
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        surrogate=StrictSurrogate(),
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": "CC(CO[Cu])CSCCOC(=O)[Au]",
            "child_smiles": "CCO",
            "explanation": "plain molecule",
        }
    ])

    assert candidates[0]["valid"] is False
    assert "Missing required marker" in candidates[0]["invalid_reason"]


def test_critic_scores_current_state_before_refinement(monkeypatch):
    critic = CriticAgent(
        model="test-model",
        api_keys={},
        task_context=GENERIC_CTX,
        reward_fn=ParetoHypervolume(),
    )
    current = PromptState.seed("generate better molecules")
    history = PromptStateHistory()
    history.add(current)

    def fake_run(initial_state):
        critic.new_state = PromptState(
            strategy_text="next strategy",
            version=current.version + 1,
            parent_version=current.version,
        )
        return critic.new_state, []

    monkeypatch.setattr(critic, "run", fake_run)
    candidates = [{
        "valid": True,
        "improvement_factor": 2.0,
        "similarity": 0.5,
        "child_property": 4.0,
        "parent_property": 2.0,
    }]

    new_state, _, _ = critic.refine(candidates, current, history)

    assert current.score == 1.0
    assert new_state.version == 1


def test_agentic_usage_summary_merge_handles_aggregated_dicts():
    total = aggregate_usage([LLMUsage("worker", 10, 5, 0.25)])
    extra = {
        "total_calls": 2,
        "total_prompt_tokens": 20,
        "total_completion_tokens": 10,
        "total_tokens": 30,
        "total_latency_s": 0.75,
        "by_model": {"critic": {"calls": 2, "tokens": 30}},
    }

    merged = _merge_usage_summary(total, extra)

    assert merged["total_calls"] == 3
    assert merged["total_prompt_tokens"] == 30
    assert merged["total_completion_tokens"] == 15
    assert merged["total_tokens"] == 45
    assert merged["by_model"]["worker"] == {"calls": 1, "tokens": 15}
    assert merged["by_model"]["critic"] == {"calls": 2, "tokens": 30}
