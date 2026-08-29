import json
from typing import List, Optional
from unittest.mock import patch

from apo.agentic_engine import _merge_usage_summaries
from apo.agents.critic import CriticAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


class StrictSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        assert isinstance(smiles_list, list), "predict must receive a list, not a raw SMILES string"
        self.calls.append(list(smiles_list))
        return [float(len(smiles)) for smiles in smiles_list]


POLYMER_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="polymer",
    smiles_markers=["[Cu]", "[Au]"],
    similarity_on_repeat_unit=True,
)

VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


def test_property_tools_use_single_and_batch_surrogate_contracts():
    surrogate = StrictSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "TestProp").execute(VALID_CHILD)
    assert single_obs.success is True
    assert surrogate.calls[-1] == [VALID_CHILD]

    batch_obs = BatchPropertyPredictorTool(surrogate, "TestProp").execute([VALID_PARENT, VALID_CHILD])
    assert batch_obs.success is True
    assert surrogate.calls[-1] == [VALID_PARENT, VALID_CHILD]
    assert [row["valid"] for row in batch_obs.result] == [True, True]


def test_worker_validation_batches_predictions_and_enforces_markers():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        surrogate=surrogate,
        parent_cache={},
    )
    worker.parent_smiles_list = [VALID_PARENT]

    candidates = worker._validate_candidates([
        {"parent_smiles": VALID_PARENT, "child_smiles": VALID_CHILD, "explanation": "valid"},
        {"parent_smiles": VALID_PARENT, "child_smiles": "CCO", "explanation": "missing markers"},
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["child_property"] is not None
    assert candidates[0]["parent_property"] is not None
    assert candidates[0]["improvement_factor"] > 0
    assert candidates[1]["valid"] is False
    assert "Missing required marker" in candidates[1]["invalid_reason"]
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_worker_parses_existing_generated_molecules_mapping_schema():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        surrogate=surrogate,
        parent_cache={},
    )
    payload = {
        "generated_molecules": {
            VALID_PARENT: {
                "smiles": [VALID_CHILD],
                "reasoning": ["added oxygen"],
            }
        }
    }

    with patch(
        "apo.agents.worker.call_llm",
        return_value=(f"```json\n{json.dumps(payload)}\n```", LLMUsage("test-model", 1, 1, 0.0)),
    ):
        parsed = worker._call_llm_for_generation()

    assert parsed == [{
        "parent_smiles": VALID_PARENT,
        "child_smiles": VALID_CHILD,
        "explanation": "added oxygen",
    }]


def test_critic_scores_current_state_before_returning_new_strategy():
    current = PromptState.seed("initial")
    history = PromptStateHistory()
    history.add(current)
    critic = CriticAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        reward_fn=ParetoHypervolume(),
    )
    candidate = {
        "valid": True,
        "improvement_factor": 2.0,
        "similarity": 0.5,
        "child_smiles": VALID_CHILD,
        "parent_smiles": VALID_PARENT,
        "child_property": 2.0,
        "parent_property": 1.0,
    }

    def fake_run(self, initial_state):
        self.new_state = PromptState(
            strategy_text="next",
            version=current.version + 1,
            parent_version=current.version,
        )
        return (self.new_state, self.analysis), []

    with patch.object(CriticAgent, "run", fake_run):
        new_state, _, _ = critic.refine([candidate], current, history)

    assert new_state.version == 1
    assert current.score == 1.0


def test_usage_summaries_merge_without_raw_usage_objects():
    first = {
        "total_calls": 1,
        "total_prompt_tokens": 3,
        "total_completion_tokens": 4,
        "total_tokens": 7,
        "total_latency_s": 0.25,
        "by_model": {"worker": {"calls": 1, "tokens": 7}},
    }
    second = {
        "total_calls": 2,
        "total_prompt_tokens": 5,
        "total_completion_tokens": 6,
        "total_tokens": 11,
        "total_latency_s": 0.75,
        "by_model": {"critic": {"calls": 2, "tokens": 11}},
    }

    merged = _merge_usage_summaries(first, second)

    assert merged["total_calls"] == 3
    assert merged["total_tokens"] == 18
    assert merged["total_latency_s"] == 1.0
    assert merged["by_model"]["worker"]["calls"] == 1
    assert merged["by_model"]["critic"]["tokens"] == 11
