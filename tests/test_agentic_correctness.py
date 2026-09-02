"""Regression tests for critical agentic workflow correctness."""
from typing import List, Optional
from unittest.mock import patch

from apo.agents.critic import CriticAgent
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext
from apo.utils.smiles_utils import canonicalize


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictSurrogate(SurrogatePredictor):
    """Surrogate that fails if callers violate the List[str] predict contract."""

    property_name = "StrictProp"
    property_units = "units"
    maximize = True

    def __init__(self, values=None):
        self.values = values or {}
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise TypeError("predict expects List[str], not str")
        self.calls.append(list(smiles_list))
        return [self.values.get(smiles, 1.0) for smiles in smiles_list]


POLYMER_CTX = TaskContext(
    property_name="StrictProp",
    property_units="units",
    maximize=True,
    molecule_type="polymer",
    smiles_markers=["[Cu]", "[Au]"],
    similarity_on_repeat_unit=True,
)


def test_property_tools_respect_surrogate_list_contract():
    surrogate = StrictSurrogate({VALID_PARENT: 2.0, VALID_CHILD: 4.0})

    single_obs = PropertyPredictorTool(surrogate, "StrictProp").execute(VALID_PARENT)
    batch_obs = BatchPropertyPredictorTool(surrogate, "StrictProp").execute(
        [VALID_PARENT, VALID_CHILD]
    )

    assert single_obs.success is True
    assert single_obs.result["StrictProp"] == 2.0
    assert batch_obs.success is True
    assert [row["property"] for row in batch_obs.result] == [2.0, 4.0]
    assert surrogate.calls == [[VALID_PARENT], [VALID_PARENT, VALID_CHILD]]


def test_worker_parses_generated_molecules_dict_and_batches_predictions():
    canonical_child = canonicalize(VALID_CHILD)
    surrogate = StrictSurrogate({VALID_PARENT: 2.0, canonical_child: 4.0})
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        surrogate=surrogate,
        parent_cache={VALID_PARENT: 2.0},
    )

    payload = {
        "generated_molecules": {
            VALID_PARENT: {
                "smiles": [VALID_CHILD],
                "reasoning": ["add ether oxygen"],
            }
        }
    }
    candidates = worker._extract_candidates_from_payload(payload)
    validated = worker._validate_candidates(candidates)

    assert len(validated) == 1
    assert validated[0]["valid"] is True
    assert validated[0]["parent_property"] == 2.0
    assert validated[0]["child_property"] == 4.0
    assert validated[0]["improvement_factor"] == 2.0
    assert surrogate.calls == [[canonical_child]]


def test_worker_computes_minimization_improvement_without_dividing_by_zero():
    ctx = TaskContext(
        property_name="StrictProp",
        property_units="units",
        maximize=False,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
    )
    surrogate = StrictSurrogate({canonicalize(VALID_CHILD): 2.0})
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=ctx,
        surrogate=surrogate,
        parent_cache={VALID_PARENT: 4.0},
    )

    validated = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "explanation": "lower property",
        }
    ])

    assert validated[0]["valid"] is True
    assert validated[0]["improvement_factor"] == 2.0


def test_critic_scores_evaluated_state_before_refinement():
    critic = CriticAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        reward_fn=ParetoHypervolume(),
    )
    current = PromptState.seed("seed")
    history = PromptStateHistory()
    history.add(current)
    candidates = [{
        "valid": True,
        "improvement_factor": 2.0,
        "similarity": 0.8,
    }]

    def fake_run(self, initial_state):
        self.new_state = PromptState(
            strategy_text="next",
            version=current.version + 1,
            parent_version=current.version,
        )
        return (self.new_state, []), []

    with patch.object(CriticAgent, "run", fake_run):
        new_state, _, usage = critic.refine(candidates, current, history)

    assert current.score == 1.6
    assert new_state.version == 1
    assert usage["total_calls"] == 0


def test_meta_formats_recent_strategies_from_history_api():
    history = PromptStateHistory()
    for i in range(4):
        history.add(PromptState(strategy_text=f"strategy {i}", version=i))

    meta = MetaAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
    )
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted
