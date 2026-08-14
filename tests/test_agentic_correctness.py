from typing import List, Optional

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
MISSING_MARKER_CHILD = "CCO"


class StrictSurrogate(SurrogatePredictor):
    property_name = "StrictProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        assert isinstance(smiles_list, list), "predict expects a list of SMILES"
        self.calls.append(list(smiles_list))
        values = {
            VALID_PARENT: 1.0,
            canonicalize(VALID_PARENT): 1.0,
            VALID_CHILD: 2.0,
            canonicalize(VALID_CHILD): 2.0,
            MISSING_MARKER_CHILD: 3.0,
        }
        return [values.get(smiles, 1.5) for smiles in smiles_list]


POLYMER_CTX = TaskContext(
    property_name="StrictProp",
    property_units="units",
    maximize=True,
    molecule_type="polymer",
    smiles_markers=["[Cu]", "[Au]"],
    similarity_on_repeat_unit=True,
)


def test_agentic_worker_uses_list_safe_predictions_and_marker_validation():
    surrogate = StrictSurrogate()
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
            "explanation": "keeps markers",
        },
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": MISSING_MARKER_CHILD,
            "explanation": "drops markers",
        },
    ])

    valid, invalid = candidates
    assert valid["valid"] is True
    assert valid["parent_property"] == 1.0
    assert valid["child_property"] == 2.0
    assert valid["improvement_factor"] == 2.0
    assert isinstance(valid["child_property"], float)
    assert invalid["valid"] is False
    assert "Missing required marker" in invalid["invalid_reason"]
    assert surrogate.calls == [[VALID_PARENT], [VALID_CHILD]]


def test_agentic_worker_parses_fenced_generated_molecules_mapping():
    text = f"""```json
{{
  "generated_molecules": {{
    "{VALID_PARENT}": {{
      "smiles": ["{VALID_CHILD}"],
      "reasoning": ["added ether"]
    }}
  }}
}}
```"""

    parsed = WorkerAgent._parse_generation_json(text)
    assert parsed["generated_molecules"][VALID_PARENT]["smiles"] == [VALID_CHILD]


def test_agentic_property_tools_respect_surrogate_contract():
    surrogate = StrictSurrogate()
    single = PropertyPredictorTool(surrogate, "StrictProp").execute(VALID_PARENT)
    batch = BatchPropertyPredictorTool(surrogate, "StrictProp").execute([VALID_PARENT, VALID_CHILD])

    assert single.success is True
    assert single.result["StrictProp"] == 1.0
    assert batch.success is True
    assert [row["property"] for row in batch.result] == [1.0, 2.0]
    assert surrogate.calls == [[VALID_PARENT], [VALID_PARENT, VALID_CHILD]]


def test_critic_scores_current_state_before_creating_next_strategy(monkeypatch):
    critic = CriticAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        reward_fn=ParetoHypervolume(),
    )
    current = PromptState.seed("current strategy")
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
    next_state, _, usage = critic.refine(
        candidates=[{
            "valid": True,
            "improvement_factor": 2.0,
            "similarity": 0.5,
            "child_property": 2.0,
            "parent_property": 1.0,
        }],
        current_state=current,
        history=history,
    )

    assert current.score == 1.0
    assert next_state.metadata["reward"] == 1.0
    assert usage["total_calls"] == 0


def test_meta_agent_formats_recent_history_without_missing_all_method():
    meta = MetaAgent(model="test-model", api_keys={}, task_context=POLYMER_CTX)
    history = PromptStateHistory()
    for i in range(4):
        history.add(PromptState(strategy_text=f"strategy {i}", version=i))
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted
    assert "v0: strategy 0" not in formatted
