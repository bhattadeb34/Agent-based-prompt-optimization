from typing import List, Optional

from apo.agentic_engine import _merge_usage_summaries
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage, aggregate_usage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictSurrogate(SurrogatePredictor):
    property_name = "Strict Property"
    property_units = "units"
    maximize = True

    def __init__(self, parent_value: float = 1.0, child_value: float = 2.0):
        self.parent_value = parent_value
        self.child_value = child_value
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise TypeError("predict expects a list of SMILES, not a scalar string")
        self.calls.append(list(smiles_list))
        return [
            self.parent_value if smiles == VALID_PARENT else self.child_value
            for smiles in smiles_list
        ]


def polymer_ctx(maximize: bool = True) -> TaskContext:
    return TaskContext(
        property_name="Strict Property",
        property_units="units",
        maximize=maximize,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_property_tools_use_surrogate_list_contract():
    surrogate = StrictSurrogate()

    single = PropertyPredictorTool(surrogate, "Strict Property")
    obs = single.execute(VALID_PARENT)
    assert obs.success is True
    assert obs.result["Strict Property"] == 1.0

    batch = BatchPropertyPredictorTool(surrogate, "Strict Property")
    obs = batch.execute([VALID_PARENT, VALID_CHILD])
    assert obs.success is True
    assert [row["property"] for row in obs.result] == [1.0, 2.0]
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_worker_validation_uses_predict_single_and_task_markers():
    surrogate = StrictSurrogate(parent_value=1.0, child_value=2.0)
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(maximize=True),
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "explanation": "valid polymer child",
        },
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": "CCO",
            "explanation": "missing polymer markers",
        },
    ])

    valid, invalid = candidates
    assert valid["valid"] is True
    assert valid["parent_property"] == 1.0
    assert valid["child_property"] == 2.0
    assert valid["improvement_factor"] == 2.0
    assert invalid["valid"] is False
    assert "Missing required marker" in invalid["invalid_reason"]
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_worker_improvement_factor_respects_minimization_direction():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(maximize=False),
        surrogate=surrogate,
        parent_cache={},
    )

    assert worker._compute_improvement_factor(parent_property=10.0, child_property=5.0) == 2.0
    assert worker._compute_improvement_factor(parent_property=10.0, child_property=0.0) == 0.0


def test_worker_parses_fenced_generated_molecules_mapping():
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(),
        surrogate=StrictSurrogate(),
        parent_cache={},
    )
    text = f"""```json
{{
  "generated_molecules": {{
    "{VALID_PARENT}": {{
      "smiles": ["{VALID_CHILD}"],
      "reasoning": ["added ether oxygen"]
    }}
  }}
}}
```"""

    parsed = worker._parse_json_response(text)
    assert VALID_PARENT in parsed["generated_molecules"]


def test_meta_formats_recent_strategies_with_history_api():
    history = PromptStateHistory()
    for version in range(4):
        history.add(PromptState(strategy_text=f"strategy {version}", version=version))

    meta = MetaAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(),
    )
    meta.history = history

    formatted = meta._format_recent_strategies()
    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted
    assert "v0: strategy 0" not in formatted


def test_agentic_usage_summaries_merge_without_llmusage_objects():
    worker_usage = aggregate_usage([LLMUsage("worker-model", 10, 5, 0.25)])
    critic_usage = aggregate_usage([LLMUsage("critic-model", 20, 10, 0.5)])

    merged = _merge_usage_summaries(worker_usage, critic_usage)
    assert merged["total_calls"] == 2
    assert merged["total_tokens"] == 45
    assert merged["by_model"]["worker-model"]["calls"] == 1
    assert merged["by_model"]["critic-model"]["tokens"] == 30
