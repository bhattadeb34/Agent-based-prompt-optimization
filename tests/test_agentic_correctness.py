"""Regression tests for agentic workflow correctness."""

from typing import List, Optional

from apo.agentic_engine import _merge_usage_summary
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage, aggregate_usage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


class StrictListSurrogate(SurrogatePredictor):
    """Surrogate that catches accidental scalar calls to predict()."""

    property_name = "Mock Property"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise TypeError("predict expects a list of SMILES")
        self.calls.append(list(smiles_list))
        return [float(len(smiles)) for smiles in smiles_list]


def test_property_tools_use_surrogate_contract():
    surrogate = StrictListSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "prop").execute("CC")
    assert single_obs.success
    assert single_obs.result["prop"] == 2.0

    batch_obs = BatchPropertyPredictorTool(surrogate, "prop").execute(["CC", "CCO"])
    assert batch_obs.success
    assert [r["property"] for r in batch_obs.result] == [2.0, 3.0]
    assert surrogate.calls == [["CC"], ["CC", "CCO"]]


def test_worker_parses_generated_molecules_mapping():
    text = """```json
{
  "generated_molecules": {
    "CC": {
      "smiles": ["CCO", "CCC"],
      "reasoning": ["add oxygen", "extend chain"]
    }
  }
}
```"""

    candidates = WorkerAgent._parse_generation_output(text)

    assert candidates == [
        {"parent_smiles": "CC", "child_smiles": "CCO", "explanation": "add oxygen"},
        {"parent_smiles": "CC", "child_smiles": "CCC", "explanation": "extend chain"},
    ]


def test_worker_validation_uses_predict_single_for_scalar_scores():
    surrogate = StrictListSurrogate()
    ctx = TaskContext(
        property_name="Mock Property",
        property_units="units",
        maximize=True,
        molecule_type="organic compound",
    )
    worker = WorkerAgent(
        model="mock-model",
        api_keys={},
        task_context=ctx,
        surrogate=surrogate,
        parent_cache={},
    )

    [candidate] = worker._validate_candidates([
        {
            "parent_smiles": "CC",
            "child_smiles": "CCO",
            "explanation": "add oxygen",
        }
    ])

    assert candidate["valid"] is True
    assert candidate["parent_property"] == 2.0
    assert candidate["child_property"] == 3.0
    assert candidate["improvement_factor"] == 1.5
    assert surrogate.calls == [["CC"], ["CCO"]]


def test_merge_usage_summary_keeps_dicts_out_of_aggregate_usage():
    base = aggregate_usage([LLMUsage("worker-model", 10, 5, 0.25)])
    critic_summary = {
        "total_calls": 2,
        "total_prompt_tokens": 20,
        "total_completion_tokens": 8,
        "total_tokens": 28,
        "total_latency_s": 0.75,
        "by_model": {"critic-model": {"calls": 2, "tokens": 28}},
    }

    merged = _merge_usage_summary(base, critic_summary)

    assert merged["total_calls"] == 3
    assert merged["total_prompt_tokens"] == 30
    assert merged["total_completion_tokens"] == 13
    assert merged["total_tokens"] == 43
    assert merged["by_model"]["worker-model"] == {"calls": 1, "tokens": 15}
    assert merged["by_model"]["critic-model"] == {"calls": 2, "tokens": 28}


def test_meta_agent_formats_recent_strategies_with_history_api():
    ctx = TaskContext(
        property_name="Mock Property",
        property_units="units",
        maximize=True,
        molecule_type="organic compound",
    )
    history = PromptStateHistory()
    for version in range(4):
        history.add(PromptState(strategy_text=f"strategy {version}", version=version))

    meta = MetaAgent(model="mock-model", api_keys={}, task_context=ctx)
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v1: strategy 1" in formatted
    assert "v2: strategy 2" in formatted
    assert "v3: strategy 3" in formatted
    assert "v0: strategy 0" not in formatted
