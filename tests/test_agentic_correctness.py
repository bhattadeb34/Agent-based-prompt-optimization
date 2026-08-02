"""Regression tests for critical agentic workflow correctness."""
from typing import List, Optional

from apo.agentic_engine import _merge_usage_summary
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"
MISSING_MARKERS = "CCO"


class StrictSurrogate(SurrogatePredictor):
    property_name = "StrictProp"
    property_units = "units"
    maximize = True

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise TypeError("predict expects a list of SMILES")
        return [float(len(smiles)) for smiles in smiles_list]


def polymer_ctx(maximize: bool = True) -> TaskContext:
    return TaskContext(
        property_name="StrictProp",
        property_units="units",
        maximize=maximize,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def make_worker(ctx: TaskContext, surrogate: SurrogatePredictor) -> WorkerAgent:
    return WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=ctx,
        surrogate=surrogate,
        parent_cache={},
    )


def test_property_tools_use_list_safe_surrogate_api():
    surrogate = StrictSurrogate()

    single = PropertyPredictorTool(surrogate, "StrictProp").execute(CHILD)
    assert single.success
    assert single.result["StrictProp"] == float(len(CHILD))

    batch = BatchPropertyPredictorTool(surrogate, "StrictProp").execute([PARENT, CHILD])
    assert batch.success
    assert [row["property"] for row in batch.result] == [float(len(PARENT)), float(len(CHILD))]


def test_worker_validation_rejects_markerless_polymer_and_scores_valid_child():
    worker = make_worker(polymer_ctx(), StrictSurrogate())

    candidates = worker._validate_candidates([
        {"parent_smiles": PARENT, "child_smiles": CHILD, "explanation": "valid"},
        {"parent_smiles": PARENT, "child_smiles": MISSING_MARKERS, "explanation": "missing markers"},
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == float(len(PARENT))
    assert candidates[0]["child_property"] == float(len(CHILD))
    assert candidates[0]["improvement_factor"] > 0
    assert candidates[0]["similarity"] > 0

    assert candidates[1]["valid"] is False
    assert "Missing required marker" in candidates[1]["invalid_reason"]


def test_worker_uses_minimization_direction_for_improvement_factor():
    worker = make_worker(polymer_ctx(maximize=False), StrictSurrogate())

    candidate = worker._validate_candidates([
        {"parent_smiles": PARENT, "child_smiles": CHILD, "explanation": "valid"},
    ])[0]

    assert candidate["valid"] is True
    assert candidate["improvement_factor"] == candidate["parent_property"] / candidate["child_property"]


def test_worker_parses_existing_generated_molecules_mapping_schema():
    worker = make_worker(polymer_ctx(), StrictSurrogate())

    parsed = worker._parse_generated_candidates({
        "generated_molecules": {
            PARENT: {
                "smiles": [CHILD],
                "reasoning": ["added ether"],
            }
        }
    })

    assert parsed == [{
        "parent_smiles": PARENT,
        "child_smiles": CHILD,
        "explanation": "added ether",
    }]


def test_meta_formats_recent_strategies_with_history_api():
    history = PromptStateHistory()
    for i in range(4):
        history.add(PromptState(strategy_text=f"strategy {i}", version=i))

    meta = MetaAgent(model="test-model", api_keys={}, task_context=polymer_ctx())
    meta.history = history

    formatted = meta._format_recent_strategies()
    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted
    assert "v0: strategy 0" not in formatted


def test_merge_usage_summary_keeps_aggregate_dicts_out_of_llm_usage_list():
    base = {
        "total_calls": 1,
        "total_prompt_tokens": 10,
        "total_completion_tokens": 5,
        "total_tokens": 15,
        "total_latency_s": 0.25,
        "by_model": {"worker": {"calls": 1, "tokens": 15}},
    }
    extra = {
        "total_calls": 2,
        "total_prompt_tokens": 20,
        "total_completion_tokens": 10,
        "total_tokens": 30,
        "total_latency_s": 0.5,
        "by_model": {"critic": {"calls": 2, "tokens": 30}},
    }

    merged = _merge_usage_summary(base, extra)

    assert merged["total_calls"] == 3
    assert merged["total_tokens"] == 45
    assert merged["avg_latency_s"] == 0.25
    assert merged["by_model"]["worker"] == {"calls": 1, "tokens": 15}
    assert merged["by_model"]["critic"] == {"calls": 2, "tokens": 30}
