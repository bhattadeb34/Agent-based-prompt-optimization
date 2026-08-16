"""Regression tests for critical agentic workflow correctness."""
from __future__ import annotations

import json
from typing import List, Optional

from apo.agentic_engine import _merge_usage_summaries
from apo.agents.critic import CriticAgent
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictListSurrogate(SurrogatePredictor):
    property_name = "StrictProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise TypeError("predict expects a list of SMILES")
        self.calls.append(list(smiles_list))
        return [float(len(smiles)) for smiles in smiles_list]


def polymer_ctx(maximize: bool = True) -> TaskContext:
    return TaskContext(
        property_name="StrictProp",
        property_units="units",
        maximize=maximize,
        molecule_type="polymer",
        domain_context="[Cu] and [Au] are backbone markers.",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_property_tools_use_strict_surrogate_api():
    surrogate = StrictListSurrogate()

    single = PropertyPredictorTool(surrogate, "StrictProp").execute(CHILD)
    assert single.success is True
    assert surrogate.calls[-1] == [CHILD]

    batch = BatchPropertyPredictorTool(surrogate, "StrictProp").execute([PARENT, CHILD])
    assert batch.success is True
    assert surrogate.calls[-1] == [PARENT, CHILD]


def test_worker_validation_uses_predict_single_and_required_markers():
    surrogate = StrictListSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(),
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": PARENT,
            "child_smiles": CHILD,
            "explanation": "valid polymer child",
        },
        {
            "parent_smiles": PARENT,
            "child_smiles": "CCO",
            "explanation": "missing polymer markers",
        },
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] is not None
    assert candidates[0]["child_property"] is not None
    assert all(isinstance(call, list) for call in surrogate.calls)
    assert candidates[1]["valid"] is False
    assert "Missing required marker" in candidates[1]["invalid_reason"]


def test_worker_parses_fenced_generated_molecules_mapping():
    payload = {
        "generated_molecules": {
            PARENT: {
                "smiles": [CHILD],
                "reasoning": ["added ether oxygen"],
            }
        }
    }

    parsed = WorkerAgent._parse_generation_output(
        "```json\n" + json.dumps(payload) + "\n```"
    )

    assert parsed == [{
        "parent_smiles": PARENT,
        "child_smiles": CHILD,
        "explanation": "added ether oxygen",
    }]


def test_critic_scores_evaluated_current_state_when_refining():
    critic = CriticAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(),
        reward_fn=ParetoHypervolume(),
    )
    current = PromptState.seed("seed strategy")
    history = PromptStateHistory()
    history.add(current)

    def fake_run(initial_state):
        critic.new_state = PromptState(
            strategy_text="next strategy",
            version=1,
            parent_version=0,
        )
        return (critic.new_state, []), []

    critic.run = fake_run
    candidates = [{
        "valid": True,
        "improvement_factor": 2.0,
        "similarity": 0.8,
    }]

    critic.refine(candidates, current, history)

    assert current.score == ParetoHypervolume().compute(candidates)


def test_meta_formats_recent_strategies_with_history_api():
    history = PromptStateHistory()
    for version in range(4):
        history.add(PromptState(strategy_text=f"strategy {version}", version=version))

    meta = MetaAgent(model="test-model", api_keys={}, task_context=polymer_ctx())
    meta.history = history

    text = meta._format_recent_strategies()

    assert "v0" not in text
    assert "v1" in text
    assert "v2" in text
    assert "v3" in text


def test_agentic_usage_summary_merges_dicts_without_llmusage_objects():
    merged = _merge_usage_summaries([
        {
            "total_calls": 1,
            "total_prompt_tokens": 10,
            "total_completion_tokens": 5,
            "total_tokens": 15,
            "total_latency_s": 0.5,
            "by_model": {"worker": {"calls": 1, "tokens": 15}},
        },
        {
            "total_calls": 2,
            "total_prompt_tokens": 20,
            "total_completion_tokens": 7,
            "total_tokens": 27,
            "total_latency_s": 1.0,
            "by_model": {"critic": {"calls": 2, "tokens": 27}},
        },
    ])

    assert merged["total_calls"] == 3
    assert merged["total_tokens"] == 42
    assert merged["by_model"]["worker"]["calls"] == 1
    assert merged["by_model"]["critic"]["tokens"] == 27
