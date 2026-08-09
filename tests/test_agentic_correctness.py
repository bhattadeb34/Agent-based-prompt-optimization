"""Regression tests for critical agentic workflow correctness."""
from __future__ import annotations

import json
from typing import List, Optional

from apo.agentic_engine import run_agentic_mode
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage, aggregate_usage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.logging.run_logger import RunLogger
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictSurrogate(SurrogatePredictor):
    """Surrogate that fails if callers pass scalar strings to predict()."""

    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.predict_calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise TypeError("predict expects a list of SMILES, not a string")
        self.predict_calls.append(list(smiles_list))
        values = []
        for smi in smiles_list:
            values.append(2.0 if "COCCO" in smi and "CSCCO" not in smi else 1.0)
        return values


def polymer_ctx() -> TaskContext:
    return TaskContext(
        property_name="TestProp",
        property_units="units",
        maximize=True,
        molecule_type="polymer",
        domain_context="[Cu] and [Au] are symbolic repeat-unit markers.",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_agentic_tools_use_surrogate_list_contract():
    surrogate = StrictSurrogate()

    scalar_obs = PropertyPredictorTool(surrogate, "TestProp").execute(VALID_PARENT)
    assert scalar_obs.success is True
    assert scalar_obs.result["TestProp"] == 1.0

    batch_obs = BatchPropertyPredictorTool(surrogate, "TestProp").execute([VALID_PARENT, VALID_CHILD])
    assert batch_obs.success is True
    assert [r["property"] for r in batch_obs.result] == [1.0, 2.0]
    assert surrogate.predict_calls == [[VALID_PARENT], [VALID_PARENT, VALID_CHILD]]


def test_worker_parses_generated_molecules_mapping_and_scores_valid_candidates():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(),
        surrogate=surrogate,
        parent_cache={},
    )

    parsed = worker._parse_json(json.dumps({
        "generated_molecules": {
            VALID_PARENT: {
                "smiles": [VALID_CHILD],
                "reasoning": ["replace sulfur with ether oxygen"],
            }
        }
    }))
    assert VALID_PARENT in parsed["generated_molecules"]

    candidates = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "explanation": "replace sulfur with ether oxygen",
        }
    ])

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["valid"] is True
    assert candidate["parent_property"] == 1.0
    assert candidate["child_property"] == 2.0
    assert candidate["improvement_factor"] == 2.0
    assert candidate["similarity"] > 0.0
    assert all(isinstance(call, list) for call in surrogate.predict_calls)


def test_meta_agent_formats_recent_history_without_missing_accessor():
    history = PromptStateHistory()
    history.add(PromptState.seed("seed strategy"))
    history.add(PromptState(strategy_text="second strategy", version=1))

    meta = MetaAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_ctx(),
    )
    meta.history = history

    formatted = meta._format_recent_strategies()
    assert "v0: seed strategy" in formatted
    assert "v1: second strategy" in formatted


def test_run_agentic_mode_logs_evaluated_state_and_merges_usage(monkeypatch, tmp_path):
    def fake_get_surrogate(name, model_base_path=""):
        return StrictSurrogate()

    def fake_generate(self, strategy, parent_smiles, n_per_molecule=4):
        self._interpretability_trace = {"steps": []}
        return [
            {
                "parent_smiles": VALID_PARENT,
                "child_smiles": VALID_CHILD,
                "valid": True,
                "parent_property": 1.0,
                "child_property": 2.0,
                "improvement_factor": 2.0,
                "similarity": 0.5,
            }
        ], [LLMUsage("worker-model", 10, 5, 0.1)]

    def fake_refine(self, candidates, current_state, history, meta_advice=""):
        self._interpretability_trace = {"steps": []}
        current_state.score = self.reward_fn.compute([c for c in candidates if c.get("valid")])
        return (
            PromptState(
                strategy_text="next strategy",
                version=current_state.version + 1,
                parent_version=current_state.version,
            ),
            {"pareto_insights": ["valid scored candidate"]},
            aggregate_usage([LLMUsage("critic-model", 20, 10, 0.2)]),
        )

    def fake_get_advice(self, history, reward_history):
        return "", aggregate_usage([LLMUsage("meta-model", 5, 5, 0.1)])

    monkeypatch.setattr("apo.agentic_engine.get_surrogate", fake_get_surrogate)
    monkeypatch.setattr("apo.agents.worker.WorkerAgent.generate", fake_generate)
    monkeypatch.setattr("apo.agents.critic.CriticAgent.refine", fake_refine)
    monkeypatch.setattr("apo.agents.meta.MetaAgent.get_advice", fake_get_advice)

    cfg = {
        "task": {"surrogate": "strict"},
        "models": {"worker": "worker-model", "critic": "critic-model", "meta": "meta-model"},
        "optimization": {
            "n_outer_epochs": 1,
            "n_per_molecule": 1,
            "batch_size": 1,
            "meta_interval": 1,
            "reward_function": "pareto_hypervolume",
        },
        "temperatures": {},
    }
    logger = RunLogger(str(tmp_path / "runs"))

    run_agentic_mode(cfg, polymer_ctx(), [VALID_PARENT], logger, api_keys={})

    records = logger.load_existing_epochs()
    assert len(records) == 1
    assert records[0]["prompt_state"]["version"] == 0
    assert records[0]["reward"] > 0.0
    assert records[0]["llm_usage"]["total_calls"] == 2
