"""Regression tests for high-impact agentic workflow correctness."""
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

from apo.agentic_engine import _merge_usage_summary, run_agentic_mode
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage, aggregate_usage
from apo.logging.run_logger import RunLogger
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


class StrictListSurrogate(SurrogatePredictor):
    """Surrogate that fails if callers use the batch API with a scalar string."""

    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise AssertionError("predict() requires a list of SMILES, not a scalar string")
        self.calls.append(list(smiles_list))
        return [float(len(smiles)) for smiles in smiles_list]


GENERIC_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="organic compound",
    domain_context="",
    smiles_markers=[],
)


def _usage(model: str = "test/model") -> LLMUsage:
    return LLMUsage(model=model, prompt_tokens=10, completion_tokens=5, latency_s=0.1)


def test_worker_validation_uses_scalar_predict_wrapper():
    surrogate = StrictListSurrogate()
    worker = WorkerAgent(
        model="test/model",
        api_keys={},
        task_context=GENERIC_CTX,
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": "CC",
            "child_smiles": "CCC",
            "explanation": "extend alkane",
        }
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 2.0
    assert candidates[0]["child_property"] == 3.0
    assert candidates[0]["improvement_factor"] == 1.5
    assert surrogate.calls == [["CC"], ["CCC"]]


def test_predictor_tools_honor_surrogate_batch_contract():
    surrogate = StrictListSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "TestProp").execute("CC")
    assert single_obs.success is True
    assert single_obs.result["TestProp"] == 2.0

    batch_obs = BatchPropertyPredictorTool(surrogate, "TestProp").execute(["CC", "CCC"])
    assert batch_obs.success is True
    assert [r["property"] for r in batch_obs.result] == [2.0, 3.0]
    assert surrogate.calls == [["CC"], ["CC", "CCC"]]


def test_usage_summary_merge_keeps_dicts_out_of_llm_usage_aggregation():
    base = aggregate_usage([_usage("worker")])
    critic_summary = aggregate_usage([_usage("critic"), _usage("critic")])

    merged = _merge_usage_summary(base, critic_summary)

    assert merged["total_calls"] == 3
    assert merged["total_tokens"] == 45
    assert merged["by_model"]["worker"]["calls"] == 1
    assert merged["by_model"]["critic"]["calls"] == 2


def test_agentic_run_persists_computed_reward_and_finishes(tmp_path):
    surrogate = StrictListSurrogate()
    logger = RunLogger(str(tmp_path / "runs"))
    cfg = {
        "models": {"worker": "test/model", "critic": "test/model", "meta": "test/model"},
        "optimization": {
            "n_outer_epochs": 1,
            "n_per_molecule": 1,
            "batch_size": 1,
            "meta_interval": 99,
            "reward_function": "pareto_hypervolume",
        },
        "task": {"surrogate": "strict"},
        "temperatures": {},
    }

    def fake_worker_llm(*, messages, **kwargs):
        prompt = messages[-1]["content"]
        if "Think step-by-step" in prompt:
            return (
                '{"reasoning_steps": ["extend chain"], "key_modifications": ["add C"], "confidence": 0.9}',
                _usage("worker"),
            )
        return (
            '{"parent_smiles": [{"parent": "CC", "candidates": [{"smiles": "CCC", "explanation": "add C"}]}]}',
            _usage("worker"),
        )

    def fake_critic_llm(*, messages, **kwargs):
        prompt = messages[-1]["content"]
        if "Analyze the experimental results" in prompt:
            return (
                '{"pareto_insights": ["longer chain"], "failure_patterns": [], '
                '"unexplored_space": ["branching"], "tradeoffs": "none", "confidence": 0.8}',
                _usage("critic"),
            )
        if "propose 2-3 alternative strategies" in prompt:
            return (
                '{"alternative_1": {"name": "Exploit", "strategy": "Extend chains", "rationale": "worked"}, '
                '"alternative_2": {"name": "Explore", "strategy": "Try branching", "rationale": "novel"}}',
                _usage("critic"),
            )
        return (
            '{"consensus": "A", "consensus_rationale": "best evidence", "confidence": 0.9}',
            _usage("critic"),
        )

    def fake_base_llm(*, messages, **kwargs):
        return (
            '{"tool": "debate_strategies", "arguments": {"strategy_a": "A", "strategy_b": "B", '
            '"context": "ctx"}, "rationale": "compare"}',
            _usage("critic"),
        )

    with patch("apo.agentic_engine.get_surrogate", return_value=surrogate), \
         patch("apo.agents.worker.call_llm", side_effect=fake_worker_llm), \
         patch("apo.agents.critic.call_llm", side_effect=fake_critic_llm), \
         patch("apo.agents.base.call_llm", side_effect=fake_base_llm), \
         patch("apo.agents.meta.MetaAgent.get_advice", return_value=("", {})):
        run_dir = run_agentic_mode(cfg, GENERIC_CTX, ["CC"], logger, api_keys={})

    history = logger.load_existing_epochs()
    assert len(history) == 1
    assert history[0]["reward"] > 0.0
    assert history[0]["prompt_state"]["score"] == history[0]["reward"]
    assert Path(run_dir).exists()
