import json
from typing import List, Optional
from unittest.mock import patch

from apo.agentic_engine import run_agentic_mode
from apo.agents.critic import CriticAgent
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.logging.run_logger import RunLogger
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


class StrictSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise TypeError("predict() expects a list of SMILES, not a string")
        self.calls.append(list(smiles_list))
        values = {"CC": 2.0, "CCC": 6.0, "CCCC": 8.0}
        return [values.get(smiles, 1.0) for smiles in smiles_list]


GENERIC_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="organic compound",
)

MOCK_USAGE = LLMUsage("test-model", 10, 5, 0.1)


def test_worker_validation_uses_single_smiles_surrogate_api():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=GENERIC_CTX,
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {"parent_smiles": "CC", "child_smiles": "CCC", "explanation": "extend chain"}
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 2.0
    assert candidates[0]["child_property"] == 6.0
    assert candidates[0]["improvement_factor"] == 3.0
    assert surrogate.calls[:2] == [["CC"], ["CCC"]]


def test_property_tools_use_list_based_surrogate_api():
    surrogate = StrictSurrogate()

    single = PropertyPredictorTool(surrogate, "TestProp").execute("CCC")
    batch = BatchPropertyPredictorTool(surrogate, "TestProp").execute(["CC", "CCCC"])

    assert single.success is True
    assert single.result["TestProp"] == 6.0
    assert batch.success is True
    assert [item["property"] for item in batch.result] == [2.0, 8.0]
    assert surrogate.calls == [["CCC"], ["CC", "CCCC"]]


def test_critic_assigns_reward_to_current_strategy():
    candidates = [{
        "parent_smiles": "CC",
        "child_smiles": "CCC",
        "valid": True,
        "parent_property": 2.0,
        "child_property": 6.0,
        "improvement_factor": 3.0,
        "similarity": 0.5,
    }]
    current = PromptState.seed("start strategy")
    history = PromptStateHistory()
    history.add(current)
    critic = CriticAgent(
        model="test-model",
        api_keys={},
        task_context=GENERIC_CTX,
        reward_fn=ParetoHypervolume(),
    )

    analysis_response = json.dumps({
        "pareto_insights": ["larger alkyl chain improved property"],
        "failure_patterns": [],
        "unexplored_space": ["branching"],
        "tradeoffs": "similarity remains acceptable",
        "confidence": 0.9,
    })
    alternatives_response = json.dumps({
        "alternative_1": {
            "name": "Extend",
            "strategy": "Extend carbon chains while preserving parent motif.",
            "rationale": "The best candidate improved by extension.",
        },
        "alternative_2": {
            "name": "Branch",
            "strategy": "Explore modest branching near the parent motif.",
            "rationale": "Branching is unexplored.",
        },
    })
    debate_response = json.dumps({
        "consensus": "A",
        "consensus_rationale": "Use the evidenced extension pattern.",
        "confidence": 0.8,
    })

    with patch(
        "apo.agents.critic.call_llm",
        side_effect=[
            (analysis_response, MOCK_USAGE),
            (alternatives_response, MOCK_USAGE),
            (debate_response, MOCK_USAGE),
        ],
    ):
        new_state, _, _ = critic.refine(candidates, current, history)

    assert current.score == 1.5
    assert new_state.version == 1
    assert new_state.score is None


def test_meta_formats_recent_strategies_from_history():
    history = PromptStateHistory()
    for idx in range(4):
        history.add(PromptState(strategy_text=f"strategy {idx}", version=idx))

    meta = MetaAgent(model="test-model", api_keys={}, task_context=GENERIC_CTX)
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted


def test_agentic_engine_logs_scored_current_state_and_completes(tmp_path):
    candidates = [{
        "parent_smiles": "CC",
        "child_smiles": "CCC",
        "valid": True,
        "parent_property": 2.0,
        "child_property": 6.0,
        "improvement_factor": 3.0,
        "similarity": 0.5,
    }]

    class FakeWorker:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def generate(self, **kwargs):
            return candidates, [MOCK_USAGE]

    class FakeCritic:
        def __init__(self, reward_fn, **kwargs):
            self.reward_fn = reward_fn
            self._interpretability_trace = {}

        def refine(self, candidates, current_state, history, meta_advice=""):
            current_state.score = self.reward_fn.compute([c for c in candidates if c.get("valid")])
            return (
                PromptState(
                    strategy_text="next strategy",
                    version=current_state.version + 1,
                    parent_version=current_state.version,
                ),
                {"pareto_insights": ["ok"]},
                {"total_calls": 1, "total_tokens": 15},
            )

    class FakeMeta:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def get_advice(self, history, reward_history):
            return "", None

    cfg = {
        "task": {"surrogate": "strict"},
        "models": {"worker": "w", "critic": "c", "meta": "m"},
        "optimization": {
            "n_outer_epochs": 1,
            "n_per_molecule": 1,
            "batch_size": 1,
            "meta_interval": 10,
            "reward_function": "pareto_hypervolume",
        },
    }
    logger = RunLogger(str(tmp_path))

    with patch("apo.agentic_engine.get_surrogate", return_value=StrictSurrogate()), \
         patch("apo.agentic_engine.WorkerAgent", FakeWorker), \
         patch("apo.agentic_engine.CriticAgent", FakeCritic), \
         patch("apo.agentic_engine.MetaAgent", FakeMeta):
        run_agentic_mode(cfg, GENERIC_CTX, ["CC"], logger, api_keys={})

    records = logger.load_existing_epochs()
    assert records[0]["reward"] == 1.5
    assert records[0]["prompt_state"]["version"] == 0
