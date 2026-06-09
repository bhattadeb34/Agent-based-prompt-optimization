from typing import List, Optional

from apo.agentic_engine import _merge_usage_summary
from apo.agents.critic import CriticAgent
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage, aggregate_usage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


class BatchOnlySurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise TypeError("predict expects a list")
        return [float(len(smiles)) for smiles in smiles_list]


GENERIC_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="organic compound",
)

POLYMER_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="polymer",
    smiles_markers=["[Cu]", "[Au]"],
    similarity_on_repeat_unit=True,
)


def test_agentic_worker_uses_single_prediction_wrapper_for_valid_candidates():
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=GENERIC_CTX,
        surrogate=BatchOnlySurrogate(),
        parent_cache={"CC": 2.0},
    )

    candidates = worker._validate_candidates([
        {"parent_smiles": "CC", "child_smiles": "CCO", "explanation": "add oxygen"}
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["child_property"] == 3.0
    assert candidates[0]["improvement_factor"] == 1.5


def test_agentic_worker_enforces_task_smiles_markers():
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        surrogate=BatchOnlySurrogate(),
        parent_cache={"CC(CO[Cu])CSCCOC(=O)[Au]": 10.0},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": "CC(CO[Cu])CSCCOC(=O)[Au]",
            "child_smiles": "CCO",
            "explanation": "missing polymer markers",
        }
    ])

    assert candidates[0]["valid"] is False
    assert "Missing required marker" in candidates[0]["invalid_reason"]


def test_agentic_property_tools_use_batch_predictor_contract():
    surrogate = BatchOnlySurrogate()

    single = PropertyPredictorTool(surrogate, "TestProp").execute("CCO")
    batch = BatchPropertyPredictorTool(surrogate, "TestProp").execute(["CC", "CCO"])

    assert single.success is True
    assert single.result["TestProp"] == 3.0
    assert batch.success is True
    assert [item["property"] for item in batch.result] == [2.0, 3.0]


def test_meta_agent_formats_recent_strategies_without_missing_history_method():
    history = PromptStateHistory()
    for version in range(4):
        history.add(PromptState(strategy_text=f"strategy {version}", version=version))

    meta = MetaAgent("test-model", {}, GENERIC_CTX)
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted
    assert "v0: strategy 0" not in formatted


class NoLLMCritic(CriticAgent):
    def run(self, initial_state):
        self.new_state = PromptState(
            strategy_text="next strategy",
            version=self.current_state.version + 1,
            parent_version=self.current_state.version,
        )
        return (self.new_state, self.analysis), []


def test_critic_persists_reward_on_evaluated_state_before_refining():
    current = PromptState.seed("current strategy")
    history = PromptStateHistory()
    history.add(current)
    critic = NoLLMCritic("test-model", {}, GENERIC_CTX, ParetoHypervolume())

    new_state, _, usage = critic.refine(
        candidates=[
            {
                "valid": True,
                "improvement_factor": 2.0,
                "similarity": 0.5,
                "child_property": 6.0,
                "parent_property": 3.0,
            }
        ],
        current_state=current,
        history=history,
    )

    assert current.score == 1.0
    assert new_state.score is None
    assert new_state.metadata["previous_reward"] == 1.0
    assert usage["total_calls"] == 0


def test_agentic_usage_summaries_merge_without_mixing_dicts_into_llm_usage():
    worker_usage = aggregate_usage([LLMUsage("worker", 10, 5, 0.5)])
    critic_usage = aggregate_usage([LLMUsage("critic", 20, 10, 1.0)])

    merged = _merge_usage_summary(worker_usage, critic_usage)

    assert merged["total_calls"] == 2
    assert merged["total_tokens"] == 45
    assert merged["by_model"]["worker"]["calls"] == 1
    assert merged["by_model"]["critic"]["tokens"] == 30
