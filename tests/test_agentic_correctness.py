from typing import List, Optional

from apo.agentic_engine import _merge_usage_summary
from apo.agents.critic import CriticAgent
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.task_context import TaskContext


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"

POLYMER_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="polymer",
    smiles_markers=["[Cu]", "[Au]"],
    similarity_on_repeat_unit=True,
)


class StrictListSurrogate:
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if isinstance(smiles_list, str):
            raise AssertionError("predict() requires a list, not a scalar string")
        self.calls.append(list(smiles_list))
        return [2.0 if smi == VALID_CHILD else 1.0 for smi in smiles_list]

    def predict_single(self, smiles: str) -> Optional[float]:
        values = self.predict([smiles])
        return values[0] if values else None


def test_agentic_property_tools_use_list_safe_predictor_api():
    surrogate = StrictListSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "TestProp").execute(VALID_CHILD)
    assert single_obs.success
    assert single_obs.result["TestProp"] == 2.0

    batch_obs = BatchPropertyPredictorTool(surrogate, "TestProp").execute([VALID_PARENT, VALID_CHILD])
    assert batch_obs.success
    assert [row["property"] for row in batch_obs.result] == [1.0, 2.0]
    assert surrogate.calls == [[VALID_CHILD], [VALID_PARENT, VALID_CHILD]]


def test_worker_validation_enforces_task_markers_and_predicts_scalars_safely():
    surrogate = StrictListSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        surrogate=surrogate,
        parent_cache={VALID_PARENT: 1.0},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "explanation": "keeps markers",
        },
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": "CCO",
            "explanation": "drops polymer markers",
        },
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["child_property"] == 2.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert candidates[1]["valid"] is False
    assert "Missing required marker" in candidates[1]["invalid_reason"]
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_worker_parses_fenced_generated_molecules_mapping():
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

    parsed = WorkerAgent._parse_llm_generation(text)
    assert parsed["generated_molecules"][VALID_PARENT]["smiles"] == [VALID_CHILD]


def test_critic_scores_evaluated_current_state_before_returning_new_strategy(monkeypatch):
    def fake_run(self, initial_state):
        self.new_state = PromptState(
            strategy_text="next strategy",
            version=self.current_state.version + 1,
            parent_version=self.current_state.version,
        )
        return (self.new_state, self.analysis), []

    monkeypatch.setattr(CriticAgent, "run", fake_run)
    critic = CriticAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        reward_fn=ParetoHypervolume(),
    )
    current = PromptState.seed("seed strategy")
    history = PromptStateHistory()
    history.add(current)

    new_state, _, usage = critic.refine(
        candidates=[{"valid": True, "improvement_factor": 2.0, "similarity": 0.5}],
        current_state=current,
        history=history,
    )

    assert current.score == 1.0
    assert new_state.score is None
    assert new_state.metadata["parent_reward"] == 1.0
    assert usage["total_calls"] == 0


def test_meta_recent_strategy_format_uses_prompt_history_api():
    history = PromptStateHistory()
    for version in range(4):
        history.add(PromptState(strategy_text=f"strategy {version}", version=version))

    meta = MetaAgent(model="test-model", api_keys={}, task_context=POLYMER_CTX)
    meta.history = history

    formatted = meta._format_recent_strategies()
    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted


def test_agentic_usage_summary_merges_dicts_without_llmusage_crash():
    worker_usage = {
        "total_calls": 1,
        "total_prompt_tokens": 10,
        "total_completion_tokens": 5,
        "total_tokens": 15,
        "total_latency_s": 0.5,
        "by_model": {"worker": {"calls": 1, "tokens": 15}},
    }
    critic_usage = {
        "total_calls": 2,
        "total_prompt_tokens": 20,
        "total_completion_tokens": 10,
        "total_tokens": 30,
        "total_latency_s": 1.0,
        "by_model": {"critic": {"calls": 2, "tokens": 30}},
    }

    merged = _merge_usage_summary(worker_usage, critic_usage)

    assert merged["total_calls"] == 3
    assert merged["total_tokens"] == 45
    assert merged["by_model"]["worker"]["calls"] == 1
    assert merged["by_model"]["critic"]["calls"] == 2
