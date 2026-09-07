import json
from typing import List, Optional

from apo.agentic_engine import run_agentic_mode
from apo.agents.base import Action, Observation
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


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"


class StrictSurrogate(SurrogatePredictor):
    property_name = "StrictProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.predict_calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise TypeError("predict expects a list of SMILES")
        self.predict_calls.append(list(smiles_list))
        return [2.0 if smi == VALID_PARENT else 4.0 for smi in smiles_list]


def _ctx(maximize=True):
    return TaskContext(
        property_name="StrictProp",
        property_units="units",
        maximize=maximize,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_agentic_worker_scores_with_list_safe_surrogate_calls():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=_ctx(),
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "explanation": "add ether oxygen",
        }
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 2.0
    assert candidates[0]["child_property"] == 4.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert all(isinstance(call, list) for call in surrogate.predict_calls)


def test_agentic_worker_rejects_missing_required_markers():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=_ctx(),
        surrogate=surrogate,
        parent_cache={VALID_PARENT: 2.0},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": "CCO",
            "explanation": "missing polymer markers",
        }
    ])

    assert candidates[0]["valid"] is False
    assert "Missing required marker" in candidates[0]["invalid_reason"]


def test_agentic_worker_parses_generated_molecules_mapping(monkeypatch):
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=_ctx(),
        surrogate=StrictSurrogate(),
        parent_cache={},
    )
    text = json.dumps({
        "generated_molecules": {
            VALID_PARENT: {
                "smiles": [VALID_CHILD],
                "reasoning": ["added ether oxygen"],
            }
        }
    })
    monkeypatch.setattr(
        "apo.agents.worker.call_llm",
        lambda **kwargs: (f"```json\n{text}\n```", LLMUsage("test-model", 1, 1, 0.1)),
    )

    candidates = worker._call_llm_for_generation()
    assert candidates == [{
        "parent_smiles": VALID_PARENT,
        "child_smiles": VALID_CHILD,
        "explanation": "added ether oxygen",
    }]


def test_agentic_property_tools_do_not_pass_scalar_strings_to_predict():
    surrogate = StrictSurrogate()

    single = PropertyPredictorTool(surrogate, "StrictProp").execute(VALID_PARENT)
    batch = BatchPropertyPredictorTool(surrogate, "StrictProp").execute([VALID_PARENT, VALID_CHILD])

    assert single.success is True
    assert batch.success is True
    assert batch.result[1]["property"] == 4.0
    assert all(isinstance(call, list) for call in surrogate.predict_calls)


def test_critic_scores_current_strategy_before_refinement(monkeypatch):
    responses = iter([
        json.dumps({"pareto_insights": ["good"], "failure_patterns": [], "unexplored_space": [], "tradeoffs": "ok", "confidence": 0.9}),
        json.dumps({
            "alternative_1": {"name": "A", "strategy": "strategy A", "rationale": "rA"},
            "alternative_2": {"name": "B", "strategy": "strategy B", "rationale": "rB"},
        }),
        json.dumps({"consensus": "A", "consensus_rationale": "best", "confidence": 0.8}),
    ])

    def fake_call_llm(**kwargs):
        return next(responses), LLMUsage("test-model", 1, 1, 0.1)

    monkeypatch.setattr("apo.agents.critic.call_llm", fake_call_llm)

    current = PromptState.seed("seed strategy")
    history = PromptStateHistory()
    history.add(current)
    critic = CriticAgent("test-model", {}, _ctx(), ParetoHypervolume())
    monkeypatch.setattr(critic, "_select_action", lambda thought: Action("noop", {}, ""))
    monkeypatch.setattr(critic, "_execute_action", lambda action: Observation(True, None))

    new_state, _, usage = critic.refine(
        candidates=[{
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "valid": True,
            "improvement_factor": 2.0,
            "similarity": 0.5,
        }],
        current_state=current,
        history=history,
    )

    assert current.score == 1.0
    assert new_state.metadata["reward"] == 1.0
    assert usage["total_calls"] == 3


def test_meta_formats_recent_history_without_missing_all_method():
    history = PromptStateHistory()
    history.add(PromptState.seed("first"))
    history.add(PromptState(strategy_text="second", version=1))

    meta = MetaAgent("test-model", {}, _ctx())
    meta.history = history

    formatted = meta._format_recent_strategies()
    assert "v0: first" in formatted
    assert "v1: second" in formatted


def test_agentic_engine_merges_dict_usages_without_final_crash(monkeypatch, tmp_path):
    class FakeWorker:
        def __init__(self, *args, **kwargs):
            self._interpretability_trace = {}

        def generate(self, strategy, parent_smiles, n_per_molecule):
            return ([{
                "parent_smiles": VALID_PARENT,
                "child_smiles": VALID_CHILD,
                "valid": True,
                "improvement_factor": 2.0,
                "similarity": 0.5,
            }], [LLMUsage("worker-model", 2, 3, 0.1)])

    class FakeCritic:
        def __init__(self, *args, **kwargs):
            self._interpretability_trace = {}

        def refine(self, candidates, current_state, history, meta_advice=""):
            current_state.score = ParetoHypervolume().compute([c for c in candidates if c.get("valid")])
            return (
                PromptState(strategy_text="next", version=current_state.version + 1),
                {},
                {
                    "total_calls": 1,
                    "total_prompt_tokens": 4,
                    "total_completion_tokens": 5,
                    "total_tokens": 9,
                    "total_latency_s": 0.2,
                    "by_model": {"critic-model": {"calls": 1, "tokens": 9}},
                },
            )

    class FakeMeta:
        def __init__(self, *args, **kwargs):
            self._interpretability_trace = {}

        def get_advice(self, history, reward_history):
            return "", {
                "total_calls": 1,
                "total_prompt_tokens": 1,
                "total_completion_tokens": 1,
                "total_tokens": 2,
                "total_latency_s": 0.1,
                "by_model": {"meta-model": {"calls": 1, "tokens": 2}},
            }

    monkeypatch.setattr("apo.agentic_engine.get_surrogate", lambda *args, **kwargs: StrictSurrogate())
    monkeypatch.setattr("apo.agentic_engine.WorkerAgent", FakeWorker)
    monkeypatch.setattr("apo.agentic_engine.CriticAgent", FakeCritic)
    monkeypatch.setattr("apo.agentic_engine.MetaAgent", FakeMeta)

    logger = RunLogger(str(tmp_path / "runs"))
    cfg = {
        "task": {"surrogate": "strict"},
        "models": {"worker": "worker-model", "critic": "critic-model", "meta": "meta-model"},
        "optimization": {
            "n_outer_epochs": 1,
            "n_per_molecule": 1,
            "batch_size": 1,
            "meta_interval": 1,
        },
    }

    run_dir = run_agentic_mode(cfg, _ctx(), [VALID_PARENT], logger, {})

    assert run_dir == str(logger.run_dir)
    assert logger.reward_history == [1.0]
