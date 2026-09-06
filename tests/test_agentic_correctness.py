from typing import List, Optional

from apo.agentic_engine import run_agentic_mode
from apo.agents.meta import MetaAgent
from apo.agents.tools import BatchPropertyPredictorTool, PropertyPredictorTool
from apo.agents.worker import WorkerAgent
from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState, PromptStateHistory
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
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise TypeError("predict expects a list of SMILES")
        self.calls.append(list(smiles_list))
        return [float(len(s)) for s in smiles_list]


POLYMER_CTX = TaskContext(
    property_name="StrictProp",
    property_units="units",
    maximize=True,
    molecule_type="polymer",
    smiles_markers=["[Cu]", "[Au]"],
    similarity_on_repeat_unit=True,
)


def test_worker_parses_generated_molecules_mapping_and_uses_predict_single():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        surrogate=surrogate,
        parent_cache={},
    )
    raw = f"""```json
    {{
      "generated_molecules": {{
        "{VALID_PARENT}": {{
          "smiles": ["{VALID_CHILD}"],
          "reasoning": ["adds ether oxygen"]
        }}
      }}
    }}
    ```"""

    parsed = worker._parse_generation_output(raw)
    candidates = worker._validate_candidates(parsed)

    assert len(candidates) == 1
    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] is not None
    assert candidates[0]["child_property"] is not None
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_worker_rejects_missing_required_markers():
    surrogate = StrictSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=POLYMER_CTX,
        surrogate=surrogate,
        parent_cache={VALID_PARENT: 1.0},
    )

    candidates = worker._validate_candidates([{
        "parent_smiles": VALID_PARENT,
        "child_smiles": "CCO",
        "explanation": "missing polymer markers",
    }])

    assert candidates[0]["valid"] is False
    assert "Missing required marker" in candidates[0]["invalid_reason"]


def test_property_tools_honor_surrogate_batch_contract():
    surrogate = StrictSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "StrictProp").execute(VALID_PARENT)
    batch_obs = BatchPropertyPredictorTool(surrogate, "StrictProp").execute([VALID_PARENT, VALID_CHILD])

    assert single_obs.success is True
    assert batch_obs.success is True
    assert [len(call) for call in surrogate.calls] == [1, 2]


def test_meta_agent_formats_recent_history_with_existing_api():
    history = PromptStateHistory()
    for i in range(4):
        history.add(PromptState(strategy_text=f"strategy {i}", version=i))

    meta = MetaAgent(model="test-model", api_keys={}, task_context=POLYMER_CTX)
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted


def test_agentic_engine_accepts_aggregate_usage_dicts(monkeypatch, tmp_path):
    class FakeWorker:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def generate(self, strategy, parent_smiles, n_per_molecule):
            return ([{
                "parent_smiles": VALID_PARENT,
                "child_smiles": VALID_CHILD,
                "parent_property": 1.0,
                "child_property": 2.0,
                "improvement_factor": 2.0,
                "similarity": 0.5,
                "valid": True,
            }], [LLMUsage("worker-model", 10, 5, 0.1)])

    class FakeCritic:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def refine(self, candidates, current_state, history, meta_advice=""):
            return (
                PromptState(
                    strategy_text="next",
                    version=current_state.version + 1,
                    parent_version=current_state.version,
                ),
                {},
                {"total_calls": 1, "total_tokens": 7, "by_model": {"critic-model": {"calls": 1, "tokens": 7}}},
            )

    class FakeMeta:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def get_advice(self, history, reward_history):
            return "", {"total_calls": 1, "total_tokens": 3, "by_model": {"meta-model": {"calls": 1, "tokens": 3}}}

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
            "reward_function": "pareto_hypervolume",
        },
    }

    run_dir = run_agentic_mode(cfg, POLYMER_CTX, [VALID_PARENT], logger, {})

    assert run_dir == str(logger.run_dir)
    assert len(logger.load_existing_epochs()) == 1
