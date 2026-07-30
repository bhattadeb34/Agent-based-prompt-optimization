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
from apo.utils.smiles_utils import canonicalize


VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD = "CC(CO[Cu])COCCOC(=O)[Au]"
MISSING_MARKER_CHILD = "CCO"


class StrictListSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def __init__(self):
        self.calls = []

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        if not isinstance(smiles_list, list):
            raise TypeError("predict expects a list of SMILES")
        self.calls.append(list(smiles_list))
        canonical_child = canonicalize(VALID_CHILD)
        return [2.0 if smi in {VALID_CHILD, canonical_child} else 1.0 for smi in smiles_list]


def polymer_context(maximize: bool = True) -> TaskContext:
    return TaskContext(
        property_name="TestProp",
        property_units="units",
        maximize=maximize,
        molecule_type="polymer",
        smiles_markers=["[Cu]", "[Au]"],
        similarity_on_repeat_unit=True,
    )


def test_worker_validation_uses_scalar_safe_surrogate_api_and_task_markers():
    surrogate = StrictListSurrogate()
    worker = WorkerAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
        surrogate=surrogate,
        parent_cache={},
    )

    candidates = worker._validate_candidates([
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": VALID_CHILD,
            "explanation": "valid marker-preserving change",
        },
        {
            "parent_smiles": VALID_PARENT,
            "child_smiles": MISSING_MARKER_CHILD,
            "explanation": "missing polymer markers",
        },
    ])

    assert candidates[0]["valid"] is True
    assert candidates[0]["parent_property"] == 1.0
    assert candidates[0]["child_property"] == 2.0
    assert candidates[0]["improvement_factor"] == 2.0
    assert candidates[1]["valid"] is False
    assert "Missing required marker" in candidates[1]["invalid_reason"]
    assert all(isinstance(call, list) for call in surrogate.calls)


def test_agentic_property_tools_use_surrogate_list_contract():
    surrogate = StrictListSurrogate()

    single_obs = PropertyPredictorTool(surrogate, "TestProp").execute(VALID_CHILD)
    batch_obs = BatchPropertyPredictorTool(surrogate, "TestProp").execute([
        VALID_PARENT,
        VALID_CHILD,
    ])

    assert single_obs.success is True
    assert single_obs.result["TestProp"] == 2.0
    assert batch_obs.success is True
    assert [r["property"] for r in batch_obs.result] == [1.0, 2.0]
    assert surrogate.calls == [[VALID_CHILD], [VALID_PARENT, VALID_CHILD]]


def test_worker_parses_legacy_generated_molecules_schema():
    payload = {
        "generated_molecules": {
            VALID_PARENT: {
                "smiles": [VALID_CHILD],
                "reasoning": ["added ether oxygen"],
            }
        }
    }

    entries = WorkerAgent._iter_parent_entries(payload)

    assert entries == [{
        "parent": VALID_PARENT,
        "candidates": [{"smiles": VALID_CHILD, "explanation": "added ether oxygen"}],
    }]


def test_meta_agent_formats_recent_strategies_from_history():
    history = PromptStateHistory()
    for version in range(4):
        history.add(PromptState(strategy_text=f"strategy {version}", version=version))

    meta = MetaAgent(
        model="test-model",
        api_keys={},
        task_context=polymer_context(),
    )
    meta.history = history

    formatted = meta._format_recent_strategies()

    assert "v1: strategy 1" in formatted
    assert "v3: strategy 3" in formatted


def test_run_agentic_mode_keeps_usage_aggregation_type_safe(monkeypatch, tmp_path):
    class FakeWorker:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def generate(self, strategy, parent_smiles, n_per_molecule):
            return ([{
                "parent_smiles": VALID_PARENT,
                "child_smiles": VALID_CHILD,
                "valid": True,
                "parent_property": 1.0,
                "child_property": 2.0,
                "improvement_factor": 2.0,
                "similarity": 0.5,
            }], [LLMUsage("worker-model", 10, 5, 0.1)])

    class FakeCritic:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def refine(self, candidates, current_state, history, meta_advice=""):
            usage = aggregate_usage([LLMUsage("critic-model", 7, 3, 0.2)])
            return (
                PromptState(
                    strategy_text="next strategy",
                    version=current_state.version + 1,
                    parent_version=current_state.version,
                ),
                {"ok": True},
                usage,
            )

    class FakeMeta:
        def __init__(self, **kwargs):
            self._interpretability_trace = {}

        def get_advice(self, history, reward_history):
            return "", aggregate_usage([LLMUsage("meta-model", 4, 2, 0.3)])

    monkeypatch.setattr("apo.agentic_engine.WorkerAgent", FakeWorker)
    monkeypatch.setattr("apo.agentic_engine.CriticAgent", FakeCritic)
    monkeypatch.setattr("apo.agentic_engine.MetaAgent", FakeMeta)
    monkeypatch.setattr(
        "apo.agentic_engine.get_surrogate",
        lambda *args, **kwargs: StrictListSurrogate(),
    )

    cfg = {
        "task": {"surrogate": "mock"},
        "models": {
            "worker": "worker-model",
            "critic": "critic-model",
            "meta": "meta-model",
        },
        "optimization": {
            "n_outer_epochs": 1,
            "n_per_molecule": 1,
            "batch_size": 1,
            "meta_interval": 1,
            "reward_function": "pareto_hypervolume",
        },
    }
    logger = RunLogger(str(tmp_path))

    run_agentic_mode(
        cfg=cfg,
        ctx=polymer_context(),
        all_smiles=[VALID_PARENT],
        logger=logger,
        api_keys={},
    )

    records = logger.load_existing_epochs()
    assert records[0]["prompt_state"]["version"] == 0
    assert records[0]["prompt_state"]["score"] == 1.0
    assert records[0]["reward"] == 1.0
    assert records[0]["llm_usage"]["total_calls"] == 2
