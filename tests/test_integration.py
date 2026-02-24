"""
Integration smoke tests — updated for TaskContext-based API.
"""
import json
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

import pytest

from apo.core.llm_client import LLMUsage
from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.logging.run_logger import RunLogger
from apo.optimizer.inner_loop import InnerLoop
from apo.optimizer.outer_loop import OuterLoop
from apo.optimizer.meta_loop import MetaLoop
from apo.surrogates.base import SurrogatePredictor
from apo.task_context import TaskContext


# ── Mock surrogate ────────────────────────────────────────────────────────────

class MockSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        return [1.0] * len(smiles_list)


# ── Shared fixtures ───────────────────────────────────────────────────────────

VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD  = "CC(CO[Cu])COCCOC(=O)[Au]"

# TaskContext with polymer markers (like conductivity task)
POLYMER_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="polymer",
    domain_context="[Cu] and [Au] are backbone markers.",
    smiles_markers=["[Cu]", "[Au]"],
    similarity_on_repeat_unit=True,
)

# TaskContext without markers (generic)
GENERIC_CTX = TaskContext(
    property_name="TestProp",
    property_units="units",
    maximize=True,
    molecule_type="organic compound",
    domain_context="",
    smiles_markers=[],
)

MOCK_INNER = json.dumps({
    "generated_molecules": {
        VALID_PARENT: {
            "smiles": [VALID_CHILD, VALID_CHILD],
            "reasoning": ["reason1", "reason2"]
        }
    }
})

MOCK_OUTER = json.dumps({
    "strategy": "Focus on ether oxygen groups.",
    "rationale": "Ether oxygen improves solvation.",
    "analysis": {
        "pareto_insights": ["Ether groups help"],
        "failure_patterns": ["Removing [Cu] breaks format"],
        "exploration_targets": ["Longer chains"],
        "chemical_hypotheses": ["PEG-like segments"],
    }
})

MOCK_META = json.dumps({
    "meta_advice": "Explore longer ether chains.",
    "strategic_assessment": {
        "progress_status": "improving",
        "key_observations": ["Ether groups help"],
        "recommended_directions": ["Sulfonate groups"],
        "warnings": [],
    }
})

MOCK_USAGE = LLMUsage("gemini/gemini-2.0-flash", 100, 50, 0.5)


@pytest.fixture()
def tmp_run_dir(tmp_path):
    return str(tmp_path / "runs")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRunLogger:
    def test_log_creates_files(self, tmp_run_dir):
        logger = RunLogger(tmp_run_dir)
        logger.save_config({"test": "config"})
        ps = PromptState.seed("test strategy")
        logger.log_epoch(
            epoch=1, prompt_state_dict=ps.to_dict(),
            candidates=[{
                "parent_smiles": VALID_PARENT, "child_smiles": VALID_CHILD,
                "improvement_factor": 1.5, "similarity": 0.8, "valid": True,
                "child_property": 1.5, "parent_property": 1.0,
                "explanation": "test", "invalid_reason": "",
            }],
            reward=0.5, pareto_data={"hypervolume": 0.5, "pareto_front": [[1.5, 0.8]], "n_evaluated": 1},
            analysis={}, meta_advice="", llm_usage={"total_calls": 1},
        )
        assert (Path(logger.run_dir) / "run_log.jsonl").exists()
        assert len(logger.load_existing_epochs()) >= 1

    def test_reward_history(self, tmp_run_dir):
        logger = RunLogger(tmp_run_dir)
        ps = PromptState.seed("strat")
        for i in range(3):
            logger.log_epoch(
                epoch=i+1, prompt_state_dict=ps.to_dict(), candidates=[],
                reward=float(i) * 0.5, pareto_data={}, analysis={},
                meta_advice="", llm_usage={},
            )
        assert logger.reward_history == [0.0, 0.5, 1.0]


class TestPromptStateHistory:
    def test_best_state(self):
        h = PromptStateHistory()
        s1 = PromptState.seed("seed"); s1.score = 0.3
        s2 = PromptState(strategy_text="v1", version=1, score=0.8, rationale="r")
        s3 = PromptState(strategy_text="v2", version=2, score=0.5, rationale="r")
        h.add(s1); h.add(s2); h.add(s3)
        assert h.best.version == 1

    def test_timeline(self):
        h = PromptStateHistory()
        for i in range(5):
            h.add(PromptState(strategy_text=f"v{i}", version=i, rationale=""))
        assert len(h.strategy_timeline()) == 5


class TestTaskContext:
    def test_polymer_ctx_markers(self):
        assert POLYMER_CTX.smiles_markers == ["[Cu]", "[Au]"]
        assert POLYMER_CTX.similarity_on_repeat_unit is True

    def test_generic_ctx_no_markers(self):
        assert GENERIC_CTX.smiles_markers == []
        assert GENERIC_CTX.similarity_on_repeat_unit is False

    def test_from_config_polymer(self):
        cfg = {
            "task": {
                "molecule_type": "polymer",
                "domain_context": "markers are [Cu] and [Au]",
                "smiles_markers": ["[Cu]", "[Au]"],
                "similarity_on_repeat_unit": True,
                "property_name": "Conductivity",
                "property_units": "mS/cm",
                "maximize": True,
            }
        }
        ctx = TaskContext.from_config(cfg)
        assert ctx.smiles_markers == ["[Cu]", "[Au]"]
        assert ctx.molecule_type == "polymer"

    def test_seed_strategy_auto(self):
        ctx = TaskContext(property_name="HOMO", property_units="eV",
                         maximize=False, molecule_type="organic compound")
        assert "minimise" in ctx.seed_strategy


class TestFullPipelineSmoke:
    """2-epoch end-to-end smoke test with polymer context (markers required)."""

    def test_two_epoch_loop_with_polymer_ctx(self, tmp_run_dir):
        surrogate = MockSurrogate()
        reward_fn = ParetoHypervolume()

        inner = InnerLoop(
            surrogate=surrogate,
            task_context=POLYMER_CTX,
            worker_model="gemini/gemini-2.0-flash",
            api_keys={},
            parent_cache={},
        )
        outer = OuterLoop(
            reward_fn=reward_fn,
            task_context=POLYMER_CTX,
            critic_model="gemini/gemini-2.0-flash",
            api_keys={},
        )
        meta = MetaLoop(
            task_context=POLYMER_CTX,
            meta_model="gemini/gemini-2.0-flash",
            api_keys={},
            meta_interval=2,
        )

        logger = RunLogger(tmp_run_dir)
        history = PromptStateHistory()
        current = PromptState.seed("Generate better polymers.")
        history.add(current)

        with patch("apo.optimizer.inner_loop.call_llm", return_value=(MOCK_INNER, MOCK_USAGE)), \
             patch("apo.optimizer.outer_loop.call_llm", return_value=(MOCK_OUTER, MOCK_USAGE)), \
             patch("apo.optimizer.meta_loop.call_llm",  return_value=(MOCK_META, MOCK_USAGE)):

            for epoch in range(1, 3):
                candidates, _ = inner.run(current.strategy_text, [VALID_PARENT], n_per_molecule=2)
                new_state, analysis, _ = outer.refine(candidates, current, history)
                pareto_data = reward_fn.pareto_data([c for c in candidates if c.get("valid")])
                logger.log_epoch(
                    epoch=epoch, prompt_state_dict=current.to_dict(),
                    candidates=candidates, reward=current.score or 0.0,
                    pareto_data=pareto_data, analysis=analysis,
                    meta_advice="", llm_usage={"total_calls": 2},
                )
                meta.maybe_get_advice(history, logger.reward_history, analysis)
                history.add(new_state)
                current = new_state

        records = logger.load_existing_epochs()
        assert len(records) == 2
        assert history.latest.version == 2
        valid = [c for rec in records for c in rec.get("candidates", []) if c.get("valid")]
        assert len(valid) > 0

    def test_generic_ctx_no_markers_allows_plain_smiles(self):
        """Generic context should NOT reject plain SMILES (no markers)."""
        surrogate = MockSurrogate()
        inner = InnerLoop(
            surrogate=surrogate,
            task_context=GENERIC_CTX,
            worker_model="gemini/gemini-2.0-flash",
            api_keys={},
            parent_cache={"CC": 1.0},
        )
        # Process a plain SMILES child (no [Cu]/[Au]) — should be valid
        candidate = inner._process_candidate("CCO", "reason", "CC", 1.0)
        assert candidate["valid"] is True, f"Expected valid: {candidate}"
