"""
Integration smoke test: 2-epoch run with a mock surrogate and mock LLM.
Validates that the full pipeline runs without error and produces correct output files.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from apo.core.prompt_state import PromptState, PromptStateHistory
from apo.core.reward import ParetoHypervolume
from apo.logging.run_logger import RunLogger
from apo.optimizer.inner_loop import InnerLoop
from apo.optimizer.outer_loop import OuterLoop
from apo.optimizer.meta_loop import MetaLoop
from apo.surrogates.base import SurrogatePredictor


# ── Mock surrogate ────────────────────────────────────────────────────────────

class MockSurrogate(SurrogatePredictor):
    property_name = "TestProp"
    property_units = "units"
    maximize = True

    def predict(self, smiles_list: List[str]) -> List[Optional[float]]:
        return [1.0] * len(smiles_list)


# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_PARENT = "CC(CO[Cu])CSCCOC(=O)[Au]"
VALID_CHILD  = "CC(CO[Cu])COCCOC(=O)[Au]"

MOCK_INNER_RESPONSE = json.dumps({
    "generated_molecules": {
        VALID_PARENT: {
            "smiles": [VALID_CHILD, VALID_CHILD],
            "reasoning": ["Added ether oxygen for better coordination",
                          "Slightly modified version"]
        }
    }
})

MOCK_OUTER_RESPONSE = json.dumps({
    "strategy": "Focus on adding ether oxygen groups for improved coordination.",
    "rationale": "Ether oxygen improves Li+ solvation.",
    "analysis": {
        "pareto_insights": ["Ether oxygen increases conductivity"],
        "failure_patterns": ["Removing [Cu] breaks polymer format"],
        "exploration_targets": ["Add more flexible linkers"],
        "chemical_hypotheses": ["PEG-like segments improve ion mobility"]
    }
})

MOCK_META_RESPONSE = json.dumps({
    "meta_advice": "Explore longer ether chains for improved flexibility and ion transport.",
    "strategic_assessment": {
        "progress_status": "improving",
        "key_observations": ["Ether groups consistently improve property"],
        "recommended_directions": ["Explore sulfonate groups"],
        "warnings": []
    }
})


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
            epoch=1,
            prompt_state_dict=ps.to_dict(),
            candidates=[{
                "parent_smiles": VALID_PARENT,
                "child_smiles": VALID_CHILD,
                "improvement_factor": 1.5,
                "similarity": 0.8,
                "valid": True,
                "child_property": 1.5,
                "parent_property": 1.0,
                "explanation": "test",
                "invalid_reason": "",
            }],
            reward=0.5,
            pareto_data={"pareto_front": [[1.5, 0.8]], "hypervolume": 0.5, "n_evaluated": 1},
            analysis={"pareto_insights": ["insight"]},
            meta_advice="",
            llm_usage={"total_calls": 1, "total_tokens": 100},
        )

        log_file = Path(logger.run_dir) / "run_log.jsonl"
        assert log_file.exists()
        records = logger.load_existing_epochs()
        assert len(records) >= 1

    def test_reward_history(self, tmp_run_dir):
        logger = RunLogger(tmp_run_dir)
        ps = PromptState.seed("strat")
        for i in range(3):
            logger.log_epoch(
                epoch=i+1,
                prompt_state_dict=ps.to_dict(),
                candidates=[],
                reward=float(i) * 0.5,
                pareto_data={},
                analysis={},
                meta_advice="",
                llm_usage={},
            )
        assert len(logger.reward_history) == 3
        assert logger.reward_history == [0.0, 0.5, 1.0]


class TestPromptStateHistory:
    def test_best_state(self):
        h = PromptStateHistory()
        s1 = PromptState.seed("seed")
        s1.score = 0.3
        s2 = PromptState(strategy_text="v1", version=1, score=0.8, rationale="r")
        s3 = PromptState(strategy_text="v2", version=2, score=0.5, rationale="r")
        h.add(s1); h.add(s2); h.add(s3)
        assert h.best.version == 1

    def test_timeline(self):
        h = PromptStateHistory()
        for i in range(5):
            h.add(PromptState(strategy_text=f"v{i}", version=i, rationale=""))
        timeline = h.strategy_timeline()
        assert len(timeline) == 5
        assert timeline[0]["version"] == 0


class TestFullPipelineSmoke:
    """End-to-end smoke test with mocked LLM calls."""

    def test_two_epoch_run(self, tmp_run_dir):
        surrogate = MockSurrogate()
        reward_fn = ParetoHypervolume()

        # Inner loop with mocked LLM
        inner = InnerLoop(
            surrogate=surrogate,
            worker_model="openai/gpt-4o",
            api_keys={},
            parent_cache={},
        )

        outer = OuterLoop(
            reward_fn=reward_fn,
            critic_model="openai/gpt-4o",
            api_keys={},
            property_name="TestProp",
            property_units="units",
        )

        meta = MetaLoop(
            meta_model="openai/gpt-4o",
            api_keys={},
            meta_interval=2,  # triggers on epoch 2
            property_name="TestProp",
        )

        logger = RunLogger(tmp_run_dir)
        history = PromptStateHistory()
        current = PromptState.seed("Generate better polymers.")
        history.add(current)

        from apo.core.llm_client import LLMUsage
        mock_usage = LLMUsage("openai/gpt-4o", 100, 50, 0.5)

        with patch("apo.optimizer.inner_loop.call_llm",
                   return_value=(MOCK_INNER_RESPONSE, mock_usage)), \
             patch("apo.optimizer.outer_loop.call_llm",
                   return_value=(MOCK_OUTER_RESPONSE, mock_usage)), \
             patch("apo.optimizer.meta_loop.call_llm",
                   return_value=(MOCK_META_RESPONSE, mock_usage)):

            for epoch in range(1, 3):
                candidates, _ = inner.run(
                    strategy=current.strategy_text,
                    parent_smiles=[VALID_PARENT],
                    n_per_molecule=2,
                )

                new_state, analysis, _ = outer.refine(
                    candidates=candidates,
                    current_state=current,
                    history=history,
                )

                pareto_data = reward_fn.pareto_data([c for c in candidates if c.get("valid")])

                logger.log_epoch(
                    epoch=epoch,
                    prompt_state_dict=current.to_dict(),
                    candidates=candidates,
                    reward=current.score or 0.0,
                    pareto_data=pareto_data,
                    analysis=analysis,
                    meta_advice="",
                    llm_usage={"total_calls": 2},
                )

                meta.maybe_get_advice(history, logger.reward_history, analysis)
                history.add(new_state)
                current = new_state

        # Assertions
        log_file = Path(logger.run_dir) / "run_log.jsonl"
        assert log_file.exists()
        records = []
        with open(log_file) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        assert len(records) == 2

        # PromptState version should have incremented
        assert history.latest.version == 2

        # Valid candidates produced
        all_candidates = []
        for rec in records:
            all_candidates.extend(rec.get("candidates", []))
        valid = [c for c in all_candidates if c.get("valid")]
        assert len(valid) > 0
