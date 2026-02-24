"""
Structured JSONL run logger.

Writes one JSON line per epoch to runs/<run_id>/run_log.jsonl.
Also saves a copy of the config and supports resuming from last epoch.

GEPA-inspired enhancement: stores full "Actionable Side Information" (failure traces,
analysis summaries) alongside reward to give the critic a complete picture.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class RunLogger:
    """
    Logs optimization progress to structured JSONL for replay and analysis.

    Each line = one epoch record:
    {
      "run_id": str,
      "epoch": int,
      "timestamp": float,
      "prompt_state": {...},        # current PromptState
      "candidates": [...],          # all candidates (valid + invalid)
      "reward": float,
      "pareto_data": {...},
      "analysis": {...},            # critic's analysis dict (ASI)
      "meta_advice": str,
      "llm_usage": {...},
      "stats": {...}                # summary stats
    }
    """

    def __init__(self, run_dir: str, run_id: Optional[str] = None):
        self.run_id = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = Path(run_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "run_log.jsonl"
        self._epoch_records: List[Dict] = []

        # Symlink "latest" → this run
        latest_link = Path(run_dir) / "latest"
        if latest_link.is_symlink():
            latest_link.unlink()
        try:
            latest_link.symlink_to(self.run_dir.resolve())
        except Exception:
            pass

        print(f"[RunLogger] Logging to: {self.run_dir}")

    def save_config(self, config: Dict) -> None:
        """Save a copy of the run config at start."""
        config_path = self.run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        print(f"[RunLogger] Config saved to {config_path}")

    def log_epoch(
        self,
        epoch: int,
        prompt_state_dict: Dict,
        candidates: List[Dict],
        reward: float,
        pareto_data: Dict,
        analysis: Dict,
        meta_advice: str,
        llm_usage: Dict,
    ) -> None:
        """Append one epoch record to the JSONL log."""
        valid = [c for c in candidates if c.get("valid")]
        invalid = [c for c in candidates if not c.get("valid")]

        # Compute summary stats
        improvements = [c.get("improvement_factor", 0.0) for c in valid]
        similarities = [c.get("similarity", 0.0) for c in valid]
        child_props = [c.get("child_property", 0.0) for c in valid if c.get("child_property")]

        stats = {
            "n_total": len(candidates),
            "n_valid": len(valid),
            "n_invalid": len(invalid),
            "validity_rate": round(len(valid) / max(len(candidates), 1), 3),
            "avg_improvement": round(_nanmean(improvements), 4),
            "max_improvement": round(max(improvements) if improvements else 0.0, 4),
            "avg_similarity": round(_nanmean(similarities), 4),
            "min_similarity": round(min(similarities) if similarities else 0.0, 4),
            "avg_child_property": round(_nanmean(child_props), 6),
            "max_child_property": round(max(child_props) if child_props else 0.0, 6),
            # Failure breakdown (GEPA-style ASI)
            "failure_reasons": _count_failure_reasons(invalid),
        }

        record = {
            "run_id": self.run_id,
            "epoch": epoch,
            "timestamp": time.time(),
            "prompt_state": prompt_state_dict,
            "candidates": candidates,  # full trace for replay
            "reward": round(reward, 6),
            "pareto_data": pareto_data,
            "analysis": analysis,      # critic's structured analysis (ASI)
            "meta_advice": meta_advice,
            "llm_usage": llm_usage,
            "stats": stats,
        }

        # Append to JSONL
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

        self._epoch_records.append(record)

        # Print summary
        print(f"\n[RunLogger] Epoch {epoch} logged:")
        print(f"  Reward: {reward:.4f} | Valid: {stats['n_valid']}/{stats['n_total']} "
              f"| AvgImprovement: {stats['avg_improvement']:.3f}x | "
              f"AvgSim: {stats['avg_similarity']:.3f}")
        if pareto_data.get("hypervolume"):
            print(f"  Pareto HV: {pareto_data['hypervolume']:.4f} | "
                  f"Pareto front size: {len(pareto_data.get('pareto_front', []))}")

    def save_prompt_history(self, history_list: List[Dict]) -> None:
        """Persist full prompt state history as JSON."""
        path = self.run_dir / "prompt_history.json"
        with open(path, "w") as f:
            json.dump(history_list, f, indent=2, default=str)

    def save_agent_trace(self, trace_name: str, trace_data: Dict) -> None:
        """Save agent interpretability trace (thought process, self-corrections, debate)."""
        traces_dir = self.run_dir / "agent_traces"
        traces_dir.mkdir(exist_ok=True)
        path = traces_dir / f"{trace_name}.json"
        with open(path, "w") as f:
            json.dump(trace_data, f, indent=2, default=str)

    def load_existing_epochs(self) -> List[Dict]:
        """Load existing epoch records from JSONL (for resume support)."""
        records = []
        if self.log_path.exists():
            with open(self.log_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        return records

    @property
    def all_records(self) -> List[Dict]:
        return list(self._epoch_records)

    @property
    def reward_history(self) -> List[float]:
        return [r["reward"] for r in self._epoch_records]


def _nanmean(vals: List[float]) -> float:
    import math
    clean = [v for v in vals if v is not None and not math.isnan(float(v))]
    return sum(clean) / len(clean) if clean else 0.0


def _count_failure_reasons(invalid: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for c in invalid:
        reason = c.get("invalid_reason", "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts
