"""PromptState dataclass and history tracking."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class PromptState:
    """
    Represents a single version of the optimization strategy (prompt).

    Every time the critic proposes a new strategy, a new PromptState is created
    with version incremented by 1. The lineage is tracked via parent_version.
    """
    strategy_text: str
    version: int = 0
    score: Optional[float] = None          # best reward seen under this strategy
    rationale: str = ""                    # why the critic proposed this
    parent_version: int = -1               # -1 = seed (no parent)
    model_used: str = ""                   # which critic model proposed this
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d

    @classmethod
    def seed(cls, strategy_text: str) -> "PromptState":
        """Create the initial seed strategy (version 0)."""
        return cls(
            strategy_text=strategy_text,
            version=0,
            rationale="Initial seed strategy",
            parent_version=-1,
        )


class PromptStateHistory:
    """
    Maintains ordered history of all PromptState versions produced during a run.
    Supports querying best, worst, timeline, and lineage.
    """

    def __init__(self):
        self._states: List[PromptState] = []

    def add(self, state: PromptState) -> None:
        self._states.append(state)

    @property
    def all_states(self) -> List[PromptState]:
        return list(self._states)

    @property
    def latest(self) -> Optional[PromptState]:
        return self._states[-1] if self._states else None

    @property
    def best(self) -> Optional[PromptState]:
        """Return the state with the highest score."""
        scored = [s for s in self._states if s.score is not None]
        if not scored:
            return self.latest
        return max(scored, key=lambda s: s.score)

    def get_recent(self, n: int) -> List[PromptState]:
        """Return the last n states."""
        return self._states[-n:]

    def reward_curve(self) -> List[Dict]:
        """Return list of {version, score} for plotting."""
        return [
            {"version": s.version, "score": s.score, "timestamp": s.timestamp}
            for s in self._states
        ]

    def strategy_timeline(self) -> List[Dict]:
        """Return concise strategy timeline for LLM context."""
        return [
            {
                "version": s.version,
                "score": s.score,
                "rationale": s.rationale[:200],
                "strategy_preview": s.strategy_text[:300],
            }
            for s in self._states
        ]

    def to_list(self) -> List[Dict]:
        return [s.to_dict() for s in self._states]

    def __len__(self) -> int:
        return len(self._states)
