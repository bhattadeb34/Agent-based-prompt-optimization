"""
Meta loop (Strategist): periodic high-level guidance. Fully TaskContext-driven.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from ..core.llm_client import LLMUsage, call_llm
from ..core.prompt_state import PromptStateHistory
from ..task_context import TaskContext


def _build_meta_system(ctx: TaskContext) -> str:
    lines = [
        f"You are a senior scientist and ML research director specialising in "
        f"{ctx.molecule_type} design.",
        "",
        f"You oversee an automated {ctx.molecule_type} optimisation pipeline. "
        f"A 'critic' LLM refines the generation strategy epoch by epoch.",
        "Your job is to provide HIGH-LEVEL STRATEGIC GUIDANCE based on the overall trajectory.",
        "",
        "You should:",
        "- Identify patterns across multiple epochs (not just the latest)",
        "- Suggest exploring fundamentally different chemical spaces if the optimisation is stuck",
        "- Warn about degenerate strategies or mode collapse",
        "- Give advice grounded in domain knowledge",
    ]
    if ctx.domain_context:
        lines += ["", "DOMAIN NOTES:", ctx.domain_context]
    return "\n".join(lines)


_META_USER_TEMPLATE = """\
Reviewing progress of a {molecule_type} optimisation targeting:
Property: {property_name}{units_str} — goal: {direction}

REWARD TRAJECTORY ({n_epochs} epochs):
{reward_trajectory}

STRATEGY EVOLUTION:
{strategy_evolution}

RECENT CRITIC ANALYSES:
{analysis_summaries}

QUESTIONS:
1. Is progress consistent, plateauing, or regressing?
2. What structural families haven't been explored?
3. Are strategies becoming degenerate (same edits repeated)?
4. What single piece of advice would have the highest impact?

Return JSON (ONLY JSON):
{{
  "meta_advice": "Concise high-level guidance in 2-5 sentences for the critic...",
  "strategic_assessment": {{
    "progress_status": "improving | plateauing | regressing",
    "key_observations": ["obs 1", "obs 2"],
    "recommended_directions": ["direction 1", "direction 2"],
    "warnings": []
  }}
}}
"""


class MetaLoop:
    def __init__(
        self,
        task_context: TaskContext,
        meta_model: str,
        api_keys: Optional[Dict[str, str]] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        meta_interval: int = 3,
    ):
        self.ctx = task_context
        self.meta_model = meta_model
        self.api_keys = api_keys or {}
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.meta_interval = meta_interval

        self._last_advice: str = ""
        self._analysis_summaries: List[str] = []
        self._epoch_counter: int = 0

    def maybe_get_advice(
        self,
        history: PromptStateHistory,
        reward_history: List[float],
        analysis: Optional[Dict] = None,
    ) -> Tuple[str, Optional[LLMUsage]]:
        self._epoch_counter += 1

        if analysis:
            summary = " | ".join(
                f"{k}: {', '.join(v[:2]) if isinstance(v, list) else str(v)}"
                for k, v in analysis.items() if v
            )
            self._analysis_summaries.append(summary[:300])

        if self._epoch_counter % self.meta_interval != 0:
            return self._last_advice, None

        print(f"\n[MetaLoop] Epoch {self._epoch_counter}: calling {self.meta_model}...")
        advice, usage = self._call_meta(history, reward_history)
        self._last_advice = advice
        return advice, usage

    def get_advice_now(
        self, history: PromptStateHistory, reward_history: List[float]
    ) -> Tuple[str, LLMUsage]:
        """Force a meta call regardless of epoch counter. Used by the agent."""
        return self._call_meta(history, reward_history)

    def _call_meta(
        self, history: PromptStateHistory, reward_history: List[float]
    ) -> Tuple[str, LLMUsage]:
        reward_str = "\n".join(f"  Epoch {i}: {r:.4f}" for i, r in enumerate(reward_history, 1)) \
                     or "  No data yet."

        states = history.get_recent(8)
        evolution_lines = []
        for s in states:
            score_str = f"{s.score:.4f}" if s.score is not None else "N/A"
            evolution_lines.append(f"  v{s.version} (score={score_str}): {s.strategy_text[:200]}...")
        evolution_str = "\n".join(evolution_lines) or "  (empty)"

        analysis_str = "\n".join(
            f"  [{i+1}] {s}" for i, s in enumerate(self._analysis_summaries[-3:])
        ) or "  (none yet)"

        units_str = f" ({self.ctx.property_units})" if self.ctx.property_units else ""
        user_content = _META_USER_TEMPLATE.format(
            molecule_type=self.ctx.molecule_type,
            property_name=self.ctx.property_name,
            units_str=units_str,
            direction=self.ctx.direction_word.upper(),
            n_epochs=len(reward_history),
            reward_trajectory=reward_str,
            strategy_evolution=evolution_str,
            analysis_summaries=analysis_str,
        )

        messages = [
            {"role": "system", "content": _build_meta_system(self.ctx)},
            {"role": "user", "content": user_content},
        ]

        text, usage = call_llm(
            model=self.meta_model,
            messages=messages,
            api_keys=self.api_keys,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=3,
        )

        parsed = _parse_json(text, "[MetaLoop]")
        advice = parsed.get("meta_advice", "")
        assessment = parsed.get("strategic_assessment", {})
        print(f"[MetaLoop] Status: {assessment.get('progress_status', 'unknown')}")
        if assessment.get("warnings"):
            print(f"[MetaLoop] Warnings: {assessment['warnings']}")
        return advice, usage


def _parse_json(text: str, tag: str = "") -> Dict:
    text = text.strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?\n?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    print(f"{tag} WARNING: Could not parse as JSON.")
    return {}
