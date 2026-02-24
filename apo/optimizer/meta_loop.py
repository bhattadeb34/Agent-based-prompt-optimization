"""
Meta loop (Strategist Agent): provides high-level strategic guidance periodically.

The meta-strategist operates every K outer epochs. It receives:
- The full strategy evolution history (with scores)
- Reward trajectory
- Analysis summaries from recent critic calls

And it returns high-level chemical intuitions/guidance that the critic
will incorporate as additional context in its next refinement.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..core.llm_client import LLMUsage, call_llm
from ..core.prompt_state import PromptState, PromptStateHistory


_META_SYSTEM = """\
You are a senior computational chemistry research director with deep expertise in \
polymer electrolyte design and machine learning-guided molecular discovery.

You oversee an automated polymer optimisation pipeline. A "critic" LLM refines \
the generation strategy epoch by epoch. Your job is to provide HIGH-LEVEL STRATEGIC \
GUIDANCE to the critic based on the overall trajectory of the optimisation.

You should:
- Identify patterns across multiple epochs (not just the latest one)
- Suggest exploration of fundamentally different chemical spaces if the optimisation is stuck
- Provide chemical hypotheses grounded in polymer science (chain flexibility, ion coordination, etc.)
- Warn about degenerate strategies or mode collapse

Your advice will be injected as context into the critic's next analysis prompt.
Keep it concise, actionable, and chemically rigorous.
"""

_META_USER_TEMPLATE = """\
You are reviewing the progress of a polymer optimisation experiment targeting:
Property: {property_name} ({property_units}) — goal: MAXIMISE

REWARD TRAJECTORY (last {n_epochs} epochs):
{reward_trajectory}

STRATEGY EVOLUTION:
{strategy_evolution}

RECENT ANALYSIS SUMMARIES FROM CRITIC:
{analysis_summaries}

QUESTIONS TO ADDRESS:
1. Is the optimisation making consistent progress or is it stuck?
2. What fundamental chemical directions haven't been explored yet?
3. Are there any degenerate patterns in the strategies (e.g. same modifications repeated)?
4. What single piece of strategic advice would have the highest impact?

Return a JSON object:
{{
  "meta_advice": "Concise high-level strategic guidance in 2-5 sentences for the critic to incorporate...",
  "strategic_assessment": {{
    "progress_status": "improving | plateauing | regressing",
    "key_observations": ["obs 1", "obs 2", "obs 3"],
    "recommended_directions": ["direction 1", "direction 2"],
    "warnings": ["warning 1 if any"]
  }}
}}
Return ONLY JSON.
"""


class MetaLoop:
    """
    Meta-strategist agent that provides high-level guidance every K epochs.
    """

    def __init__(
        self,
        meta_model: str,
        api_keys: Optional[Dict[str, str]] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        meta_interval: int = 3,
        property_name: str = "Property",
        property_units: str = "",
    ):
        self.meta_model = meta_model
        self.api_keys = api_keys or {}
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.meta_interval = meta_interval
        self.property_name = property_name
        self.property_units = property_units

        self._last_advice: str = ""
        self._analysis_summaries: List[str] = []
        self._epoch_counter: int = 0

    def maybe_get_advice(
        self,
        history: PromptStateHistory,
        reward_history: List[float],
        analysis: Optional[Dict] = None,
    ) -> Tuple[str, Optional[LLMUsage]]:
        """
        Check if it's time to call the meta-strategist and return advice.

        Returns:
            (advice_str, usage_or_None)
        """
        self._epoch_counter += 1

        # Collect analysis summaries
        if analysis:
            summary = " | ".join(
                f"{k}: {', '.join(v[:2]) if isinstance(v, list) else v}"
                for k, v in analysis.items()
                if v
            )
            self._analysis_summaries.append(summary[:300])

        # Only call meta every K epochs
        if self._epoch_counter % self.meta_interval != 0:
            return self._last_advice, None

        print(f"\n[MetaLoop] Epoch {self._epoch_counter}: calling {self.meta_model} for strategic guidance...")
        advice, usage = self._call_meta(history, reward_history)
        self._last_advice = advice
        return advice, usage

    def _call_meta(
        self,
        history: PromptStateHistory,
        reward_history: List[float],
    ) -> Tuple[str, LLMUsage]:
        """Call the meta-strategist LLM."""
        # Format reward trajectory
        reward_lines = [
            f"  Epoch {i}: {r:.4f}"
            for i, r in enumerate(reward_history, 1)
        ]
        reward_str = "\n".join(reward_lines) if reward_lines else "  No data yet."

        # Format strategy evolution (last 8 states)
        states = history.get_recent(8)
        evolution_lines = []
        for s in states:
            score_str = f"{s.score:.4f}" if s.score is not None else "N/A"
            evolution_lines.append(f"  v{s.version} (score={score_str}): {s.strategy_text[:200]}...")
        evolution_str = "\n".join(evolution_lines) if evolution_lines else "  (empty)"

        # Format analysis summaries (last 3)
        analysis_str = "\n".join(
            f"  [{i+1}] {s}" for i, s in enumerate(self._analysis_summaries[-3:])
        ) or "  (none yet)"

        user_content = _META_USER_TEMPLATE.format(
            property_name=self.property_name,
            property_units=self.property_units,
            n_epochs=len(reward_history),
            reward_trajectory=reward_str,
            strategy_evolution=evolution_str,
            analysis_summaries=analysis_str,
        )

        messages = [
            {"role": "system", "content": _META_SYSTEM},
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

        parsed = self._parse(text)
        advice = parsed.get("meta_advice", "")
        assessment = parsed.get("strategic_assessment", {})

        print(f"[MetaLoop] Strategic assessment: {assessment.get('progress_status', 'unknown')}")
        if assessment.get("warnings"):
            print(f"[MetaLoop] Warnings: {assessment['warnings']}")

        return advice, usage

    @staticmethod
    def _parse(text: str) -> Dict:
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
        print("[MetaLoop] WARNING: Could not parse meta-strategist response.")
        return {}
