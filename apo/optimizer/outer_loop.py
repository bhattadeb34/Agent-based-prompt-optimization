"""
Outer loop (Critic Agent): analyses candidate results and refines the strategy prompt.

The critic:
1. Aggregates candidate data from the inner loop
2. Computes Pareto front and reward
3. Formats a rich analysis prompt including strategy history + meta-advice
4. Calls the critic LLM → produces refined PromptState
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.llm_client import LLMUsage, call_llm
from ..core.prompt_state import PromptState, PromptStateHistory
from ..core.reward import RewardFunction


# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates (strategy refinement)
# ──────────────────────────────────────────────────────────────────────────────

_CRITIC_SYSTEM = """\
You are an expert computational chemist and machine learning strategist.
Your role is to analyse the results of a polymer generation experiment and \
propose an improved STRATEGY for the next round.

The strategy you write will be given verbatim to a worker LLM that generates polymer SMILES.
It must be:
- Specific and actionable (concrete structural modifications to try)
- Chemical precise (mention functional groups, substituents, ring systems)
- Grounded in the data you observe (explain why patterns succeeded/failed)

[Cu] and [Au] are polymer backbone connection points, NOT actual metal atoms.
"""

_CRITIC_USER_TEMPLATE = """\
TASK: Optimise {property_name} ({property_units}) while maintaining Tanimoto similarity.

═══════════════════════════════════
CURRENT STRATEGY (Version {version}):
{current_strategy}
═══════════════════════════════════

EPOCH PERFORMANCE:
- Total candidates generated: {n_total}
- Valid candidates: {n_valid}
- Current reward ({reward_name}): {current_reward:.4f}
- Reward history: {reward_history}
- Avg improvement factor: {avg_improvement:.3f}×
- Avg Tanimoto similarity: {avg_similarity:.3f}

PARETO FRONT ({n_pareto} non-dominated solutions):
{pareto_str}

BEST CANDIDATES THIS EPOCH:
{best_candidates_str}

FAILURE PATTERNS:
{failure_str}

RECENT STRATEGY HISTORY:
{history_str}

{meta_advice_block}

═══════════════════════════════════
ANALYSIS FRAMEWORK:
1. PARETO ANALYSIS: What patterns do the non-dominated solutions share?
2. FAILURE ANALYSIS: What SMILES edits consistently failed? Why?
3. EXPLORATION GAPS: What regions of chemical space are unexplored?
4. NEW STRATEGY: Provide a refined, concrete, actionable strategy.

Return JSON with this exact format:
{{
  "strategy": "Detailed strategy text with step-by-step instructions and specific SMILES motifs...",
  "rationale": "Brief explanation of why this strategy is an improvement",
  "analysis": {{
    "pareto_insights": ["insight 1", "insight 2"],
    "failure_patterns": ["pattern 1", "pattern 2"],
    "exploration_targets": ["target 1", "target 2"],
    "chemical_hypotheses": ["hypothesis 1", "hypothesis 2"]
  }}
}}
Return ONLY the JSON object, no other text.
"""


def _format_pareto(pareto_candidates: List[Dict], property_name: str, property_units: str,
                   max_show: int = 8) -> str:
    if not pareto_candidates:
        return "  No Pareto-optimal candidates found."
    ranked = sorted(pareto_candidates,
                    key=lambda x: x.get("improvement_factor", 0) * x.get("similarity", 0),
                    reverse=True)
    lines = []
    for i, c in enumerate(ranked[:max_show], 1):
        lines.append(
            f"  {i}. Child:  {c.get('child_smiles', 'N/A')}\n"
            f"     Parent: {c.get('parent_smiles', 'N/A')}\n"
            f"     {property_name}: {c.get('parent_property', 0):.3e} → "
            f"{c.get('child_property', 0):.3e} {property_units} "
            f"({c.get('improvement_factor', 0):.2f}×)\n"
            f"     Similarity: {c.get('similarity', 0):.3f}"
        )
    if len(pareto_candidates) > max_show:
        lines.append(f"  ... and {len(pareto_candidates) - max_show} more")
    return "\n".join(lines)


def _format_best(candidates: List[Dict], property_name: str, property_units: str,
                 n: int = 5) -> str:
    valid = [c for c in candidates if c.get("valid")]
    if not valid:
        return "  No valid candidates."
    top = sorted(valid, key=lambda x: x.get("improvement_factor", 0), reverse=True)[:n]
    lines = []
    for i, c in enumerate(top, 1):
        lines.append(
            f"  {i}. {c.get('child_smiles', 'N/A')}\n"
            f"     {property_name}: {c.get('child_property', 0):.3e} {property_units} "
            f"(improvement: {c.get('improvement_factor', 0):.2f}×, sim: {c.get('similarity', 0):.3f})"
        )
    return "\n".join(lines)


def _format_failures(candidates: List[Dict], n: int = 5) -> str:
    invalid = [c for c in candidates if not c.get("valid")]
    if not invalid:
        return "  None (all candidates were valid SMILES)."
    counts: Dict[str, int] = {}
    for c in invalid:
        reason = c.get("invalid_reason", "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    lines = [f"  {v}× {k}" for k, v in sorted(counts.items(), key=lambda x: -x[1])[:n]]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Outer loop runner
# ──────────────────────────────────────────────────────────────────────────────

class OuterLoop:
    """
    Critic agent that analyses results and proposes improved strategy.
    """

    def __init__(
        self,
        reward_fn: RewardFunction,
        critic_model: str,
        api_keys: Optional[Dict[str, str]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        property_name: str = "Property",
        property_units: str = "",
    ):
        self.reward_fn = reward_fn
        self.critic_model = critic_model
        self.api_keys = api_keys or {}
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.property_name = property_name
        self.property_units = property_units
        self._reward_history: List[float] = []

    def refine(
        self,
        candidates: List[Dict],
        current_state: PromptState,
        history: PromptStateHistory,
        meta_advice: str = "",
    ) -> Tuple[PromptState, Dict, LLMUsage]:
        """
        Analyse candidates and return a refined PromptState.

        Returns:
            (new_prompt_state, analysis_dict, llm_usage)
        """
        # Compute reward and update history
        valid_candidates = [c for c in candidates if c.get("valid")]
        reward = self.reward_fn.compute(valid_candidates)
        self._reward_history.append(reward)

        # Update score on current state
        current_state.score = reward

        # Get Pareto data
        pareto_data = self.reward_fn.pareto_data(valid_candidates)
        pareto_candidates = self._extract_pareto_candidates(candidates, pareto_data)

        # Build summary stats
        improvements = [c.get("improvement_factor", 0) for c in valid_candidates]
        similarities = [c.get("similarity", 0) for c in valid_candidates]
        avg_imp = float(np.nanmean(improvements)) if improvements else 0.0
        avg_sim = float(np.nanmean(similarities)) if similarities else 0.0

        # Format history (last 5 strategies)
        recent = history.get_recent(5)
        hist_lines = []
        for s in recent:
            score_str = f"{s.score:.4f}" if s.score is not None else "N/A"
            hist_lines.append(f"  v{s.version}: score={score_str} — {s.strategy_text[:150]}...")
        history_str = "\n".join(hist_lines) if hist_lines else "  (first epoch)"

        # Meta advice block
        if meta_advice:
            meta_block = f"\nMETA-STRATEGIST ADVICE:\n{meta_advice}\n"
        else:
            meta_block = ""

        # Reward history string
        reward_hist_str = ", ".join(f"{r:.4f}" for r in self._reward_history[-6:])

        # Build messages
        user_content = _CRITIC_USER_TEMPLATE.format(
            property_name=self.property_name,
            property_units=self.property_units,
            version=current_state.version,
            current_strategy=current_state.strategy_text,
            n_total=len(candidates),
            n_valid=len(valid_candidates),
            reward_name=self.reward_fn.__class__.__name__,
            current_reward=reward,
            reward_history=reward_hist_str,
            avg_improvement=avg_imp,
            avg_similarity=avg_sim,
            n_pareto=len(pareto_candidates),
            pareto_str=_format_pareto(pareto_candidates, self.property_name, self.property_units),
            best_candidates_str=_format_best(candidates, self.property_name, self.property_units),
            failure_str=_format_failures(candidates),
            history_str=history_str,
            meta_advice_block=meta_block,
        )

        messages = [
            {"role": "system", "content": _CRITIC_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        print(f"\n[OuterLoop] Calling {self.critic_model} to refine strategy "
              f"(epoch reward={reward:.4f})...")
        text, usage = call_llm(
            model=self.critic_model,
            messages=messages,
            api_keys=self.api_keys,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=3,
        )

        # Parse response
        parsed = self._parse_response(text)
        new_strategy = parsed.get("strategy", current_state.strategy_text)
        rationale = parsed.get("rationale", "")
        analysis = parsed.get("analysis", {})

        new_state = PromptState(
            strategy_text=new_strategy,
            version=current_state.version + 1,
            rationale=rationale,
            parent_version=current_state.version,
            model_used=self.critic_model,
            metadata={
                "reward": reward,
                "pareto_hypervolume": pareto_data.get("hypervolume", 0.0),
                "n_valid": len(valid_candidates),
                "avg_improvement": avg_imp,
                "avg_similarity": avg_sim,
            },
        )

        return new_state, analysis, usage

    @staticmethod
    def _extract_pareto_candidates(
        candidates: List[Dict], pareto_data: Dict
    ) -> List[Dict]:
        """Extract candidates on the Pareto front."""
        front_pts = pareto_data.get("pareto_front", [])
        if not front_pts:
            return []
        front_set = {(round(p[0], 4), round(p[1], 4)) for p in front_pts}
        result = []
        for c in candidates:
            if not c.get("valid"):
                continue
            pt = (round(c.get("improvement_factor", 0), 4), round(c.get("similarity", 0), 4))
            if pt in front_set:
                result.append(c)
        return result

    @staticmethod
    def _parse_response(text: str) -> Dict:
        import re
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
        print(f"[OuterLoop] WARNING: Could not parse critic response as JSON.")
        return {}
