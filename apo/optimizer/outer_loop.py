"""
Outer loop (Critic Agent): analyses candidate results and refines the strategy.

Fully domain-agnostic — all molecule/property language injected via TaskContext.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.llm_client import LLMUsage, call_llm
from ..core.prompt_state import PromptState, PromptStateHistory
from ..core.reward import RewardFunction
from ..task_context import TaskContext


def _build_critic_system(ctx: TaskContext) -> str:
    lines = [
        f"You are an expert scientist and machine learning strategist "
        f"specialising in {ctx.molecule_type} design.",
        "Your role is to analyse the results of a molecular generation experiment "
        f"and propose an improved STRATEGY for generating better {ctx.molecule_type}s.",
        "",
        "The strategy you write will be given verbatim to a worker LLM.",
        "It must be:",
        "- Specific and actionable (name concrete structural modifications to try)",
        "- Chemically precise (mention functional groups, motifs, connectivity)",
        "- Grounded in the data (explain why patterns succeeded or failed)",
    ]
    if ctx.domain_context:
        lines += ["", "DOMAIN NOTES:", ctx.domain_context]
    return "\n".join(lines)


_CRITIC_USER_TEMPLATE = """\
TASK: {direction} {property_name}{units_str} while maintaining structural similarity.

═══════════════════
CURRENT STRATEGY (v{version}):
{current_strategy}
═══════════════════

EPOCH PERFORMANCE:
- Total candidates: {n_total}  |  Valid: {n_valid}
- Reward ({reward_name}): {current_reward:.4f}   History: {reward_history}
- Avg improvement: {avg_improvement:.3f}×  |  Avg similarity: {avg_similarity:.3f}

PARETO FRONT ({n_pareto} non-dominated solutions):
{pareto_str}

BEST CANDIDATES (top by improvement):
{best_candidates_str}

FAILURE PATTERNS (Actionable Side Information):
{failure_str}

RECENT STRATEGY HISTORY:
{history_str}
{meta_advice_block}
═══════════════════
ANALYSIS FRAMEWORK:
1. PARETO: What do the non-dominated solutions have in common?
2. FAILURES: What edits consistently failed? Why?
3. GAPS: What structural families haven't been explored?
4. NEW STRATEGY: Concrete, step-by-step instructions for the next epoch.

Return JSON (ONLY JSON, no other text):
{{
  "strategy": "Detailed step-by-step strategy text...",
  "rationale": "Why this is an improvement over the previous strategy",
  "analysis": {{
    "pareto_insights": ["insight 1", "insight 2"],
    "failure_patterns": ["pattern 1"],
    "exploration_targets": ["target 1"],
    "chemical_hypotheses": ["hypothesis 1"]
  }}
}}
"""


def _fmt_pareto(pareto_candidates, ctx: TaskContext, max_show=6):
    if not pareto_candidates:
        return "  None."
    ranked = sorted(pareto_candidates,
                    key=lambda c: c.get("improvement_factor", 0) * c.get("similarity", 0),
                    reverse=True)
    units_str = f" {ctx.property_units}" if ctx.property_units else ""
    lines = []
    for i, c in enumerate(ranked[:max_show], 1):
        lines.append(
            f"  {i}. {c.get('child_smiles','N/A')}\n"
            f"     {ctx.property_name}: {c.get('parent_property',0):.3e} → "
            f"{c.get('child_property',0):.3e}{units_str} "
            f"({c.get('improvement_factor',0):.2f}×)  sim={c.get('similarity',0):.3f}"
        )
    return "\n".join(lines)


def _fmt_best(candidates, ctx: TaskContext, n=5):
    valid = sorted([c for c in candidates if c.get("valid")],
                   key=lambda c: c.get("improvement_factor", 0), reverse=True)[:n]
    if not valid:
        return "  None."
    units_str = f" {ctx.property_units}" if ctx.property_units else ""
    lines = [
        f"  {i+1}. {c.get('child_smiles','N/A')}\n"
        f"     {ctx.property_name}={c.get('child_property',0):.3e}{units_str} "
        f"improvement={c.get('improvement_factor',0):.2f}× sim={c.get('similarity',0):.3f}"
        for i, c in enumerate(valid)
    ]
    return "\n".join(lines)


def _fmt_failures(candidates, n=5):
    invalid = [c for c in candidates if not c.get("valid")]
    if not invalid:
        return "  None (all valid)."
    counts: Dict[str, int] = {}
    for c in invalid:
        r = c.get("invalid_reason", "unknown")
        counts[r] = counts.get(r, 0) + 1
    return "\n".join(f"  {v}× {k}"
                     for k, v in sorted(counts.items(), key=lambda x: -x[1])[:n])


class OuterLoop:
    """Critic agent that analyses results and proposes improved strategy."""

    def __init__(
        self,
        reward_fn: RewardFunction,
        task_context: TaskContext,
        critic_model: str,
        api_keys: Optional[Dict[str, str]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ):
        self.reward_fn = reward_fn
        self.ctx = task_context
        self.critic_model = critic_model
        self.api_keys = api_keys or {}
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._reward_history: List[float] = []

    def refine(
        self,
        candidates: List[Dict],
        current_state: PromptState,
        history: PromptStateHistory,
        meta_advice: str = "",
    ) -> Tuple[PromptState, Dict, LLMUsage]:
        valid_candidates = [c for c in candidates if c.get("valid")]
        reward = self.reward_fn.compute(valid_candidates)
        self._reward_history.append(reward)
        current_state.score = reward

        pareto_data = self.reward_fn.pareto_data(valid_candidates)
        pareto_candidates = _extract_pareto_candidates(candidates, pareto_data)

        improvements = [c.get("improvement_factor", 0) for c in valid_candidates]
        similarities = [c.get("similarity", 0) for c in valid_candidates]
        avg_imp = float(np.nanmean(improvements)) if improvements else 0.0
        avg_sim = float(np.nanmean(similarities)) if similarities else 0.0

        recent = history.get_recent(5)
        hist_lines = []
        for s in recent:
            score_str = f"{s.score:.4f}" if s.score is not None else "N/A"
            hist_lines.append(f"  v{s.version}: score={score_str} — {s.strategy_text[:150]}...")
        history_str = "\n".join(hist_lines) if hist_lines else "  (first epoch)"

        meta_block = f"\nMETA-STRATEGIST ADVICE:\n{meta_advice}\n" if meta_advice else ""
        reward_hist_str = ", ".join(f"{r:.4f}" for r in self._reward_history[-6:])
        units_str = f" ({self.ctx.property_units})" if self.ctx.property_units else ""

        user_content = _CRITIC_USER_TEMPLATE.format(
            direction=self.ctx.direction_word.capitalize(),
            property_name=self.ctx.property_name,
            units_str=units_str,
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
            pareto_str=_fmt_pareto(pareto_candidates, self.ctx),
            best_candidates_str=_fmt_best(candidates, self.ctx),
            failure_str=_fmt_failures(candidates),
            history_str=history_str,
            meta_advice_block=meta_block,
        )

        messages = [
            {"role": "system", "content": _build_critic_system(self.ctx)},
            {"role": "user", "content": user_content},
        ]

        print(f"\n[OuterLoop] Calling {self.critic_model} to refine strategy "
              f"(reward={reward:.4f})...")
        text, usage = call_llm(
            model=self.critic_model,
            messages=messages,
            api_keys=self.api_keys,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=3,
        )

        parsed = _parse_json(text, "[OuterLoop]")
        new_state = PromptState(
            strategy_text=parsed.get("strategy", current_state.strategy_text),
            version=current_state.version + 1,
            rationale=parsed.get("rationale", ""),
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
        return new_state, parsed.get("analysis", {}), usage


def _extract_pareto_candidates(candidates, pareto_data):
    front_pts = {(round(p[0], 4), round(p[1], 4)) for p in pareto_data.get("pareto_front", [])}
    if not front_pts:
        return []
    return [
        c for c in candidates
        if c.get("valid") and
        (round(c.get("improvement_factor", 0), 4), round(c.get("similarity", 0), 4)) in front_pts
    ]


def _parse_json(text: str, tag: str = "") -> Dict:
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
    print(f"{tag} WARNING: Could not parse response as JSON.")
    return {}
