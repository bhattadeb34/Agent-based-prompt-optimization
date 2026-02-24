"""
Inner loop (Worker Agent): generates candidate SMILES guided by the current strategy prompt.

Fully domain-agnostic — all molecule-type assumptions come from TaskContext.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..core.llm_client import LLMUsage, call_llm
from ..surrogates.base import SurrogatePredictor
from ..task_context import TaskContext
from ..utils.smiles_utils import (
    canonicalize,
    compute_similarity,
    validate_smiles,
)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builders (domain-driven, no hardcoding)
# ──────────────────────────────────────────────────────────────────────────────

def _build_system_prompt(ctx: TaskContext) -> str:
    """Build a domain-specific system prompt from TaskContext."""
    lines = [
        f"You are an expert chemist specialising in {ctx.molecule_type} design.",
        f"Your task is to generate novel {ctx.molecule_type} SMILES that "
        f"{ctx.direction_word_us} a target property "
        f"while maintaining structural similarity to the parent molecules.",
    ]
    if ctx.domain_context:
        lines += ["", "DOMAIN-SPECIFIC NOTES:", ctx.domain_context]
    if ctx.example_smiles:
        lines += ["", f"Example valid {ctx.molecule_type} SMILES:"]
        lines += [f"  {s}" for s in ctx.example_smiles[:4]]
    return "\n".join(lines)


_GENERATION_USER_TEMPLATE = """\
TARGET PROPERTY: {property_name}{units_str}
OBJECTIVE: {direction} {property_name} while preserving structural similarity.

CURRENT STRATEGY:
{strategy}

PARENT MOLECULES AND THEIR {prop_upper} VALUES:
{molecule_pairs}

TASK: For each parent, generate {n_per_molecule} candidate {molecule_type}s.

Return a JSON object with this exact structure:
{{
  "generated_molecules": {{
    "<parent_smiles_1>": {{
      "smiles": ["candidate_1", "candidate_2", ...],
      "reasoning": ["why_candidate_1_is_good", "why_candidate_2_is_good", ...]
    }},
    "<parent_smiles_2>": {{
      "smiles": [...],
      "reasoning": [...]
    }}
  }}
}}

Rules:
- Each parent must have exactly {n_per_molecule} entries in both lists.
- Use ONLY valid SMILES notation (no spaces).
{marker_rules}
- Return ONLY valid JSON, no other text.
"""


def build_generation_prompt(
    strategy: str,
    parent_data: List[Tuple[str, Optional[float]]],
    n_per_molecule: int,
    ctx: TaskContext,
) -> List[Dict[str, str]]:
    """Build OpenAI-format messages for the worker LLM."""
    units_str = f" ({ctx.property_units})" if ctx.property_units else ""
    molecule_pairs = "\n".join(
        f"{i+1}. SMILES: {smiles}\n"
        f"   {ctx.property_name}: {val:.4e}{units_str}"
        if val is not None
        else f"{i+1}. SMILES: {smiles}\n   {ctx.property_name}: UNKNOWN"
        for i, (smiles, val) in enumerate(parent_data)
    )

    # Marker rules only if needed
    if ctx.smiles_markers:
        marker_rules = "- Every generated SMILES MUST contain: " + \
                       ", ".join(f"'{m}'" for m in ctx.smiles_markers)
    else:
        marker_rules = "- Generate valid, chemically-sensible SMILES."

    user_content = _GENERATION_USER_TEMPLATE.format(
        property_name=ctx.property_name,
        units_str=units_str,
        direction=ctx.direction_word_us.capitalize(),
        strategy=strategy,
        molecule_pairs=molecule_pairs,
        n_per_molecule=n_per_molecule,
        molecule_type=ctx.molecule_type,
        prop_upper=ctx.property_name.upper(),
        marker_rules=marker_rules,
    )

    return [
        {"role": "system", "content": _build_system_prompt(ctx)},
        {"role": "user", "content": user_content},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Inner loop runner
# ──────────────────────────────────────────────────────────────────────────────

class InnerLoop:
    """Worker agent that generates candidate molecules given a strategy."""

    def __init__(
        self,
        surrogate: SurrogatePredictor,
        task_context: TaskContext,
        worker_model: str,
        api_keys: Optional[Dict[str, str]] = None,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        parent_cache: Optional[Dict[str, float]] = None,
    ):
        self.surrogate = surrogate
        self.ctx = task_context
        self.worker_model = worker_model
        self.api_keys = api_keys or {}
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._parent_cache: Dict[str, float] = parent_cache if parent_cache is not None else {}

    def run(
        self,
        strategy: str,
        parent_smiles: List[str],
        n_per_molecule: int = 3,
    ) -> Tuple[List[Dict[str, Any]], List[LLMUsage]]:
        """
        Generate candidates for all parent SMILES.

        Returns:
            (candidates, usages)
        """
        parent_values = self._get_parent_properties(parent_smiles)
        parent_data = list(zip(parent_smiles, parent_values))

        messages = build_generation_prompt(
            strategy=strategy,
            parent_data=parent_data,
            n_per_molecule=n_per_molecule,
            ctx=self.ctx,
        )

        print(f"\n[InnerLoop] Calling {self.worker_model} — "
              f"{n_per_molecule} candidates × {len(parent_smiles)} parents ...")
        text, usage = call_llm(
            model=self.worker_model,
            messages=messages,
            api_keys=self.api_keys,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=3,
        )

        parsed = self._parse_llm_output(text)
        parent_map = {p: v for p, v in parent_data}

        all_candidates: List[Dict[str, Any]] = []
        for parent_raw, gen_dict in parsed.get("generated_molecules", {}).items():
            parent_canonical = canonicalize(parent_raw) or parent_raw
            parent_val = parent_map.get(parent_raw) or parent_map.get(parent_canonical)
            if parent_val is None:
                print(f"[InnerLoop] Warning: no parent value for {parent_raw[:40]}...")
                continue

            smiles_list = gen_dict.get("smiles", [])
            reasoning_list = gen_dict.get("reasoning", [""] * len(smiles_list))

            for child_smiles, reason in zip(smiles_list, reasoning_list):
                candidate = self._process_candidate(child_smiles, reason,
                                                    parent_canonical, parent_val)
                all_candidates.append(candidate)

        print(f"[InnerLoop] Generated {len(all_candidates)} candidates "
              f"({sum(1 for c in all_candidates if c['valid'])} valid)")
        return all_candidates, [usage]

    def _get_parent_properties(self, smiles_list: List[str]) -> List[Optional[float]]:
        missing = [s for s in smiles_list if (canonicalize(s) or s) not in self._parent_cache]
        if missing:
            preds = self.surrogate.predict(missing)
            for s, v in zip(missing, preds):
                self._parent_cache[canonicalize(s) or s] = v
        return [self._parent_cache.get(canonicalize(s) or s) for s in smiles_list]

    def _process_candidate(
        self,
        child_smiles: str,
        explanation: str,
        parent_smiles: str,
        parent_value: Optional[float],
    ) -> Dict[str, Any]:
        base = {
            "parent_smiles": parent_smiles,
            "child_smiles": child_smiles,
            "explanation": explanation,
            "parent_property": parent_value,
            "child_property": None,
            "improvement_factor": 0.0,
            "similarity": 0.0,
            "valid": False,
            "invalid_reason": "",
        }

        ok, reason = validate_smiles(child_smiles, required_markers=self.ctx.smiles_markers)
        if not ok:
            base["invalid_reason"] = reason
            return base

        child_canonical = canonicalize(child_smiles)
        if child_canonical is None:
            base["invalid_reason"] = "canonicalization failed"
            return base

        similarity = compute_similarity(
            child_canonical, parent_smiles,
            similarity_on_repeat_unit=self.ctx.similarity_on_repeat_unit,
            marker_strip_tokens=self.ctx.marker_strip_tokens,
        )

        try:
            preds = self.surrogate.predict([child_canonical])
            child_val = preds[0] if preds else None
        except Exception as e:
            base["invalid_reason"] = f"prediction error: {e}"
            return base

        if child_val is None:
            base["invalid_reason"] = "surrogate returned None"
            return base

        if parent_value and abs(parent_value) > 1e-15:
            improvement = (child_val / parent_value) if self.ctx.maximize \
                          else (parent_value / child_val)
        else:
            improvement = 0.0

        base.update({
            "child_smiles": child_canonical,
            "child_property": child_val,
            "improvement_factor": round(improvement, 6),
            "similarity": round(similarity, 6),
            "valid": True,
        })
        return base

    @staticmethod
    def _parse_llm_output(text: str) -> Dict:
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
        print(f"[InnerLoop] WARNING: Could not parse LLM output as JSON.\n{text[:300]}")
        return {"generated_molecules": {}}
