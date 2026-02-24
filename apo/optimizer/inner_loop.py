"""
Inner loop (Worker Agent): generates candidate polymer SMILES guided by the current strategy prompt.

The worker:
1. Formats a generation prompt from (strategy, parent SMILES list, property values)
2. Calls the configured worker LLM to get candidate SMILES + explanations
3. Validates each SMILES (RDKit)
4. Predicts properties via the surrogate
5. Computes Tanimoto similarity
6. Returns structured CandidateResult dicts
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..core.llm_client import LLMUsage, call_llm
from ..surrogates.base import SurrogatePredictor
from ..utils.smiles_utils import (
    canonicalize,
    tanimoto_on_repeat_unit,
    validate_polymer_smiles,
)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates (generation-focused)
# ──────────────────────────────────────────────────────────────────────────────

_GENERATION_SYSTEM = """\
You are an expert computational chemist specialising in polymer electrolyte design.
Your task is to generate novel polymer SMILES that maximise a target property \
while maintaining structural similarity to the parent polymers.

IMPORTANT NOTATION:
- [Cu] and [Au] are SYMBOLIC PLACEHOLDERS for polymer backbone connection points.
  They are NOT actual copper or gold atoms.
- Every generated SMILES MUST contain exactly one [Cu] and one [Au].

Example valid polymer SMILES:
  CC(CO[Cu])CSCCOC(=O)[Au]
  CC(CN(C)CCOC(=O)[Au])N[Cu]
  O=C([Au])NCCCCNC(=O)CCCN[Cu]
"""

_GENERATION_USER_TEMPLATE = """\
TARGET PROPERTY: {property_name} ({property_units})
OBJECTIVE: Maximise {property_name} improvement while preserving Tanimoto similarity.

CURRENT STRATEGY:
{strategy}

PARENT MOLECULES AND THEIR {prop_upper} VALUES:
{molecule_pairs}

TASK: For each parent, generate {n_per_molecule} candidate polymers.

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
- Each parent must have exactly {n_per_molecule} entries in smiles and reasoning lists.
- All SMILES must contain [Cu] and [Au] connection points.
- Use ONLY valid SMILES notation (no spaces).
- Return ONLY valid JSON, no other text.
"""


def build_generation_prompt(
    strategy: str,
    parent_data: List[Tuple[str, Optional[float]]],
    n_per_molecule: int,
    property_name: str,
    property_units: str,
) -> Tuple[List[Dict[str, str]], str]:
    """
    Build the message list for the worker LLM.

    Returns:
        (messages, molecule_pairs_str) — messages is OpenAI-format list
    """
    molecule_pairs = "\n".join(
        f"{i+1}. Parent SMILES: {smiles}\n"
        f"   Current {property_name}: {val:.4e} {property_units}"
        if val is not None
        else f"{i+1}. Parent SMILES: {smiles}\n   Current {property_name}: UNKNOWN"
        for i, (smiles, val) in enumerate(parent_data)
    )

    user_content = _GENERATION_USER_TEMPLATE.format(
        property_name=property_name,
        property_units=property_units,
        strategy=strategy,
        molecule_pairs=molecule_pairs,
        n_per_molecule=n_per_molecule,
        prop_upper=property_name.upper(),
    )

    messages = [
        {"role": "system", "content": _GENERATION_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    return messages, molecule_pairs


# ──────────────────────────────────────────────────────────────────────────────
# Inner loop runner
# ──────────────────────────────────────────────────────────────────────────────

class InnerLoop:
    """
    Worker agent that generates candidate molecules given a strategy.
    """

    def __init__(
        self,
        surrogate: SurrogatePredictor,
        worker_model: str,
        api_keys: Optional[Dict[str, str]] = None,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        parent_cache: Optional[Dict[str, float]] = None,
    ):
        self.surrogate = surrogate
        self.worker_model = worker_model
        self.api_keys = api_keys or {}
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Shared parent property cache (persists across epochs)
        self._parent_cache: Dict[str, float] = parent_cache if parent_cache is not None else {}

    # ── public API ────────────────────────────────────────────────────────────

    def run(
        self,
        strategy: str,
        parent_smiles: List[str],
        n_per_molecule: int = 3,
    ) -> Tuple[List[Dict[str, Any]], List[LLMUsage]]:
        """
        Generate candidates for all parent SMILES.

        Returns:
            (candidates, usages) where each candidate is a dict:
              {parent_smiles, child_smiles, parent_property, child_property,
               improvement_factor, similarity, valid, explanation}
        """
        # 1. Get parent properties (cached)
        parent_values = self._get_parent_properties(parent_smiles)
        parent_data = list(zip(parent_smiles, parent_values))

        # 2. Build and send prompt
        messages, _ = build_generation_prompt(
            strategy=strategy,
            parent_data=parent_data,
            n_per_molecule=n_per_molecule,
            property_name=self.surrogate.property_name,
            property_units=self.surrogate.property_units,
        )

        print(f"\n[InnerLoop] Calling {self.worker_model} to generate {n_per_molecule} candidates per parent...")
        text, usage = call_llm(
            model=self.worker_model,
            messages=messages,
            api_keys=self.api_keys,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=3,
        )

        # 3. Parse LLM output
        parsed = self._parse_llm_output(text)

        # 4. Process each candidate
        all_candidates: List[Dict[str, Any]] = []
        parent_map = {p: v for p, v in parent_data}

        for parent_raw, gen_dict in parsed.get("generated_molecules", {}).items():
            # Match to canonical parent
            parent_canonical = canonicalize(parent_raw) or parent_raw
            parent_val = parent_map.get(parent_raw) or parent_map.get(parent_canonical)
            if parent_val is None:
                print(f"[InnerLoop] Warning: no parent value for {parent_raw[:40]}... skipping")
                continue

            smiles_list = gen_dict.get("smiles", [])
            reasoning_list = gen_dict.get("reasoning", [""] * len(smiles_list))

            for child_smiles, reason in zip(smiles_list, reasoning_list):
                candidate = self._process_candidate(
                    child_smiles=child_smiles,
                    explanation=reason,
                    parent_smiles=parent_canonical,
                    parent_value=parent_val,
                )
                all_candidates.append(candidate)

        print(f"[InnerLoop] Generated {len(all_candidates)} candidates "
              f"({sum(1 for c in all_candidates if c['valid'])} valid)")
        return all_candidates, [usage]

    # ── private helpers ───────────────────────────────────────────────────────

    def _get_parent_properties(self, smiles_list: List[str]) -> List[Optional[float]]:
        """Batch-fetch parent properties with caching."""
        missing = [s for s in smiles_list if canonicalize(s) not in self._parent_cache]
        if missing:
            preds = self.surrogate.predict(missing)
            for s, v in zip(missing, preds):
                key = canonicalize(s) or s
                self._parent_cache[key] = v
        results = []
        for s in smiles_list:
            key = canonicalize(s) or s
            results.append(self._parent_cache.get(key))
        return results

    def _process_candidate(
        self,
        child_smiles: str,
        explanation: str,
        parent_smiles: str,
        parent_value: Optional[float],
    ) -> Dict[str, Any]:
        """Validate + score a single generated SMILES."""
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

        # Validate
        ok, reason = validate_polymer_smiles(child_smiles)
        if not ok:
            base["invalid_reason"] = reason
            return base

        child_canonical = canonicalize(child_smiles)
        if child_canonical is None:
            base["invalid_reason"] = "canonicalization failed"
            return base

        # Compute similarity
        similarity = tanimoto_on_repeat_unit(child_canonical, parent_smiles)

        # Predict property
        try:
            preds = self.surrogate.predict([child_canonical])
            child_val = preds[0] if preds else None
        except Exception as e:
            base["invalid_reason"] = f"prediction error: {e}"
            return base

        if child_val is None:
            base["invalid_reason"] = "surrogate returned None"
            return base

        # Compute improvement factor
        if parent_value and abs(parent_value) > 1e-15:
            if self.surrogate.maximize:
                improvement = child_val / parent_value
            else:
                improvement = parent_value / child_val  # minimisation
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
        """Extract JSON from raw LLM text."""
        text = text.strip()
        # Strip markdown fences
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
        print(f"[InnerLoop] WARNING: Could not parse LLM output as JSON. Raw:\n{text[:500]}")
        return {"generated_molecules": {}}
