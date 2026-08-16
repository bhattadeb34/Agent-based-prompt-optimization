"""
Worker Agent: Generates candidate SMILES with self-correction and retry logic.

Key agentic features:
1. **Thought process**: Analyzes strategy before generating
2. **Self-validation**: Checks own outputs before returning
3. **Auto-retry**: Regenerates invalid SMILES with corrected approach
4. **Tool use**: Can query chemistry knowledge, validate SMILES, check similarity
5. **Full interpretability**: Logs all thoughts, actions, self-corrections

Differences from original InnerLoop:
- Original: Direct LLM call → parse → done (no retry)
- Agentic: Think → Generate → Validate → Self-correct → Retry if needed
"""
from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import Action, Observation, ReActAgent, Thought, Tool
from .tools import (
    BatchPropertyPredictorTool,
    ChemistryKnowledgeTool,
    SimilarityCalculatorTool,
    SMILESRepairTool,
    SMILESValidatorTool,
)
from ..core.llm_client import LLMUsage, call_llm
from ..task_context import TaskContext
from ..utils.smiles_utils import canonicalize, compute_similarity, validate_smiles


class WorkerAgent(ReActAgent):
    """
    Agentic SMILES generator with self-correction.

    Goal: Generate n_per_molecule valid candidate SMILES per parent that:
      1. Follow the strategy
      2. Pass RDKit validation
      3. Have reasonable similarity to parent (0.3-0.9)
      4. Maximize predicted property improvement
    """

    def __init__(
        self,
        model: str,
        api_keys: Dict[str, str],
        task_context: TaskContext,
        surrogate,
        parent_cache: Dict[str, float],
        temperature: float = 0.7,
        max_retries_per_batch: int = 3,
    ):
        self.ctx = task_context
        self.surrogate = surrogate
        self.parent_cache = parent_cache
        self.max_retries = max_retries_per_batch

        # Current generation task
        self.strategy: str = ""
        self.parent_smiles_list: List[str] = []
        self.n_per_molecule: int = 4
        self.generated_candidates: List[Dict] = []

        # Interpretability: full trace
        self.generation_trace: List[Dict] = []  # ALL thoughts/actions/retries

        super().__init__(
            model=model,
            api_keys=api_keys,
            temperature=temperature,
            max_iterations=max_retries_per_batch + 1,  # 1 initial + N retries
            allow_backtracking=True,
        )

    def _init_tools(self) -> List[Tool]:
        """Tools available to Worker agent."""
        return [
            SMILESValidatorTool(required_markers=self.ctx.smiles_markers),
            SMILESRepairTool(),
            SimilarityCalculatorTool(
                similarity_on_repeat_unit=self.ctx.similarity_on_repeat_unit,
                marker_strip_tokens=self.ctx.marker_strip_tokens,
            ),
            ChemistryKnowledgeTool(),
            BatchPropertyPredictorTool(self.surrogate, self.ctx.property_name),
        ]

    def _get_system_prompt(self) -> str:
        """Worker's role definition."""
        domain_notes = f"\nDOMAIN CONTEXT:\n{self.ctx.domain_context}" if self.ctx.domain_context else ""
        return f"""\
You are a SMILES Generation Agent specializing in {self.ctx.molecule_type} design.

Goal: Generate valid {self.ctx.molecule_type} SMILES that {self.ctx.direction_word} {self.ctx.property_name}.

CRITICAL CONSTRAINTS:
- SMILES MUST be RDKit-parseable (no syntax errors)
- SMILES MUST contain required markers: {self.ctx.smiles_markers}
- Similarity to parent: 0.3-0.9 (balance novelty & similarity)

Available tools:
- validate_smiles: Check if SMILES are valid before submitting
- repair_smiles: Fix common syntax errors
- calculate_similarity: Check if similarity is in range
- query_chemistry_knowledge: Ask about functional groups, motifs
- batch_predict_property: Estimate property values

WORKFLOW:
1. Analyze strategy → Decide what structural changes to make
2. Generate candidate SMILES
3. Validate with tools BEFORE submitting
4. If invalid → Repair or regenerate
5. Return only valid, on-strategy candidates
{domain_notes}"""

    def _format_state(self) -> str:
        """Current generation task state."""
        return f"""\
STRATEGY: {self.strategy[:500]}

PARENT MOLECULES ({len(self.parent_smiles_list)}):
{chr(10).join(f"  {i+1}. {s}" for i, s in enumerate(self.parent_smiles_list[:5]))}

TASK: Generate {self.n_per_molecule} candidates per parent
TOTAL TARGET: {self.n_per_molecule * len(self.parent_smiles_list)} candidates

GENERATED SO FAR: {len(self.generated_candidates)}
VALID SO FAR: {sum(1 for c in self.generated_candidates if c.get('valid'))}
"""

    def _is_goal_met(self) -> Tuple[bool, str]:
        """Check if we've generated enough valid candidates."""
        target = self.n_per_molecule * len(self.parent_smiles_list)
        n_valid = sum(1 for c in self.generated_candidates if c.get("valid"))

        if n_valid >= target:
            return True, f"Generated {n_valid}/{target} valid candidates"

        # Also stop if we've tried too many times
        if len(self.steps) >= self.max_iterations:
            return True, f"Max iterations reached. Valid: {n_valid}/{target}"

        return False, ""

    def _extract_final_result(self) -> Any:
        """Return generated candidates."""
        return self.generated_candidates

    def generate(
        self,
        strategy: str,
        parent_smiles: List[str],
        n_per_molecule: int = 4,
    ) -> Tuple[List[Dict], List[LLMUsage]]:
        """
        Main entry point: Generate candidates for given parents.

        Returns:
            (candidates, usages)
            where candidates is list of dicts with keys:
              - parent_smiles, child_smiles, explanation
              - parent_property, child_property, improvement_factor
              - similarity, valid, invalid_reason
        """
        # Reset state
        self.strategy = strategy
        self.parent_smiles_list = parent_smiles
        self.n_per_molecule = n_per_molecule
        self.generated_candidates = []
        self.generation_trace = []
        self.steps = []
        self.all_usages = []

        print(f"\n[WorkerAgent] Starting generation: {len(parent_smiles)} parents × {n_per_molecule} = {len(parent_smiles) * n_per_molecule} target")

        # Run ReAct loop
        result, steps = self.run(initial_state="")

        # Save interpretability trace
        self._save_trace_to_disk()

        return self.generated_candidates, self.all_usages

    def _generate_thought(self, iteration: int) -> Thought:
        """Override: Worker-specific thought generation."""
        # First iteration: analyze strategy
        if iteration == 0:
            return self._initial_analysis()
        # Later iterations: reflect on failures and adjust
        else:
            return self._reflect_on_failures()

    def _initial_analysis(self) -> Thought:
        """First thought: Analyze strategy before generating."""
        prompt = f"""\
You are about to generate {self.ctx.molecule_type} SMILES.

STRATEGY:
{self.strategy}

PARENT MOLECULES:
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(self.parent_smiles_list[:3]))}

Think step-by-step:
1. What are the key structural features to focus on? (functional groups, motifs)
2. What modifications will likely {self.ctx.direction_word} {self.ctx.property_name}?
3. How can I ensure similarity stays in 0.3-0.9 range?
4. What common errors should I avoid? (valence, syntax, markers)

Return JSON:
{{
  "reasoning_steps": ["step 1", "step 2", ...],
  "key_modifications": ["mod 1", "mod 2"],
  "confidence": 0.0-1.0
}}
"""

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt},
        ]

        text, usage = call_llm(
            model=self.model,
            messages=messages,
            api_keys=self.api_keys,
            temperature=0.3,  # Lower temp for analysis
            max_tokens=512,
        )
        self.all_usages.append(usage)

        try:
            data = json.loads(text)
            thought = Thought(
                content=text,
                reasoning_steps=data.get("reasoning_steps", []),
                confidence=data.get("confidence", 1.0),
            )
            # Log for interpretability
            self.generation_trace.append({
                "type": "initial_thought",
                "iteration": 0,
                "reasoning": data.get("reasoning_steps", []),
                "key_modifications": data.get("key_modifications", []),
            })
            return thought
        except json.JSONDecodeError:
            return Thought(content=text)

    def _reflect_on_failures(self) -> Thought:
        """Reflect on what went wrong and adjust approach."""
        last_step = self.steps[-1] if self.steps else None
        if not last_step or last_step.observation.success:
            return Thought(content="Previous generation succeeded, continuing...")

        prompt = f"""\
Your previous generation had issues:
{last_step.observation.error}

Current stats:
- Generated: {len(self.generated_candidates)}
- Valid: {sum(1 for c in self.generated_candidates if c.get('valid'))}
- Target: {len(self.parent_smiles_list) * self.n_per_molecule}

Analyze what went wrong and how to fix it:

Return JSON:
{{
  "failure_analysis": "what went wrong",
  "correction_strategy": "how to fix it",
  "confidence": 0.0-1.0
}}
"""

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt},
        ]

        text, usage = call_llm(
            model=self.model,
            messages=messages,
            api_keys=self.api_keys,
            temperature=0.3,
            max_tokens=512,
        )
        self.all_usages.append(usage)

        try:
            data = json.loads(text)
            thought = Thought(
                content=text,
                reasoning_steps=[data.get("failure_analysis", ""), data.get("correction_strategy", "")],
                confidence=data.get("confidence", 0.5),
            )
            # Log self-correction
            self.generation_trace.append({
                "type": "self_correction",
                "iteration": len(self.steps),
                "failure_analysis": data.get("failure_analysis"),
                "correction_strategy": data.get("correction_strategy"),
            })
            return thought
        except json.JSONDecodeError:
            return Thought(content=text)

    def _select_action(self, thought: Thought) -> Action:
        """Override: Always generate candidates (this is Worker's job)."""
        # For Worker, the main action is always "generate_smiles"
        # But we wrap it with validation
        return Action(
            tool_name="generate_and_validate",
            arguments={},
            rationale="Generate batch of candidate SMILES and validate them",
        )

    def _execute_action(self, action: Action) -> Observation:
        """Override: Custom generation + validation logic."""
        # Generate candidates using LLM
        candidates_raw = self._call_llm_for_generation()

        # Validate using tools
        validated_candidates = self._validate_candidates(candidates_raw)

        # Add to results
        self.generated_candidates.extend(validated_candidates)

        n_valid = sum(1 for c in validated_candidates if c.get("valid"))
        n_total = len(validated_candidates)

        if n_valid == 0:
            return Observation(
                success=False,
                result=validated_candidates,
                error=f"0/{n_total} candidates were valid. Need to regenerate with fixes.",
                metadata={"n_valid": n_valid, "n_total": n_total},
            )

        return Observation(
            success=True,
            result=validated_candidates,
            metadata={"n_valid": n_valid, "n_total": n_total, "validity_rate": n_valid / n_total},
        )

    def _call_llm_for_generation(self) -> List[Dict]:
        """Call LLM to generate candidate SMILES."""
        # Build prompt for generation
        examples = self._build_examples()

        prompt = f"""\
Generate {self.n_per_molecule} candidate {self.ctx.molecule_type} SMILES for each parent.

STRATEGY:
{self.strategy}

CONSTRAINTS:
- Each SMILES MUST contain: {self.ctx.smiles_markers}
- Follow the strategy precisely
- Aim for similarity 0.3-0.9 to parent
- Ensure RDKit-valid syntax

{examples}

Return JSON (ONLY JSON, no other text):
{{
  "parent_smiles": [
    {{
      "parent": "CC...",
      "candidates": [
        {{"smiles": "CCC...", "explanation": "Added ether oxygen..."}},
        {{"smiles": "CC(O)...", "explanation": "Introduced hydroxyl..."}},
        ...
      ]
    }},
    ...
  ]
}}
"""

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt},
        ]

        text, usage = call_llm(
            model=self.model,
            messages=messages,
            api_keys=self.api_keys,
            temperature=self.temperature,
            max_tokens=4096,
        )
        self.all_usages.append(usage)

        return self._parse_generation_output(text)

    def _validate_candidates(self, candidates_raw: List[Dict]) -> List[Dict]:
        """Validate candidates using RDKit and surrogate."""
        validator = next((t for t in self.tools if t.name == "validate_smiles"), None)
        if not validator:
            # No validation tool, return as-is
            for c in candidates_raw:
                c["valid"] = True
            return candidates_raw

        if not candidates_raw:
            return []

        # Extract SMILES
        smiles_list = [c["child_smiles"] for c in candidates_raw]

        # Validate
        obs = validator.execute(smiles_list=smiles_list)
        validation_results = obs.result if obs.success else []

        # Merge validation results back
        validated = []
        for i, (cand, val_result) in enumerate(zip(candidates_raw, validation_results)):
            cand["valid"] = val_result.get("valid", False)
            if not cand["valid"]:
                cand["invalid_reason"] = val_result.get("error", "unknown")

            # Get parent and child properties
            parent_smiles_raw = cand["parent_smiles"]
            child_smiles_raw = cand["child_smiles"]
            parent_smiles = canonicalize(parent_smiles_raw) or parent_smiles_raw

            marker_ok, marker_reason = validate_smiles(
                child_smiles_raw,
                required_markers=self.ctx.smiles_markers,
            )
            if cand["valid"] and not marker_ok:
                cand["valid"] = False
                cand["invalid_reason"] = marker_reason

            child_smiles = canonicalize(child_smiles_raw) if cand["valid"] else None
            if cand["valid"] and child_smiles is None:
                cand["valid"] = False
                cand["invalid_reason"] = "canonicalization failed"

            if cand["valid"]:
                cand["parent_smiles"] = parent_smiles
                cand["child_smiles"] = child_smiles

                if parent_smiles not in self.parent_cache:
                    try:
                        self.parent_cache[parent_smiles] = self.surrogate.predict_single(parent_smiles)
                    except Exception:
                        self.parent_cache[parent_smiles] = None

                cand["parent_property"] = self.parent_cache.get(parent_smiles)

                try:
                    cand["child_property"] = self.surrogate.predict_single(child_smiles)
                except Exception as e:
                    cand["child_property"] = None
                    cand["valid"] = False
                    cand["invalid_reason"] = f"Prediction failed: {str(e)}"

                if cand["valid"] and (cand["parent_property"] is None or cand["child_property"] is None):
                    cand["valid"] = False
                    cand["invalid_reason"] = "surrogate returned None"

                if cand["valid"]:
                    parent_value = cand["parent_property"]
                    child_value = cand["child_property"]
                    if self.ctx.maximize:
                        denom = parent_value
                        cand["improvement_factor"] = child_value / denom if abs(denom) > 1e-15 else 0.0
                    else:
                        denom = child_value
                        cand["improvement_factor"] = parent_value / denom if abs(denom) > 1e-15 else 0.0

                    cand["similarity"] = compute_similarity(
                        child_smiles,
                        parent_smiles,
                        similarity_on_repeat_unit=self.ctx.similarity_on_repeat_unit,
                        marker_strip_tokens=self.ctx.marker_strip_tokens,
                    )

            if not cand["valid"]:
                cand.setdefault("parent_property", None)
                cand["child_property"] = None
                cand["improvement_factor"] = 0.0
                cand["similarity"] = 0.0

            validated.append(cand)

        return validated

    @staticmethod
    def _parse_generation_output(text: str) -> List[Dict]:
        """Parse supported worker JSON schemas without crashing on model drift."""
        text = text.strip()
        if "```" in text:
            text = re.sub(r"```(?:json)?\n?", "", text).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                print(f"[WorkerAgent] JSON parse failed, no object found")
                return []
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                print(f"[WorkerAgent] JSON parse failed")
                return []

        candidates = []
        generated = data.get("generated_molecules")
        if isinstance(generated, dict):
            for parent, gen_dict in generated.items():
                smiles_list = gen_dict.get("smiles", []) if isinstance(gen_dict, dict) else []
                reasoning_list = gen_dict.get("reasoning", [""] * len(smiles_list)) if isinstance(gen_dict, dict) else []
                for child_smiles, explanation in zip(smiles_list, reasoning_list):
                    candidates.append({
                        "parent_smiles": parent,
                        "child_smiles": child_smiles,
                        "explanation": explanation,
                    })
            return candidates

        parent_entries = generated if isinstance(generated, list) else data.get("parent_smiles", [])
        if not isinstance(parent_entries, list):
            return []

        for parent_entry in parent_entries:
            if not isinstance(parent_entry, dict):
                continue
            parent = parent_entry.get("parent", "")
            for cand in parent_entry.get("candidates", []):
                if not isinstance(cand, dict):
                    continue
                candidates.append({
                    "parent_smiles": parent,
                    "child_smiles": cand.get("smiles", ""),
                    "explanation": cand.get("explanation", ""),
                })
        return candidates

    def _build_examples(self) -> str:
        """Build few-shot examples from parent SMILES."""
        if not self.parent_smiles_list:
            return ""

        examples = ["PARENT MOLECULES:"]
        for i, parent in enumerate(self.parent_smiles_list[:3], 1):
            examples.append(f"{i}. {parent}")

        return "\n".join(examples)

    def _save_trace_to_disk(self):
        """Save full interpretability trace to JSON."""
        trace_data = {
            "strategy": self.strategy,
            "parents": self.parent_smiles_list,
            "n_per_molecule": self.n_per_molecule,
            "generation_trace": self.generation_trace,
            "steps": [
                {
                    "iteration": s.iteration,
                    "thought": {
                        "reasoning": s.thought.reasoning_steps,
                        "confidence": s.thought.confidence,
                    },
                    "action": {
                        "tool": s.action.tool_name,
                        "args": s.action.arguments,
                        "rationale": s.action.rationale,
                    },
                    "observation": {
                        "success": s.observation.success,
                        "error": s.observation.error,
                        "metadata": s.observation.metadata,
                    },
                }
                for s in self.steps
            ],
            "final_results": {
                "n_generated": len(self.generated_candidates),
                "n_valid": sum(1 for c in self.generated_candidates if c.get("valid")),
                "n_retries": len(self.steps) - 1,
            },
        }

        # This will be saved by RunLogger later
        self._interpretability_trace = trace_data
        return trace_data
