"""
Critic Agent: Analyzes results and proposes improved strategies through debate/negotiation.

Key agentic features:
1. **Multi-perspective analysis**: Considers multiple improvement angles
2. **Debate mode**: Can argue with itself about strategy tradeoffs
3. **Evidence-based reasoning**: Grounds decisions in Pareto front data
4. **Strategy versioning**: Tracks evolution of strategy with rationale
5. **Full interpretability**: Logs all reasoning, rejected alternatives, confidence scores

Differences from original OuterLoop:
- Original: Single LLM call → parse strategy → done
- Agentic: Analyze → Generate alternatives → Debate → Select best → Explain
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .base import Action, Observation, ReActAgent, Thought, Tool
from ..core.llm_client import LLMUsage, call_llm
from ..core.prompt_state import PromptState, PromptStateHistory
from ..core.reward import RewardFunction
from ..task_context import TaskContext


class DebateTool(Tool):
    """Internal debate between two perspectives on strategy."""

    def __init__(self, model: str, api_keys: Dict[str, str]):
        self.model = model
        self.api_keys = api_keys

    @property
    def name(self) -> str:
        return "debate_strategies"

    @property
    def description(self) -> str:
        return (
            "Debate between two alternative strategies. "
            "Returns pros/cons and consensus recommendation."
        )

    @property
    def parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "strategy_a": {"type": "string", "description": "First strategy option"},
                "strategy_b": {"type": "string", "description": "Second strategy option"},
                "context": {"type": "string", "description": "Analysis context (Pareto data, failures)"},
            },
            "required": ["strategy_a", "strategy_b", "context"],
        }

    def execute(self, strategy_a: str, strategy_b: str, context: str) -> Observation:
        """Run multi-turn debate between perspectives."""
        debate_prompt = f"""\
You are a scientific debate moderator. Two strategies have been proposed:

STRATEGY A:
{strategy_a}

STRATEGY B:
{strategy_b}

CONTEXT:
{context}

Simulate a debate:
1. Perspective A argues for Strategy A
2. Perspective B argues for Strategy B
3. Rebuttal from A
4. Rebuttal from B
5. Consensus recommendation

Return JSON:
{{
  "perspective_a_argument": "...",
  "perspective_b_argument": "...",
  "perspective_a_rebuttal": "...",
  "perspective_b_rebuttal": "...",
  "consensus": "A" or "B" or "hybrid",
  "consensus_rationale": "...",
  "confidence": 0.0-1.0
}}
"""

        messages = [{"role": "user", "content": debate_prompt}]

        text, usage = call_llm(
            model=self.model,
            messages=messages,
            api_keys=self.api_keys,
            temperature=0.4,
            max_tokens=1024,
        )

        try:
            debate_result = json.loads(text)
            return Observation(
                success=True,
                result=debate_result,
                metadata={"llm_usage": usage},
            )
        except json.JSONDecodeError:
            return Observation(
                success=False,
                result=None,
                error=f"Could not parse debate output: {text[:200]}",
            )


class CriticAgent(ReActAgent):
    """
    Agentic strategy refiner with multi-perspective analysis.

    Goal: Analyze candidate results and propose an improved strategy that:
      1. Builds on successful patterns (Pareto front)
      2. Avoids repeated failures
      3. Explores new chemical space when stagnant
      4. Maintains domain validity
    """

    def __init__(
        self,
        model: str,
        api_keys: Dict[str, str],
        task_context: TaskContext,
        reward_fn: RewardFunction,
        temperature: float = 0.3,
    ):
        self.ctx = task_context
        self.reward_fn = reward_fn

        # Current refinement task
        self.candidates: List[Dict] = []
        self.current_state: Optional[PromptState] = None
        self.history: Optional[PromptStateHistory] = None
        self.meta_advice: str = ""

        # Results
        self.new_state: Optional[PromptState] = None
        self.analysis: Dict = {}

        # Interpretability: full reasoning trace
        self.refinement_trace: List[Dict] = []

        super().__init__(
            model=model,
            api_keys=api_keys,
            temperature=temperature,
            max_iterations=5,  # Analyze → Generate alternatives → Debate → Select → Finalize
            allow_backtracking=True,
        )

    def _init_tools(self) -> List[Tool]:
        """Tools available to Critic agent."""
        return [
            DebateTool(self.model, self.api_keys),
        ]

    def _get_system_prompt(self) -> str:
        """Critic's role definition."""
        return f"""\
You are a Strategy Refinement Agent for {self.ctx.molecule_type} optimization.

Goal: Analyze experimental results and propose an improved generation strategy.

Your responsibilities:
1. Identify patterns in successful candidates (Pareto front analysis)
2. Diagnose failure modes and suggest fixes
3. Balance exploration (new motifs) vs exploitation (refine current approach)
4. Ground recommendations in chemical principles

Available tools:
- debate_strategies: Compare alternative strategies through structured debate

WORKFLOW:
1. Analyze Pareto front → Extract successful patterns
2. Analyze failures → Identify what to avoid
3. Generate 2-3 alternative strategies
4. Debate alternatives → Select best
5. Return refined strategy with full rationale
"""

    def _format_state(self) -> str:
        """Current refinement task state."""
        valid = [c for c in self.candidates if c.get("valid")]
        pareto = self.reward_fn.pareto_data(valid) if valid else {}

        return f"""\
CURRENT STRATEGY (v{self.current_state.version if self.current_state else 0}):
{self.current_state.strategy_text[:300] if self.current_state else 'N/A'}...

RESULTS THIS EPOCH:
- Candidates: {len(self.candidates)} ({len(valid)} valid)
- Reward: {self.current_state.score if self.current_state else 0:.4f}
- Pareto front: {len(pareto.get('pareto_front', []))} solutions

TOP IMPROVEMENTS:
{self._format_top_candidates(valid[:3])}

FAILURE SUMMARY:
{self._format_failures()}

META ADVICE:
{self.meta_advice if self.meta_advice else "None"}
"""

    def _format_top_candidates(self, candidates: List[Dict]) -> str:
        """Format top candidates for display."""
        if not candidates:
            return "None"
        lines = []
        for i, c in enumerate(candidates, 1):
            lines.append(
                f"{i}. {c.get('child_smiles', '')[:40]}... "
                f"improvement={c.get('improvement_factor', 0):.2f}× "
                f"sim={c.get('similarity', 0):.2f}"
            )
        return "\n".join(lines)

    def _format_failures(self) -> str:
        """Format failure patterns."""
        invalid = [c for c in self.candidates if not c.get("valid")]
        if not invalid:
            return "No failures"

        failure_counts: Dict[str, int] = {}
        for c in invalid:
            reason = c.get("invalid_reason", "unknown")
            failure_counts[reason] = failure_counts.get(reason, 0) + 1

        return ", ".join(f"{count}× {reason}" for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1])[:3])

    def _is_goal_met(self) -> Tuple[bool, str]:
        """Check if we've refined the strategy."""
        if self.new_state is not None:
            return True, f"Strategy refined to v{self.new_state.version}"
        if len(self.steps) >= self.max_iterations:
            return True, "Max iterations reached"
        return False, ""

    def _extract_final_result(self) -> Any:
        """Return refined strategy."""
        return (self.new_state, self.analysis)

    def refine(
        self,
        candidates: List[Dict],
        current_state: PromptState,
        history: PromptStateHistory,
        meta_advice: str = "",
    ) -> Tuple[PromptState, Dict, LLMUsage]:
        """
        Main entry point: Refine strategy based on results.

        Returns:
            (new_state, analysis, usage)
        """
        # Reset state
        self.candidates = candidates
        self.current_state = current_state
        self.history = history
        self.meta_advice = meta_advice
        self.new_state = None
        self.analysis = {}
        self.refinement_trace = []
        self.steps = []
        self.all_usages = []

        print(f"\n[CriticAgent] Refining strategy v{current_state.version} → v{current_state.version + 1}")

        # Run ReAct loop
        result, steps = self.run(initial_state="")

        # Save interpretability trace
        self._save_trace_to_disk()

        # Aggregate usage
        from ..core.llm_client import aggregate_usage
        total_usage = aggregate_usage(self.all_usages)

        return self.new_state, self.analysis, total_usage

    def _generate_thought(self, iteration: int) -> Thought:
        """Override: Critic-specific thought generation."""
        if iteration == 0:
            return self._analyze_results()
        elif iteration == 1:
            return self._generate_alternatives()
        else:
            return Thought(content="Finalizing strategy selection")

    def _analyze_results(self) -> Thought:
        """Step 1: Analyze Pareto front and failures."""
        valid = [c for c in self.candidates if c.get("valid")]
        pareto = self.reward_fn.pareto_data(valid) if valid else {}

        prompt = f"""\
Analyze the experimental results:

PARETO FRONT ({len(pareto.get('pareto_front', []))} solutions):
{json.dumps(pareto.get('pareto_front', [])[:5], indent=2)}

TOP CANDIDATES:
{self._format_top_candidates(valid[:5])}

FAILURES:
{self._format_failures()}

Perform structured analysis:
1. What structural patterns appear in Pareto-optimal molecules?
2. What common failures occurred and why?
3. What chemical space remains unexplored?
4. What tradeoffs exist (improvement vs similarity)?

Return JSON:
{{
  "pareto_insights": ["insight 1", "insight 2", ...],
  "failure_patterns": ["pattern 1", "pattern 2", ...],
  "unexplored_space": ["area 1", "area 2", ...],
  "tradeoffs": "description of key tradeoffs",
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
            temperature=0.2,  # Lower temp for analysis
            max_tokens=1024,
        )
        self.all_usages.append(usage)

        try:
            data = json.loads(text)
            self.analysis = {
                "pareto_insights": data.get("pareto_insights", []),
                "failure_patterns": data.get("failure_patterns", []),
                "exploration_targets": data.get("unexplored_space", []),
                "chemical_hypotheses": [],  # Will be filled by strategy generation
            }

            # Log for interpretability
            self.refinement_trace.append({
                "step": "analyze_results",
                "iteration": 0,
                "analysis": self.analysis,
                "confidence": data.get("confidence", 1.0),
            })

            return Thought(
                content=text,
                reasoning_steps=[
                    f"Pareto: {len(data.get('pareto_insights', []))} insights",
                    f"Failures: {len(data.get('failure_patterns', []))} patterns",
                ],
                confidence=data.get("confidence", 1.0),
            )
        except json.JSONDecodeError:
            return Thought(content=text)

    def _generate_alternatives(self) -> Thought:
        """Step 2: Generate multiple alternative strategies."""
        prompt = f"""\
Based on the analysis, propose 2-3 alternative strategies for the next epoch.

ANALYSIS:
{json.dumps(self.analysis, indent=2)}

CURRENT STRATEGY (v{self.current_state.version}):
{self.current_state.strategy_text}

META ADVICE:
{self.meta_advice if self.meta_advice else "None"}

Generate alternatives that:
1. **Exploit**: Refine current successful patterns
2. **Explore**: Try fundamentally new approaches
3. **Hybrid**: Balance exploitation and exploration

Return JSON:
{{
  "alternative_1": {{
    "name": "Exploit (refine current)",
    "strategy": "detailed strategy text...",
    "rationale": "why this approach"
  }},
  "alternative_2": {{
    "name": "Explore (new direction)",
    "strategy": "detailed strategy text...",
    "rationale": "why this approach"
  }},
  "alternative_3": {{
    "name": "Hybrid",
    "strategy": "detailed strategy text...",
    "rationale": "why this approach"
  }}
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
            temperature=0.5,  # Higher temp for creativity
            max_tokens=2048,
        )
        self.all_usages.append(usage)

        try:
            alternatives = json.loads(text)

            # Log alternatives for interpretability
            self.refinement_trace.append({
                "step": "generate_alternatives",
                "iteration": 1,
                "alternatives": alternatives,
            })

            # Now run debate to select best
            selected = self._run_debate(alternatives)

            # Create new state
            self.new_state = PromptState(
                strategy_text=selected["strategy"],
                version=self.current_state.version + 1,
                rationale=selected["rationale"],
                parent_version=self.current_state.version,
                model_used=self.model,
            )

            return Thought(
                content=f"Selected: {selected['name']}",
                reasoning_steps=["Generated 3 alternatives", "Ran debate", f"Selected: {selected['name']}"],
                confidence=selected.get("confidence", 0.8),
            )
        except json.JSONDecodeError:
            # Fallback: use text as new strategy
            self.new_state = PromptState(
                strategy_text=text[:500],
                version=self.current_state.version + 1,
                rationale="Fallback (JSON parse failed)",
                parent_version=self.current_state.version,
                model_used=self.model,
            )
            return Thought(content=text)

    def _run_debate(self, alternatives: Dict) -> Dict:
        """Run debate between top 2 alternatives."""
        alt_list = [v for k, v in alternatives.items() if isinstance(v, dict)]
        if len(alt_list) < 2:
            # Not enough alternatives, return first
            return alt_list[0] if alt_list else {"strategy": "fallback", "rationale": "no alternatives", "name": "fallback"}

        # Debate top 2
        debate_tool = next((t for t in self.tools if t.name == "debate_strategies"), None)
        if not debate_tool:
            # No debate tool, return first alternative
            return alt_list[0]

        context = f"Analysis:\n{json.dumps(self.analysis, indent=2)}"

        obs = debate_tool.execute(
            strategy_a=alt_list[0]["strategy"],
            strategy_b=alt_list[1]["strategy"],
            context=context,
        )

        if not obs.success:
            # Debate failed, return first alternative
            return alt_list[0]

        debate_result = obs.result
        consensus = debate_result.get("consensus", "A")

        # Log debate for interpretability
        self.refinement_trace.append({
            "step": "debate",
            "iteration": 2,
            "debate_transcript": debate_result,
            "selected": consensus,
        })

        # Get usage from metadata
        if "llm_usage" in obs.metadata:
            self.all_usages.append(obs.metadata["llm_usage"])

        # Return selected alternative
        if consensus == "A":
            selected = alt_list[0]
        elif consensus == "B":
            selected = alt_list[1]
        else:  # Hybrid
            # Merge both strategies
            selected = {
                "name": "Hybrid (debate consensus)",
                "strategy": f"{alt_list[0]['strategy']}\n\nAND\n\n{alt_list[1]['strategy']}",
                "rationale": debate_result.get("consensus_rationale", "Hybrid approach"),
                "confidence": debate_result.get("confidence", 0.7),
            }

        return selected

    def _save_trace_to_disk(self):
        """Save full interpretability trace."""
        trace_data = {
            "current_strategy_version": self.current_state.version if self.current_state else 0,
            "new_strategy_version": self.new_state.version if self.new_state else 0,
            "refinement_trace": self.refinement_trace,
            "analysis": self.analysis,
            "steps": [
                {
                    "iteration": s.iteration,
                    "thought": {
                        "reasoning": s.thought.reasoning_steps,
                        "confidence": s.thought.confidence,
                    },
                    "action": {
                        "tool": s.action.tool_name,
                        "rationale": s.action.rationale,
                    },
                    "observation": {
                        "success": s.observation.success,
                    },
                }
                for s in self.steps
            ],
        }

        self._interpretability_trace = trace_data
        return trace_data
