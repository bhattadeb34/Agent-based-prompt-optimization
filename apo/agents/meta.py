"""
Meta-Strategist Agent: High-level strategic guidance when optimization plateaus.

Key agentic features:
1. **Trend Analysis**: Detects plateaus, oscillations, degradation
2. **Pattern Recognition**: Identifies repeated failure modes
3. **Strategic Pivots**: Recommends major direction changes
4. **Confidence Assessment**: Only intervenes when high confidence
5. **Full interpretability**: Logs trend analysis, pivot reasoning

Differences from original MetaLoop:
- Original: Simple interval check (every 3 epochs)
- Agentic: Analyzes trend, only intervenes when needed, explains reasoning
"""
from __future__ import annotations

import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .base import Action, Observation, ReActAgent, Thought, Tool
from ..core.llm_client import LLMUsage, call_llm
from ..core.prompt_state import PromptStateHistory
from ..task_context import TaskContext


class TrendAnalyzerTool(Tool):
    """Analyze reward trend to detect plateaus, oscillations, degradation."""

    @property
    def name(self) -> str:
        return "analyze_trend"

    @property
    def description(self) -> str:
        return "Analyze reward history to detect plateaus, oscillations, or degradation."

    @property
    def parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "reward_history": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of reward values over epochs",
                },
                "window_size": {
                    "type": "integer",
                    "description": "Number of recent epochs to analyze (default 3)",
                    "default": 3,
                },
            },
            "required": ["reward_history"],
        }

    def execute(self, reward_history: List[float], window_size: int = 3) -> Observation:
        """Detect trend patterns."""
        if len(reward_history) < window_size:
            return Observation(
                success=True,
                result={"pattern": "insufficient_data", "confidence": 0.0},
                metadata={"n_epochs": len(reward_history)},
            )

        recent = reward_history[-window_size:]
        mean_recent = np.mean(recent)
        std_recent = np.std(recent)

        # Calculate trend
        if len(reward_history) >= 2:
            delta = reward_history[-1] - reward_history[-2]
            delta_pct = (delta / reward_history[-2] * 100) if reward_history[-2] != 0 else 0
        else:
            delta = 0
            delta_pct = 0

        # Detect patterns
        pattern = "unknown"
        confidence = 0.5

        if std_recent < 0.05 * mean_recent and len(recent) >= 3:
            # Low variance = plateau
            pattern = "plateau"
            confidence = 0.9

        elif std_recent > 0.3 * mean_recent:
            # High variance = oscillation
            pattern = "oscillation"
            confidence = 0.8

        elif delta < -0.1 * mean_recent:
            # Recent drop = degradation
            pattern = "degradation"
            confidence = 0.85

        elif delta > 0.1 * mean_recent:
            # Recent gain = improving
            pattern = "improving"
            confidence = 0.8

        else:
            # Stable
            pattern = "stable"
            confidence = 0.7

        return Observation(
            success=True,
            result={
                "pattern": pattern,
                "confidence": confidence,
                "mean_recent": mean_recent,
                "std_recent": std_recent,
                "delta": delta,
                "delta_pct": delta_pct,
            },
            metadata={"window_size": window_size, "n_epochs": len(reward_history)},
        )


class MetaAgent(ReActAgent):
    """
    Meta-Strategist agent for high-level guidance.

    Goal: Detect when optimization is stuck and recommend strategic pivots.

    Only intervenes when:
      1. Plateau detected (low variance across epochs)
      2. Degradation detected (reward dropping)
      3. Repeated failures (same pattern multiple epochs)
    """

    def __init__(
        self,
        model: str,
        api_keys: Dict[str, str],
        task_context: TaskContext,
        temperature: float = 0.4,
        intervention_threshold: float = 0.7,
    ):
        self.ctx = task_context
        self.intervention_threshold = intervention_threshold

        # Current analysis task
        self.history: Optional[PromptStateHistory] = None
        self.reward_history: List[float] = []
        self.recent_analysis: Optional[Dict] = None

        # Results
        self.advice: str = ""
        self.should_intervene: bool = False

        # Interpretability
        self.meta_trace: List[Dict] = []

        super().__init__(
            model=model,
            api_keys=api_keys,
            temperature=temperature,
            max_iterations=3,  # Analyze → Decide → Generate advice
            allow_backtracking=False,
        )

    def _init_tools(self) -> List[Tool]:
        """Tools available to Meta agent."""
        return [
            TrendAnalyzerTool(),
        ]

    def _get_system_prompt(self) -> str:
        """Meta-strategist's role definition."""
        return f"""\
You are a Meta-Strategist for {self.ctx.molecule_type} optimization.

Goal: Provide high-level strategic guidance when optimization gets stuck.

Your responsibilities:
1. Analyze reward trends (plateaus, oscillations, degradation)
2. Identify repeated failure patterns across epochs
3. Recommend strategic pivots (explore new chemical space, change approach)
4. Only intervene when high confidence (>{self.intervention_threshold})

Available tools:
- analyze_trend: Detect plateau/oscillation/degradation patterns

WORKFLOW:
1. Analyze reward trend
2. If stuck (plateau/degradation) → Recommend pivot
3. If improving/stable → No intervention needed
"""

    def _format_state(self) -> str:
        """Current meta-analysis task state."""
        return f"""\
REWARD HISTORY:
{' → '.join(f'{r:.4f}' for r in self.reward_history[-10:])}

RECENT STRATEGIES (last 3):
{self._format_recent_strategies()}

ANALYSIS SO FAR:
{json.dumps(self.recent_analysis, indent=2) if self.recent_analysis else 'None'}
"""

    def _format_recent_strategies(self) -> str:
        """Format recent strategy evolution."""
        if not self.history:
            return "No history"

        recent = self.history.all()[-3:]
        lines = []
        for state in recent:
            lines.append(f"v{state.version}: {state.strategy_text[:100]}...")
        return "\n".join(lines)

    def _is_goal_met(self) -> Tuple[bool, str]:
        """Check if we've made a decision."""
        if self.advice:
            return True, f"Advice generated: {self.advice[:100]}"
        if len(self.steps) >= self.max_iterations:
            return True, "Max iterations reached"
        return False, ""

    def _extract_final_result(self) -> Any:
        """Return advice string."""
        return self.advice

    def get_advice(
        self,
        history: PromptStateHistory,
        reward_history: List[float],
    ) -> Tuple[str, Optional[LLMUsage]]:
        """
        Main entry point: Analyze state and provide advice if needed.

        Returns:
            (advice_string, usage_or_none)
        """
        # Reset state
        self.history = history
        self.reward_history = reward_history
        self.advice = ""
        self.should_intervene = False
        self.recent_analysis = None
        self.meta_trace = []
        self.steps = []
        self.all_usages = []

        print(f"\n[MetaAgent] Analyzing trends...")

        # Run ReAct loop
        result, steps = self.run(initial_state="")

        # Save interpretability trace
        self._save_trace_to_disk()

        # Aggregate usage
        if self.all_usages:
            from ..core.llm_client import aggregate_usage
            total_usage = aggregate_usage(self.all_usages)
            return self.advice, total_usage
        else:
            return self.advice, None

    def _generate_thought(self, iteration: int) -> Thought:
        """Override: Meta-specific thought generation."""
        if iteration == 0:
            return self._analyze_trend()
        elif iteration == 1:
            return self._decide_intervention()
        else:
            return self._generate_advice_text()

    def _analyze_trend(self) -> Thought:
        """Step 1: Analyze reward trend."""
        analyzer = next((t for t in self.tools if t.name == "analyze_trend"), None)
        if not analyzer:
            return Thought(content="No trend analyzer available")

        obs = analyzer.execute(reward_history=self.reward_history, window_size=3)

        if not obs.success:
            return Thought(content="Trend analysis failed")

        self.recent_analysis = obs.result

        # Log for interpretability
        self.meta_trace.append({
            "step": "analyze_trend",
            "iteration": 0,
            "trend_pattern": obs.result["pattern"],
            "confidence": obs.result["confidence"],
            "metrics": {
                "mean_recent": obs.result.get("mean_recent", 0),
                "std_recent": obs.result.get("std_recent", 0),
                "delta_pct": obs.result.get("delta_pct", 0),
            },
        })

        return Thought(
            content=json.dumps(obs.result),
            reasoning_steps=[
                f"Pattern: {obs.result['pattern']}",
                f"Confidence: {obs.result['confidence']:.2f}",
            ],
            confidence=obs.result["confidence"],
        )

    def _decide_intervention(self) -> Thought:
        """Step 2: Decide if intervention is needed."""
        if not self.recent_analysis:
            self.should_intervene = False
            return Thought(content="No analysis available, skipping intervention")

        pattern = self.recent_analysis["pattern"]
        confidence = self.recent_analysis["confidence"]

        # Intervention criteria
        needs_help = pattern in ["plateau", "degradation", "oscillation"]
        high_confidence = confidence > self.intervention_threshold

        self.should_intervene = needs_help and high_confidence

        # Log decision
        self.meta_trace.append({
            "step": "decide_intervention",
            "iteration": 1,
            "should_intervene": self.should_intervene,
            "reason": f"Pattern={pattern}, Confidence={confidence:.2f}",
        })

        if not self.should_intervene:
            self.advice = ""  # No advice needed
            return Thought(
                content=f"No intervention needed: {pattern} with confidence {confidence:.2f}",
                confidence=confidence,
            )

        return Thought(
            content=f"Intervention needed: {pattern} detected with {confidence:.2f} confidence",
            reasoning_steps=[
                f"Pattern: {pattern}",
                f"Confidence: {confidence:.2f} > threshold {self.intervention_threshold}",
            ],
            confidence=confidence,
        )

    def _generate_advice_text(self) -> Thought:
        """Step 3: Generate specific advice."""
        if not self.should_intervene:
            return Thought(content="No advice needed")

        pattern = self.recent_analysis["pattern"]

        prompt = f"""\
The optimization is {pattern}. Provide strategic guidance.

REWARD HISTORY:
{' → '.join(f'{r:.4f}' for r in self.reward_history[-5:])}

TREND ANALYSIS:
{json.dumps(self.recent_analysis, indent=2)}

RECENT STRATEGIES:
{self._format_recent_strategies()}

Based on the {pattern} pattern, what strategic pivot do you recommend?

Consider:
1. If plateau → Explore fundamentally new chemical space
2. If degradation → Revert to earlier successful approach
3. If oscillation → Increase consistency, reduce exploration variance

Return JSON:
{{
  "advice": "Specific strategic recommendation...",
  "rationale": "Why this pivot is needed",
  "expected_outcome": "What this should achieve",
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
            temperature=self.temperature,
            max_tokens=512,
        )
        self.all_usages.append(usage)

        try:
            data = json.loads(text)
            self.advice = data.get("advice", "")

            # Log advice generation
            self.meta_trace.append({
                "step": "generate_advice",
                "iteration": 2,
                "advice": self.advice,
                "rationale": data.get("rationale", ""),
                "expected_outcome": data.get("expected_outcome", ""),
                "confidence": data.get("confidence", 0.7),
            })

            return Thought(
                content=self.advice,
                reasoning_steps=[data.get("rationale", "")],
                confidence=data.get("confidence", 0.7),
            )
        except json.JSONDecodeError:
            self.advice = text[:300]
            return Thought(content=text)

    def _save_trace_to_disk(self):
        """Save full interpretability trace."""
        trace_data = {
            "reward_history": self.reward_history,
            "intervention_decision": self.should_intervene,
            "meta_trace": self.meta_trace,
            "final_advice": self.advice,
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
                }
                for s in self.steps
            ],
        }

        self._interpretability_trace = trace_data
        return trace_data
