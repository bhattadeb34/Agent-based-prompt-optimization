"""
Base ReAct Agent with Thought-Action-Observation loop.

Key differences from simple orchestrator:
- Agents have internal thought process before each action
- Agents can self-correct based on observations
- Agents have access to tools they can dynamically choose
- Agents maintain working memory of recent actions/observations
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.llm_client import LLMUsage, call_llm


@dataclass
class Thought:
    """Internal reasoning before taking action."""
    content: str
    reasoning_steps: List[str] = field(default_factory=list)
    confidence: float = 1.0  # 0-1


@dataclass
class Action:
    """Tool invocation with arguments."""
    tool_name: str
    arguments: Dict[str, Any]
    rationale: str = ""


@dataclass
class Observation:
    """Result from executing an action."""
    success: bool
    result: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    """Single thought-action-observation cycle."""
    thought: Thought
    action: Action
    observation: Observation
    iteration: int


class Tool(ABC):
    """Abstract base class for agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for LLM to reference."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this tool does."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict:
        """JSON schema for tool parameters."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Observation:
        """Execute the tool and return observation."""
        pass

    def to_schema(self) -> Dict:
        """Convert to OpenAI function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


class ReActAgent(ABC):
    """
    Base agent with ReAct (Reasoning + Acting) loop.

    Agent loop:
        1. THOUGHT: Analyze current state, decide what to do next
        2. ACTION: Choose a tool and arguments
        3. OBSERVATION: Execute tool, observe result
        4. REFLECTION: Evaluate if goal is met, adjust if needed
        5. Repeat until goal achieved or max iterations
    """

    def __init__(
        self,
        model: str,
        api_keys: Dict[str, str],
        temperature: float = 0.3,
        max_iterations: int = 10,
        allow_backtracking: bool = True,
    ):
        self.model = model
        self.api_keys = api_keys
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.allow_backtracking = allow_backtracking

        # Working memory
        self.steps: List[Step] = []
        self.all_usages: List[LLMUsage] = []

        # Tools available to this agent
        self.tools: List[Tool] = self._init_tools()

    @abstractmethod
    def _init_tools(self) -> List[Tool]:
        """Initialize tools available to this agent."""
        pass

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """System prompt defining agent's role and capabilities."""
        pass

    @abstractmethod
    def _format_state(self) -> str:
        """Format current state for LLM context."""
        pass

    @abstractmethod
    def _is_goal_met(self) -> Tuple[bool, str]:
        """Check if agent's goal is achieved. Returns (done, reason)."""
        pass

    def run(self, initial_state: str) -> Tuple[Any, List[Step]]:
        """
        Execute ReAct loop until goal met or max iterations.

        Returns:
            (final_result, steps_taken)
        """
        iteration = 0

        while iteration < self.max_iterations:
            # 1. THOUGHT
            thought = self._generate_thought(iteration)

            # 2. ACTION
            action = self._select_action(thought)

            # 3. OBSERVATION
            observation = self._execute_action(action)

            # 4. RECORD
            step = Step(
                thought=thought,
                action=action,
                observation=observation,
                iteration=iteration,
            )
            self.steps.append(step)

            # 5. REFLECTION
            done, reason = self._is_goal_met()
            if done:
                print(f"[{self.__class__.__name__}] Goal met: {reason}")
                break

            # 6. SELF-CORRECTION (if observation indicates failure)
            if not observation.success and self.allow_backtracking:
                corrected_action = self._self_correct(observation, action)
                if corrected_action:
                    print(f"[{self.__class__.__name__}] Self-correcting: {corrected_action.tool_name}")
                    observation = self._execute_action(corrected_action)
                    self.steps[-1].observation = observation

            iteration += 1

        # Extract final result from last successful observation
        final_result = self._extract_final_result()
        return final_result, self.steps

    def _generate_thought(self, iteration: int) -> Thought:
        """Generate thought process using LLM."""
        # Build context from working memory
        memory_str = self._format_memory()
        state_str = self._format_state()

        prompt = f"""\
CURRENT STATE:
{state_str}

MEMORY (last 3 steps):
{memory_str}

Think step-by-step about what to do next:
1. What have I accomplished so far?
2. What is my current goal?
3. What tool should I use next and why?
4. What could go wrong?

Return JSON:
{{
  "reasoning_steps": ["step 1", "step 2", ...],
  "next_action": "which tool to call",
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
            return Thought(
                content=text,
                reasoning_steps=data.get("reasoning_steps", []),
                confidence=data.get("confidence", 1.0),
            )
        except json.JSONDecodeError:
            return Thought(content=text, reasoning_steps=[text])

    def _select_action(self, thought: Thought) -> Action:
        """Select action (tool + args) based on thought."""
        tool_schemas = [t.to_schema() for t in self.tools]

        prompt = f"""\
Based on your thought process, select a tool to call.

Thought: {thought.content}

Available tools:
{json.dumps([t["function"]["name"] for t in tool_schemas], indent=2)}

Return JSON:
{{
  "tool": "tool_name",
  "arguments": {{}},
  "rationale": "why this action"
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
            max_tokens=256,
        )
        self.all_usages.append(usage)

        try:
            data = json.loads(text)
            return Action(
                tool_name=data.get("tool", ""),
                arguments=data.get("arguments", {}),
                rationale=data.get("rationale", ""),
            )
        except json.JSONDecodeError:
            # Fallback: call first available tool
            return Action(tool_name=self.tools[0].name if self.tools else "", arguments={})

    def _execute_action(self, action: Action) -> Observation:
        """Execute the selected action."""
        tool = next((t for t in self.tools if t.name == action.tool_name), None)
        if not tool:
            return Observation(
                success=False,
                result=None,
                error=f"Tool '{action.tool_name}' not found",
            )

        try:
            return tool.execute(**action.arguments)
        except Exception as e:
            return Observation(
                success=False,
                result=None,
                error=str(e),
            )

    def _self_correct(self, failed_observation: Observation, failed_action: Action) -> Optional[Action]:
        """Attempt to correct a failed action."""
        prompt = f"""\
Your action failed:
  Tool: {failed_action.tool_name}
  Args: {json.dumps(failed_action.arguments)}
  Error: {failed_observation.error}

Analyze what went wrong and propose a corrected action.

Return JSON:
{{
  "analysis": "what went wrong",
  "corrected_tool": "tool_name",
  "corrected_arguments": {{}},
  "confidence": 0.0-1.0
}}

Or return {{"skip": true}} if correction is not possible.
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
            if data.get("skip"):
                return None
            return Action(
                tool_name=data.get("corrected_tool", ""),
                arguments=data.get("corrected_arguments", {}),
                rationale=f"Self-correction: {data.get('analysis', '')}",
            )
        except json.JSONDecodeError:
            return None

    def _format_memory(self) -> str:
        """Format recent steps for context."""
        recent = self.steps[-3:]
        if not recent:
            return "No previous steps"

        lines = []
        for s in recent:
            lines.append(
                f"Step {s.iteration}: {s.action.tool_name} → "
                f"{'✓' if s.observation.success else '✗'} {str(s.observation.result)[:100]}"
            )
        return "\n".join(lines)

    @abstractmethod
    def _extract_final_result(self) -> Any:
        """Extract final result from step history."""
        pass

    def total_usage(self) -> Dict:
        """Aggregate LLM usage stats."""
        from ..core.llm_client import aggregate_usage
        return aggregate_usage(self.all_usages)
