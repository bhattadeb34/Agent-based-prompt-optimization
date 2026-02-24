"""
APO Tool-Calling Orchestrator Agent.

Instead of a hardcoded `for epoch in range(N)` loop, this module exposes optimization
primitives as structured tool definitions and lets an LLM decide the execution flow.

Agent loop:
  1. Show orchestrator the current state (reward history, strategy, latest batch stats)
  2. LLM emits a tool_call (generate_candidates, refine_strategy, get_meta_advice, ...)
  3. Execute the tool and feed results back  
  4. Repeat until agent calls done() or hits budget

Tools available:
  - generate_candidates(n_per_molecule, notes)
  - refine_strategy(notes)
  - get_meta_advice()
  - set_strategy(strategy_text, rationale)
  - done(reason)
"""
from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from .core.llm_client import LLMUsage, aggregate_usage, call_llm, _inject_api_keys
from .core.prompt_state import PromptState, PromptStateHistory
from .core.reward import RewardFunction
from .logging.run_logger import RunLogger
from .optimizer.inner_loop import InnerLoop
from .optimizer.meta_loop import MetaLoop
from .optimizer.outer_loop import OuterLoop
from .task_context import TaskContext


# ──────────────────────────────────────────────────────────────────────────────
# Tool schema (OpenAI function-calling format — also works with LiteLLM)
# ──────────────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_candidates",
            "description": (
                "Call the worker LLM to generate candidate molecules for the current strategy. "
                "Returns validity stats and top improvements."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n_per_molecule": {
                        "type": "integer",
                        "description": "Candidates to generate per parent SMILES (1-5).",
                        "default": 3,
                    },
                    "notes": {
                        "type": "string",
                        "description": "Extra notes appended to strategy for this call only.",
                        "default": "",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refine_strategy",
            "description": (
                "Run the critic LLM to analyse the latest candidates and produce an improved strategy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "notes": {
                        "type": "string",
                        "description": "Optional extra instructions for the critic.",
                        "default": "",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_meta_advice",
            "description": (
                "Call the meta-strategist for high-level strategic guidance. "
                "Useful when reward is plateauing or you want to explore new directions."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_strategy",
            "description": "Manually override the current strategy text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy_text": {"type": "string", "description": "New strategy text."},
                    "rationale": {"type": "string", "description": "Why you're overriding.", "default": ""},
                },
                "required": ["strategy_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "End the run. Call when converged, budget exhausted, or no improvement possible.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string", "description": "Why run is complete."}},
                "required": ["reason"],
            },
        },
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ──────────────────────────────────────────────────────────────────────────────

def _agent_system_prompt(ctx: TaskContext, budget: int) -> str:
    domain_notes = ("DOMAIN NOTES:\n" + ctx.domain_context) if ctx.domain_context else ""
    return f"""\
You are an Agentic Prompt Optimisation Controller for {ctx.molecule_type} discovery.

Goal: {ctx.direction_word} {ctx.property_name} ({ctx.property_units}) by iteratively improving
the generation strategy used by a worker LLM.

Budget: {budget} tool calls. Use them wisely.

TOOLS:
- generate_candidates(n_per_molecule=3, notes="")  — run worker LLM
- refine_strategy(notes="")                        — run critic LLM, update strategy
- get_meta_advice()                                — high-level guidance from meta-strategist
- set_strategy(strategy_text, rationale="")        — directly set strategy
- done(reason)                                     — finish the run

TYPICAL FLOW: generate_candidates → refine_strategy → (repeat) → done
You MAY deviate: call get_meta_advice if stuck, call generate_candidates twice before refining
if the first batch was poor, or stop early if reward is satisfactory.

ALWAYS call done() when finished.

{domain_notes}"""


def _state_message(
    current_state: PromptState,
    reward_history: List[float],
    last_candidates: Optional[List[Dict]],
    meta_advice: str,
    budget_remaining: int,
    ctx: TaskContext,
) -> str:
    reward_str = " → ".join(f"{r:.4f}" for r in reward_history) if reward_history else "none"
    lines = [
        "══ CURRENT STATE ══",
        f"Budget remaining: {budget_remaining} tool calls",
        f"Reward history: {reward_str}",
        f"Current strategy (v{current_state.version}): {current_state.strategy_text[:400]}...",
        "",
    ]
    if len(reward_history) >= 2:
        delta = reward_history[-1] - reward_history[-2]
        lines.append(f"Last reward delta: {delta:+.4f}")

    if last_candidates:
        valid = [c for c in last_candidates if c.get("valid")]
        avg_imp = sum(c.get("improvement_factor", 0) for c in valid) / max(len(valid), 1)
        avg_sim = sum(c.get("similarity", 0) for c in valid) / max(len(valid), 1)
        lines.append(
            f"Last batch: {len(valid)}/{len(last_candidates)} valid, "
            f"avg_improvement={avg_imp:.3f}×, avg_sim={avg_sim:.3f}"
        )
        top3 = sorted(valid, key=lambda c: c.get("improvement_factor", 0), reverse=True)[:3]
        for i, c in enumerate(top3, 1):
            lines.append(
                f"  #{i}: ...{c.get('child_smiles','')[:40]} "
                f"improvement={c.get('improvement_factor',0):.2f}× "
                f"property={c.get('child_property',0):.3e}"
            )

    if meta_advice:
        lines += ["", f"Meta advice: {meta_advice[:200]}"]

    lines += ["", "What tool do you want to call next?"]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────────────────────────

def _try_parse_json_tool_call(text: str) -> Optional[Dict]:
    """Try parsing {tool/name, args/arguments} JSON from model text output."""
    if not text:
        return None
    clean = text.strip()
    if "```" in clean:
        clean = re.sub(r"```(?:json)?\n?", "", clean).strip()
    try:
        parsed = json.loads(clean)
        tool_name = parsed.get("tool") or parsed.get("name", "")
        args = parsed.get("args") or parsed.get("arguments", {}) or {}
        if tool_name:
            return {"name": tool_name, "arguments": args}
    except Exception:
        m = re.search(r'"tool"\s*:\s*"(\w+)"', clean)
        if m:
            return {"name": m.group(1), "arguments": {}}
    return None


def _failure_summary(candidates: List[Dict]) -> str:
    invalid = [c for c in candidates if not c.get("valid")]
    if not invalid:
        return "none"
    counts: Dict[str, int] = {}
    for c in invalid:
        r = c.get("invalid_reason", "unknown")
        counts[r] = counts.get(r, 0) + 1
    return ", ".join(f"{v}×{k}" for k, v in sorted(counts.items(), key=lambda x: -x[1])[:3])


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator Agent
# ──────────────────────────────────────────────────────────────────────────────

class OrchestratorAgent:
    """
    LLM-based orchestrator that drives optimization via tool calling.

    The LLM decides when to generate candidates, refine strategy, call meta-advice,
    and when to stop — instead of a fixed for-epoch loop.
    """

    def __init__(
        self,
        inner: InnerLoop,
        outer: OuterLoop,
        meta: MetaLoop,
        logger: RunLogger,
        history: PromptStateHistory,
        reward_fn: RewardFunction,
        task_context: TaskContext,
        orchestrator_model: str,
        api_keys: Optional[Dict[str, str]] = None,
        parent_smiles: Optional[List[str]] = None,
        batch_size: int = 4,
        tool_budget: int = 20,
    ):
        self.inner = inner
        self.outer = outer
        self.meta = meta
        self.logger = logger
        self.history = history
        self.reward_fn = reward_fn
        self.ctx = task_context
        self.orchestrator_model = orchestrator_model
        self.api_keys = api_keys or {}
        self.parent_smiles = parent_smiles or []
        self.batch_size = batch_size
        self.tool_budget = tool_budget

        self._current_state: PromptState = history.latest or PromptState.seed(task_context.seed_strategy)
        self._last_candidates: List[Dict] = []
        self._meta_advice: str = ""
        self._all_usages: List[LLMUsage] = []
        self._epoch: int = 0
        self._messages: List[Dict] = []

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> str:
        """Execute the agent loop. Returns run directory path."""
        print(f"\n{'═'*65}")
        print(f"  APO ORCHESTRATOR AGENT  ({self.orchestrator_model})")
        print(f"  Tool budget: {self.tool_budget} | Property: {self.ctx.property_name}")
        print(f"{'═'*65}\n")

        self._messages = [
            {"role": "system", "content": _agent_system_prompt(self.ctx, self.tool_budget)},
        ]
        budget_remaining = self.tool_budget

        while budget_remaining > 0:
            state_msg = _state_message(
                current_state=self._current_state,
                reward_history=self.logger.reward_history,
                last_candidates=self._last_candidates or None,
                meta_advice=self._meta_advice,
                budget_remaining=budget_remaining,
                ctx=self.ctx,
            )
            self._messages.append({"role": "user", "content": state_msg})

            print(f"\n[Agent] Calling {self.orchestrator_model} (budget={budget_remaining})...")
            try:
                text, usage, tool_call = self._call_orchestrator()
            except Exception as e:
                print(f"[Agent] Orchestrator error: {e}. Stopping.")
                break

            self._all_usages.append(usage)
            budget_remaining -= 1

            if tool_call is None:
                print(f"[Agent] No tool call returned. Text: {text[:200]}. Stopping.")
                break

            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("arguments", {}) or {}
            print(f"[Agent] Tool: {tool_name}({json.dumps(tool_args)[:100]})")

            result = self._execute_tool(tool_name, tool_args)

            self._messages.append({
                "role": "assistant",
                "content": json.dumps({"tool": tool_name, "args": tool_args}),
            })
            self._messages.append({
                "role": "user",
                "content": f"Tool result:\n{result}",
            })

            if tool_name == "done":
                print(f"[Agent] Done: {tool_args.get('reason', '')}")
                break

        used = self.tool_budget - budget_remaining
        print(f"\n[Agent] Run complete. {used}/{self.tool_budget} tools used.")
        return str(self.logger.run_dir)

    # ── LLM call logic ────────────────────────────────────────────────────────

    def _call_orchestrator(self) -> Tuple[str, LLMUsage, Optional[Dict]]:
        """Call orchestrator LLM. Returns (text, usage, tool_call_or_None)."""
        import litellm

        _inject_api_keys(self.api_keys)

        try:
            response = litellm.completion(
                model=self.orchestrator_model,
                messages=self._messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=1024,
            )
            msg = response.choices[0].message
            usage_obj = response.usage or {}
            usage = LLMUsage(
                model=self.orchestrator_model,
                prompt_tokens=getattr(usage_obj, "prompt_tokens", 0),
                completion_tokens=getattr(usage_obj, "completion_tokens", 0),
                latency_s=0.0,
            )

            # Native tool_calls
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                return "", usage, {"name": name, "arguments": args}

            # Some models return JSON text even with tool_choice=auto
            text_content = msg.content or ""
            parsed = _try_parse_json_tool_call(text_content)
            if parsed:
                return "", usage, parsed

            return text_content, usage, None

        except Exception:
            # Fallback: ask for JSON {tool, args}
            return self._json_fallback()

    def _json_fallback(self) -> Tuple[str, LLMUsage, Optional[Dict]]:
        """Fallback for models that don't support native tool calling."""
        tool_names = [t["function"]["name"] for t in TOOLS]
        suffix = (
            f'\n\nRespond with ONLY a JSON object: '
            f'{{"tool": "<one of {tool_names}>", "args": {{...}}}}'
        )
        msgs = list(self._messages)
        if msgs and msgs[-1]["role"] == "user":
            msgs[-1] = {**msgs[-1], "content": msgs[-1]["content"] + suffix}

        text, usage = call_llm(
            model=self.orchestrator_model,
            messages=msgs,
            api_keys=self.api_keys,
            temperature=0.3,
            max_tokens=512,
        )

        parsed = _try_parse_json_tool_call(text)
        if parsed:
            return "", usage, parsed
        return text, usage, None

    # ── Tool implementations ──────────────────────────────────────────────────

    def _execute_tool(self, name: str, args: Dict) -> str:
        if name == "generate_candidates":
            return self._tool_generate_candidates(**args)
        elif name == "refine_strategy":
            return self._tool_refine_strategy(**args)
        elif name == "get_meta_advice":
            return self._tool_get_meta_advice()
        elif name == "set_strategy":
            return self._tool_set_strategy(**args)
        elif name == "done":
            return f"Run finalised: {args.get('reason', '')}"
        return f"Unknown tool: {name}"

    def _tool_generate_candidates(self, n_per_molecule: int = 3, notes: str = "") -> str:
        self._epoch += 1
        strategy = self._current_state.strategy_text
        if notes:
            strategy = f"{strategy}\n\nADDITIONAL NOTES:\n{notes}"

        batch = random.sample(self.parent_smiles, min(self.batch_size, len(self.parent_smiles)))
        candidates, usages = self.inner.run(
            strategy=strategy,
            parent_smiles=batch,
            n_per_molecule=n_per_molecule,
        )
        self._all_usages.extend(usages)
        self._last_candidates = candidates

        valid = [c for c in candidates if c.get("valid")]
        improvements = [c.get("improvement_factor", 0) for c in valid]
        avg_imp = sum(improvements) / max(len(improvements), 1)
        avg_sim = sum(c.get("similarity", 0) for c in valid) / max(len(valid), 1)

        return (
            f"Generated {len(candidates)} candidates: {len(valid)} valid.\n"
            f"Avg improvement: {avg_imp:.3f}×  Avg similarity: {avg_sim:.3f}\n"
            f"Top improvement: {max(improvements, default=0):.3f}×\n"
            f"Failure reasons: {_failure_summary(candidates)}"
        )

    def _tool_refine_strategy(self, notes: str = "") -> str:
        if not self._last_candidates:
            return "No candidates to analyse. Call generate_candidates first."

        new_state, analysis, usage = self.outer.refine(
            candidates=self._last_candidates,
            current_state=self._current_state,
            history=self.history,
            meta_advice=self._meta_advice + (f"\n{notes}" if notes else ""),
        )
        self._all_usages.append(usage)

        reward = self._current_state.score or 0.0
        pareto_data = self.reward_fn.pareto_data(
            [c for c in self._last_candidates if c.get("valid")]
        )
        self.logger.log_epoch(
            epoch=self._epoch,
            prompt_state_dict=self._current_state.to_dict(),
            candidates=self._last_candidates,
            reward=reward,
            pareto_data=pareto_data,
            analysis=analysis,
            meta_advice=self._meta_advice,
            llm_usage=aggregate_usage([usage]),
        )
        self.history.add(new_state)
        self._current_state = new_state

        return (
            f"Strategy refined to v{new_state.version}.\n"
            f"Epoch reward: {reward:.4f} | Pareto HV: {pareto_data.get('hypervolume', 0):.4f}\n"
            f"Rationale: {new_state.rationale[:300]}\n"
            f"New strategy preview: {new_state.strategy_text[:300]}..."
        )

    def _tool_get_meta_advice(self) -> str:
        advice, usage = self.meta.get_advice_now(self.history, self.logger.reward_history)
        if usage:
            self._all_usages.append(usage)
        self._meta_advice = advice
        return f"Meta advice:\n{advice}"

    def _tool_set_strategy(self, strategy_text: str, rationale: str = "") -> str:
        new_state = PromptState(
            strategy_text=strategy_text,
            version=self._current_state.version + 1,
            rationale=rationale or "Manually set by agent",
            parent_version=self._current_state.version,
            model_used="agent_override",
        )
        self.history.add(new_state)
        self._current_state = new_state
        return f"Strategy set to v{new_state.version}: {strategy_text[:200]}..."

    # ── Summary ───────────────────────────────────────────────────────────────

    def total_usage(self) -> Dict:
        return aggregate_usage(self._all_usages)
