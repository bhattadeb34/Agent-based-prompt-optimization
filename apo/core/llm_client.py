"""LiteLLM-based universal LLM client with retry, token tracking, and structured output."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import litellm
from litellm import completion

litellm.drop_params = True  # silently ignore unsupported params per model


class LLMUsage:
    """Tracks token usage + latency for a single call."""

    def __init__(self, model: str, prompt_tokens: int, completion_tokens: int, latency_s: float):
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.latency_s = latency_s

    def to_dict(self) -> Dict:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_s": round(self.latency_s, 3),
        }


def call_llm(
    model: str,
    messages: List[Dict[str, str]],
    api_keys: Optional[Dict[str, str]] = None,
    temperature: float = 0.5,
    max_tokens: int = 4096,
    response_format: Optional[Dict] = None,
    max_retries: int = 3,
) -> Tuple[str, LLMUsage]:
    """
    Universal LLM call via LiteLLM.

    Args:
        model: LiteLLM model string, e.g. "openai/gpt-4o", "gemini/gemini-2.0-flash",
               "anthropic/claude-3-opus-20240229"
        messages: OpenAI-format message list
        api_keys: Dict with keys OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY.
                  Falls back to environment variables if None.
        temperature: Sampling temperature
        max_tokens: Max output tokens
        response_format: Optional dict like {"type": "json_object"} for JSON mode
        max_retries: Number of retry attempts on failure

    Returns:
        (text_content, LLMUsage)
    """
    if api_keys:
        _inject_api_keys(api_keys)

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            resp = completion(**kwargs)
            latency = time.time() - t0

            text = resp.choices[0].message.content or ""
            usage = resp.usage or {}
            llm_usage = LLMUsage(
                model=model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                latency_s=latency,
            )
            return text, llm_usage

        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"[LLMClient] Attempt {attempt}/{max_retries} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}") from last_err


def call_llm_json(
    model: str,
    messages: List[Dict[str, str]],
    api_keys: Optional[Dict[str, str]] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    max_retries: int = 3,
) -> Tuple[Dict, LLMUsage]:
    """
    Call LLM and parse JSON from the response.
    Tries JSON mode first, then falls back to extracting JSON from text.
    """
    # Try with JSON response format hint (not all models support it)
    try:
        text, usage = call_llm(
            model=model,
            messages=messages,
            api_keys=api_keys,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            max_retries=max_retries,
        )
    except Exception:
        # Fallback: plain text, we'll parse JSON manually
        text, usage = call_llm(
            model=model,
            messages=messages,
            api_keys=api_keys,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=None,
            max_retries=max_retries,
        )

    parsed = _extract_json(text)
    return parsed, usage


def _extract_json(text: str) -> Dict:
    """Extract JSON from LLM output, handling markdown code fences."""
    text = text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        inner = "\n".join(lines[1:]) if len(lines) > 1 else text
        if inner.rstrip().endswith("```"):
            inner = "\n".join(inner.split("\n")[:-1])
        text = inner.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try finding JSON object in text
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Could not extract JSON from LLM response:\n{text[:500]}")


def _inject_api_keys(api_keys: Dict[str, str]) -> None:
    """Set API keys as environment variables for LiteLLM."""
    key_map = {
        "OPENAI_API_KEY": "OPENAI_API_KEY",
        "GOOGLE_API_KEY": "GEMINI_API_KEY",  # LiteLLM uses GEMINI_API_KEY for gemini/
        "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
    }
    for src, dst in key_map.items():
        if src in api_keys and api_keys[src]:
            os.environ[dst] = api_keys[src]
    # Also set GOOGLE_API_KEY directly (some versions use it)
    if "GOOGLE_API_KEY" in api_keys and api_keys["GOOGLE_API_KEY"]:
        os.environ["GOOGLE_API_KEY"] = api_keys["GOOGLE_API_KEY"]


# Public alias used by agent.py
_inject_env_keys = _inject_api_keys


def aggregate_usage(usages: List[LLMUsage]) -> Dict:
    """Aggregate multiple LLMUsage objects into a summary dict."""
    if not usages:
        return {"total_calls": 0, "total_tokens": 0, "total_latency_s": 0.0}
    return {
        "total_calls": len(usages),
        "total_prompt_tokens": sum(u.prompt_tokens for u in usages),
        "total_completion_tokens": sum(u.completion_tokens for u in usages),
        "total_tokens": sum(u.total_tokens for u in usages),
        "total_latency_s": round(sum(u.latency_s for u in usages), 3),
        "avg_latency_s": round(sum(u.latency_s for u in usages) / len(usages), 3),
        "by_model": _group_by_model(usages),
    }


def _group_by_model(usages: List[LLMUsage]) -> Dict:
    groups: Dict[str, Dict] = {}
    for u in usages:
        if u.model not in groups:
            groups[u.model] = {"calls": 0, "tokens": 0}
        groups[u.model]["calls"] += 1
        groups[u.model]["tokens"] += u.total_tokens
    return groups
