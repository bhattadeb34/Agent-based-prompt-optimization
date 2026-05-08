"""Utilities for combining aggregated LLM usage summaries."""
from __future__ import annotations

from typing import Any, Dict, Optional


def merge_usage_summaries(base: Dict[str, Any], extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge aggregated LLM usage dictionaries without mutating inputs."""
    merged: Dict[str, Any] = {
        "total_calls": base.get("total_calls", 0),
        "total_prompt_tokens": base.get("total_prompt_tokens", 0),
        "total_completion_tokens": base.get("total_completion_tokens", 0),
        "total_tokens": base.get("total_tokens", 0),
        "total_latency_s": base.get("total_latency_s", 0.0),
        "by_model": {
            model: {"calls": stats.get("calls", 0), "tokens": stats.get("tokens", 0)}
            for model, stats in base.get("by_model", {}).items()
        },
    }

    if not extra:
        merged["avg_latency_s"] = _average_latency(merged)
        return merged

    merged["total_calls"] += extra.get("total_calls", 0)
    merged["total_prompt_tokens"] += extra.get("total_prompt_tokens", 0)
    merged["total_completion_tokens"] += extra.get("total_completion_tokens", 0)
    merged["total_tokens"] += extra.get("total_tokens", 0)
    merged["total_latency_s"] = round(
        merged["total_latency_s"] + extra.get("total_latency_s", 0.0),
        3,
    )

    by_model = merged["by_model"]
    for model, stats in extra.get("by_model", {}).items():
        if model not in by_model:
            by_model[model] = {"calls": 0, "tokens": 0}
        by_model[model]["calls"] += stats.get("calls", 0)
        by_model[model]["tokens"] += stats.get("tokens", 0)

    merged["avg_latency_s"] = _average_latency(merged)
    return merged


def _average_latency(usage: Dict[str, Any]) -> float:
    calls = usage.get("total_calls", 0)
    if not calls:
        return 0.0
    return round(usage.get("total_latency_s", 0.0) / calls, 3)
