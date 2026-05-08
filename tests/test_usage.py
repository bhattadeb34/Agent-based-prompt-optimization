"""Tests for LLM usage aggregation helpers."""

from apo.core.usage import merge_usage_summaries


def test_merge_usage_summaries_combines_aggregated_agent_usages():
    base = {
        "total_calls": 1,
        "total_prompt_tokens": 100,
        "total_completion_tokens": 50,
        "total_tokens": 150,
        "total_latency_s": 0.5,
        "avg_latency_s": 0.5,
        "by_model": {
            "gemini/gemini-2.0-flash": {"calls": 1, "tokens": 150},
        },
    }
    extra = {
        "total_calls": 2,
        "total_prompt_tokens": 25,
        "total_completion_tokens": 15,
        "total_tokens": 40,
        "total_latency_s": 0.7,
        "avg_latency_s": 0.35,
        "by_model": {
            "gemini/gemini-2.0-flash": {"calls": 1, "tokens": 10},
            "gpt-4o-mini": {"calls": 1, "tokens": 30},
        },
    }

    merged = merge_usage_summaries(base, extra)

    assert merged["total_calls"] == 3
    assert merged["total_prompt_tokens"] == 125
    assert merged["total_completion_tokens"] == 65
    assert merged["total_tokens"] == 190
    assert merged["total_latency_s"] == 1.2
    assert merged["avg_latency_s"] == 0.4
    assert merged["by_model"]["gemini/gemini-2.0-flash"] == {"calls": 2, "tokens": 160}
    assert merged["by_model"]["gpt-4o-mini"] == {"calls": 1, "tokens": 30}
    assert base["total_calls"] == 1
