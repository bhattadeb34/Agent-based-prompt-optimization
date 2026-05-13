"""Tests for LLM usage accounting helpers."""

from apo.core.llm_client import LLMUsage, aggregate_usage


def test_aggregate_usage_merges_raw_and_aggregated_records():
    raw_usage = LLMUsage("worker-model", 10, 5, 0.2)
    critic_summary = {
        "total_calls": 2,
        "total_prompt_tokens": 30,
        "total_completion_tokens": 15,
        "total_tokens": 45,
        "total_latency_s": 0.6,
        "by_model": {"critic-model": {"calls": 2, "tokens": 45}},
    }

    summary = aggregate_usage([raw_usage, critic_summary])

    assert summary["total_calls"] == 3
    assert summary["total_prompt_tokens"] == 40
    assert summary["total_completion_tokens"] == 20
    assert summary["total_tokens"] == 60
    assert summary["total_latency_s"] == 0.8
    assert summary["avg_latency_s"] == 0.267
    assert summary["by_model"] == {
        "worker-model": {"calls": 1, "tokens": 15},
        "critic-model": {"calls": 2, "tokens": 45},
    }
