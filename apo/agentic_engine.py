"""
Simple Agentic Engine (without LangGraph initially).

Replaces the linear loop with agentic agents that:
- Self-correct invalid SMILES
- Debate strategy alternatives
- Detect plateaus and pivot

This is a simpler version without LangGraph state machine.
LangGraph version will be added later for advanced conditional flows.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agents import WorkerAgent, CriticAgent, MetaAgent
from .core.llm_client import LLMUsage, aggregate_usage
from .core.prompt_state import PromptState, PromptStateHistory
from .core.reward import get_reward_function
from .logging.run_logger import RunLogger
from .surrogates.registry import get_surrogate
from .task_context import TaskContext


def _merge_usage_summaries(base: Dict[str, Any], extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge pre-aggregated usage dicts returned by agentic sub-agents."""
    if not extra:
        return base

    merged = dict(base)
    for key in (
        "total_calls",
        "total_tokens",
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_latency_s",
    ):
        merged[key] = merged.get(key, 0) + extra.get(key, 0)

    by_model = dict(merged.get("by_model", {}))
    for model, stats in extra.get("by_model", {}).items():
        model_stats = by_model.setdefault(model, {"calls": 0, "tokens": 0})
        model_stats["calls"] = model_stats.get("calls", 0) + stats.get("calls", 0)
        model_stats["tokens"] = model_stats.get("tokens", 0) + stats.get("tokens", 0)
    if by_model:
        merged["by_model"] = by_model

    total_calls = merged.get("total_calls", 0)
    if total_calls:
        merged["avg_latency_s"] = round(merged.get("total_latency_s", 0.0) / total_calls, 3)
    return merged


def run_agentic_mode(
    cfg: Dict,
    ctx: TaskContext,
    all_smiles: List[str],
    logger: RunLogger,
    api_keys: Dict[str, str],
) -> str:
    """
    Run optimization with agentic agents.

    Returns: run_dir path
    """
    print("\n" + "=" * 70)
    print("  AGENTIC PROMPT OPTIMIZATION")
    print("=" * 70)

    model_cfg = cfg["models"]
    opt_cfg = cfg["optimization"]
    temp_cfg = cfg.get("temperatures", {})

    # Load surrogate
    surrogate_name = cfg["task"]["surrogate"]
    model_base_path = cfg["task"].get("model_base_path", "")
    print(f"[APO Agentic] Loading surrogate: {surrogate_name}")
    surrogate = get_surrogate(surrogate_name, model_base_path=model_base_path)

    # Reward function
    reward_name = opt_cfg.get("reward_function", "pareto_hypervolume")
    reward_fn = get_reward_function(reward_name)

    # Initialize history
    history = PromptStateHistory()
    current_state = PromptState.seed(ctx.seed_strategy)
    history.add(current_state)

    # Parent cache
    parent_cache: Dict[str, float] = {}

    # Initialize agents
    print(f"\n[APO Agentic] Initializing agents...")
    worker = WorkerAgent(
        model=model_cfg["worker"],
        api_keys=api_keys,
        task_context=ctx,
        surrogate=surrogate,
        parent_cache=parent_cache,
        temperature=temp_cfg.get("worker", 0.7),
        max_retries_per_batch=opt_cfg.get("max_retries_per_batch", 3),
    )

    critic = CriticAgent(
        model=model_cfg["critic"],
        api_keys=api_keys,
        task_context=ctx,
        reward_fn=reward_fn,
        temperature=temp_cfg.get("critic", 0.3),
    )

    meta = MetaAgent(
        model=model_cfg["meta"],
        api_keys=api_keys,
        task_context=ctx,
        temperature=temp_cfg.get("meta", 0.4),
    )

    # Optimization parameters
    n_epochs = opt_cfg.get("n_outer_epochs", 10)
    n_per_mol = opt_cfg.get("n_per_molecule", 4)
    batch_size = opt_cfg.get("batch_size", 6)
    meta_interval = opt_cfg.get("meta_interval", 3)

    print(f"[APO Agentic] Config: {n_epochs} epochs, {n_per_mol} candidates/parent, batch={batch_size}")
    print(f"[APO Agentic] Models: Worker={model_cfg['worker']}, Critic={model_cfg['critic']}, Meta={model_cfg['meta']}")

    total_usage = aggregate_usage([])
    meta_advice = ""

    # Main optimization loop
    for epoch in range(1, n_epochs + 1):
        print(f"\n{'═'*65}")
        print(f"  EPOCH {epoch}/{n_epochs}  |  Strategy v{current_state.version}")
        print(f"{'═'*65}")

        # Sample batch
        batch = random.sample(all_smiles, min(batch_size, len(all_smiles)))

        # Worker agent: Generate candidates
        candidates, worker_usages = worker.generate(
            strategy=current_state.strategy_text,
            parent_smiles=batch,
            n_per_molecule=n_per_mol,
        )

        print(f"[Worker] Generated {len(candidates)} candidates, "
              f"{sum(1 for c in candidates if c.get('valid'))} valid")

        # Critic agent: Refine strategy
        new_state, analysis, critic_usage = critic.refine(
            candidates=candidates,
            current_state=current_state,
            history=history,
            meta_advice=meta_advice,
        )

        print(f"[Critic] Refined strategy to v{new_state.version}")

        # Calculate reward
        reward = current_state.score or 0.0
        pareto_data = reward_fn.pareto_data([c for c in candidates if c.get("valid")])

        # Log epoch
        # critic/meta usages are already aggregated dicts; worker usages are LLMUsage objects.
        epoch_usage = aggregate_usage(worker_usages)
        epoch_usage = _merge_usage_summaries(epoch_usage, critic_usage)

        logger.log_epoch(
            epoch=epoch,
            prompt_state_dict=current_state.to_dict(),
            candidates=candidates,
            reward=reward,
            pareto_data=pareto_data,
            analysis=analysis,
            meta_advice=meta_advice,
            llm_usage=epoch_usage,
        )

        # Save agent traces
        logger.save_agent_trace(f"worker_epoch_{epoch}", worker._interpretability_trace)
        logger.save_agent_trace(f"critic_epoch_{epoch}", critic._interpretability_trace)

        # Meta agent: Get advice if needed
        if epoch % meta_interval == 0 or epoch == n_epochs:
            meta_advice, meta_usage = meta.get_advice(history, logger.reward_history)
            epoch_usage = _merge_usage_summaries(epoch_usage, meta_usage)
            if meta_advice:
                print(f"[Meta] Advice: {meta_advice[:200]}...")
                logger.save_agent_trace(f"meta_epoch_{epoch}", meta._interpretability_trace)

        total_usage = _merge_usage_summaries(total_usage, epoch_usage)

        # Update history
        history.add(new_state)
        current_state = new_state

    # Final summary
    logger.save_prompt_history(history.to_list())

    print(f"\n{'='*70}")
    print("  AGENTIC OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Run dir   : {logger.run_dir}")
    if history.best:
        score_str = f"{history.best.score:.4f}" if history.best.score is not None else "N/A"
        print(f"  Best score: {score_str} (v{history.best.version})")
    print(f"\n  LLM Usage:")
    print(f"    Calls   : {total_usage.get('total_calls', 0)}")
    print(f"    Tokens  : {total_usage.get('total_tokens', 0):,}")
    for model, stats in total_usage.get("by_model", {}).items():
        print(f"    [{model}] {stats['calls']} calls, {stats['tokens']:,} tokens")

    return str(logger.run_dir)
