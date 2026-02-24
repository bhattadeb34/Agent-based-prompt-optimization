"""
Main optimization engine — supports both:
  1. AGENT mode:  LLM orchestrator with tool calling drives the loop
  2. LOOP mode:   Classic fixed for-epoch loop (original behaviour, still useful)

The mode is selected via config: `optimization.mode: agent` (default) or `loop`.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .core.llm_client import LLMUsage, aggregate_usage
from .core.prompt_state import PromptState, PromptStateHistory
from .core.reward import get_reward_function
from .logging.knowledge_extractor import extract_knowledge
from .logging.run_logger import RunLogger
from .optimizer.inner_loop import InnerLoop
from .optimizer.meta_loop import MetaLoop
from .optimizer.outer_loop import OuterLoop
from .surrogates.registry import get_surrogate
from .task_context import TaskContext


# ──────────────────────────────────────────────────────────────────────────────
# Data + key loading
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(cfg: Dict) -> List[str]:
    task = cfg["task"]
    data_path = task["dataset"]
    smiles_col = task.get("smiles_column", "mol_smiles")
    sample_size = task.get("sample_size")

    df = pd.read_csv(data_path)
    if smiles_col not in df.columns:
        raise ValueError(
            f"Column '{smiles_col}' not found in {data_path}. "
            f"Available: {list(df.columns)}"
        )
    df = df.dropna(subset=[smiles_col])
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    smiles_list = df[smiles_col].tolist()
    print(f"[APO] Loaded {len(smiles_list)} parent SMILES from {data_path}")
    return smiles_list


def load_api_keys(keys_path: str) -> Dict[str, str]:
    keys: Dict[str, str] = {}
    if not os.path.exists(keys_path):
        print(f"[APO] Warning: API keys file not found at {keys_path}")
        return keys
    with open(keys_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                keys[k.strip()] = v.strip().strip("'\"")
    return keys


def _normalise_api_keys(raw: Dict[str, str]) -> Dict[str, str]:
    mapping = {
        "GOOGLE_GEMINI_API_KEY": "GOOGLE_API_KEY",
        "openai_GPT_api_key": "OPENAI_API_KEY",
        "CLAUDE_API_KEY": "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY": "GOOGLE_API_KEY",
        "OPENAI_API_KEY": "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
    }
    return {mapping.get(k, k): v for k, v in raw.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Component factory
# ──────────────────────────────────────────────────────────────────────────────

def _build_components(cfg: Dict, api_keys: Dict[str, str], all_smiles: List[str]):
    """Build all optimizer components from config."""
    task_cfg = cfg["task"]
    model_cfg = cfg["models"]
    opt_cfg = cfg["optimization"]
    temp_cfg = cfg.get("temperatures", {})

    # Surrogate
    surrogate_name = task_cfg["surrogate"]
    model_base_path = task_cfg.get("model_base_path", "")
    print(f"[APO] Loading surrogate: {surrogate_name}")
    surrogate = get_surrogate(surrogate_name, model_base_path=model_base_path)

    # Task context — ALL domain knowledge lives here
    ctx = TaskContext.from_config(cfg, surrogate=surrogate)
    print(f"[APO] Task: {ctx.property_name} ({ctx.property_units}) | "
          f"molecule_type={ctx.molecule_type}")
    if ctx.smiles_markers:
        print(f"[APO] Required SMILES markers: {ctx.smiles_markers}")
    else:
        print(f"[APO] No SMILES markers required (generic mode)")

    # Reward
    reward_name = opt_cfg.get("reward_function", "pareto_hypervolume")
    reward_kwargs = opt_cfg.get("reward_function_kwargs", {}) or {}
    reward_fn = get_reward_function(reward_name, **reward_kwargs)

    # Run logger
    run_dir_base = cfg.get("output", {}).get("run_dir", "./runs")
    logger = RunLogger(run_dir_base)
    logger.save_config(cfg)

    # History + seed state
    history = PromptStateHistory()
    current_state = PromptState.seed(ctx.seed_strategy)
    history.add(current_state)

    # Loop components
    parent_cache: Dict[str, float] = {}

    inner = InnerLoop(
        surrogate=surrogate,
        task_context=ctx,
        worker_model=model_cfg["worker"],
        api_keys=api_keys,
        temperature=temp_cfg.get("worker", 0.85),
        parent_cache=parent_cache,
    )

    outer = OuterLoop(
        reward_fn=reward_fn,
        task_context=ctx,
        critic_model=model_cfg["critic"],
        api_keys=api_keys,
        temperature=temp_cfg.get("critic", 0.2),
    )

    meta = MetaLoop(
        task_context=ctx,
        meta_model=model_cfg["meta"],
        api_keys=api_keys,
        temperature=temp_cfg.get("meta", 0.3),
        meta_interval=opt_cfg.get("meta_interval", 3),
    )

    return ctx, surrogate, reward_fn, logger, history, current_state, inner, outer, meta


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def run(cfg: Dict, api_keys_path: str = "api_keys.txt") -> str:
    """
    Run prompt optimization from config dict.

    Mode is chosen by cfg['optimization']['mode']:
      'agent'  — LLM orchestrator with tool calling (default, recommended)
      'loop'   — classic fixed-epoch loop
    """
    print("\n" + "=" * 70)
    print("  AGENTIC PROMPT OPTIMISATION FRAMEWORK")
    print("=" * 70)

    raw_keys = load_api_keys(api_keys_path)
    api_keys = _normalise_api_keys(raw_keys)
    print(f"[APO] API keys found: {[k for k, v in api_keys.items() if v]}")

    all_smiles = load_dataset(cfg)

    ctx, surrogate, reward_fn, logger, history, current_state, inner, outer, meta = \
        _build_components(cfg, api_keys, all_smiles)

    opt_cfg = cfg["optimization"]
    mode = opt_cfg.get("mode", "agent")
    model_cfg = cfg["models"]
    out_cfg = cfg.get("output", {})

    if mode == "agentic":
        # NEW: Agentic workflow with ReAct agents, self-correction, debate
        from .agentic_engine import run_agentic_mode
        run_dir = run_agentic_mode(
            cfg=cfg, ctx=ctx, all_smiles=all_smiles,
            logger=logger, api_keys=api_keys,
        )
    elif mode == "agent":
        run_dir = _run_agent_mode(
            cfg=cfg, ctx=ctx, inner=inner, outer=outer, meta=meta,
            logger=logger, history=history, current_state=current_state,
            reward_fn=reward_fn, all_smiles=all_smiles,
            model_cfg=model_cfg, api_keys=api_keys, opt_cfg=opt_cfg,
        )
    else:
        run_dir = _run_loop_mode(
            cfg=cfg, ctx=ctx, inner=inner, outer=outer, meta=meta,
            logger=logger, history=history, current_state=current_state,
            reward_fn=reward_fn, all_smiles=all_smiles,
            opt_cfg=opt_cfg,
        )

    # Knowledge extraction
    if out_cfg.get("extract_knowledge", True):
        knowledge_model = model_cfg.get("knowledge_extractor", model_cfg.get("meta"))
        print("\n[APO] Extracting knowledge...")
        try:
            extract_knowledge(
                run_log_path=str(logger.log_path),
                extractor_model=knowledge_model,
                api_keys=api_keys,
                task_context=ctx,
            )
        except Exception as e:
            print(f"[APO] Warning: knowledge extraction failed: {e}")

    return run_dir


def _run_agent_mode(
    cfg, ctx, inner, outer, meta, logger, history, current_state,
    reward_fn, all_smiles, model_cfg, api_keys, opt_cfg,
) -> str:
    from .agent import OrchestratorAgent

    # tool_budget = 2 tools per epoch × n_epochs (default) + some extra
    n_epochs = opt_cfg.get("n_outer_epochs", 5)
    tool_budget = opt_cfg.get("tool_budget", n_epochs * 3)
    batch_size = opt_cfg.get("batch_size", 4)
    orchestrator_model = model_cfg.get("orchestrator", model_cfg["meta"])

    print(f"\n[APO] Mode: AGENT | Orchestrator: {orchestrator_model} | Budget: {tool_budget} tools")

    agent = OrchestratorAgent(
        inner=inner,
        outer=outer,
        meta=meta,
        logger=logger,
        history=history,
        reward_fn=reward_fn,
        task_context=ctx,
        orchestrator_model=orchestrator_model,
        api_keys=api_keys,
        parent_smiles=all_smiles,
        batch_size=batch_size,
        tool_budget=tool_budget,
    )
    agent._current_state = current_state

    run_dir = agent.run()
    _print_summary(logger, history, agent.total_usage())
    return run_dir


def _run_loop_mode(
    cfg, ctx, inner, outer, meta, logger, history, current_state,
    reward_fn, all_smiles, opt_cfg,
) -> str:
    n_epochs = opt_cfg.get("n_outer_epochs", 10)
    n_per_mol = opt_cfg.get("n_per_molecule", 3)
    batch_size = opt_cfg.get("batch_size", 4)

    print(f"\n[APO] Mode: LOOP | {n_epochs} epochs | {n_per_mol} candidates/parent")

    all_usages: List[LLMUsage] = []
    meta_advice = ""

    for epoch in range(1, n_epochs + 1):
        print(f"\n{'─'*65}")
        print(f"  EPOCH {epoch}/{n_epochs}  |  Strategy v{current_state.version}")
        print(f"{'─'*65}")

        batch = random.sample(all_smiles, min(batch_size, len(all_smiles)))

        candidates, inner_usages = inner.run(current_state.strategy_text, batch, n_per_mol)
        all_usages.extend(inner_usages)

        new_state, analysis, critic_usage = outer.refine(candidates, current_state, history, meta_advice)
        all_usages.append(critic_usage)

        reward = current_state.score or 0.0
        pareto_data = reward_fn.pareto_data([c for c in candidates if c.get("valid")])
        logger.log_epoch(
            epoch=epoch,
            prompt_state_dict=current_state.to_dict(),
            candidates=candidates,
            reward=reward,
            pareto_data=pareto_data,
            analysis=analysis,
            meta_advice=meta_advice,
            llm_usage=aggregate_usage(inner_usages + [critic_usage]),
        )

        meta_advice, meta_usage = meta.maybe_get_advice(history, logger.reward_history, analysis)
        if meta_usage:
            all_usages.append(meta_usage)

        history.add(new_state)
        current_state = new_state

    logger.save_prompt_history(history.to_list())
    _print_summary(logger, history, aggregate_usage(all_usages))
    return str(logger.run_dir)


def _print_summary(logger, history, total_usage):
    print(f"\n{'='*70}")
    print("  OPTIMISATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Run dir   : {logger.run_dir}")
    if history.best:
        score_str = f"{history.best.score:.4f}" if history.best.score is not None else "N/A"
        print(f"  Best score: {score_str} (v{history.best.version})")
        print(f"  Best strat: {history.best.strategy_text[:250]}...")
    print(f"\n  LLM Usage:")
    print(f"    Calls   : {total_usage.get('total_calls', 0)}")
    print(f"    Tokens  : {total_usage.get('total_tokens', 0):,}")
    print(f"    Latency : {total_usage.get('total_latency_s', 0):.1f}s")
    for model, stats in total_usage.get("by_model", {}).items():
        print(f"    [{model}] {stats['calls']} calls, {stats['tokens']:,} tokens")
