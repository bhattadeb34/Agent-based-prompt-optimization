"""
Main optimization orchestrator — ties together all three loops.

This is the core engine called by run_optimization.py.
"""
from __future__ import annotations

import os
import random
import time
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


def load_dataset(cfg: Dict) -> List[str]:
    """Load parent SMILES from CSV."""
    task = cfg["task"]
    data_path = task["dataset"]
    smiles_col = task.get("smiles_column", "mol_smiles")
    sample_size = task.get("sample_size")

    df = pd.read_csv(data_path)
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in {data_path}. "
                         f"Available: {list(df.columns)}")

    # Drop NaN SMILES
    df = df.dropna(subset=[smiles_col])

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    smiles_list = df[smiles_col].tolist()
    print(f"[APO] Loaded {len(smiles_list)} parent SMILES from {data_path}")
    return smiles_list


def load_api_keys(keys_path: str) -> Dict[str, str]:
    """Load API keys from a simple KEY='value' text file."""
    keys = {}
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
    """Map raw key names to standard names expected by llm_client."""
    mapping = {
        "GOOGLE_GEMINI_API_KEY": "GOOGLE_API_KEY",
        "openai_GPT_api_key": "OPENAI_API_KEY",
        "CLAUDE_API_KEY": "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY": "GOOGLE_API_KEY",
        "OPENAI_API_KEY": "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
    }
    norm = {}
    for k, v in raw.items():
        std_key = mapping.get(k, k)
        norm[std_key] = v
    return norm


def run(cfg: Dict, api_keys_path: str = "api_keys.txt", resume_run_id: Optional[str] = None) -> str:
    """
    Run the full agentic prompt optimisation loop.

    Args:
        cfg: Parsed YAML config dict.
        api_keys_path: Path to api_keys.txt.
        resume_run_id: Optional run ID to resume from (not yet implemented).

    Returns:
        Path to the run directory.
    """
    print("\n" + "=" * 70)
    print("  AGENTIC PROMPT OPTIMISATION FRAMEWORK")
    print("=" * 70)

    # ── Load API keys ────────────────────────────────────────────────────────
    raw_keys = load_api_keys(api_keys_path)
    api_keys = _normalise_api_keys(raw_keys)
    print(f"[APO] Loaded API keys: {[k for k, v in api_keys.items() if v]}")

    # ── Config extraction ────────────────────────────────────────────────────
    task_cfg = cfg["task"]
    model_cfg = cfg["models"]
    opt_cfg = cfg["optimization"]
    temp_cfg = cfg.get("temperatures", {})
    out_cfg = cfg.get("output", {})

    property_name_key = task_cfg["property"]
    surrogate_name = task_cfg["surrogate"]
    model_base_path = task_cfg.get("model_base_path", "")

    worker_model = model_cfg["worker"]
    critic_model = model_cfg["critic"]
    meta_model = model_cfg["meta"]
    knowledge_model = model_cfg.get("knowledge_extractor", meta_model)

    n_epochs = opt_cfg["n_outer_epochs"]
    n_per_mol = opt_cfg["n_per_molecule"]
    batch_size = opt_cfg.get("batch_size", 4)
    meta_interval = opt_cfg.get("meta_interval", 3)
    reward_name = opt_cfg.get("reward_function", "pareto_hypervolume")
    reward_kwargs = opt_cfg.get("reward_function_kwargs", {})
    seed_strategy = opt_cfg.get("seed_strategy", "Generate polymer SMILES with higher target property.")

    t_worker = temp_cfg.get("worker", 0.85)
    t_critic = temp_cfg.get("critic", 0.2)
    t_meta = temp_cfg.get("meta", 0.3)

    # ── Initialise surrogate ─────────────────────────────────────────────────
    print(f"\n[APO] Initialising surrogate: {surrogate_name}")
    surrogate = get_surrogate(surrogate_name, model_base_path=model_base_path)
    property_name = surrogate.property_name
    property_units = surrogate.property_units
    print(f"[APO] Surrogate: {property_name} ({property_units})")

    # ── Initialise reward ────────────────────────────────────────────────────
    reward_fn = get_reward_function(reward_name, **reward_kwargs)
    print(f"[APO] Reward function: {reward_name}")

    # ── Load dataset ─────────────────────────────────────────────────────────
    all_smiles = load_dataset(cfg)

    # ── Initialise logger ────────────────────────────────────────────────────
    run_dir_base = out_cfg.get("run_dir", "./runs")
    logger = RunLogger(run_dir_base)
    logger.save_config(cfg)

    # ── Initialise prompt state history ──────────────────────────────────────
    history = PromptStateHistory()
    current_state = PromptState.seed(seed_strategy)
    history.add(current_state)
    print(f"\n[APO] Seed strategy:\n  {seed_strategy[:200]}...")

    # ── Initialise optimizer components ─────────────────────────────────────
    parent_cache: Dict[str, float] = {}

    inner = InnerLoop(
        surrogate=surrogate,
        worker_model=worker_model,
        api_keys=api_keys,
        temperature=t_worker,
        parent_cache=parent_cache,
    )

    outer = OuterLoop(
        reward_fn=reward_fn,
        critic_model=critic_model,
        api_keys=api_keys,
        temperature=t_critic,
        property_name=property_name,
        property_units=property_units,
    )

    meta = MetaLoop(
        meta_model=meta_model,
        api_keys=api_keys,
        temperature=t_meta,
        meta_interval=meta_interval,
        property_name=property_name,
        property_units=property_units,
    )

    all_usages: List[LLMUsage] = []
    meta_advice = ""

    # ── Main optimisation loop ───────────────────────────────────────────────
    print(f"\n[APO] Starting {n_epochs} epoch optimisation loop...\n")
    for epoch in range(1, n_epochs + 1):
        print(f"\n{'─' * 70}")
        print(f"  EPOCH {epoch}/{n_epochs}  |  Strategy v{current_state.version}")
        print(f"{'─' * 70}")

        # Sample a batch of parent SMILES
        batch = random.sample(all_smiles, min(batch_size, len(all_smiles)))

        # 1. INNER LOOP — generate candidates
        t0 = time.time()
        candidates, inner_usages = inner.run(
            strategy=current_state.strategy_text,
            parent_smiles=batch,
            n_per_molecule=n_per_mol,
        )
        inner_time = time.time() - t0
        all_usages.extend(inner_usages)

        # 2. OUTER LOOP — compute reward, refine strategy
        new_state, analysis, critic_usage = outer.refine(
            candidates=candidates,
            current_state=current_state,
            history=history,
            meta_advice=meta_advice,
        )
        all_usages.append(critic_usage)

        reward = current_state.score or 0.0
        pareto_data = reward_fn.pareto_data([c for c in candidates if c.get("valid")])

        # 3. LOG epoch
        epoch_usage = aggregate_usage(inner_usages + [critic_usage])
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

        # 4. META LOOP — periodic strategic guidance
        meta_advice, meta_usage = meta.maybe_get_advice(
            history=history,
            reward_history=logger.reward_history,
            analysis=analysis,
        )
        if meta_usage:
            all_usages.append(meta_usage)
            if meta_advice:
                print(f"\n[MetaAdvice] → {meta_advice[:200]}...")

        # 5. Advance state
        history.add(new_state)
        current_state = new_state

    # ── Save prompt history ──────────────────────────────────────────────────
    logger.save_prompt_history(history.to_list())

    # ── Print final summary ──────────────────────────────────────────────────
    total_usage = aggregate_usage(all_usages)
    print(f"\n{'=' * 70}")
    print("  OPTIMISATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Run directory : {logger.run_dir}")
    print(f"  Epochs        : {n_epochs}")
    if history.best:
        print(f"  Best reward   : {history.best.score:.4f} (v{history.best.version})")
        print(f"  Best strategy :\n    {history.best.strategy_text[:300]}...")
    print(f"\n  LLM Usage Summary:")
    print(f"    Total calls   : {total_usage.get('total_calls', 0)}")
    print(f"    Total tokens  : {total_usage.get('total_tokens', 0):,}")
    print(f"    Total latency : {total_usage.get('total_latency_s', 0):.1f}s")
    for model, stats in total_usage.get("by_model", {}).items():
        print(f"    [{model}] {stats['calls']} calls, {stats['tokens']:,} tokens")

    # ── Knowledge extraction ─────────────────────────────────────────────────
    if out_cfg.get("extract_knowledge", True):
        print(f"\n[APO] Extracting knowledge from run log...")
        try:
            knowledge_path = extract_knowledge(
                run_log_path=str(logger.log_path),
                extractor_model=knowledge_model,
                api_keys=api_keys,
                property_name=property_name,
                property_units=property_units,
            )
            print(f"[APO] Knowledge saved to: {knowledge_path}")
        except Exception as e:
            print(f"[APO] Warning: knowledge extraction failed: {e}")

    return str(logger.run_dir)
