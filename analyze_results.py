#!/usr/bin/env python3
"""
analyze_results.py — Post-run analysis CLI.

Reads a run_log.jsonl file and prints:
- Reward curve
- Best strategy found
- Pareto front stats
- Top candidates
- Token usage summary

Usage:
    python analyze_results.py runs/run_20260223_200000/run_log.jsonl
    python analyze_results.py runs/latest/run_log.jsonl --extract-knowledge
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Analyse APO run results")
    parser.add_argument("log_path", help="Path to run_log.jsonl")
    parser.add_argument(
        "--extract-knowledge",
        action="store_true",
        help="Re-run knowledge extraction (requires API keys)",
    )
    parser.add_argument("--api-keys", default="api_keys.txt")
    parser.add_argument("--knowledge-model", default="openai/gpt-4o")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"ERROR: {log_path} not found")
        return

    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        print("No records found.")
        return

    print(f"\n{'='*65}")
    print(f"  RUN ANALYSIS: {log_path.parent.name}")
    print(f"{'='*65}")
    print(f"  Epochs logged    : {len(records)}")

    # Reward curve
    rewards = [r.get("reward", 0.0) for r in records]
    print(f"\n  Reward Curve:")
    for i, (rec, rw) in enumerate(zip(records, rewards), 1):
        hv = rec.get("pareto_data", {}).get("hypervolume", 0)
        print(f"    Epoch {i:2d}: reward={rw:.4f}  pareto_hv={hv:.4f}")

    best_idx = rewards.index(max(rewards))
    best_rec = records[best_idx]
    best_ps = best_rec.get("prompt_state", {})
    print(f"\n  Best epoch: {best_idx+1} (reward={max(rewards):.4f})")
    print(f"  Best strategy (v{best_ps.get('version', '?')}):")
    strat = best_ps.get("strategy_text", "")
    print(f"    {strat[:400]}{'...' if len(strat) > 400 else ''}")

    # All candidates
    all_candidates = []
    for rec in records:
        all_candidates.extend(rec.get("candidates", []))
    valid = [c for c in all_candidates if c.get("valid")]
    print(f"\n  Total candidates : {len(all_candidates)}")
    print(f"  Valid candidates : {len(valid)}")
    print(f"  Validity rate    : {len(valid)/max(len(all_candidates),1):.1%}")

    # Top 5
    top5 = sorted(valid, key=lambda c: c.get("improvement_factor", 0), reverse=True)[:5]
    print(f"\n  Top 5 by improvement factor:")
    for i, c in enumerate(top5, 1):
        print(f"    {i}. {c.get('child_smiles','')[:55]}...")
        print(f"       improvement={c.get('improvement_factor',0):.2f}x  "
              f"similarity={c.get('similarity',0):.3f}  "
              f"property={c.get('child_property',0):.3e}")

    # LLM usage
    total_tokens = sum(
        rec.get("llm_usage", {}).get("total_tokens", 0) for rec in records
    )
    total_calls = sum(
        rec.get("llm_usage", {}).get("total_calls", 0) for rec in records
    )
    total_latency = sum(
        rec.get("llm_usage", {}).get("total_latency_s", 0) for rec in records
    )
    print(f"\n  LLM Usage:")
    print(f"    Total calls   : {total_calls}")
    print(f"    Total tokens  : {total_tokens:,}")
    print(f"    Total latency : {total_latency:.1f}s")

    # Knowledge file
    knowledge_file = log_path.parent / "knowledge.md"
    if knowledge_file.exists():
        print(f"\n  Knowledge doc  : {knowledge_file}")
    elif args.extract_knowledge:
        print(f"\n  Extracting knowledge...")
        from apo.engine import load_api_keys, _normalise_api_keys
        from apo.logging.knowledge_extractor import extract_knowledge
        api_keys = _normalise_api_keys(load_api_keys(args.api_keys))
        # Infer property name from first record
        ps = records[0].get("prompt_state", {})
        prop_name = "Property"
        extract_knowledge(
            run_log_path=args.log_path,
            extractor_model=args.knowledge_model,
            api_keys=api_keys,
            property_name=prop_name,
        )

    print()


if __name__ == "__main__":
    main()
