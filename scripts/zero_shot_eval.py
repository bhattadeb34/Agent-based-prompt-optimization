"""
Zero-shot evaluation script.

Uses the best strategy prompt discovered during optimization runs
the same Gemini Flash worker LLM (non-agentic, single call) on the TEST set,
and compares against a naive baseline prompt.

Usage:
    python scripts/zero_shot_eval.py \
        --run-dir results/conductivity_experiment/latest \
        --test-csv data/test.csv \
        --api-keys api_keys.txt \
        --out results/zero_shot_eval.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from apo.core.llm_client import call_llm, _inject_api_keys
from apo.surrogates.registry import get_surrogate
from apo.utils.smiles_utils import validate_smiles, canonicalize, compute_similarity

SURROGATE_NAME = "gnn_conductivity"

MODEL_BASE = "/noether/s0/dxb5775/prompt-optimization-work-jan-8/htpmd/ml_models"
WORKER_MODEL = "gemini/gemini-2.0-flash"
N_PER_PARENT = 5  # candidates per parent
SMILES_MARKERS = ["[Cu]", "[Au]"]

BASELINE_STRATEGY = """Generate polymer repeat-unit SMILES that may improve Li-ion conductivity
compared to the parent. Include exactly one [Cu] and one [Au] marker."""


def load_api_keys(path: str) -> Dict[str, str]:
    keys = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                keys[k.strip()] = v.strip().strip("'\"")
    normalised = {
        "GOOGLE_GEMINI_API_KEY": "GOOGLE_API_KEY",
        "openai_GPT_api_key": "OPENAI_API_KEY",
        "CLAUDE_API_KEY": "ANTHROPIC_API_KEY",
    }
    return {normalised.get(k, k): v for k, v in keys.items()}


def load_best_strategy(run_dir: Path) -> Tuple[str, float]:
    """Load best strategy from run_log.jsonl; return (strategy_text, best_reward)."""
    log = run_dir / "run_log.jsonl"
    if not log.exists():
        # Try best_strategy.txt
        txt = run_dir / "best_strategy.txt"
        if txt.exists():
            return txt.read_text(), 0.0
        raise FileNotFoundError(f"No run_log.jsonl or best_strategy.txt in {run_dir}")

    records = []
    with open(log) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    if not records:
        raise ValueError("Run log is empty")

    best = max(records, key=lambda r: r.get("reward", 0.0))
    strategy = best.get("prompt_state", {}).get("strategy_text", "")
    return strategy, best.get("reward", 0.0)


def generate_for_parent(
    parent_smiles: str,
    parent_conductivity: float,
    strategy: str,
    model: str,
    api_keys: Dict,
    n: int = N_PER_PARENT,
) -> List[str]:
    """Call worker LLM. Returns list of raw SMILES strings (may include invalids)."""
    system = (
        "You are an expert polymer chemist. Generate novel polymer electrolyte SMILES "
        "to maximise Li-ion conductivity. Every SMILES must contain exactly one [Cu] and one [Au]."
    )
    user = (
        f"STRATEGY:\n{strategy}\n\n"
        f"PARENT SMILES: {parent_smiles}\n"
        f"PARENT CONDUCTIVITY: {parent_conductivity:.4e} mS/cm\n\n"
        f"Generate {n} candidate polymer SMILES (one per line, no numbering, no explanations)."
    )
    text, _ = call_llm(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        api_keys=api_keys,
        temperature=0.85,
        max_tokens=512,
    )
    # Parse SMILES: one per line, skip empty / non-SMILES lines
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    candidates = []
    for line in lines:
        # Strip leading bullets / numbers
        clean = line.lstrip("•-–*0123456789). ").strip()
        if clean:
            candidates.append(clean)
    return candidates[:n]


def evaluate_candidates(
    parent_smiles: str,
    parent_conductivity: float,
    raw_candidates: List[str],
    predictor,
) -> List[Dict]:
    results = []
    for smi in raw_candidates:
        ok, reason = validate_smiles(smi, required_markers=SMILES_MARKERS)
        if not ok:
            results.append({"smiles": smi, "valid": False, "reason": reason,
                            "conductivity": None, "improvement": None, "similarity": None})
            continue
        canon = canonicalize(smi)
        if not canon:
            results.append({"smiles": smi, "valid": False, "reason": "canon failed",
                            "conductivity": None, "improvement": None, "similarity": None})
            continue
        try:
            pred = predictor.predict([canon])[0]
        except Exception as e:
            results.append({"smiles": canon, "valid": False, "reason": str(e),
                            "conductivity": None, "improvement": None, "similarity": None})
            continue

        improvement = (pred / parent_conductivity) if parent_conductivity > 0 else 0.0
        similarity = compute_similarity(canon, parent_smiles,
                                        similarity_on_repeat_unit=True,
                                        marker_strip_tokens=SMILES_MARKERS)
        results.append({
            "smiles": canon,
            "valid": True,
            "reason": "",
            "conductivity": pred,
            "improvement": improvement,
            "similarity": similarity,
        })
    return results


def run_eval(strategy: str, label: str, test_df, predictor, api_keys, model) -> Dict:
    print(f"\n[ZeroShot] Evaluating strategy: {label}")
    all_results = []
    for i, row in test_df.iterrows():
        parent = row["mol_smiles"]
        parent_cond = row["conductivity"]
        print(f"  Parent {i+1}/{len(test_df)}: {parent[:50]}...")
        raw = generate_for_parent(parent, parent_cond, strategy, model, api_keys)
        evals = evaluate_candidates(parent, parent_cond, raw, predictor)
        for e in evals:
            e["parent_smiles"] = parent
            e["parent_conductivity"] = parent_cond
        all_results.extend(evals)
        time.sleep(0.3)  # gentle rate limiting

    valid = [r for r in all_results if r["valid"]]
    invalid = [r for r in all_results if not r["valid"]]
    improvements = [r["improvement"] for r in valid]
    similarities = [r["similarity"] for r in valid]

    summary = {
        "label": label,
        "n_parents": len(test_df),
        "n_total": len(all_results),
        "n_valid": len(valid),
        "validity_rate": len(valid) / max(len(all_results), 1),
        "avg_improvement": float(np.mean(improvements)) if improvements else 0.0,
        "std_improvement": float(np.std(improvements)) if improvements else 0.0,
        "max_improvement": float(np.max(improvements)) if improvements else 0.0,
        "avg_similarity": float(np.mean(similarities)) if similarities else 0.0,
        "pct_improving": float(np.mean([i > 1.0 for i in improvements])) if improvements else 0.0,
        "failure_reasons": {},
        "results": all_results,
    }
    reason_counts: Dict[str, int] = {}
    for r in invalid:
        k = r.get("reason", "unknown")
        reason_counts[k] = reason_counts.get(k, 0) + 1
    summary["failure_reasons"] = reason_counts
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="results/conductivity_experiment/latest")
    parser.add_argument("--test-csv", default="data/test.csv")
    parser.add_argument("--api-keys", default="api_keys.txt")
    parser.add_argument("--model", default=WORKER_MODEL)
    parser.add_argument("--n-per-parent", type=int, default=N_PER_PARENT)
    parser.add_argument("--out", default="results/zero_shot_eval.json")
    args = parser.parse_args()

    api_keys = load_api_keys(args.api_keys)
    _inject_api_keys(api_keys)

    run_dir = Path(args.run_dir)
    if run_dir.is_symlink():
        run_dir = run_dir.resolve()

    print(f"[ZeroShot] Loading best strategy from {run_dir}...")
    best_strategy, best_reward = load_best_strategy(run_dir)
    print(f"[ZeroShot] Best reward in run: {best_reward:.4f}")
    print(f"[ZeroShot] Strategy preview: {best_strategy[:200]}...")

    test_df = pd.read_csv(args.test_csv)
    print(f"[ZeroShot] Test set: {len(test_df)} molecules")

    print("[ZeroShot] Loading GNN predictor...")
    predictor = get_surrogate(SURROGATE_NAME, model_base_path=MODEL_BASE)

    # Evaluate baseline and optimized strategy
    baseline = run_eval(BASELINE_STRATEGY, "baseline", test_df, predictor, api_keys, args.model)
    optimized = run_eval(best_strategy, "optimized", test_df, predictor, api_keys, args.model)

    output = {
        "model": args.model,
        "run_dir": str(run_dir),
        "best_run_reward": best_reward,
        "best_strategy_text": best_strategy,
        "baseline_strategy_text": BASELINE_STRATEGY,
        "baseline": {k: v for k, v in baseline.items() if k != "results"},
        "optimized": {k: v for k, v in optimized.items() if k != "results"},
        "baseline_results": baseline["results"],
        "optimized_results": optimized["results"],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("ZERO-SHOT EVALUATION RESULTS")
    print(f"{'='*60}")
    for label, res in [("Baseline", baseline), ("Optimized", optimized)]:
        print(f"\n{label}:")
        print(f"  Validity:        {res['n_valid']}/{res['n_total']} ({res['validity_rate']:.1%})")
        print(f"  Avg improvement: {res['avg_improvement']:.3f}×")
        print(f"  % improving:     {res['pct_improving']:.1%}")
        print(f"  Avg similarity:  {res['avg_similarity']:.3f}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
