#!/usr/bin/env python
"""
Export the best strategy prompt from a completed APO run.

The exported prompt can be used directly as a zero-shot generation prompt,
or dropped into a new config as `optimization.seed_strategy` to start the
next run from the best discovered strategy.

Usage:
    python export_best_prompt.py                          # use runs/latest
    python export_best_prompt.py runs/run_20260223_205356
    python export_best_prompt.py --as-yaml                # emit YAML snippet
    python export_best_prompt.py --all                    # show all strategies, ranked
"""
import argparse
import json
import sys
from pathlib import Path


def load_records(log_path: Path):
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def find_best(records):
    """Return (best_record, best_strategy_text, best_reward)."""
    scored = [(r, r.get("reward", 0.0)) for r in records]
    if not scored:
        return None, "", 0.0
    best_record, best_reward = max(scored, key=lambda x: x[1])
    ps = best_record.get("prompt_state", {})
    strategy = ps.get("strategy_text", "")
    return best_record, strategy, best_reward


def main():
    parser = argparse.ArgumentParser(description="Export best strategy prompt from an APO run.")
    parser.add_argument(
        "run_dir", nargs="?", default="runs/latest",
        help="Path to run directory (default: runs/latest)",
    )
    parser.add_argument(
        "--as-yaml", action="store_true",
        help="Emit a YAML snippet you can paste into a config file",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Show ALL strategies ranked by reward",
    )
    parser.add_argument(
        "--out", default=None,
        help="Write exported prompt to this file (default: print to stdout)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if run_dir.is_symlink():
        run_dir = run_dir.resolve()

    log_path = run_dir / "run_log.jsonl"
    if not log_path.exists():
        print(f"ERROR: No run_log.jsonl found at {log_path}", file=sys.stderr)
        sys.exit(1)

    records = load_records(log_path)
    if not records:
        print("ERROR: Log file is empty.", file=sys.stderr)
        sys.exit(1)

    # ── Show all strategies ───────────────────────────────────────────────────
    if args.all:
        print(f"\nAll strategies from {run_dir} ({len(records)} epochs):\n")
        ranked = sorted(records, key=lambda r: r.get("reward", 0.0), reverse=True)
        for i, rec in enumerate(ranked, 1):
            ps = rec.get("prompt_state", {})
            reward = rec.get("reward", 0.0)
            v = ps.get("version", rec.get("epoch", "?"))
            stats = rec.get("stats", {})
            print(f"{'═'*70}")
            print(f"  #{i}  Epoch {rec['epoch']}  Strategy v{v}  Reward={reward:.4f}")
            print(f"  Valid: {stats.get('n_valid','?')}/{stats.get('n_total','?')}  "
                  f"AvgImprovement: {stats.get('avg_improvement','?')}x  "
                  f"AvgSim: {stats.get('avg_similarity','?')}")
            print(f"\n{ps.get('strategy_text','')}\n")
        return

    # ── Best strategy ─────────────────────────────────────────────────────────
    best_record, strategy, reward = find_best(records)
    if not strategy:
        print("ERROR: No strategy found in log.", file=sys.stderr)
        sys.exit(1)

    ps = best_record.get("prompt_state", {})
    stats = best_record.get("stats", {})
    epoch = best_record.get("epoch")
    v = ps.get("version", epoch)

    header = (
        f"# Best strategy from: {run_dir}\n"
        f"# Epoch: {epoch}  |  Strategy v{v}  |  Reward: {reward:.4f}\n"
        f"# Valid: {stats.get('n_valid','?')}/{stats.get('n_total','?')}  "
        f"AvgImprovement: {stats.get('avg_improvement','?')}x  "
        f"AvgSim: {stats.get('avg_similarity','?')}\n"
    )

    if args.as_yaml:
        # YAML snippet for direct paste into next config
        # Indent strategy text as a YAML block scalar
        indented = "\n".join("    " + line for line in strategy.split("\n"))
        output = (
            f"{header}\n"
            f"# Paste this into your YAML config under optimization:\n"
            f"optimization:\n"
            f"  seed_strategy: |\n"
            f"{indented}\n"
        )
    else:
        output = f"{header}\n{strategy}\n"

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(output)
        print(f"Exported to {out_path}")
        print(f"\nBest reward: {reward:.4f} (epoch {epoch}, v{v})")
        print(f"Strategy preview: {strategy[:300]}...")
    else:
        print(output)

    # Always write a copy alongside the run log for convenience
    default_out = run_dir / "best_strategy.txt"
    default_out.write_text(f"{header}\n{strategy}\n")

    yaml_out = run_dir / "best_strategy.yaml"
    indented = "\n".join("    " + line for line in strategy.split("\n"))
    yaml_out.write_text(
        f"# Best strategy from run: {run_dir}\n"
        f"# Reward: {reward:.4f}  Epoch: {epoch}  Strategy version: {v}\n\n"
        f"optimization:\n"
        f"  seed_strategy: |\n"
        f"{indented}\n"
    )

    if not args.out:
        print(f"\n[Saved] {default_out}", file=sys.stderr)
        print(f"[Saved] {yaml_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
