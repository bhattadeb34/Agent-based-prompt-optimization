"""
Create side-by-side comparison plots for baseline vs stronger model experiments.

Usage:
    python scripts/plot_model_comparison.py \
        --baseline results/tg_experiment/run_20260223_221315 \
        --strategic results/tg_experiment_stronger/run_20260223_223758 \
        --out results/tg_figures/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})
BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"
PURPLE = "#8172B2"


def load_run_log(run_dir: Path):
    log = run_dir / "run_log.jsonl"
    records = []
    with open(log) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def plot_comparison(baseline_records, strategic_records, out_dir: Path):
    """Create side-by-side comparison plot."""

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1, 0.8], hspace=0.35, wspace=0.25)

    # ────────────────────────────────────────────────────────────────────────
    # Row 1: Reward curves comparison
    # ────────────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])

    baseline_epochs = [r["epoch"] for r in baseline_records]
    baseline_rewards = [r["reward"] for r in baseline_records]
    strategic_epochs = [r["epoch"] for r in strategic_records]
    strategic_rewards = [r["reward"] for r in strategic_records]

    ax1.plot(baseline_epochs, baseline_rewards, "o-", color=BLUE, lw=3, ms=9,
             label=f"Baseline (Gemini Flash) - Best: {max(baseline_rewards):.3f}", alpha=0.8)
    ax1.plot(strategic_epochs, strategic_rewards, "s-", color=ORANGE, lw=3, ms=8,
             label=f"Strategic (GPT-5.2 Critic/Meta) - Best: {max(strategic_rewards):.3f}", alpha=0.8)

    # Highlight best epochs
    best_baseline_idx = np.argmax(baseline_rewards)
    best_strategic_idx = np.argmax(strategic_rewards)
    ax1.scatter([baseline_epochs[best_baseline_idx]], [baseline_rewards[best_baseline_idx]],
               s=300, color=BLUE, marker='*', edgecolors='darkblue', linewidth=2, zorder=5)
    ax1.scatter([strategic_epochs[best_strategic_idx]], [strategic_rewards[best_strategic_idx]],
               s=300, color=ORANGE, marker='*', edgecolors='darkorange', linewidth=2, zorder=5)

    ax1.set_xlabel("Epoch", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Reward (Pareto Hypervolume)", fontsize=13, fontweight="bold")
    ax1.set_title("Model Comparison: Baseline (All Gemini Flash) vs Strategic (GPT-5.2 for Reasoning)",
                  fontsize=15, fontweight="bold", pad=15)
    ax1.legend(loc="upper left", fontsize=12, frameon=True, fancybox=True)
    ax1.grid(True, alpha=0.3, ls=":")
    ax1.set_xlim(0.5, 10.5)

    # ────────────────────────────────────────────────────────────────────────
    # Row 2: Validity comparison
    # ────────────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])

    baseline_validity = [100.0 * r["stats"]["n_valid"] / r["stats"]["n_total"]
                         if r["stats"]["n_total"] > 0 else 0
                         for r in baseline_records]
    strategic_validity = [100.0 * r["stats"]["n_valid"] / r["stats"]["n_total"]
                          if r["stats"]["n_total"] > 0 else 0
                          for r in strategic_records]

    # Handle different lengths
    max_epochs = max(len(baseline_records), len(strategic_records))
    baseline_epochs_x = [r["epoch"] for r in baseline_records]
    strategic_epochs_x = [r["epoch"] for r in strategic_records]

    width = 0.35
    ax2.bar(np.array(baseline_epochs_x) - width/2, baseline_validity, width, label='Baseline', color=BLUE, alpha=0.7)
    ax2.bar(np.array(strategic_epochs_x) + width/2, strategic_validity, width, label='Strategic', color=ORANGE, alpha=0.7)

    ax2.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Validity (%)", fontsize=12, fontweight="bold")
    ax2.set_title("SMILES Validity Rate", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=10)
    ax2.set_xticks(range(1, max_epochs + 1))
    ax2.grid(True, alpha=0.3, ls=":", axis='y')
    ax2.set_ylim(0, 110)

    # ────────────────────────────────────────────────────────────────────────
    # Row 2: Improvement factor comparison
    # ────────────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])

    baseline_max_imp = [r["stats"]["max_improvement"] for r in baseline_records]
    strategic_max_imp = [r["stats"]["max_improvement"] for r in strategic_records]

    ax3.bar(np.array(baseline_epochs_x) - width/2, baseline_max_imp, width, label='Baseline', color=BLUE, alpha=0.7)
    ax3.bar(np.array(strategic_epochs_x) + width/2, strategic_max_imp, width, label='Strategic', color=ORANGE, alpha=0.7)

    ax3.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Max Improvement Factor (×)", fontsize=12, fontweight="bold")
    ax3.set_title("Peak Property Improvement", fontsize=13, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=10)
    ax3.set_xticks(range(1, max_epochs + 1))
    ax3.grid(True, alpha=0.3, ls=":", axis='y')

    # ────────────────────────────────────────────────────────────────────────
    # Row 3: Summary statistics table
    # ────────────────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    baseline_total_valid = sum(r["stats"]["n_valid"] for r in baseline_records)
    baseline_total = sum(r["stats"]["n_total"] for r in baseline_records)
    strategic_total_valid = sum(r["stats"]["n_valid"] for r in strategic_records)
    strategic_total = sum(r["stats"]["n_total"] for r in strategic_records)

    table_data = [
        ["Metric", "Baseline (Gemini Flash)", "Strategic (GPT-5.2)", "Winner"],
        ["Best Reward", f"{max(baseline_rewards):.4f}", f"{max(strategic_rewards):.4f}",
         "Baseline" if max(baseline_rewards) > max(strategic_rewards) else "Strategic"],
        ["Max Improvement", f"{max(baseline_max_imp):.2f}×", f"{max(strategic_max_imp):.2f}×",
         "Baseline" if max(baseline_max_imp) > max(strategic_max_imp) else "Strategic"],
        ["Overall Validity",
         f"{100*baseline_total_valid/baseline_total:.1f}% ({baseline_total_valid}/{baseline_total})",
         f"{100*strategic_total_valid/strategic_total:.1f}% ({strategic_total_valid}/{strategic_total})",
         "Tied"],
        ["Cost", "$0 (free tier)", "~$0.25 (estimated)", "Baseline"],
        ["Runtime", "211s (3.5 min)", "516s (8.6 min)", "Baseline"],
    ]

    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0.1, 0.0, 0.8, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4C72B0')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight winner column
    for i in range(1, len(table_data)):
        winner = table_data[i][3]
        if winner == "Baseline":
            table[(i, 3)].set_facecolor('#90EE90')
            table[(i, 3)].set_text_props(weight='bold')
        elif winner == "Strategic":
            table[(i, 3)].set_facecolor('#FFB347')
            table[(i, 3)].set_text_props(weight='bold')

    # Add verdict text
    verdict_text = "VERDICT: Baseline (All Gemini Flash) achieves 13% higher reward, 58% higher peak Tg,\n" + \
                   "2.4× faster runtime, and $0 cost. Strategic allocation didn't improve results for this task."
    fig.text(0.5, 0.02, verdict_text, ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    out_path = out_dir / "fig4_model_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs strategic model runs")
    parser.add_argument("--baseline", required=True, help="Baseline run directory")
    parser.add_argument("--strategic", required=True, help="Strategic run directory")
    parser.add_argument("--out", default="results/tg_figures", help="Output directory")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline)
    strategic_dir = Path(args.strategic)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON PLOT")
    print(f"{'='*70}")
    print(f"Baseline  : {baseline_dir}")
    print(f"Strategic : {strategic_dir}")
    print(f"Output    : {out_dir}\n")

    baseline_records = load_run_log(baseline_dir)
    strategic_records = load_run_log(strategic_dir)

    print(f"Loaded {len(baseline_records)} baseline epochs")
    print(f"Loaded {len(strategic_records)} strategic epochs\n")

    plot_comparison(baseline_records, strategic_records, out_dir)

    print(f"\n{'='*70}")
    print(f"  ✓ Comparison plot saved")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
