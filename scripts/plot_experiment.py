"""
Plot all experiment results:
  Fig 1: Epoch vs performance (dual y-axis: conductivity improvement + Tanimoto)
  Fig 2: Train/test conductivity distribution
  Fig 3: Optimized prompt text visualization
  Fig 4: Zero-shot comparison (baseline vs optimized)

Usage:
    python scripts/plot_experiment.py \
        --run-dir results/conductivity_experiment/latest \
        --zero-shot results/zero_shot_eval.json \
        --train-csv data/train.csv \
        --test-csv data/test.csv \
        --out results/figures/
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Epoch performance (dual y-axis)
# ─────────────────────────────────────────────────────────────────────────────

def plot_epoch_performance(records, out_dir: Path):
    epochs         = [r["epoch"] for r in records]
    avg_imp        = [r["stats"]["avg_improvement"] for r in records]
    max_imp        = [r["stats"]["max_improvement"] for r in records]
    avg_sim        = [r["stats"]["avg_similarity"]  for r in records]
    rewards        = [r["reward"] for r in records]
    n_valid        = [r["stats"]["n_valid"] for r in records]
    n_total        = [r["stats"]["n_total"] for r in records]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    # Conductivity improvement (left axis)
    l1, = ax1.plot(epochs, avg_imp, "o-", color=BLUE, lw=2, ms=7, label="Avg improvement (×)")
    l2, = ax1.plot(epochs, max_imp, "s--", color=BLUE, lw=1.5, ms=6, alpha=0.6, label="Max improvement (×)")
    ax1.fill_between(epochs, avg_imp, 1.0, alpha=0.12, color=BLUE)
    ax1.axhline(1.0, color="grey", lw=0.8, ls="--", alpha=0.5)

    # Tanimoto similarity (right axis)
    l3, = ax2.plot(epochs, avg_sim, "^-", color=ORANGE, lw=2, ms=7, label="Avg Tanimoto sim")
    ax2.fill_between(epochs, avg_sim, alpha=0.10, color=ORANGE)

    # Validity as bar background
    validity = [v / t for v, t in zip(n_valid, n_total)]
    ax1.bar(epochs, [max(max_imp) * 1.05] * len(epochs), width=0.8,
            color="lightgrey", alpha=0.25, zorder=0, label="_nolegend_")
    for i, (e, v) in enumerate(zip(epochs, validity)):
        ax1.text(e, 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=7.5,
                 color="dimgrey", rotation=0)

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Conductivity improvement factor (×)", color=BLUE, fontsize=11)
    ax2.set_ylabel("Avg Tanimoto similarity", color=ORANGE, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax2.set_ylim(0, 1.05)

    handles = [l1, l2, l3]
    ax1.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.7)
    ax1.set_title("APO Optimization Progress — Conductivity vs. Structural Similarity",
                  fontsize=12, pad=12)
    ax1.set_xticks(epochs)

    plt.tight_layout()
    p = out_dir / "fig1_epoch_performance.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved {p}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Train / Test distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_distribution(train_csv, test_csv, out_dir: Path):
    train = pd.read_csv(train_csv)
    test  = pd.read_csv(test_csv)
    log_train = np.log10(train["conductivity"])
    log_test  = np.log10(test["conductivity"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: overlapping histograms
    ax = axes[0]
    bins = np.linspace(min(log_train.min(), log_test.min()),
                       max(log_train.max(), log_test.max()), 25)
    ax.hist(log_train, bins=bins, color=BLUE,   alpha=0.65, edgecolor="white",
            label=f"Train (n={len(train)})")
    ax.hist(log_test,  bins=bins, color=ORANGE, alpha=0.65, edgecolor="white",
            label=f"Test  (n={len(test)})")
    ax.axvline(log_train.mean(), color=BLUE,   lw=2, ls="--", alpha=0.8)
    ax.axvline(log_test.mean(),  color=ORANGE, lw=2, ls="--", alpha=0.8)
    ax.set_xlabel("log₁₀(Conductivity / mS cm⁻¹)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Train / Test Conductivity Distribution", fontsize=11)
    ax.legend(fontsize=9)

    # Right: violin / box
    ax2 = axes[1]
    parts = ax2.violinplot([log_train, log_test], positions=[1, 2],
                           showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], [BLUE, ORANGE]):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    parts["cmedians"].set_colors(["white", "white"])
    parts["cmedians"].set_linewidth(2)
    for key in ["cbars", "cmins", "cmaxes"]:
        parts[key].set_colors(["#555", "#555"])

    ax2.set_xticks([1, 2])
    ax2.set_xticklabels([f"Train\n(n={len(train)})", f"Test\n(n={len(test)})"])
    ax2.set_ylabel("log₁₀(Conductivity / mS cm⁻¹)", fontsize=11)
    ax2.set_title("Conductivity Violin Plot", fontsize=11)

    # Stats annotation
    for x, vals, col in [(1, log_train, BLUE), (2, log_test, ORANGE)]:
        ax2.text(x, vals.max() + 0.02,
                 f"μ={vals.mean():.2f}\nσ={vals.std():.2f}",
                 ha="center", va="bottom", fontsize=8.5, color=col)

    plt.tight_layout()
    p = out_dir / "fig2_distribution.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved {p}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Optimized prompt visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_best_prompt(records, out_dir: Path):
    best = max(records, key=lambda r: r.get("reward", 0.0))
    strategy = best.get("prompt_state", {}).get("strategy_text", "")
    epoch = best["epoch"]
    reward = best.get("reward", 0.0)

    # Wrap text
    wrapped = textwrap.fill(strategy, width=90)

    fig, ax = plt.subplots(figsize=(11, max(4, min(10, len(strategy) // 80))))
    ax.axis("off")

    # Title box
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("#F7F9FC")

    # Header
    header = (f"🏆  Best Strategy  |  Epoch {epoch}  |  Reward {reward:.4f}")
    ax.text(0.5, 0.97, header, transform=ax.transAxes, fontsize=13, fontweight="bold",
            ha="center", va="top", color="#1a1a2e")

    # Divider
    # Divider line
    ax.plot([0.02, 0.98], [0.93, 0.93], color="#4C72B0", linewidth=1.5,
            transform=ax.transAxes, clip_on=False)

    # Strategy text
    ax.text(0.03, 0.88, wrapped, transform=ax.transAxes, fontsize=9.5,
            ha="left", va="top", color="#2d2d2d",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#CBD5E1", linewidth=1.2))

    # Reward history mini-bar below
    rewards = [r.get("reward", 0.0) for r in records]
    epochs  = [r["epoch"] for r in records]

    inset = fig.add_axes([0.03, 0.02, 0.94, 0.10])
    colors = [GREEN if r == max(rewards) else BLUE for r in rewards]
    inset.bar(epochs, rewards, color=colors, width=0.6, alpha=0.85, edgecolor="white")
    inset.set_title("Reward per epoch", fontsize=8, pad=3)
    inset.set_xticks(epochs)
    inset.tick_params(labelsize=7.5)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    inset.set_ylabel("Reward", fontsize=8)

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    p = out_dir / "fig3_best_prompt.png"
    fig.savefig(p, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved {p}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Zero-shot comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_zero_shot(zs_path: Path, out_dir: Path):
    if not zs_path.exists():
        print(f"[Plot] Skipping Fig 4 — {zs_path} not found")
        return None

    with open(zs_path) as f:
        data = json.load(f)

    baseline  = data["baseline"]
    optimized = data["optimized"]

    def extract_arrays(side_data):
        recs = data[f"{side_data['label']}_results"]
        valid = [r for r in recs if r.get("valid")]
        return (
            [r["improvement"] for r in valid],
            [r["similarity"]  for r in valid],
            [r["conductivity"] for r in valid],
        )

    b_imp, b_sim, b_cond = extract_arrays(baseline)
    o_imp, o_sim, o_cond = extract_arrays(optimized)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: improvement factor distribution
    ax = axes[0]
    all_imp = b_imp + o_imp
    bins = np.linspace(min(all_imp + [0.8]), max(all_imp + [1.0]) * 1.05, 20)
    ax.hist(b_imp, bins=bins, color=BLUE,   alpha=0.7, label="Baseline", edgecolor="white")
    ax.hist(o_imp, bins=bins, color=GREEN,  alpha=0.7, label="Optimized", edgecolor="white")
    ax.axvline(1.0, color="grey", ls="--", lw=1.2, label="No change")
    ax.set_xlabel("Conductivity improvement factor (×)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("A.  Improvement Distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel B: scatter improvement vs similarity
    ax = axes[1]
    ax.scatter(b_sim, b_imp, color=BLUE,  alpha=0.6, s=50, label="Baseline",  edgecolors="white")
    ax.scatter(o_sim, o_imp, color=GREEN, alpha=0.7, s=55, label="Optimized", edgecolors="white", marker="^")
    ax.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.8)
    ax.set_xlabel("Tanimoto similarity to parent", fontsize=10)
    ax.set_ylabel("Conductivity improvement (×)", fontsize=10)
    ax.set_title("B.  Improvement vs. Similarity", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel C: summary bar chart
    ax = axes[2]
    metrics = ["Avg improvement (×)", "Avg similarity", "Valid %", "% Improving"]
    b_vals  = [baseline["avg_improvement"], baseline["avg_similarity"],
               baseline["validity_rate"],   baseline["pct_improving"]]
    o_vals  = [optimized["avg_improvement"], optimized["avg_similarity"],
               optimized["validity_rate"],   optimized["pct_improving"]]
    x = np.arange(len(metrics))
    w = 0.35
    bars1 = ax.bar(x - w/2, b_vals, w, label="Baseline",  color=BLUE,  alpha=0.8, edgecolor="white")
    bars2 = ax.bar(x + w/2, o_vals, w, label="Optimized", color=GREEN, alpha=0.8, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8.5, rotation=12, ha="right")
    ax.set_title("C.  Summary Metrics", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    for bar in [*bars1, *bars2]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7.5)

    fig.suptitle(f"Zero-Shot Evaluation  —  Baseline vs APO-Optimized Prompt  ({data['model']})",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    p = out_dir / "fig4_zero_shot_comparison.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved {p}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir",    default="results/conductivity_experiment/latest")
    parser.add_argument("--zero-shot",  default="results/zero_shot_eval.json")
    parser.add_argument("--train-csv",  default="data/train.csv")
    parser.add_argument("--test-csv",   default="data/test.csv")
    parser.add_argument("--out",        default="results/figures")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(args.run_dir)
    if run_dir.is_symlink():
        run_dir = run_dir.resolve()

    records = load_run_log(run_dir)
    print(f"[Plot] Loaded {len(records)} epoch records from {run_dir}")

    plot_epoch_performance(records, out_dir)
    plot_distribution(args.train_csv, args.test_csv, out_dir)
    plot_best_prompt(records, out_dir)
    plot_zero_shot(Path(args.zero_shot), out_dir)

    print(f"\n[Plot] All figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
