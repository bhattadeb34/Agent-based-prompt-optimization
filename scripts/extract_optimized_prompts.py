"""
Extract and analyze optimized prompts from APO runs.

This script extracts:
1. All strategy evolution (prompt history)
2. Best strategies by epoch
3. Meta-strategist interventions
4. Critic's analysis and rationale for each refinement
5. Strategy patterns and trends
"""
import json
import sys
from pathlib import Path
from typing import Dict, List


def load_run_data(run_dir: str) -> Dict:
    """Load all data from a run directory."""
    run_path = Path(run_dir)

    data = {
        "run_dir": run_dir,
        "run_id": run_path.name,
        "epochs": [],
        "prompt_history": [],
        "best_epoch": None,
        "config": {},
    }

    # Load config
    config_path = run_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            data["config"] = json.load(f)

    # Load run log
    log_path = run_path / "run_log.jsonl"
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                if line.strip():
                    data["epochs"].append(json.loads(line))

    # Load prompt history
    history_path = run_path / "prompt_history.json"
    if history_path.exists():
        with open(history_path) as f:
            data["prompt_history"] = json.load(f)

    # Find best epoch
    if data["epochs"]:
        data["best_epoch"] = max(data["epochs"], key=lambda e: e.get("reward", 0))

    return data


def extract_strategy_evolution(data: Dict) -> List[Dict]:
    """Extract strategy evolution with rationale."""
    evolution = []

    for state in data["prompt_history"]:
        evolution.append({
            "version": state["version"],
            "strategy_text": state["strategy_text"],
            "rationale": state.get("rationale", ""),
            "parent_version": state.get("parent_version"),
            "model_used": state.get("model_used", ""),
        })

    return evolution


def extract_meta_interventions(data: Dict) -> List[Dict]:
    """Extract meta-strategist interventions."""
    interventions = []

    for epoch_data in data["epochs"]:
        if epoch_data.get("meta_advice"):
            interventions.append({
                "epoch": epoch_data["epoch"],
                "advice": epoch_data["meta_advice"],
                "reward_history": data["epochs"][:epoch_data["epoch"]],
                "reward_before": epoch_data.get("reward", 0),
            })

    return interventions


def extract_critic_analysis(data: Dict) -> List[Dict]:
    """Extract critic's analysis for each epoch."""
    analyses = []

    for epoch_data in data["epochs"]:
        if epoch_data.get("analysis"):
            analyses.append({
                "epoch": epoch_data["epoch"],
                "reward": epoch_data.get("reward", 0),
                "analysis": epoch_data["analysis"],
                "strategy_version": epoch_data["prompt_state"].get("version"),
            })

    return analyses


def generate_report(data: Dict, output_path: str):
    """Generate comprehensive prompt optimization report."""
    evolution = extract_strategy_evolution(data)
    interventions = extract_meta_interventions(data)
    analyses = extract_critic_analysis(data)

    report = []
    report.append("=" * 80)
    report.append(f"  OPTIMIZED PROMPTS & DECISIONS REPORT")
    report.append("=" * 80)
    report.append(f"Run: {data['run_id']}")
    report.append(f"Task: {data['config'].get('task', {}).get('name', 'Unknown')}")
    report.append(f"Total Epochs: {len(data['epochs'])}")
    if data["best_epoch"]:
        report.append(f"Best Reward: {data['best_epoch'].get('reward', 0):.4f} (Epoch {data['best_epoch']['epoch']})")
    report.append("")

    # Strategy Evolution
    report.append("=" * 80)
    report.append("  STRATEGY EVOLUTION")
    report.append("=" * 80)
    report.append("")

    for ev in evolution:
        report.append(f"VERSION {ev['version']} (by {ev['model_used']})")
        report.append("-" * 80)
        report.append(f"Strategy:")
        report.append(f"{ev['strategy_text']}")
        report.append("")
        if ev['rationale']:
            report.append(f"Rationale: {ev['rationale']}")
            report.append("")
        report.append("")

    # Meta Interventions
    if interventions:
        report.append("=" * 80)
        report.append("  META-STRATEGIST INTERVENTIONS")
        report.append("=" * 80)
        report.append("")

        for intervention in interventions:
            report.append(f"EPOCH {intervention['epoch']}")
            report.append("-" * 80)
            report.append(f"Advice: {intervention['advice']}")
            report.append(f"Reward Before: {intervention['reward_before']:.4f}")
            report.append("")

    # Critic Analysis
    report.append("=" * 80)
    report.append("  CRITIC ANALYSIS (Selected Epochs)")
    report.append("=" * 80)
    report.append("")

    # Show first, middle, and last epochs
    selected_analyses = []
    if len(analyses) > 0:
        selected_analyses.append(analyses[0])  # First
    if len(analyses) > 5:
        selected_analyses.append(analyses[len(analyses)//2])  # Middle
    if len(analyses) > 1:
        selected_analyses.append(analyses[-1])  # Last

    for analysis in selected_analyses:
        report.append(f"EPOCH {analysis['epoch']} (Strategy v{analysis['strategy_version']}, Reward: {analysis['reward']:.4f})")
        report.append("-" * 80)

        ana = analysis["analysis"]
        if ana.get("pareto_insights"):
            report.append("Pareto Insights:")
            for insight in ana["pareto_insights"]:
                report.append(f"  • {insight}")
            report.append("")

        if ana.get("failure_patterns"):
            report.append("Failure Patterns:")
            for pattern in ana["failure_patterns"]:
                report.append(f"  • {pattern}")
            report.append("")

        if ana.get("exploration_targets"):
            report.append("Exploration Targets:")
            for target in ana["exploration_targets"]:
                report.append(f"  • {target}")
            report.append("")

        report.append("")

    # Best Strategy
    if data["best_epoch"]:
        report.append("=" * 80)
        report.append("  BEST STRATEGY")
        report.append("=" * 80)
        report.append("")
        best_state = data["best_epoch"]["prompt_state"]
        report.append(f"Epoch: {data['best_epoch']['epoch']}")
        report.append(f"Version: {best_state['version']}")
        report.append(f"Reward: {data['best_epoch']['reward']:.4f}")
        report.append("")
        report.append(f"Strategy:")
        report.append(f"{best_state['strategy_text']}")
        report.append("")
        if best_state.get('rationale'):
            report.append(f"Rationale: {best_state['rationale']}")
        report.append("")

    # Save report
    report_text = "\n".join(report)
    with open(output_path, "w") as f:
        f.write(report_text)

    print(f"Report saved to: {output_path}")
    return report_text


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_optimized_prompts.py <run_dir>")
        print("Example: python extract_optimized_prompts.py results/tg_experiment_full/latest")
        sys.exit(1)

    run_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else f"{run_dir}/OPTIMIZED_PROMPTS_REPORT.txt"

    print(f"Loading run data from: {run_dir}")
    data = load_run_data(run_dir)

    print(f"Generating report...")
    report = generate_report(data, output_path)

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"Total strategies generated: {len(data['prompt_history'])}")
    print(f"Total epochs: {len(data['epochs'])}")
    print(f"Meta interventions: {len(extract_meta_interventions(data))}")
    if data["best_epoch"]:
        print(f"Best reward: {data['best_epoch']['reward']:.4f} (Epoch {data['best_epoch']['epoch']})")


if __name__ == "__main__":
    main()
