"""
Knowledge Extractor: post-run LLM analysis that distills what the agent learned.

Inspired by GEPA's reflection approach — reads the FULL run trace (all strategies,
rewards, Pareto data, failure patterns) and synthesises human-readable insights.
Output is a structured knowledge.md document.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from ..core.llm_client import call_llm


_EXTRACTOR_SYSTEM = """\
You are a senior computational scientist analysing the results of an AI-driven \
polymer discovery experiment. You have access to the complete trace of an \
iterative prompt optimisation run: every strategy tried, every molecule generated, \
every success and failure.

Your task is to synthesise this data into a concise, high-value knowledge document \
that a human chemist could use to:
1. Understand what the AI discovered about the structure-property relationship
2. Start a future optimisation run from a much better position
3. Identify promising chemical hypotheses for experimental validation

Be specific. Cite actual SMILES fragments and property values where relevant.
"""

_EXTRACTOR_USER_TEMPLATE = """\
EXPERIMENT SUMMARY
==================
Property Optimised: {property_name} ({property_units})
Total Epochs: {n_epochs}
Total Candidates Generated: {n_total_candidates}
Valid Candidates: {n_valid_candidates}
Validity Rate: {validity_rate:.1%}

REWARD TRAJECTORY:
{reward_trajectory}

STRATEGY EVOLUTION (all {n_strategies} versions):
{strategy_evolution}

TOP PERFORMING CANDIDATES (best improvement × similarity):
{top_candidates}

WORST FAILURES (most common invalid SMILES reasons):
{failure_summary}

CRITIC ANALYSES (chemical insights from each epoch):
{critic_analyses}

META-STRATEGIST ADVICE GIVEN:
{meta_advice_log}

Based on this complete run, produce a knowledge document with the following sections.
Return a JSON object:
{{
  "executive_summary": "2-3 sentence overview of what was achieved and key finding",
  "best_strategy_found": "The exact strategy text that achieved the highest reward",
  "best_strategy_score": <float>,
  "what_worked": [
    "Specific structural modification X consistently improved property by Y factor because...",
    "..."
  ],
  "what_failed": [
    "Approach X failed because... (N occurrences)",
    "..."
  ],
  "key_chemical_insights": [
    "Insight about structure-property relationship 1",
    "Insight 2"
  ],
  "smiles_motifs_to_explore": [
    "CC(O[Cu])...[Au] — ether oxygen density improves coordination",
    "..."
  ],
  "recommended_next_strategy": "Concrete starting strategy for the next optimisation run...",
  "prompt_evolution_table": [
    {{"version": 0, "score": null, "key_change": "seed", "preview": "..."}},
    {{"version": 1, "score": 1.23, "key_change": "added ether groups", "preview": "..."}}
  ],
  "lessons_for_future_runs": [
    "Increase n_per_molecule to at least 5 for better Pareto coverage",
    "..."
  ]
}}
Return ONLY the JSON object.
"""


def extract_knowledge(
    run_log_path: str,
    extractor_model: str,
    api_keys: Optional[Dict[str, str]] = None,
    property_name: str = "Property",
    property_units: str = "",
) -> str:
    """
    Read a run_log.jsonl file and produce a knowledge.md document.

    Returns:
        Path to the written knowledge.md file.
    """
    log_path = Path(run_log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Run log not found: {run_log_path}")

    # Load all epoch records
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
        print("[KnowledgeExtractor] No records found in run log.")
        return ""

    # Aggregate data
    all_candidates = []
    all_strategies = []
    all_analyses = []
    all_meta_advices = []
    rewards = []

    for rec in records:
        all_candidates.extend(rec.get("candidates", []))
        ps = rec.get("prompt_state", {})
        all_strategies.append({
            "version": ps.get("version", rec["epoch"]),
            "score": rec.get("reward"),
            "strategy_text": ps.get("strategy_text", ""),
            "rationale": ps.get("rationale", ""),
        })
        if rec.get("analysis"):
            all_analyses.append(rec["analysis"])
        if rec.get("meta_advice"):
            all_meta_advices.append(rec["meta_advice"])
        rewards.append(rec.get("reward", 0.0))

    valid_candidates = [c for c in all_candidates if c.get("valid")]
    invalid_candidates = [c for c in all_candidates if not c.get("valid")]

    # Top candidates
    top = sorted(valid_candidates,
                 key=lambda c: c.get("improvement_factor", 0) * c.get("similarity", 0),
                 reverse=True)[:10]
    top_str = "\n".join(
        f"  {i+1}. Child: {c.get('child_smiles', 'N/A')}\n"
        f"     Improvement: {c.get('improvement_factor', 0):.2f}x | "
        f"Similarity: {c.get('similarity', 0):.3f} | "
        f"{property_name}: {c.get('child_property', 0):.3e} {property_units}"
        for i, c in enumerate(top)
    ) or "  None."

    # Failure summary
    failure_counts: Dict[str, int] = {}
    for c in invalid_candidates:
        reason = c.get("invalid_reason", "unknown")
        failure_counts[reason] = failure_counts.get(reason, 0) + 1
    failure_str = "\n".join(
        f"  {v}× {k}" for k, v in sorted(failure_counts.items(), key=lambda x: -x[1])[:8]
    ) or "  None."

    # Strategy evolution
    evo_str = "\n".join(
        f"  v{s['version']} (score={s['score']:.4f if s['score'] is not None else 'N/A'}): "
        f"{s['strategy_text'][:200]}..."
        for s in all_strategies
    )

    # Reward trajectory
    reward_str = " → ".join(f"{r:.4f}" for r in rewards)

    # Critic analyses
    analysis_entries = []
    for i, a in enumerate(all_analyses[:6]):
        insights = a.get("pareto_insights", []) + a.get("chemical_hypotheses", [])
        if insights:
            analysis_entries.append(f"  Epoch {i+1}: {'; '.join(insights[:3])}")
    analysis_str = "\n".join(analysis_entries) or "  (none)"

    # Meta advice
    meta_str = "\n".join(f"  [{i+1}] {m}" for i, m in enumerate(all_meta_advices)) or "  (none)"

    validity_rate = len(valid_candidates) / max(len(all_candidates), 1)

    user_content = _EXTRACTOR_USER_TEMPLATE.format(
        property_name=property_name,
        property_units=property_units,
        n_epochs=len(records),
        n_total_candidates=len(all_candidates),
        n_valid_candidates=len(valid_candidates),
        validity_rate=validity_rate,
        reward_trajectory=reward_str,
        strategy_evolution=evo_str,
        n_strategies=len(all_strategies),
        top_candidates=top_str,
        failure_summary=failure_str,
        critic_analyses=analysis_str,
        meta_advice_log=meta_str,
    )

    messages = [
        {"role": "system", "content": _EXTRACTOR_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    print(f"\n[KnowledgeExtractor] Calling {extractor_model} to extract run knowledge...")
    text, usage = call_llm(
        model=extractor_model,
        messages=messages,
        api_keys=api_keys,
        temperature=0.2,
        max_tokens=4096,
        max_retries=3,
    )

    # Parse and render to markdown
    parsed = _parse_json(text)
    md_path = log_path.parent / "knowledge.md"
    _render_markdown(parsed, md_path, property_name)
    print(f"[KnowledgeExtractor] Knowledge document saved to: {md_path}")
    print(f"[KnowledgeExtractor] LLM usage: {usage.to_dict()}")
    return str(md_path)


def _render_markdown(data: Dict, path: Path, property_name: str) -> None:
    """Render the extracted knowledge as a human-readable Markdown file."""
    lines = [
        f"# Optimisation Knowledge Report — {property_name}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        data.get("executive_summary", ""),
        "",
        f"**Best strategy score**: `{data.get('best_strategy_score', 'N/A')}`",
        "",
        "---",
        "",
        "## Best Strategy Found",
        "",
        "```",
        data.get("best_strategy_found", ""),
        "```",
        "",
        "---",
        "",
        "## What Worked ✅",
        "",
    ]
    for item in data.get("what_worked", []):
        lines.append(f"- {item}")

    lines += ["", "---", "", "## What Failed ❌", ""]
    for item in data.get("what_failed", []):
        lines.append(f"- {item}")

    lines += ["", "---", "", "## Key Chemical Insights 🔬", ""]
    for item in data.get("key_chemical_insights", []):
        lines.append(f"- {item}")

    lines += ["", "---", "", "## SMILES Motifs to Explore 🧬", ""]
    for item in data.get("smiles_motifs_to_explore", []):
        lines.append(f"- `{item}`")

    lines += ["", "---", "", "## Recommended Starting Strategy for Next Run", "",
              data.get("recommended_next_strategy", ""), ""]

    lines += ["", "---", "", "## Prompt Evolution Table", "",
              "| Version | Score | Key Change | Strategy Preview |",
              "|---------|-------|------------|-----------------|"]
    for row in data.get("prompt_evolution_table", []):
        score = f"{row.get('score', 'N/A'):.4f}" if isinstance(row.get("score"), (int, float)) else "N/A"
        lines.append(
            f"| v{row.get('version','?')} | {score} | "
            f"{row.get('key_change', '')[:40]} | {row.get('preview', '')[:60]} |"
        )

    lines += ["", "---", "", "## Lessons for Future Runs 📚", ""]
    for item in data.get("lessons_for_future_runs", []):
        lines.append(f"- {item}")

    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def _parse_json(text: str) -> Dict:
    import re
    text = text.strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?\n?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    print("[KnowledgeExtractor] WARNING: Could not parse extractor response.")
    return {}
