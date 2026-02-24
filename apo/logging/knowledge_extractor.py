"""Knowledge extractor — fully TaskContext-driven, no hardcoded domain language."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from ..core.llm_client import call_llm
from ..task_context import TaskContext


def _build_extractor_system(ctx: TaskContext) -> str:
    lines = [
        f"You are a senior scientist analysing results of an AI-driven "
        f"{ctx.molecule_type} discovery experiment.",
        "",
        "You have access to the complete optimisation trace: every strategy tried, "
        "every molecule generated, every success and failure.",
        "",
        "Synthesise this into a concise, high-value knowledge document that a human "
        "scientist could use to:",
        "1. Understand what the AI discovered about the structure-property relationship",
        "2. Start a future run from a much better position",
        "3. Identify promising structural motifs for experimental validation",
    ]
    if ctx.domain_context:
        lines += ["", "DOMAIN NOTES:", ctx.domain_context]
    return "\n".join(lines)


_EXTRACTOR_USER_TEMPLATE = """\
EXPERIMENT SUMMARY
==================
Property: {property_name}{units_str} | Goal: {direction}
Molecule type: {molecule_type}
Epochs: {n_epochs} | Total candidates: {n_total} | Valid: {n_valid} ({validity_rate:.1%})

REWARD TRAJECTORY: {reward_trajectory}

STRATEGY EVOLUTION ({n_strategies} versions):
{strategy_evolution}

TOP CANDIDATES (best improvement × similarity):
{top_candidates}

FAILURE BREAKDOWN:
{failure_summary}

CRITIC ANALYSES (chemical insights per epoch):
{critic_analyses}

META-STRATEGIST ADVICE GIVEN:
{meta_advice_log}

Return JSON (ONLY JSON):
{{
  "executive_summary": "2-3 sentence overview",
  "best_strategy_found": "exact strategy text that scored highest",
  "best_strategy_score": <float or null>,
  "what_worked": ["specific structural modification X improved Y because...", ...],
  "what_failed": ["approach X failed because... (N occurrences)", ...],
  "key_insights": ["structure-property insight 1", "insight 2"],
  "smiles_motifs_to_explore": ["motif 1 — rationale", "motif 2"],
  "recommended_next_strategy": "Concrete strategy for next run...",
  "prompt_evolution_table": [
    {{"version": 0, "score": null, "key_change": "seed", "preview": "..."}},
    {{"version": 1, "score": 1.23, "key_change": "added ether groups", "preview": "..."}}
  ],
  "lessons_for_future_runs": ["lesson 1", "lesson 2"]
}}
"""


def extract_knowledge(
    run_log_path: str,
    extractor_model: str,
    task_context: Optional[TaskContext] = None,
    api_keys: Optional[Dict[str, str]] = None,
    # Backward-compatible kwargs (ignored if task_context provided)
    property_name: str = "Property",
    property_units: str = "",
) -> str:
    """
    Read a run_log.jsonl and produce knowledge.md.

    Args:
        task_context: If provided, uses it for all domain context.
                      If None, falls back to property_name/property_units kwargs.
    Returns:
        Path to written knowledge.md.
    """
    # Build minimal context if not provided
    if task_context is None:
        task_context = TaskContext(
            property_name=property_name,
            property_units=property_units,
            molecule_type="molecule",
        )

    log_path = Path(run_log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Run log not found: {run_log_path}")

    records = _load_records(log_path)
    if not records:
        print("[KnowledgeExtractor] No records found.")
        return ""

    # Aggregate
    all_candidates = []
    all_strategies = []
    all_analyses = []
    all_meta = []
    rewards = []

    for rec in records:
        all_candidates.extend(rec.get("candidates", []))
        ps = rec.get("prompt_state", {})
        all_strategies.append({
            "version": ps.get("version", rec["epoch"]),
            "score": rec.get("reward"),
            "strategy_text": ps.get("strategy_text", ""),
        })
        if rec.get("analysis"):
            all_analyses.append(rec["analysis"])
        if rec.get("meta_advice"):
            all_meta.append(rec["meta_advice"])
        rewards.append(rec.get("reward", 0.0))

    valid = [c for c in all_candidates if c.get("valid")]
    invalid = [c for c in all_candidates if not c.get("valid")]
    units_str = f" ({task_context.property_units})" if task_context.property_units else ""

    top = sorted(valid,
                 key=lambda c: c.get("improvement_factor", 0) * c.get("similarity", 0),
                 reverse=True)[:10]
    top_str = "\n".join(
        f"  {i+1}. {c.get('child_smiles','N/A')}\n"
        f"     improvement={c.get('improvement_factor',0):.2f}× "
        f"sim={c.get('similarity',0):.3f} "
        f"{task_context.property_name}={c.get('child_property',0):.3e}"
        for i, c in enumerate(top)
    ) or "  None."

    failure_counts: Dict[str, int] = {}
    for c in invalid:
        r = c.get("invalid_reason", "unknown")
        failure_counts[r] = failure_counts.get(r, 0) + 1
    failure_str = "\n".join(
        f"  {v}× {k}" for k, v in sorted(failure_counts.items(), key=lambda x: -x[1])[:8]
    ) or "  None."

    evo_str = "\n".join(
        f"  v{s['version']} (score={'%.4f' % s['score'] if s['score'] is not None else 'N/A'}): "
        f"{s['strategy_text'][:200]}..."
        for s in all_strategies
    )

    analyses_str = "\n".join(
        f"  [{i+1}] " + " | ".join(
            f"{k}: {', '.join(v[:2]) if isinstance(v, list) else str(v)}"
            for k, v in a.items() if v
        )[:250]
        for i, a in enumerate(all_analyses[:6])
    ) or "  (none)"

    meta_str = "\n".join(f"  [{i+1}] {m}" for i, m in enumerate(all_meta)) or "  (none)"

    user_content = _EXTRACTOR_USER_TEMPLATE.format(
        property_name=task_context.property_name,
        units_str=units_str,
        direction=task_context.direction_word.upper(),
        molecule_type=task_context.molecule_type,
        n_epochs=len(records),
        n_total=len(all_candidates),
        n_valid=len(valid),
        validity_rate=len(valid) / max(len(all_candidates), 1),
        reward_trajectory=" → ".join(f"{r:.4f}" for r in rewards),
        strategy_evolution=evo_str,
        n_strategies=len(all_strategies),
        top_candidates=top_str,
        failure_summary=failure_str,
        critic_analyses=analyses_str,
        meta_advice_log=meta_str,
    )

    messages = [
        {"role": "system", "content": _build_extractor_system(task_context)},
        {"role": "user", "content": user_content},
    ]

    print(f"[KnowledgeExtractor] Calling {extractor_model}...")
    text, usage = call_llm(
        model=extractor_model,
        messages=messages,
        api_keys=api_keys,
        temperature=0.2,
        max_tokens=4096,
        max_retries=3,
    )
    print(f"[KnowledgeExtractor] Usage: {usage.to_dict()}")

    parsed = _parse_json(text)
    md_path = log_path.parent / "knowledge.md"
    _render_markdown(parsed, md_path, task_context)
    print(f"[KnowledgeExtractor] Saved: {md_path}")
    return str(md_path)


def _render_markdown(data: Dict, path: Path, ctx: TaskContext) -> None:
    lines = [
        f"# Optimisation Knowledge Report — {ctx.property_name}",
        f"*{ctx.molecule_type.capitalize()} | {ctx.direction_word} {ctx.property_name}*",
        "",
        "---",
        "## Executive Summary",
        "",
        data.get("executive_summary", ""),
        "",
        f"**Best score**: `{data.get('best_strategy_score', 'N/A')}`",
        "",
        "---",
        "## Best Strategy Found",
        "",
        "```",
        data.get("best_strategy_found", ""),
        "```",
        "",
        "---",
        "## What Worked ✅",
        "",
    ]
    for item in data.get("what_worked", []):
        lines.append(f"- {item}")

    lines += ["", "---", "## What Failed ❌", ""]
    for item in data.get("what_failed", []):
        lines.append(f"- {item}")

    lines += ["", "---", "## Key Insights 🔬", ""]
    for item in data.get("key_insights", []):
        lines.append(f"- {item}")

    lines += ["", "---", "## SMILES/Structural Motifs to Explore 🧬", ""]
    for item in data.get("smiles_motifs_to_explore", []):
        lines.append(f"- `{item}`")

    lines += ["", "---", "## Recommended Next Strategy", "", data.get("recommended_next_strategy", ""), ""]

    lines += ["", "---", "## Prompt Evolution", "",
              "| Version | Score | Key Change | Preview |",
              "|---------|-------|------------|---------|"]
    for row in data.get("prompt_evolution_table", []):
        score = f"{row.get('score', 'N/A'):.4f}" if isinstance(row.get("score"), (int, float)) else "N/A"
        lines.append(
            f"| v{row.get('version','?')} | {score} | "
            f"{str(row.get('key_change', ''))[:40]} | {str(row.get('preview', ''))[:60]} |"
        )

    lines += ["", "---", "## Lessons for Future Runs 📚", ""]
    for item in data.get("lessons_for_future_runs", []):
        lines.append(f"- {item}")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def _load_records(log_path: Path) -> List[Dict]:
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


def _parse_json(text: str) -> Dict:
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
    print("[KnowledgeExtractor] WARNING: Could not parse response.")
    return {}
