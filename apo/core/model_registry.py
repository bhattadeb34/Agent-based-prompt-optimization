"""
Model registry for programmatically selecting LLM models.

Provides:
- Model tier definitions (fast/balanced/premium)
- Model aliases (latest/stable)
- Programmatic model selection
- Provider-specific fallbacks

Usage:
    from apo.core.model_registry import get_model, ModelTier

    # Get latest premium model from preferred provider
    model = get_model(tier=ModelTier.PREMIUM, provider="openai")

    # Get all models for a tier
    models = get_models_for_tier(ModelTier.FAST)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class ModelTier(Enum):
    """Model performance tiers."""
    FAST = "fast"           # Quick, cheap, good for simple tasks
    BALANCED = "balanced"   # Balance of quality/speed/cost
    PREMIUM = "premium"     # Highest quality, slower, expensive
    REASONING = "reasoning" # Extended thinking models


class Provider(Enum):
    """LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class ModelSpec:
    """Model specification."""
    name: str
    provider: Provider
    tier: ModelTier
    litellm_id: str
    context_window: int
    supports_tools: bool
    cost_per_1m_input: float  # USD
    cost_per_1m_output: float # USD
    notes: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# Model Registry — Update this when new models are released
# ══════════════════════════════════════════════════════════════════════════════

MODELS: Dict[str, ModelSpec] = {
    # ──────────────────────────────────────────────────────────────────────────
    # OpenAI Models
    # ──────────────────────────────────────────────────────────────────────────
    "gpt-5.2-latest": ModelSpec(
        name="gpt-5.2-latest",
        provider=Provider.OPENAI,
        tier=ModelTier.PREMIUM,
        litellm_id="openai/gpt-5.2-2025-12-11",
        context_window=200000,
        supports_tools=True,
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
        notes="Latest GPT-5 model (Dec 2025)"
    ),
    "gpt-4o": ModelSpec(
        name="gpt-4o",
        provider=Provider.OPENAI,
        tier=ModelTier.BALANCED,
        litellm_id="openai/gpt-4o",
        context_window=128000,
        supports_tools=True,
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
        notes="GPT-4 Optimized (stable)"
    ),
    "gpt-4o-mini": ModelSpec(
        name="gpt-4o-mini",
        provider=Provider.OPENAI,
        tier=ModelTier.FAST,
        litellm_id="openai/gpt-4o-mini",
        context_window=128000,
        supports_tools=True,
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60,
        notes="Fast, cheap, good tool-calling"
    ),
    "o1-mini": ModelSpec(
        name="o1-mini",
        provider=Provider.OPENAI,
        tier=ModelTier.REASONING,
        litellm_id="openai/o1-mini",
        context_window=128000,
        supports_tools=False,
        cost_per_1m_input=3.00,
        cost_per_1m_output=12.00,
        notes="Reasoning model, no tool calling"
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Google Models
    # ──────────────────────────────────────────────────────────────────────────
    "gemini-3.1-pro": ModelSpec(
        name="gemini-3.1-pro",
        provider=Provider.GOOGLE,
        tier=ModelTier.PREMIUM,
        litellm_id="gemini/gemini-3.1-pro-preview",
        context_window=1000000,
        supports_tools=True,
        cost_per_1m_input=0.00,  # Preview pricing
        cost_per_1m_output=0.00,
        notes="Latest Gemini Pro (preview, Feb 2026)"
    ),
    "gemini-2.0-flash": ModelSpec(
        name="gemini-2.0-flash",
        provider=Provider.GOOGLE,
        tier=ModelTier.FAST,
        litellm_id="gemini/gemini-2.0-flash",
        context_window=1000000,
        supports_tools=True,
        cost_per_1m_input=0.00,  # Free tier
        cost_per_1m_output=0.00,
        notes="Fast, free, multimodal"
    ),
    "gemini-2.0-flash-thinking": ModelSpec(
        name="gemini-2.0-flash-thinking",
        provider=Provider.GOOGLE,
        tier=ModelTier.REASONING,
        litellm_id="gemini/gemini-2.0-flash-thinking-exp-1219",
        context_window=32000,
        supports_tools=True,
        cost_per_1m_input=0.00,
        cost_per_1m_output=0.00,
        notes="Extended thinking mode (experimental)"
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Anthropic Models
    # ──────────────────────────────────────────────────────────────────────────
    "claude-3.5-sonnet": ModelSpec(
        name="claude-3.5-sonnet",
        provider=Provider.ANTHROPIC,
        tier=ModelTier.PREMIUM,
        litellm_id="anthropic/claude-3-5-sonnet-20241022",
        context_window=200000,
        supports_tools=True,
        cost_per_1m_input=3.00,
        cost_per_1m_output=15.00,
        notes="Best Claude for reasoning"
    ),
    "claude-3.5-haiku": ModelSpec(
        name="claude-3.5-haiku",
        provider=Provider.ANTHROPIC,
        tier=ModelTier.FAST,
        litellm_id="anthropic/claude-3-5-haiku-20241022",
        context_window=200000,
        supports_tools=True,
        cost_per_1m_input=0.80,
        cost_per_1m_output=4.00,
        notes="Fast Claude variant"
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Model Selection Functions
# ══════════════════════════════════════════════════════════════════════════════

def get_model(
    tier: Optional[ModelTier] = None,
    provider: Optional[Provider] = None,
    supports_tools: Optional[bool] = None,
    prefer_latest: bool = True,
) -> str:
    """
    Get best model matching criteria.

    Args:
        tier: Model performance tier
        provider: Preferred provider
        supports_tools: Filter by tool-calling support
        prefer_latest: Prefer latest/preview models

    Returns:
        LiteLLM model ID (e.g., "openai/gpt-4o")

    Examples:
        >>> get_model(tier=ModelTier.PREMIUM, provider=Provider.OPENAI)
        'openai/gpt-5.2-2025-12-11'

        >>> get_model(tier=ModelTier.FAST, supports_tools=True)
        'openai/gpt-4o-mini'
    """
    candidates = list(MODELS.values())

    # Filter by criteria
    if tier:
        candidates = [m for m in candidates if m.tier == tier]
    if provider:
        candidates = [m for m in candidates if m.provider == provider]
    if supports_tools is not None:
        candidates = [m for m in candidates if m.supports_tools == supports_tools]

    if not candidates:
        raise ValueError(f"No models found matching tier={tier}, provider={provider}, tools={supports_tools}")

    # Prefer latest if requested
    if prefer_latest:
        latest = [m for m in candidates if "latest" in m.name or "preview" in m.name]
        if latest:
            candidates = latest

    # Return first match (sorted by cost if multiple)
    candidates.sort(key=lambda m: m.cost_per_1m_input)
    return candidates[0].litellm_id


def get_models_for_tier(tier: ModelTier) -> List[ModelSpec]:
    """Get all models in a given tier."""
    return [m for m in MODELS.values() if m.tier == tier]


def get_model_preset(preset: str) -> Dict[str, str]:
    """
    Get pre-defined model configurations for all APO roles.

    Args:
        preset: One of ["fast", "balanced", "premium", "strategic"]

    Returns:
        Dict mapping role → LiteLLM model ID

    Presets:
        - fast: All cheap/free models, lowest cost
        - balanced: Mix of fast/balanced models
        - premium: All premium models, highest quality
        - strategic: Expensive models only where critical (critic/meta),
                    cheap models for simple tasks (worker/orchestrator)

    Examples:
        >>> get_model_preset("strategic")
        {
            "worker": "gemini/gemini-2.0-flash",        # FREE, generates many SMILES
            "critic": "openai/gpt-5.2-2025-12-11",       # EXPENSIVE, critical reasoning
            "meta": "openai/gpt-5.2-2025-12-11",         # EXPENSIVE, strategic guidance
            "orchestrator": "openai/gpt-4o-mini",        # CHEAP, simple tool calling
            "knowledge_extractor": "openai/gpt-4o",      # BALANCED, runs once at end
        }
    """
    presets = {
        "fast": {
            "worker": get_model(tier=ModelTier.FAST, provider=Provider.GOOGLE),
            "critic": get_model(tier=ModelTier.FAST, provider=Provider.OPENAI, supports_tools=True),
            "meta": get_model(tier=ModelTier.FAST, provider=Provider.OPENAI),
            "orchestrator": get_model(tier=ModelTier.FAST, provider=Provider.OPENAI, supports_tools=True),
            "knowledge_extractor": get_model(tier=ModelTier.FAST, provider=Provider.OPENAI),
        },
        "balanced": {
            "worker": get_model(tier=ModelTier.BALANCED, provider=Provider.GOOGLE),
            "critic": get_model(tier=ModelTier.BALANCED, provider=Provider.OPENAI),
            "meta": get_model(tier=ModelTier.BALANCED, provider=Provider.OPENAI),
            "orchestrator": get_model(tier=ModelTier.FAST, provider=Provider.OPENAI, supports_tools=True),
            "knowledge_extractor": get_model(tier=ModelTier.BALANCED, provider=Provider.OPENAI),
        },
        "premium": {
            "worker": get_model(tier=ModelTier.PREMIUM, provider=Provider.GOOGLE),
            "critic": get_model(tier=ModelTier.PREMIUM, provider=Provider.OPENAI),
            "meta": get_model(tier=ModelTier.PREMIUM, provider=Provider.OPENAI),
            "orchestrator": get_model(tier=ModelTier.BALANCED, provider=Provider.OPENAI, supports_tools=True),
            "knowledge_extractor": get_model(tier=ModelTier.PREMIUM, provider=Provider.OPENAI),
        },
        "strategic": {
            # STRATEGIC: Use expensive models only where they add most value
            "worker": "gemini/gemini-2.0-flash",        # FREE - generates 240 SMILES/run (bulk task)
            "critic": "openai/gpt-5.2-2025-12-11",       # PREMIUM - refines strategy (critical reasoning, 10 calls/run)
            "meta": "openai/gpt-5.2-2025-12-11",         # PREMIUM - strategic pivots (critical, 3 calls/run)
            "orchestrator": "openai/gpt-4o-mini",        # FAST - simple tool routing (10 calls/run)
            "knowledge_extractor": "openai/gpt-4o",      # BALANCED - final analysis (1 call/run)
        },
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(presets.keys())}")

    return presets[preset]


def list_all_models() -> None:
    """Print all available models grouped by tier."""
    print("\n" + "="*80)
    print("  AVAILABLE MODELS")
    print("="*80 + "\n")

    for tier in ModelTier:
        models = get_models_for_tier(tier)
        if not models:
            continue

        print(f"{'─'*80}")
        print(f"  {tier.value.upper()} TIER ({len(models)} models)")
        print(f"{'─'*80}\n")

        for m in sorted(models, key=lambda x: (x.provider.value, x.name)):
            cost_str = f"${m.cost_per_1m_input:.2f}/${m.cost_per_1m_output:.2f} per 1M tokens" if m.cost_per_1m_input > 0 else "FREE"
            tools_str = "✓" if m.supports_tools else "✗"

            print(f"  • {m.name:30s} [{m.provider.value:10s}] Tools:{tools_str}  {cost_str}")
            print(f"    LiteLLM ID: {m.litellm_id}")
            if m.notes:
                print(f"    {m.notes}")
            print()

    print("="*80 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI for testing
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "list":
        list_all_models()
    elif len(sys.argv) > 1 and sys.argv[1] == "preset":
        preset_name = sys.argv[2] if len(sys.argv) > 2 else "latest"
        print(f"\nPreset '{preset_name}':")
        for role, model_id in get_model_preset(preset_name).items():
            print(f"  {role:20s} : {model_id}")
        print()
    else:
        print("Usage:")
        print("  python model_registry.py list           # List all models")
        print("  python model_registry.py preset latest   # Show preset config")
