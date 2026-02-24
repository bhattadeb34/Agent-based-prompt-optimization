#!/usr/bin/env python3
"""
run_optimization.py — Main CLI entrypoint for the Agentic Prompt Optimisation framework.

Usage:
    conda activate li_llm_optimization
    python run_optimization.py --config config/conductivity_task.yaml

    # Override specific settings without editing YAML:
    python run_optimization.py --config config/conductivity_task.yaml \\
        --override optimization.n_outer_epochs=3 \\
        --override optimization.n_per_molecule=2 \\
        --override task.sample_size=5

    # Use a specific API keys file:
    python run_optimization.py --config config/conductivity_task.yaml \\
        --api-keys /path/to/api_keys.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def apply_override(cfg: dict, dotted_key: str, value: str) -> None:
    """Apply a dot-notation override like 'optimization.n_outer_epochs=3'."""
    keys = dotted_key.split(".")
    d = cfg
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    # Auto-cast: try int, then float, then leave as string
    last = keys[-1]
    try:
        d[last] = int(value)
    except ValueError:
        try:
            d[last] = float(value)
        except ValueError:
            if value.lower() in ("true", "false"):
                d[last] = value.lower() == "true"
            else:
                d[last] = value


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Prompt Optimisation for Polymer Property Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML task config (e.g. config/conductivity_task.yaml)",
    )
    parser.add_argument(
        "--api-keys",
        default=None,
        help="Path to api_keys.txt (default: looks for api_keys.txt next to config)",
    )
    parser.add_argument(
        "--override",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help="Override a YAML config key (e.g. --override optimization.n_outer_epochs=3). "
             "Can be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and dataset but don't run the optimisation.",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Apply overrides
    for override in args.override:
        if "=" not in override:
            print(f"ERROR: --override must be KEY=VALUE format, got: {override}")
            sys.exit(1)
        key, _, val = override.partition("=")
        apply_override(cfg, key.strip(), val.strip())
        print(f"[CLI] Override: {key} = {val}")

    # Resolve API keys path
    if args.api_keys:
        api_keys_path = args.api_keys
    else:
        # Try next to config, then cwd
        candidate = config_path.parent.parent / "api_keys.txt"
        api_keys_path = str(candidate) if candidate.exists() else "api_keys.txt"

    print(f"[CLI] Config  : {config_path}")
    print(f"[CLI] API keys: {api_keys_path}")

    if args.dry_run:
        print("\n[DRY RUN] Config is valid. Dataset preview:")
        from apo.engine import load_dataset, load_api_keys, _normalise_api_keys
        smiles = load_dataset(cfg)
        print(f"  {len(smiles)} parent SMILES loaded.")
        print(f"  First 3: {smiles[:3]}")
        keys = _normalise_api_keys(load_api_keys(api_keys_path))
        print(f"  API keys found: {[k for k, v in keys.items() if v]}")
        print("\n[DRY RUN] All checks passed. Ready to run.")
        return

    # Run
    from apo.engine import run
    run_dir = run(cfg, api_keys_path=api_keys_path)
    print(f"\n[CLI] Run complete. Output: {run_dir}")


if __name__ == "__main__":
    main()
