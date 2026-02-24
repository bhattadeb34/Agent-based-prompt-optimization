# APO — Developer Log

> **Purpose**: Complete record of what was built, why each decision was made, what went wrong, and how to continue. Written for a future developer (or future you) picking this up cold.

---

## Table of Contents

1. [Project Goal](#project-goal)
2. [Repository Layout](#repository-layout)
3. [Architecture](#architecture)
4. [Build History — What We Did and In What Order](#build-history)
5. [Design Decisions](#design-decisions)
6. [What Worked](#what-worked)
7. [What Failed or Needed Fixing](#what-failed-or-needed-fixing)
8. [Live Run Results](#live-run-results)
9. [How to Continue](#how-to-continue)
10. [Dependencies](#dependencies)

---

## Project Goal

Build a **reusable, property-agnostic Agentic Prompt Optimization (APO) framework** for molecular discovery.

The core idea: instead of writing a new LLM prompting pipeline for each property (conductivity, Tg, HOMO/LUMO, solubility…), build a single framework where:

- The **chemistry domain knowledge** lives in a YAML config (`task:` block → `TaskContext`)
- The **LLM orchestration** is handled by an agent that decides when to generate molecules, when to refine its strategy, when to call a high-level advisor, and when to stop
- The **surrogate model** (GNN or anything else) is pluggable
- New properties = new YAML + surrogate registration, zero code changes

Inspired by [GEPA](https://github.com/gepa-ai/gepa) — using full execution traces (Actionable Side Information) to give the critic LLM complete context about what failed and why.

---

## Repository Layout

```
agentic-prompt-optimization/
├── apo/
│   ├── __init__.py
│   ├── agent.py              ← OrchestratorAgent (tool-calling LLM drives the loop)
│   ├── engine.py             ← Main run() entry point; builds all components from config
│   ├── task_context.py       ← TaskContext dataclass (all domain knowledge lives here)
│   ├── core/
│   │   ├── llm_client.py     ← LiteLLM wrapper with retry, token tracking, JSON parsing
│   │   ├── prompt_state.py   ← PromptState + PromptStateHistory
│   │   └── reward.py         ← ParetoHypervolume, WeightedSum, PropertyOnly reward fns
│   ├── optimizer/
│   │   ├── inner_loop.py     ← Worker agent: generates SMILES given strategy
│   │   ├── outer_loop.py     ← Critic agent: analyses results, refines strategy
│   │   └── meta_loop.py      ← Strategist: periodic high-level guidance
│   ├── logging/
│   │   ├── run_logger.py     ← Writes run_log.jsonl (one line per epoch)
│   │   └── knowledge_extractor.py  ← Synthesises run → knowledge.md
│   ├── surrogates/
│   │   ├── base.py           ← Abstract SurrogatePredictor
│   │   ├── registry.py       ← @register decorator, get_surrogate()
│   │   └── gnn_predictor.py  ← GNN for conductivity; also template for new predictors
│   └── utils/
│       └── smiles_utils.py   ← Validation, canonicalization, Tanimoto, compute_similarity()
├── config/
│   ├── conductivity_task.yaml   ← Polymer electrolyte conductivity (polymer markers)
│   ├── tg_task.yaml             ← Placeholder for glass transition temperature
│   └── generic_smiles_task.yaml ← Template for any non-polymer property
├── tests/
│   ├── test_reward.py        ← 13 tests for all reward functions
│   ├── test_predictor.py     ← 9 tests for surrogate interface + registry
│   └── test_integration.py   ← 10 tests including full 2-epoch pipeline smoke test
├── run_optimization.py       ← Primary CLI entrypoint
├── analyze_results.py        ← Analyze a completed run (reward curve, stats, re-extract)
├── export_best_prompt.py     ← Export best strategy for zero-shot use
├── requirements.txt
├── README.md
└── DEVLOG.md                 ← This file
```

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │     OrchestratorAgent (engine.py)        │
                    │     LLM with 5 tools:                    │
                    │       generate_candidates(n, notes)      │
                    │       refine_strategy(notes)             │
                    │       get_meta_advice()                  │
                    │       set_strategy(text, rationale)      │
                    │       done(reason)                       │
                    └───────────────┬──────────────────────────┘
                                    │ tool calls
            ┌───────────────────────┼──────────────────────────┐
            ▼                       ▼                          ▼
   InnerLoop (Worker)       OuterLoop (Critic)        MetaLoop (Strategist)
   LLM generates SMILES     LLM analyses results       LLM gives guidance
   given current strategy   and proposes new strat.    every N epochs or
            │                       │                   on-demand by agent
            ▼                       ▼
   SurrogatePredictor       RewardFunction
   (GNN / custom plug-in)   (Pareto HV etc.)
            │
            ▼
   TaskContext (from YAML)
   - property_name/units
   - molecule_type
   - domain_context (injected into all prompts)
   - smiles_markers (structural requirements)
   - similarity_on_repeat_unit
   - seed_strategy
```

**Two modes** (`optimization.mode` in config):
- `agent` — LLM orchestrator decides the flow (default, recommended)
- `loop`  — classic fixed `for epoch in range(N)` loop (reproducible)

---

## Build History

### Phase 1: Core Framework (Session 1)

Started from scratch. Ported ideas from `prompt-optimization-work-jan-8` (original polymer-specific code) into a clean reusable library.

**Built in order:**
1. `apo/core/llm_client.py` — LiteLLM wrapper, retry, token tracking
2. `apo/core/prompt_state.py` — versioned strategy state + history
3. `apo/core/reward.py` — ParetoHypervolume (primary), WeightedSum, PropertyOnly
4. `apo/surrogates/` — abstract base + registry + GNN predictor (ported from original)
5. `apo/optimizer/inner_loop.py` — worker LLM prompt builder + SMILES validator
6. `apo/optimizer/outer_loop.py` — critic LLM + Pareto ASI formatting
7. `apo/optimizer/meta_loop.py` — periodic strategist
8. `apo/logging/` — JSONL logger + knowledge extractor
9. `apo/engine.py` — orchestrator gluing everything
10. `run_optimization.py`, `analyze_results.py` — CLI tools
11. Unit + integration tests (27 tests)

### Phase 2: Property-Agnostic Refactor (Session 2)

**Problem identified**: Phase 1 code had hardcoded polymer-specific language throughout (`[Cu]`/`[Au]` markers in prompts, "polymer electrolyte" in system prompts, polymer-specific SMILES validation). Couldn't be reused for other properties without code changes.

**Solution**: `TaskContext` dataclass — all domain knowledge moves into YAML config.

**Changed files:**
- Created `apo/task_context.py`
- Rewrote all 4 prompt builders (inner, outer, meta, extractor) to use `ctx.*`
- `smiles_utils.py` — `required_markers` optional param, `compute_similarity()`
- `engine.py` — builds `TaskContext` from config, passes everywhere
- Added `config/generic_smiles_task.yaml`

### Phase 3: Tool-Calling Agent (Session 2, late)

**User request**: "can we have an orchestrator agent using LLMs with tool calling"

**Built `apo/agent.py`:**
- 5 tool definitions in OpenAI function-calling schema
- `OrchestratorAgent` with native `litellm.completion(tools=TOOLS)` + JSON-text fallback
- `engine.py` updated: `mode: agent` delegates to `OrchestratorAgent`, `mode: loop` keeps original fixed-epoch loop

### Phase 4: Export + GitHub (Session 2, end)

- `export_best_prompt.py` — extract best strategy from run log for zero-shot use
- `.gitignore` — excludes api_keys, runs/, weights, data/
- `DEVLOG.md` — this file
- Push to `git@github.com:bhattadeb34/Agent-based-prompt-optimization.git`

---

## Design Decisions

### `TaskContext` as single source of truth
Every LLM system prompt, SMILES validator, and similarity function receives a `TaskContext` object. No other code knows what domain it's in. Result: zero code changes to add a new property — just write a new YAML.

### Tool-calling over hardcoded loop
The original design ran a fixed `for epoch in range(N)` loop. The agent approach lets the LLM decide:
- Call `generate_candidates` twice before refining if the first batch was mostly invalid
- Call `get_meta_advice` early if reward is flat
- Call `done()` early if reward is satisfactory or if the last batch was useless
In the live test, the agent correctly self-terminated when 0/6 candidates were valid, reasoning that "further refinement is unlikely to yield better results."

### JSON-text fallback for Gemini
Gemini Flash (and some other models) don't always return structured `tool_calls` even with `tool_choice="auto"`. The fallback asks for `{"tool": "...", "args": {...}}` JSON text. `_try_parse_json_tool_call()` handles both native tool_calls and text JSON.

### Pareto + Hypervolume reward
Two objectives: improvement factor (property improvement over parent) and Tanimoto similarity to parent. Pareto hypervolume rewards solutions that are both better AND structurally close. This avoids mode collapse toward extreme improvement at zero similarity.

### GEPA-style Actionable Side Information (ASI)
Every epoch logs the full candidate trace including failure reasons, not just the reward. The critic LLM receives this full picture ("30× missing marker, 5× invalid SMILES, 2× prediction error") and can reason about systematic failures.

---

## What Worked

| Thing | Why it worked |
|-------|---------------|
| LiteLLM for model agnosticism | Single interface for OpenAI, Gemini, Anthropic; retry built-in |
| `TaskContext` from YAML | Clean separation of domain vs framework logic |
| Tool-calling agent | Agent self-diagnosed failure (0 valid candidates) and stopped without human intervention |
| Pareto hypervolume reward | Prevents reward hacking (pure improvement at 0 similarity) |
| JSONL logging with full traces | Critic has complete picture of failures, not just the reward scalar |
| JSON-text fallback for Gemini | Gemini Flash doesn't always return structured tool_calls; fallback caught `{"tool": "refine_strategy"}` as text and parsed it correctly |
| Test mocking strategy | Patching `call_llm` at the module level lets integration tests validate the full pipeline without real API calls |

---

## What Failed or Needed Fixing

### 1. F-string backslash restriction (Python 3.10)
Python < 3.12 disallows backslash inside f-string expressions. Expressions like:
```python
f"{('NOTES:\n' + x) if x else ''}"
```
Fix: precompute the string before the f-string:
```python
notes = ("NOTES:\n" + x) if x else ""
f"{notes}"
```
This hit us in `meta_loop.py`, `outer_loop.py`, and `agent.py`.

### 2. `multi_replace_file_content` indentation bug
When adding `_try_parse_json_tool_call` to `agent.py` via multi-replace, the tool accidentally placed the new module-level function inside the class body at wrong indentation, then left `_execute_tool` and all following methods outside the class. Fixed by rewriting the whole file cleanly.

### 3. `PromptState.score` can be `None`
When the first strategy hasn't been evaluated yet, `score` is `None`. Several f-strings tried `f"{score:.4f}"` and crashed. Fix: always guard with `if score is not None else "N/A"`.

### 4. Gemini tool calling behavior
Gemini Flash returns tool calls as text JSON (`{"tool": "...", "args": {...}}`) instead of native `tool_calls` objects, even with `tool_choice="auto"`. The native path raises an exception, which was caught, but the fallback path was also not being triggered correctly — the text was being returned as "no tool call" and the loop stopped. Fixed by calling `_try_parse_json_tool_call(text_content)` before returning `None` on the no-tool-call path.

### 5. `hypervolume` assertion in test
Single-point Pareto front: original implementation tried to compute hypervolume with 0 reference points, raising an assertion. Fix: return 0.0 for sets smaller than 2 non-dominated points.

### 6. Surrogate `__init__` vs registration
`GNNPredictor` was auto-registered by module import, but the module wasn't being imported unless explicitly referenced. Fix: `engine.py` imports `apo.surrogates.gnn_predictor` at startup via `get_surrogate()`.

---

## Live Run Results

**Configuration**: `conductivity_task.yaml` · All 4 tiers = `gemini/gemini-2.0-flash` · Budget = 8 tools

| Tool # | Action | Outcome |
|--------|--------|---------|
| 1 | `generate_candidates(n=3)` | 6/6 valid |
| 2 | `refine_strategy()` | reward = 0.9380 |
| 3 | `generate_candidates(n=3)` | 3/6 valid, avg 2.1× improvement |
| 4 | `refine_strategy()` | reward = 1.0525 |
| 5 | `generate_candidates(n=3)` | 6/6 valid, avg 2.0× improvement |
| 6 | `refine_strategy()` | reward = **1.0981** (best) |
| 7 | `generate_candidates(n=3)` | 0/6 valid (RDKit parse errors) |
| 8 | `done("zero valid candidates, further refinement unlikely")` | ✅ self-terminated |

**Total: 15 LLM calls · 27,491 tokens · 33.5 seconds · $0.00 (Gemini free tier)**

Best strategy extracted (for zero-shot re-use):
```
Based on successful candidates, focus on:
1. Ether chain length 2–4 ether linkages (COC)
2. Tertiary amine -N(C)C positioned 2–4 carbons from [Cu]
3. Both OC(=O)[Au] and C(=O)O[Au] ester linkages
4. Short alkyl chains (2–4 carbons), no nitrile groups
General template: R-N(C)C-Alkylene-(COC)n-[Cu]-Alkylene-OC(=O)[Au]
```

---

## How to Continue

### Add a new property

1. Write a surrogate predictor:
```python
# apo/surrogates/my_predictor.py
from .base import SurrogatePredictor
from .registry import register

@register("my_property_gnn")
class MyPredictor(SurrogatePredictor):
    property_name = "HOMO"
    property_units = "eV"
    maximize = False

    def predict(self, smiles_list):
        ...  # your model here
        return [float(v) for v in values]
```

2. Write a config:
```yaml
# config/homo_task.yaml
task:
  surrogate: my_property_gnn
  dataset: /path/to/data.csv
  smiles_column: smiles
  molecule_type: "organic molecule"
  domain_context: "Generate drug-like SMILES with MW < 500."
  smiles_markers: []          # no markers for plain organic SMILES
  similarity_on_repeat_unit: false

models:
  worker: gemini/gemini-2.0-flash
  critic: openai/gpt-4o-mini
  meta: openai/gpt-4o
  orchestrator: openai/gpt-4o-mini

optimization:
  mode: agent
  tool_budget: 20
```

3. Run:
```bash
python run_optimization.py --config config/homo_task.yaml --api-keys api_keys.txt
```

### Resume from best strategy
```bash
# Export
python export_best_prompt.py runs/latest --as-yaml

# New run seeded from it — paste best_strategy.yaml content into config
# or use --override:
python run_optimization.py --config config/conductivity_task.yaml \
  --api-keys api_keys.txt \
  --override optimization.seed_strategy="$(cat runs/latest/best_strategy.txt)"
```

### Run all tests
```bash
conda activate li_llm_optimization
cd agentic-prompt-optimization
python -m pytest tests/ -v
# Expected: 32 passed
```

### Analyze a completed run
```bash
python analyze_results.py runs/latest
python analyze_results.py runs/latest --re-extract-knowledge  # regenerate knowledge.md
python export_best_prompt.py runs/latest --all  # all strategies ranked
```

---

## Dependencies

```
litellm>=1.0         # model-agnostic LLM interface (OpenAI/Gemini/Anthropic)
rdkit-pypi           # SMILES validation, fingerprints
torch                # GNN surrogate
torch-geometric      # GNN layers
numpy, pandas        # data handling
PyYAML               # config loading
pytest               # testing
```

Conda env: `li_llm_optimization` (on the cluster at `/noether/s0/wdxb5775/anaconda3/envs/`)

API keys file (NOT in git): `api_keys.txt` — format:
```
GOOGLE_GEMINI_API_KEY=...
openai_GPT_api_key=...
CLAUDE_API_KEY=...
```

---

## Known Limitations / Future Work

- **Marker detection is still heuristic** — the system trusts the LLM to include `[Cu]`/`[Au]`; a future version could attempt marker insertion as a post-processing step on otherwise-valid SMILES.
- **No warm-start resume** — if a run crashes mid-epoch, there's no checkpoint-based resume. `RunLogger.load_existing_epochs()` exists for this but isn't wired into `engine.py` yet.
- **Tool budget is fixed** — the agent has a hard cap. A token-budget version (stop when `total_tokens > N`) would be more cost-transparent.
- **Knowledge extractor produces Markdown** — could also emit a structured JSON for downstream consumption by other tools.
- **No multi-property optimization** — each config optimizes one property. Multi-objective joint optimization (maximize conductivity AND minimize Tg) is a possible extension.
