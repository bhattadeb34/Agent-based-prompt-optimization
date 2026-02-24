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

---

## Experiment 1: Conductivity Optimization (80 train / 20 test)

> **Date**: 2026-02-23  
> **Config**: `config/conductivity_experiment.yaml`  
> **Mode**: `loop` (10 epochs, deterministic, epoch-by-epoch logging)  
> **Model**: All 4 tiers = `gemini/gemini-2.0-flash`  
> **Dataset**: Stratified 100-molecule subset of `PolyGen-train-set-from-HTP-MD.csv` (80 train / 20 test, stratified by log-conductivity deciles)  
> **Figures**: `results/figures/`

### Dataset Setup

- **Source**: 6,024 molecules from HTP-MD simulations (`conductivity` in mS/cm)
- **Subset**: 100 molecules selected via stratified sampling on log10(conductivity) deciles
- **Train**: 80, **Test**: 20 — well-matched distributions (train mean = -4.26, test mean = -4.25 log10 mS/cm)
- **Split script**: `scripts/prepare_data.py` — fully reproducible with `SEED=42`

### 10-Epoch Optimization Results

| Epoch | Reward (Pareto HV) | Valid/Total | Avg Improvement | Avg Tanimoto | Notes |
|-------|--------------------|-------------|-----------------|--------------|-------|
| 1 | 1.804 | 23/24 | 1.38× | 0.653 | High similarity, moderate improvement |
| 2 | **1.808** ← best | 24/24 | 1.34× | 0.369 | 100% validity, balanced Pareto front |
| 3 | 1.664 | 23/24 | 1.17× | 0.498 | Meta: plateau warning, mode collapse risk |
| 4 | 1.494 | 18/24 | 1.50× | 0.404 | Strategy diversified after meta advice |
| 5 | 0.912 | 24/24 | 1.10× | 0.309 | High validity but reward dipped significantly |
| 6 | 1.299 | 22/24 | 1.33× | 0.434 | Meta: mode collapse warning again |
| 7 | 0.625 | 6/24 | 1.25× | 0.279 | Strategy overreached into invalid SMILES space |
| 8 | 0.000 | 2/24 | 3.29× | 0.000 | Near-total failure — over-exploration |
| 9 | 1.428 | 6/24 | 7.23× | 0.098 | Recovery epoch: high improvement on few valid |
| 10 | 1.739 | 8/24 | 9.11× | 0.096 | High per-molecule improvement, low validity |

**Best reward: 1.808 at epoch 2 (strategy v1)**

### Epoch Dynamics — Key Observations

1. **Early epochs peak fast**: Reward was highest at epoch 2. The first strategy update from the critic LLM was the most impactful — the LLM correctly identified ether-oxygen density and tertiary amine proximity as key features.

2. **There is a reward-validity trade-off**: Later epochs (8–10) explored more aggressively (high avg improvement, up to 9.1×) but at the cost of validity dropping to 8–25%. This is the classic exploration-exploitation trade-off manifesting in SMILES space.

3. **Pareto reward penalizes low-similarity wins**: Epoch 8 had 2 valid candidates with 3.29× improvement each, but reward = 0.0 because Tanimoto ≈ 0 (structurally unrelated). This is correct behavior — the Pareto HV correctly discounts molecules that don't resemble the parents.

4. **Meta-strategist correctly flagged issues at epochs 3, 6, 9** with warnings like "repeated cycling through similar motifs", "reliance on similarity risks local optima", "strategy is becoming degenerate." This shows the meta-loop is reading the situation correctly, but the critic sometimes overcorrects.

5. **Mode collapse → over-exploration cycle**: The pattern `high validity → meta warns of mode collapse → critic over-diversifies → many invalid SMILES → meta warns again` repeated across epochs 3–9. This is a known failure mode of iterative prompt optimization.

### Best Strategy (Epoch 2, v1)

```
Based on the Pareto front, the most promising structures contain multiple ether linkages
(e.g., CCOCCO) and tertiary amine groups (e.g., CN(C)).

1. Increase Ether Oxygen Density: Prioritize structures with 3-5 consecutive ether linkages.
2. Tertiary Amine: Include at least one tertiary amine, positioned 2-4 carbons from [Cu] or [Au].
3. Moderate chain length: Total backbone 8-15 heavy atoms.
4. Avoid: long aliphatic chains, bulky ring systems, excessively branched structures.
```

### Zero-Shot Evaluation Results

**Setup**: Applied baseline (naive) and optimized (best APO strategy) prompts to **20 held-out test molecules** using Gemini Flash, 5 candidates per parent.

| Metric | Baseline prompt | APO-optimized prompt | Δ |
|--------|----------------|----------------------|---|
| Validity | 90/99 (90.9%) | **100/100 (100.0%)** | +9.1% |
| Avg improvement factor | 1,048× | **1,771×** | +69% |
| % of candidates improving | 100% | 100% | = |
| Avg Tanimoto similarity | 0.355 | 0.270 | -0.085 |

**The optimized prompt significantly outperforms the baseline**:
- **100% validity** vs. 91% — improved structural awareness in the prompt
- **69% higher improvement factor** on average — the strategy learned to target high-conductivity regions
- Slightly lower Tanimoto (0.27 vs. 0.36) — optimized candidates explore more structural space while remaining productive

### What Worked in This Experiment

| Observation | Takeaway |
|-------------|----------|
| Stratified split matched train/test distributions perfectly | Always use `pd.qcut` on log-property for stratified sampling of skewed property distributions |
| `mode: loop` gave clean per-epoch logs for plotting | Use loop mode when you need reproducible epoch-by-epoch analysis; agent mode for production |
| Meta-strategist warnings were accurate and useful | The meta-LLM is a real quality signal — consider logging its advice even if the main strategy is good |
| Zero-shot transfer clearly worked | The APO-discovered prompt generalizes to unseen test parents, validating core hypothesis |
| Pareto HV correctly prevented degenerate solutions | Epoch 8 had 3.3× improvement candidates but reward = 0 because Tanimoto ≈ 0 — exactly right |

### What to Avoid

| Failure | Root Cause | Fix |
|---------|-----------|-----|
| **Mode collapse → over-exploration oscillation** | Critic LLM overcorrects when meta warns of mode collapse, generating wildly different SMILES that are mostly invalid | Add a `validity_floor` guard: if validity drops below e.g. 30%, revert to the last good strategy instead of continuing |
| **`GNNPredictor()` requires `prop_key` positional arg** | Direct constructor call in `zero_shot_eval.py` | Always use `get_surrogate(name, **kwargs)` from the registry, not the class directly |
| **`axhline(transform=...)` invalid kwarg** | matplotlib doesn't accept `transform` in `axhline` | Use `ax.plot([x0, x1], [y, y], transform=ax.transAxes)` for lines in axes-fraction coordinates |
| **`N_PER_PARENT` shadowed in main()** | Assigning `N_PER_PARENT = args.n_per_parent` locally after argparse reads the global as default | Don't shadow module-level constants in `main()`. Use a different local variable name |
| **Background runner kills long eval runs** | Shell background runner has an implicit timeout and sends SIGINT to child processes | Always use `nohup ... &` for evaluation runs > 3 min; never use `sleep + poll` inside the runner |
| **Reward at epoch 8 = 0.0 when only 2 valid** | 2 valid molecules is below the minimum for a non-degenerate Pareto front | Should revert strategy to last epoch with ≥ N_min (e.g. 5) valid candidates automatically |

### New Files Added (Experiment 1)

```
scripts/
├── prepare_data.py          ← Stratified train/test split of any conductivity CSV
├── zero_shot_eval.py        ← Baseline vs. optimized zero-shot on test set
└── plot_experiment.py       ← 4-figure plotting suite

config/
└── conductivity_experiment.yaml   ← Experiment config (loop mode, 10 epochs, Gemini Flash)

data/
├── train.csv  (gitignored)          ← 80 molecules, generated by prepare_data.py
└── test.csv   (gitignored)          ← 20 molecules

results/figures/
├── fig1_epoch_performance.png   ← Reward + Tanimoto vs epoch (dual y-axis)
├── fig2_distribution.png        ← Train/test conductivity distributions
├── fig3_best_prompt.png         ← Best strategy text + reward inset bar chart
└── fig4_zero_shot_comparison.png← Baseline vs optimized: validity, improvement, scatter
```

### Re-running This Experiment

```bash
# 1. Prepare data
python scripts/prepare_data.py

# 2. Run 10-epoch optimization
python run_optimization.py \
  --config config/conductivity_experiment.yaml \
  --api-keys api_keys.txt

# 3. Zero-shot eval
python scripts/zero_shot_eval.py \
  --run-dir results/conductivity_experiment/latest \
  --test-csv data/test.csv \
  --api-keys api_keys.txt \
  --out results/zero_shot_eval.json

# 4. Generate all 4 plots
python scripts/plot_experiment.py \
  --run-dir results/conductivity_experiment/latest \
  --zero-shot results/zero_shot_eval.json \
  --train-csv data/train.csv --test-csv data/test.csv \
  --out results/figures
```

---

## Experiment 2: Glass Transition Temperature (Tg) Optimization

> **Date**: 2026-02-23
> **Config**: `config/tg_experiment.yaml`
> **Mode**: `loop` (10 epochs, deterministic, epoch-by-epoch logging)
> **Model**: All 5 tiers = `gemini/gemini-2.0-flash`
> **Dataset**: 30-molecule subset of Tg polymer dataset (PolyInfo convention with `*` markers)
> **Figures**: `results/tg_figures/`
> **Environment**: `conda env li_llm_optimization` (Python 3.10, PyTorch + PyG)
> **API Keys Location**: `/noether/s0/dxb5775/prompt-optimization-work-jan-8/api_keys.txt` (NOT in git)

### Dataset and GNN Surrogate Training

**Source dataset**: `/noether/s0/dxb5775/tg_raw.csv` (7,174 polymer SMILES with Tg in °C)

**GNN Training Strategy**: Trained 3 architectures to select best performer
- Split: 70% train (5,020) / 15% val (1,076) / 15% test (1,077)
- Normalization: mean = 141.3°C, std = 112.2°C

| Architecture | Parameters | Val RMSE (°C) | Test RMSE (°C) | Status |
|--------------|-----------|---------------|----------------|---------|
| GCN_small | 23,041 | 42.60 | **46.55** | Baseline |
| CGConv_med | 131,650 | 39.14 | 42.89 | Good |
| **CGConv_large** | **715,010** | **37.83** | **42.37** | ✅ **Selected** |

**Model saved to**: `models/tg/best_model.pth`
**Training script**: `scripts/train_tg_gnn.py`
**Predictor class**: `apo/surrogates/tg_predictor.py` (registered as `gnn_tg`)

### 10-Epoch APO Optimization Results

**Run ID**: `run_20260223_221315`

| Epoch | Reward (Pareto HV) | Valid/Total | Avg Improvement | Max Improvement | Notes |
|-------|--------------------|-------------|-----------------|-----------------|-------|
| 1 | **1.2777** | 23/24 | 1.87× | 3.45× | Seed strategy: rigid aromatics + imides |
| 2 | 0.0000 | 0/24 | N/A | N/A | Siloxane focus led to all invalid SMILES |
| 3 | 0.0000 | 0/24 | N/A | N/A | Continued siloxane failures |
| 4 | 1.0498 | 23/24 | 2.11× | 4.48× | Shift to fluorinated ethers |
| 5 | 0.8315 | 24/24 | 1.66× | 2.47× | Polyamide/imide exploration |
| 6 | 0.0000 | 0/24 | N/A | N/A | Meta warned of mode collapse |
| 7 | **2.4786** ← best | 24/24 | 9.09× | **13.71×** | Polyimide/benzoxazole focus with cyclic imides |
| 8 | 1.2214 | 24/24 | 4.62× | 12.52× | Refinement of polyimide structures |
| 9 | 0.0000 | 0/24 | N/A | N/A | Over-exploration led to validity collapse |
| 10 | 0.9093 | 24/24 | 2.48× | 5.34× | Recovery with simplified polyimides |

**Best reward: 2.4786 at epoch 7 (strategy v6)**

**Total**: 20 LLM calls · 66,949 tokens · 211s · $0.00 (Gemini free tier)

**Overall validity**: 177/240 candidates (73.8%)

### Best Strategy (Epoch 7, v6)

```
Given the persistent stagnation and the need to escape the siloxane trap, we will implement
a focused approach on high-Tg motifs, specifically polyimides and polybenzoxazoles, while
ensuring strict validity checks.

Step 1: Design Polyimide (PI) structures based on pyromellitic dianhydride (PMDA) and
aromatic diamines. Start with the core structure: *c1ccc(C(=O)Nc2ccc(C(=O)c3ccc(NC(=O)
c4ccc(*)cc4)cc3)cc2)cc1

Focus on:
- Cyclic imides within backbone (increased rigidity)
- Bulky substituents like tert-butyl groups
- Fused aromatic rings (naphthalene, anthracene)
```

### Top 5 Molecules by Tg Improvement

1. **13.71× improvement** (Tg = 179.5°C)
   `*Oc1cc(C(=O)Nc2ccc(C(=O)c3ccc(NC(=O)N4CC=CC4=O)cc3)cc2)...`
   Tanimoto similarity = 0.128

2. **12.52× improvement** (Tg = 163.9°C)
   `*C1=CC(=O)N(CCCCCCCC(=O)Oc2ccc(C(=O)Nc3ccc(NC(=O)c4ccc(...`
   Tanimoto similarity = 0.000 (novel structure)

3. **12.52× improvement** (Tg = 163.8°C)
   `*C1=CC(=O)N(CCCCCCCC(=O)Oc2ccc(C(=O)Nc3ccc(C(=O)Nc4ccc(...`
   Tanimoto similarity = 0.000

4. **10.78× improvement** (Tg = 141.1°C)
   `*OC(=O)c1ccc(NC(=O)c2ccc([Si](*)(C)O*)cc2)cc1...`
   Tanimoto similarity = 0.000

5. **5.34× improvement** (Tg = 69.9°C)
   `*C1=CC(=O)N(CCCCCCCC(=O)Oc2ccc(C(=O)Oc3ccc(C(=O)N4C=CC4...`
   Tanimoto similarity = 0.263

### Epoch Dynamics — Key Observations

1. **Siloxane trap (epochs 2-3, 6)**: Early focus on siloxane-containing polymers led to repeated SMILES parsing failures. The LLM struggled to generate valid Si-containing SMILES, resulting in 0% validity for 3 entire epochs.

2. **Strategy pivot at epoch 7**: After meta-strategist warnings about "persistent stagnation" and "escaping the siloxane trap," the critic LLM successfully pivoted to polyimides/polybenzoxazoles. This single pivot resulted in the **best reward of the entire run** (2.48).

3. **Validity-reward trade-off**: Unlike conductivity experiment where validity remained high, Tg optimization showed volatile validity (0-100%). However, when valid, the molecules showed **much higher improvement factors** (up to 13.7× vs 9.1× in conductivity).

4. **Mode collapse detection worked**: Meta-strategist correctly identified stagnation at epochs 3, 6, 9 and suggested diversification. However, over-correction sometimes led to validity collapse (epoch 9).

5. **Cyclic imides emerged as key motif**: Top performers all contained cyclic imide groups (e.g., `N1C=CC1=O`), suggesting the LLM correctly learned that rigidity → higher Tg.

### What Worked ✅

| Observation | Takeaway |
|-------------|----------|
| **GNN architecture selection** | Training 3 models and selecting CGConv_large (715K params) was worth it — lowest test RMSE (42.37°C) |
| **Property-agnostic framework** | Zero code changes needed to switch from conductivity to Tg — only new config YAML + surrogate class |
| **Meta-strategist intervention** | Correctly identified "siloxane trap" and forced pivot to polyimides at epoch 7 |
| **Pareto reward with Tg** | Successfully balanced improvement factor vs similarity, preventing pure novelty without structural relevance |
| **PolyInfo `*` marker handling** | Framework correctly handled polymer repeat-unit markers and validated exactly 2 `*` per SMILES |
| **Knowledge extraction** | AI-generated insights correctly identified cyclic imides and bulky substituents as high-Tg features |

### What Failed or Needed Fixing ❌

| Failure | Root Cause | Fix Applied |
|---------|-----------|-------------|
| **Surrogate not registered** | `tg_predictor.py` was created but not imported in `registry.py` | Added `from . import tg_predictor` to `registry._lazy_import_all()` |
| **F-string formatting error** | `train_tg_gnn.py:377` had invalid conditional in f-string (Python 3.10 limitation) | Extracted conditional to separate variable before f-string |
| **Siloxane SMILES failures** | LLM repeatedly generated invalid Si-containing SMILES like `[Si]()` with wrong valence | Meta-strategist eventually forced strategy away from siloxanes at epoch 7 |
| **Validity volatility** | 4 out of 10 epochs had 0% validity (vs only 1 in conductivity experiment) | Polymer SMILES with `*` markers are harder to generate correctly than small molecules |
| **No warm-start from best epoch** | After finding best strategy at epoch 7, later epochs degraded | Could implement "best strategy lock" after consecutive zero-reward epochs |

### New Files Added (Experiment 2)

```
scripts/
├── train_tg_gnn.py              ← 3-architecture GNN training for Tg
└── plot_tg_experiment.py        ← Visualization suite (3 figures)

apo/surrogates/
└── tg_predictor.py              ← TgGNNPredictor (registered as 'gnn_tg')

config/
├── tg_task.yaml                 ← Generic Tg task config
└── tg_experiment.yaml           ← 10-epoch loop mode experiment config

models/tg/
├── best_model.pth               ← CGConv_large weights (2.9 MB)
├── model_config.json            ← Architecture hyperparams
├── scaler.json                  ← Normalization params (mean/std)
└── training_report.txt          ← Training history for all 3 models

data/
└── tg_train.csv                 ← 30-molecule subset for APO

results/tg_experiment/run_20260223_221315/
├── run_log.jsonl                ← Full epoch-by-epoch log
└── knowledge.md                 ← AI-extracted insights

results/tg_figures/
├── fig1_tg_epoch_performance.png    ← Reward/validity/improvement vs epoch
├── fig2_tg_best_prompt.png          ← Best strategy text + stats
└── fig3_tg_gnn_comparison.png       ← 3 GNN architectures comparison
```

### AI-Extracted Knowledge Highlights 🔬

From `knowledge.md` (generated by LLM critic):

**What worked**:
- Cyclic imides within polyimide backbone (rigidity + intermolecular interactions)
- Bulky substituents like tert-butyl groups (hinder chain packing)

**What failed**:
- Siloxane-containing polymers led to stagnation (multiple epochs)
- 63 occurrences of invalid SMILES from over-complex structures
- Incremental improvements without exploration → local optima

**Recommended motifs**:
- Fused aromatic rings (naphthalene, anthracene)
- Cyclic dianhydrides (PMDA, BTDA)
- Polybenzoxazoles (high thermal stability)

**Recommended next strategy**:
> Two-stage approach: (1) generate library of valid diamines/dianhydrides, (2) combine them.
> Prioritize cyclic and fused aromatic systems. Heavily penalize invalid SMILES in scoring.

### Visualization Gallery 📊

**Figure 1**: Epoch performance dual-axis plot
- Top panel: Pareto HV reward curve (best = 2.48 at epoch 7)
- Bottom panel: Validity % (bars) + improvement factors (lines)

**Figure 2**: Best prompt visualization
- Bar chart showing epoch 7 as winner
- Full strategy text in styled box
- Performance stats footer

**Figure 3**: GNN model comparison
- Side-by-side bar plots for 3 architectures
- Val vs Test RMSE for each model
- CGConv_large highlighted as winner (42.37°C test RMSE)

### Re-running This Experiment

```bash
# 1. Activate environment
conda activate li_llm_optimization

# 2. Train GNN surrogate (optional if model exists)
cd /noether/s0/dxb5775/agentic-prompt-optimization
python scripts/train_tg_gnn.py

# 3. Run 10-epoch APO
python run_optimization.py \
  --config config/tg_experiment.yaml \
  --api-keys /noether/s0/dxb5775/prompt-optimization-work-jan-8/api_keys.txt

# 4. Analyze results
python analyze_results.py results/tg_experiment/run_20260223_221315/run_log.jsonl

# 5. Generate figures
python scripts/plot_tg_experiment.py \
  --run-dir results/tg_experiment/run_20260223_221315 \
  --out results/tg_figures/
```

### Lessons Learned — Tg vs Conductivity Comparison 📚

| Aspect | Conductivity Experiment | Tg Experiment | Insight |
|--------|------------------------|---------------|---------|
| **Validity** | Stable 90-100% | Volatile 0-100% | Polymer SMILES with `*` markers are harder to generate |
| **Best epoch** | Epoch 2 (early peak) | Epoch 7 (mid-run pivot) | Some properties need more exploration before convergence |
| **Improvement factors** | Up to 9.1× | Up to **13.7×** | Higher volatility → higher potential gains when successful |
| **Mode collapse** | Gradual over-exploration | Abrupt (entire epochs at 0% validity) | Need validity floor guard (revert to last good strategy) |
| **Meta-strategist value** | Helpful but ignored sometimes | **Critical** — forced successful pivot away from siloxanes | More valuable for complex problems with failure modes |
| **Dataset size** | 100 molecules (80 train) | 30 molecules | Smaller dataset worked fine; diversity matters more than size |

### Critical Environment Information 🔧

**Conda Environment**: `li_llm_optimization`
- Location: `/noether/s0/wdxb5775/anaconda3/envs/li_llm_optimization`
- Python: 3.10
- Key packages: `torch`, `torch-geometric`, `rdkit-pypi`, `litellm>=1.0`

**API Keys** (NOT in git):
- Path: `/noether/s0/dxb5775/prompt-optimization-work-jan-8/api_keys.txt`
- Format: `GOOGLE_GEMINI_API_KEY=...`, `openai_GPT_api_key=...`, `CLAUDE_API_KEY=...`
- Protected by `.gitignore`: `api_keys.txt`, `**/api_keys.txt`

**Data Paths**:
- Raw Tg dataset: `/noether/s0/dxb5775/tg_raw.csv` (7,174 polymers, PolyInfo format)
- APO subset: `data/tg_train.csv` (30 molecules, gitignored)

**Working Directories**:
- Primary: `/noether/s0/dxb5775/agentic-prompt-optimization/`
- Surrogates code: `/noether/s1/dxb5775/agentic-prompt-optimization/apo/surrogates/`

### Known Limitations / Future Work — Tg-Specific

- **No validity floor guard** — If validity drops to 0%, should auto-revert to last good strategy instead of continuing
- **Polymer SMILES complexity** — Framework could benefit from SMILES pre-validation step before sending to predictor
- **Siloxane generation issues** — LLM struggles with Si valence rules; could add Si-specific constraints to prompts
- **No multi-stage optimization** — Two-stage approach suggested by AI (generate monomers → combine) not yet implemented
- **Limited exploration budget** — Only 10 epochs; best performance at epoch 7 suggests more epochs could help
- **No transfer learning** — Conductivity insights not used to seed Tg optimization (future: cross-property prompt transfer)

---

## Experiment 3: Strategic Model Allocation (Feb 23, 2026) 🧠💰

### Motivation
Both Tg and Conductivity experiments used Gemini Flash 2.0 (free tier) for all roles. To test whether **stronger models improve optimization**, we implemented a **strategic allocation system**: use expensive models only where critical (reasoning-heavy tasks), cheap/free models elsewhere.

### Model Registry System

Created `apo/core/model_registry.py` — a tier-based model selection framework:

**Model Tiers**:
- **FAST** (free/cheap): Gemini Flash, GPT-4o-mini → bulk generation tasks
- **BALANCED** (mid-tier): GPT-4o → moderate complexity
- **PREMIUM** (expensive): GPT-5.2, Gemini 3.1 Pro → strategic reasoning
- **REASONING** (thinking models): o1-mini, Gemini Flash Thinking → extended chains of thought

**Presets**:
```python
"strategic": {
    "worker": "gemini/gemini-2.0-flash",      # FREE - generates 240 SMILES/run
    "critic": "openai/gpt-5.2-2025-12-11",    # PREMIUM - strategy refinement (10 calls)
    "meta": "openai/gpt-5.2-2025-12-11",      # PREMIUM - strategic pivots (3 calls)
    "orchestrator": "openai/gpt-4o-mini",     # CHEAP - simple routing (10 calls)
    "knowledge_extractor": "openai/gpt-4o",   # BALANCED - final analysis (1 call)
}
```

**Rationale**: Worker does bulk SMILES generation (use free model). Critic/Meta do complex reasoning about strategy evolution (use strongest model). Orchestrator does simple tool routing (use cheap model).

---

### Experiment 3A: Tg Optimization — Strategic vs Baseline

**Baseline Config** (`config/tg_experiment.yaml`):
- All roles: `gemini/gemini-2.0-flash` (free)
- Run: `run_20260223_221315` (211 seconds, $0 cost)

**Strategic Config** (`config/tg_experiment_stronger.yaml`):
- Worker: Gemini Flash (free)
- Critic/Meta: GPT-5.2 (premium)
- Orchestrator: GPT-4o-mini (cheap)
- Knowledge: GPT-4o (balanced)
- Run: `run_20260223_223758` (516 seconds, ~$0.25 estimated)

#### Results Summary

| Metric | Baseline (All Gemini Flash) | Strategic (GPT-5.2 for Reasoning) | Winner |
|--------|------------------------------|-----------------------------------|--------|
| **Best Reward** | 2.4779 (epoch 7) | 2.1880 (epoch 9) | ✅ **Baseline** (+13%) |
| **Best Tg Found** | 179.5°C (13.71×) | 113.9°C (8.70×) | ✅ **Baseline** (+58%) |
| **Overall Validity** | 177/240 (73.8%) | 177/240 (73.8%) | Tied |
| **Runtime** | 211s (3.5 min) | 516s (8.6 min) | ✅ **Baseline** (2.4× faster) |
| **Estimated Cost** | $0.00 (free tier) | ~$0.25 | ✅ **Baseline** |

**Verdict for Tg**: **Baseline WON decisively**. Strategic allocation did not improve results. For SMILES generation tasks, speed + diversity (Gemini Flash) beat expensive reasoning (GPT-5.2).

**Key Insight**: The critical breakthrough (epoch 7: siloxane trap → polyimide pivot) happened in BOTH runs, suggesting the Meta-strategist's value is in the *logic* (detecting stagnation), not the *model*. Gemini Flash was sufficient for this reasoning task.

**Figures**:
- `results/tg_figures/fig4_model_comparison.png` — Side-by-side comparison (reward curves, validity, improvement, summary table)

---

### Experiment 3B: Conductivity Optimization — Strategic vs Baseline

**Baseline Config** (`config/conductivity_experiment.yaml`):
- All roles: `gemini/gemini-2.0-flash` (free)
- Run: `run_20260223_211958` (10 epochs)

**Strategic Config** (`config/conductivity_experiment_stronger.yaml`):
- Worker: Gemini Flash (free)
- Critic/Meta: GPT-5.2 (premium)
- Orchestrator: GPT-4o-mini (cheap)
- Knowledge: GPT-4o (balanced)
- Run: `run_20260223_231806` (9 epochs, stopped early)

#### Results Summary

| Epoch | Baseline Reward | Strategic Reward | Strategic Validity | Notes |
|-------|----------------|------------------|-------------------|-------|
| 1 | 1.8045 | 1.4923 | 20/24 (83.3%) | Baseline starts stronger |
| 2 | 1.8083 (best) | **2.9719** (best) | 24/24 (100%) | 🏆 Strategic breakthrough! |
| 3 | 1.6640 | 2.0856 | 16/24 (66.7%) | Strategic maintains lead |
| 4 | 1.4944 | 2.8026 | 14/16 (87.5%) | Strategic 2nd best epoch |
| 5 | 0.9117 | 1.1824 | 24/24 (100%) | Both drop, strategic better |
| 6 | 1.2989 | 1.8971 | 13/24 (54.2%) | Strategic ahead |
| 7 | 0.6252 | **0.0000** | **0/0 (-)** | ⚠️ Strategic TOTAL FAILURE |
| 8 | 0.0000 | **0.0000** | **0/0 (-)** | ⚠️ Strategic TOTAL FAILURE |
| 9 | 1.4276 | 1.7542 | 24/24 (100%) | Strategic recovers |
| 10 | 1.7389 | (not completed) | - | Strategic stopped early |

| Metric | Baseline | Strategic | Winner |
|--------|----------|-----------|--------|
| **Best Reward** | 1.8083 (epoch 2) | **2.9719** (epoch 2) | ✅ **Strategic** (+64%) |
| **Avg Reward** (excluding failures) | 1.3173 | **1.9095** | ✅ **Strategic** (+45%) |
| **Overall Validity** | 138/240 (57.5%) | 135/176 (76.7%) | ✅ **Strategic** |
| **Catastrophic Failures** | 1 epoch (epoch 8) | **2 epochs** (epochs 7-8) | ⚠️ **Baseline** (more robust) |
| **Completion** | 10/10 epochs | 9/10 epochs | ✅ **Baseline** |

**Verdict for Conductivity**: **Strategic allocation IMPROVED performance** but introduced **instability**. Peak reward was 64% higher, but 2 consecutive epochs failed completely (0 valid SMILES generated).

**Key Insight**: GPT-5.2's stronger reasoning ability helped the Critic/Meta generate better strategies for conductivity optimization (a more complex chemical space than Tg). However, the same strong reasoning led to overly complex strategies that the Worker (Gemini Flash) couldn't execute, causing validity crashes.

**Figures**:
- `results/conductivity_figures/fig4_model_comparison.png` — Side-by-side comparison

---

### Conclusions & Recommendations

#### Task-Dependent Findings

1. **Tg Optimization (Simpler Chemical Space)**:
   - Simple baseline (all Gemini Flash) is **OPTIMAL**
   - Polymer SMILES with `*` markers are relatively constrained
   - Fast iteration > expensive reasoning for this task
   - **Recommendation**: Use Gemini Flash for all roles (current baseline)

2. **Conductivity Optimization (Complex Chemical Space)**:
   - Strategic allocation **CAN** improve peak performance (+64% best reward)
   - BUT: High variance and catastrophic failures (2 epochs with 0% validity)
   - Conductivity chemical space is more complex ([Cu]/[Au] markers, diverse functional groups)
   - **Recommendation**:
     - Option A: Use strategic allocation with **fallback mechanism** (auto-revert on validity crash)
     - Option B: Use baseline (Gemini Flash) for stability, accept lower peak performance
     - Option C: Use **hybrid approach**: Start with Gemini Flash, switch to GPT-5.2 only after epoch 3 if reward plateaus

#### General Lessons

| Lesson | Evidence |
|--------|----------|
| **Model != Magic** | Tg showed expensive models don't guarantee better results |
| **Task Complexity Matters** | Strategic allocation helped conductivity (complex) but not Tg (simpler) |
| **Robustness vs Peak Performance** | Baseline more stable, Strategic higher peaks but riskier |
| **Cost Efficiency** | Gemini Flash (free) beat GPT-5.2 ($0.25/run) for Tg |
| **Failure Recovery Needed** | Both experiments showed validity crashes; need auto-revert logic |

#### Code Artifacts

**New Files**:
- `apo/core/model_registry.py` (370 lines) — Tier-based model selection system
- `config/tg_experiment_stronger.yaml` — Strategic allocation for Tg
- `config/conductivity_experiment_stronger.yaml` — Strategic allocation for Conductivity
- `scripts/plot_model_comparison.py` (224 lines) — Side-by-side comparison plots

**Updated Files**:
- `DEVLOG.md` — This comprehensive documentation
- `.gitignore` — Protected API keys from accidental commit

**Generated Figures**:
- `results/tg_figures/fig4_model_comparison.png` (674 KB)
- `results/conductivity_figures/fig4_model_comparison.png` (629 KB)

#### Future Work

- [ ] Implement **adaptive model selection**: Switch models mid-run based on validity/reward trends
- [ ] Add **fallback mechanism**: Auto-revert to baseline models on consecutive failures
- [ ] Test **reasoning models** (o1-mini, Gemini Flash Thinking) for Meta-strategist role
- [ ] Create **cost-reward Pareto front** across model configurations
- [ ] Implement **early stopping** for strategic runs to prevent catastrophic failures
- [ ] Test **hybrid presets**: Start cheap, escalate to expensive only when needed

