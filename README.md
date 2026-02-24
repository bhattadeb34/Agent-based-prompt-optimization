# Agentic Prompt Optimization (APO)

A production-quality, model-agnostic framework for **iterative prompt optimization** guided by surrogate property predictors. Designed for polymer electrolyte discovery but works for any molecular property task.

---

## Key Features

- 🔄 **3-tier meta-optimization**: Worker (generates molecules) → Critic (refines strategy) → Strategist (high-level guidance)
- 🔌 **Model-agnostic**: Uses [LiteLLM](https://github.com/BerriAI/litellm) — swap any model in one YAML line (`openai/gpt-4o`, `gemini/gemini-2.0-flash`, `anthropic/claude-3-opus`, etc.)
- 🧪 **Pluggable surrogates**: Abstract `SurrogatePredictor` interface with registry — swap GNN→Tg model with one config change
- 📊 **Pareto-aware optimization**: Maximises hypervolume of (property improvement, Tanimoto similarity) Pareto front
- 📝 **Structured logging**: JSONL run logs with full candidate traces, reward history, and Actionable Side Information (failure patterns)
- 🧠 **Knowledge extraction**: End-of-run LLM synthesis producing `knowledge.md` — what the agent learned and where to go next

---

## Architecture

```
config/conductivity_task.yaml   ← task, model, optimization config
    │
run_optimization.py             ← CLI entrypoint
    │
apo/engine.py                   ← orchestration loop
    │
    ├── optimizer/inner_loop.py  ← Worker: LLM generates SMILES
    │       + surrogate.predict() + validate + Tanimoto
    │
    ├── optimizer/outer_loop.py  ← Critic: analyses Pareto front → refined strategy
    │       + reward_fn.compute() + PromptState versioning
    │
    ├── optimizer/meta_loop.py   ← Strategist: high-level guidance every K epochs  
    │
    ├── surrogates/              ← Pluggable property predictors
    │   ├── gnn_predictor.py     ← GNN for conductivity, diffusivity, etc.
    │   └── registry.py          ← @register("name") decorator
    │
    ├── core/
    │   ├── llm_client.py        ← LiteLLM wrapper with retry + token tracking
    │   ├── prompt_state.py      ← PromptState dataclass + history
    │   └── reward.py            ← ParetoHypervolume, WeightedSum, PropertyOnly
    │
    └── logging/
        ├── run_logger.py        ← JSONL run logger
        └── knowledge_extractor.py ← End-of-run knowledge synthesis
```

---

## Quick Start

```bash
# Activate environment
conda activate li_llm_optimization

# Install litellm (if not already)
pip install litellm pyyaml

# Dry run to validate config and dataset
python run_optimization.py --config config/conductivity_task.yaml --dry-run \
    --api-keys /path/to/api_keys.txt

# Run full optimization (10 epochs, 3 candidates/molecule)
python run_optimization.py --config config/conductivity_task.yaml \
    --api-keys /path/to/api_keys.txt

# Quick test run (override YAML settings)
python run_optimization.py --config config/conductivity_task.yaml \
    --api-keys /path/to/api_keys.txt \
    --override optimization.n_outer_epochs=3 \
    --override optimization.n_per_molecule=2 \
    --override task.sample_size=5

# Analyse results
python analyze_results.py runs/latest/run_log.jsonl
```

---

## Config Reference (`config/conductivity_task.yaml`)

```yaml
task:
  surrogate: gnn_conductivity           # swap with "gnn_li_diffusivity", or your custom model
  model_base_path: /path/to/gnn/weights

models:
  worker: gemini/gemini-2.0-flash       # any LiteLLM model string
  critic: openai/gpt-4o-mini
  meta: openai/gpt-4o

optimization:
  n_outer_epochs: 10
  n_per_molecule: 3
  meta_interval: 3                      # meta-strategist fires every 3 epochs
  reward_function: pareto_hypervolume   # or: weighted_sum, property_only

output:
  extract_knowledge: true               # runs knowledge.md synthesis at end
```

---

## Adding a New Surrogate

```python
from apo.surrogates.base import SurrogatePredictor
from apo.surrogates.registry import register

@register("my_tg_model")
class TgPredictor(SurrogatePredictor):
    property_name = "Glass Transition Temperature"
    property_units = "K"
    maximize = True

    def __init__(self, model_base_path: str):
        # load your model weights here
        ...

    def predict(self, smiles_list):
        # return list of float values
        ...
```

Then in YAML: `surrogate: my_tg_model`

---

## Output Files

After each run, `runs/<run_id>/` contains:

| File | Description |
|------|------------|
| `run_log.jsonl` | One JSON line per epoch: full candidate traces, rewards, Pareto data |
| `config.json` | Snapshot of config used for this run |
| `prompt_history.json` | All PromptState versions with scores and rationales |
| `knowledge.md` | LLM-synthesised knowledge report (what worked, what failed, next strategy) |

---

## Running Tests

```bash
cd agentic-prompt-optimization
python -m pytest tests/ -v
# 27 tests, all passing (unit + integration with mocked LLMs)
```

---

## GEPA-Inspired Design Principles

This framework draws from [GEPA](https://github.com/gepa-ai/gepa)'s core ideas:
- **Actionable Side Information (ASI)**: failure traces (invalid SMILES reasons, prediction errors) are fed back to the critic verbatim
- **Pareto-aware selection**: candidates are evaluated on a multi-objective front, not a single scalar
- **Reflective text evolution**: the critic reads full execution traces, not just summary statistics

---

## API Keys

Expected in `api_keys.txt` (same format as original repo):
```
GOOGLE_GEMINI_API_KEY='...'
openai_GPT_api_key='...'
CLAUDE_API_KEY='...'
```
