# %% [markdown]
# # Music Preference Lens: Mechanistic Probe Pilot
#
# This notebook is the first mechanistic-interpretability pilot for Music
# Preference Lens.
#
# The goal is not to prove that a model "understands music taste". The goal is
# narrower:
#
# > When a recommendation prompt changes from one listening context to another,
# > do target dimensions such as `rain_night_drive`, `high_energy_workout`, or
# > `instrumental_focus` show corresponding movement in model-internal features?
#
# Start with the hand-curated six-probe pilot:
#
# - `data/mechanistic_pilot_specs.jsonl`
# - `runs/mechanistic_probe_pairs.jsonl`
# - `runs/feature_observations_template.csv`

# %%
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path("..").resolve()
if not (PROJECT_ROOT / "data" / "mechanistic_pilot_specs.jsonl").exists():
    PROJECT_ROOT = Path(".").resolve()

PILOT_SPECS = PROJECT_ROOT / "data" / "mechanistic_pilot_specs.jsonl"
MODEL_RESOURCES = PROJECT_ROOT / "config" / "model_resources.json"
DIMENSIONS = PROJECT_ROOT / "config" / "interpretability_dimensions.json"

print(PROJECT_ROOT)

# %%
def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


pilot_specs = load_jsonl(PILOT_SPECS)
model_resources = json.loads(MODEL_RESOURCES.read_text(encoding="utf-8"))
dimensions = json.loads(DIMENSIONS.read_text(encoding="utf-8"))["dimensions"]

len(pilot_specs), list(model_resources["resources"])[:3], len(dimensions)

# %% [markdown]
# ## 1. Inspect Probe Contrasts
#
# Each probe has:
#
# - original text
# - counterfactual text
# - added dimensions
# - removed dimensions
# - expected candidate movement

# %%
for spec in pilot_specs:
    print("\n" + "=" * 88)
    print(spec["probe_id"])
    print("added:", spec["added_dimensions"])
    print("removed:", spec["removed_dimensions"])
    print("expected:", spec["expected_effect"])

# %% [markdown]
# ## 2. Pick The First Model
#
# Recommended first pass:
#
# - `gemma_scope_2_270m_it` for the cheapest workflow validation.
# - `gemma_scope_2_1b_it` for a stronger Gemma pilot.
# - `qwen_scope_qwen3_1_7b_base` for a Chinese-oriented compact pass.
# - `qwen_scope_qwen3_8b_base` once compute is available.
#
# The code below only selects the config. Actual model loading depends on your
# local GPU / Colab / HF Jobs environment.

# %%
model_key = "gemma_scope_2_270m_it"
model_cfg = model_resources["resources"][model_key]
model_cfg

# %% [markdown]
# ## 3. Minimal Readout Contract
#
# Whatever tool is used (SAELens, TransformerLens, Gemma Scope, Qwen-Scope, or
# Neuronpedia exports), the result should fill this shape:
#
# ```python
# {
#   "probe_id": "...",
#   "dimension_id": "...",
#   "feature_id": "...",
#   "feature_label": "...",
#   "original_activation": 0.0,
#   "counterfactual_activation": 0.0,
#   "observed_delta": 0.0,
#   "observed_direction": "increase|decrease|flat",
#   "evidence_note": "..."
# }
# ```
#
# For the first pilot, feature labels are tentative. We care more about whether
# context/constraint contrasts produce stable movement than whether any single
# feature label is perfect.

# %%
def expected_rows(spec):
    for dimension in spec["target_dimensions"]:
        if dimension in spec["added_dimensions"]:
            expected = "increase"
        elif dimension in spec["removed_dimensions"]:
            expected = "decrease"
        else:
            expected = "inspect"
        yield {
            "probe_id": spec["probe_id"],
            "dimension_id": dimension,
            "expected_direction": expected,
        }


observation_plan = [row for spec in pilot_specs for row in expected_rows(spec)]
observation_plan[:10], len(observation_plan)

# %% [markdown]
# ## 4. Placeholder: Load Model And SAE
#
# This section is intentionally a placeholder. Keep model-specific code in one
# cell so we can swap Gemma Scope and Qwen-Scope without rewriting the analysis.
#
# Pseudocode:
#
# ```python
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from sae_lens import SAE
#
# tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
# model = AutoModelForCausalLM.from_pretrained(
#     model_cfg["base_model"],
#     device_map="auto",
#     torch_dtype="auto",
# )
# sae = SAE.from_pretrained(model_cfg["sae_repo"], ...)
# ```
#
# Exact SAE loading arguments vary by repo; verify the model card before running.

# %%
# Placeholder only:
print("Base model:", model_cfg["base_model"])
print("SAE repo:", model_cfg["sae_repo"])

# %% [markdown]
# ## 5. Placeholder: Compute Activations
#
# A model-specific helper should:
#
# 1. tokenize `original_text` and `counterfactual_text`,
# 2. collect activations at the SAE hook point,
# 3. encode activations through the SAE,
# 4. aggregate feature activations over the relevant final/context tokens,
# 5. compare original versus counterfactual.
#
# Keep the aggregation simple for the pilot:
#
# - mean activation over all prompt tokens, or
# - mean activation over context sentence tokens, or
# - final-token activation before answer generation.

# %%
def compare_probe_placeholder(spec):
    return {
        "probe_id": spec["probe_id"],
        "status": "placeholder",
        "original_chars": len(spec["original_text"]),
        "counterfactual_chars": len(spec["counterfactual_text"]),
        "target_dimensions": spec["target_dimensions"],
    }


placeholder_results = [compare_probe_placeholder(spec) for spec in pilot_specs]
placeholder_results

# %% [markdown]
# ## 6. Analysis Questions
#
# After filling real feature observations, answer:
#
# 1. Do added dimensions tend to increase?
# 2. Do removed dimensions tend to decrease?
# 3. Which dimensions are easy to observe?
# 4. Which dimensions remain behaviorally useful but mechanistically unclear?
# 5. Which recommendation explanations look plausible but lack matching
#    internal movement?
