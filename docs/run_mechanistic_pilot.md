# Running The Mechanistic Pilot

This document explains how to move from the prepared probe specs to an actual
SAE/J-lens style run.

## Prepared Inputs

- `data/mechanistic_pilot_specs.jsonl`
  - Six hand-curated contrast probes.
- `runs/mechanistic_probe_pairs.jsonl`
  - Original/counterfactual text rows for programmatic model runs.
- `runs/mechanistic_probe_pairs.csv`
  - Same pair data for manual review.
- `runs/feature_observations_template.csv`
  - Observation table to fill after feature analysis.
- `data/identity_probe_tasks.jsonl`
  - Auxiliary identity and role-binding task pairs.
- `runs/identity_probe_specs.jsonl`
  - Generated routine-vs-complex identity probe specs after export.
- `notebooks/mechanistic_probe_pilot.py`
  - Jupytext-style notebook scaffold.

## Recommended Model Order

1. `gemma_scope_2_270m_it`
   - Base: `google/gemma-3-270m-it`
   - SAE: `google/gemma-scope-2-270m-it`
   - Use this as the first smoke test for the workflow.

2. `gemma_scope_2_1b_it`
   - Base: `google/gemma-3-1b-it`
   - SAE: `google/gemma-scope-2-1b-it`
   - Use this after the 270M workflow runs.

3. `qwen_scope_qwen3_1_7b_base`
   - Base: `Qwen/Qwen3-1.7B-Base`
   - SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
   - Use this for a compact Chinese-relevant pass.

4. `qwen_scope_qwen3_8b_base`
   - Base: `Qwen/Qwen3-8B-Base`
   - SAE: `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50`
   - Use this after the smaller workflow is stable.

## Local Run Path

Use local execution for validation, prompt export, dry-runs, and report
generation. Use local real runs only if you have enough GPU memory and a clean
Python 3.11/3.12 environment.

1. Install dependencies in a dedicated environment.
2. Open `notebooks/mechanistic_probe_pilot.py` as a notebook.
3. Load one base model and one SAE resource.
4. Run the six probes.
5. Fill `runs/feature_observations_template.csv`.

Keep the first run small:

- one model,
- one layer or SAE source,
- six probes,
- one aggregation method.

## Colab / Remote Notebook Path

This is the easiest first option if local GPU is not ready.

1. Upload the repository folder or just the prepared JSONL
   files.
2. Run the notebook with `gemma_scope_2_270m_it`.
3. Export the filled feature observation CSV.
4. Bring the CSV back into this repo.

## Hugging Face Jobs Path

Use this when the workflow is stable and you want reproducibility. This should
be the default path for real activation/SAE runs if local GPU is not ready.

1. Package the probe specs and runner script.
2. Run on a small GPU job for Gemma 270M or 1B.
3. Save observations as a Hub artifact or dataset before the job exits.
4. Repeat with Qwen 1.7B or Qwen 8B.

Important: Hugging Face Jobs environments are ephemeral. Any observation CSV or
report must be pushed back to the Hub or another persistent store during the
job. Treat this repo as the control plane and report workspace, not as the
heavy compute environment.

## Identity Probe Pack

To generate the auxiliary identity/role-binding prompts and probe specs:

```powershell
python scripts/export_j_space_tasks.py --tasks data/identity_probe_tasks.jsonl --prompt-pack runs/identity_probe_prompt_pack.jsonl --csv runs/identity_probe_tasks.csv --markdown reports/identity_probe_task_summary.md --title "Identity Probe Task Summary"
python scripts/build_j_space_probe_specs.py --tasks data/identity_probe_tasks.jsonl --output runs/identity_probe_specs.jsonl
python scripts/run_sae_probe.py --mode plan --specs runs/identity_probe_specs.jsonl --plan-output reports/identity_sae_probe_run_plan.md
python scripts/run_sae_probe.py --mode dry-run --specs runs/identity_probe_specs.jsonl --output runs/identity_sae_probe_observations_dry_run.csv
```

Identity probes should be reported separately from the music recommendation
results. They test role binding and non-overclaiming, not playlist quality.

## Qwen-Scope Path

Qwen-Scope publishes TopK SAE checkpoints as per-layer files such as
`layer5.sae.pt`. Use the dedicated runner instead of the SAELens release runner:

```powershell
python scripts/run_qwen_scope_probe.py --mode plan --specs runs/identity_probe_specs.jsonl --plan-output reports/qwen_scope_identity_probe_run_plan.md
python scripts/run_qwen_scope_probe.py --mode dry-run --specs runs/identity_probe_specs.jsonl --output runs/qwen_scope_identity_probe_observations_dry_run.csv
```

For a real run, use a GPU environment:

```powershell
python scripts/run_qwen_scope_probe.py --mode run --specs runs/identity_probe_specs.jsonl --model-key qwen_scope_qwen3_1_7b_base --layer 5 --report-top-k 20 --output runs/qwen_scope_identity_probe_observations.csv
```

The first Hugging Face smoke run used `Qwen/Qwen3-1.7B-Base` with
`Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50` at layer 5 and completed in about one
minute on `t4-small`.

## What To Measure First

For each probe and target dimension:

- expected direction: increase/decrease/inspect
- observed activation delta
- top feature IDs that moved
- tentative feature label
- whether the movement matches behavioral ranking movement

## Success Criteria

The first pilot succeeds if at least some clean probes show aligned movement:

- `high_energy_workout` or `rave_hard_energy` increases when context changes to
  gym/rave.
- `instrumental_focus` increases under no-vocals study context.
- `party_warmup_groove` decreases when context changes to rave or low-presence
  background.

## Failure Is Still Useful

Useful negative findings:

- behavior changes but no feature group moves,
- explanations cite a factor but no matching feature movement appears,
- feature movement appears but does not map cleanly to product-readable labels,
- Chinese context terms are weaker than English tags in Gemma but stronger in
  Qwen.
