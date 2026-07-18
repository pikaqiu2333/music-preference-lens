# Generated-Reason Interpretability Plan

## Purpose

This project is intentionally not moving toward a music recommendation
benchmark right now. The near-term research question is narrower:

> When a model generates a recommendation reason, do the internal features
> during prompt reading and answer generation align with the factors named in
> that reason?

Music recommendation remains the testbed because it contains subjective,
context-sensitive reasons, but the artifact we want is an interpretability
method rather than a leaderboard.

## Current Evidence

The phrase-level Qwen-Scope probe found separable layer 24 features for several
product-readable music dimensions:

| Working Dimension | Candidate Features |
|---|---|
| high energy / rave | `layer24.feature7884`, `layer24.feature32078` |
| quiet / low presence | negative `layer24.feature32078`, positive `layer24.feature30584` |
| emotional vocal / lyric salience | `layer24.feature8706` |
| no vocals / focus constraint | `layer24.feature24979`, `layer24.feature24916` |

Identity/source-binding controls stayed near zero, which makes them useful
negative controls for generated-answer runs.

## Next Question

The phrase-level run only measured prompt-side spans. The next step is to
inspect generated answers:

- If the model writes "high energy" or "rave pressure", do energy features move
  during the generated reason?
- If the model writes "emotional vocal" or "story-driven chorus", does
  `feature8706` move?
- If the model writes "no vocals", "focus", or "no lyric attention", do
  `feature24979` and `feature24916` move while vocal/emotional features drop?
- If the answer sounds plausible but the expected feature movement is missing,
  is that an explanation-faithfulness failure case?

## Experiment Shape

Each generated-reason probe should contain:

- stable user profile,
- original listening need,
- counterfactual listening need,
- unchanged candidate cards,
- expected reason factors for each variant,
- candidate feature hypotheses,
- generated JSON answer from the model,
- activation or SAE-feature traces over both prompt and answer tokens.

The analysis unit is not "correct recommendation". It is:

> reason factor named by the model -> expected feature movement -> observed
> movement during generation.

## Preferred Metrics

### Reason-Feature Alignment

For each reason factor, map it to candidate features and check whether those
features increase on the relevant generated reason span.

Examples:

- `high energy` -> `feature32078` and `feature7884`
- `emotional vocal` -> `feature8706`
- `no vocals` -> `feature24979` and `feature24916`

### Prompt-Answer Continuity

Compare feature movement in the current-need phrase span with movement in the
generated reason. A clean case should show the same direction in both places.

### Missing Signal

Flag cases where the model names a factor but the expected feature movement is
weak, absent, or contradicted by another feature.

### Control Features

Keep identity/source-binding features as negative controls:

- `layer24.feature19603`
- `layer24.feature30224`
- `layer24.feature22232`

They should not explain music preference reasons.

## What We Are Not Doing Yet

We are not building a benchmark or ranking models by music taste. We are also
not asking humans to decide the single correct track. Human judgment can come
later as qualitative auditing, but the current contribution is the
feature-alignment method.

## Immediate Deliverables

1. `data/qwen_scope_music_generation_specs.jsonl`
2. `prompts/generated_reason_probe_prompt.md`
3. `scripts/export_generation_reason_prompts.py`
4. `runs/qwen_scope_music_generation_prompt_pack.jsonl`
5. `reports/qwen_scope_music_generation_probe_plan.md`

After those are in place, run a Hugging Face job that records feature
activations across generated answer tokens and summarizes reason-feature
alignment.
