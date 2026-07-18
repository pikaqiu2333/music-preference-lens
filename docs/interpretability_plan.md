# Interpretability Plan

## Core Shift

The project is not primarily a music recommendation benchmark. It is an
interpretability project that uses music recommendation as a rich, subjective
testbed.

The key question is:

> When a model says it recommends a track because of mood, context, taste, or
> constraints, can we observe matching sensitivity in either behavior or
> internal representations?

## Current Priority

The current priority is generated-reason interpretability, not benchmark
construction. Behavioral ranking changes are useful only when they create
answer text and contrastive spans that can be inspected with SAE features.

The near-term loop is:

1. matched prompt phrase -> candidate feature movement,
2. generated recommendation reason -> answer-token feature movement,
3. reason text -> expected feature label,
4. alignment or missing-signal case.

## Two-Layer Explanation

### Layer 1: Behavioral Trace Generation

Behavioral probes are used to generate recommendation reasons under controlled
counterfactuals. They are not treated as a final benchmark score.

Example:

- Original: rainy night drive, lonely but not too loud.
- Counterfactual: gym sprint, high energy and strong beat.
- Expected behavior: high-energy EDM should rise; rainy/night/solitude reasons
  should stop dominating.

This layer can be run with Codex during research design.

### Layer 2: Mechanistic Sensitivity

Mechanistic probes ask whether an open-weight model's internal signals move
with the same factors.

Example:

- Compare activations for `rainy night drive` versus `gym sprint`.
- Look for features associated with low energy, night, driving, solitude,
  high energy, rave, or workout.
- Compare whether the cited explanation factors correspond to feature movement.

This layer needs open-weight models and feature tools such as Gemma Scope,
Qwen-Scope, SAELens, TransformerLens, or Neuronpedia exports.

### Layer 3: Workspace Pressure

Workspace/J-space signals may be weak for routine recommendation tasks. Add a
third layer that compares simple tag-matching tasks with complex tasks that
require planning, conflict resolution, evidence integration, and repair.

Use `data/j_space_complexity_tasks.jsonl` to test whether more complex
recommendation tasks produce clearer internal movement than routine baselines.

## Why Music Recommendation Is a Good Testbed

Music recommendation is useful for interpretability because it combines:

- Hard constraints: "no vocals", "not too loud", "Chinese lyrics".
- Soft context: "rainy night drive", "party warm-up", "deep work".
- Taste priors: electronic, City Pop, funk, ambient.
- Subjective plausibility: multiple tracks can be defensible.
- Explanation risk: recommendation copy can sound persuasive even when it is
  weakly grounded.

This lets us avoid pretending there is a single correct song while still asking
whether the model uses the factors it claims to use.

## Recommended Model Path

### Exploratory Design

Use Codex to:

- expand cases,
- create counterfactuals,
- produce manual baseline rankings,
- define failure modes,
- write analysis notes.

This stage is not a final quantitative result; it is study design.

### Mechanistic Pilot

Use a small open-weight model first:

- Gemma 3 1B/4B + Gemma Scope 2 for low-cost feature probing.
- Then Qwen3-8B-Base + Qwen-Scope for Chinese music scenarios.

The first mechanistic pilot should use 3-5 cases, not the full casebook.

## Probe Units

Each mechanistic probe should include:

- `original_text`: profile/context/candidates for the original case.
- `counterfactual_text`: the minimally changed variant.
- `target_dimensions`: expected concepts that should change.
- `candidate_focus`: which candidate should rise or fall.
- `behavioral_observation`: ranking/score/reason movement from Codex or another
  model.
- `mechanistic_observation`: later filled with activation/feature movement.

## Target Dimensions

Initial dimensions:

- `low_energy_quiet`
- `high_energy_workout`
- `rain_night_drive`
- `solitude_melancholy`
- `chinese_vocal`
- `instrumental_focus`
- `lyric_narrative`
- `party_warmup_groove`
- `rave_hard_energy`
- `novelty_discovery`
- `weak_evidence_overjustification`

These are intentionally product-readable. Later, each dimension can map to one
or more SAE feature IDs.

## Mechanistic Metrics

### Feature Activation Delta

For a target dimension, compare activation between original and counterfactual
prompts.

Example:

- `high_energy_workout` should increase from rainy-drive prompt to gym prompt.
- `rain_night_drive` should decrease from rainy-drive prompt to gym prompt.

### Explanation-Feature Alignment

If the output reason cites a factor, check whether related feature activations
move under the corresponding counterfactual.

Example:

- Reason cites "not too loud".
- Removing or flipping that constraint should reduce low-energy/quiet feature
  advantage or increase high-energy candidate score.

### Feature-Behavior Alignment

Check whether candidate score changes align with expected feature movement.

Example:

- Hard techno score rises under rave context.
- `rave_hard_energy` features rise in the same prompt contrast.

### Missing Signal

Flag cases where the model confidently cites a factor but feature movement is
absent or inconsistent.

This is the strongest interpretability-oriented failure mode.

## What Counts As A Finding

A useful finding does not need to prove that the model "really understands"
music taste. A useful finding can be:

- a factor that reliably moves both behavior and internal features,
- a factor that moves behavior but not internal features,
- a factor that appears in explanations without corresponding sensitivity,
- a class of prompts where the model over-justifies weak track evidence,
- a difference between English-heavy and Chinese-heavy music context features.

## First Pilot

Use the existing three cases:

1. Rainy night drive versus gym sprint.
2. Study background versus emotional lyric listening.
3. Party warm-up versus rave peak.

These have clean contrast directions and should reveal whether the method can
detect context/energy shifts.

## Complexity Pilot (Later)

After the simple pilot, compare matched routine/complex task pairs:

1. Quiet study direct pick versus tool-conflict repair.
2. 2am rave direct pick versus feedback-based party repair.
3. Night-drive rerank versus sequence-level playlist planning.

This tests whether the workspace signal appears mainly when the model must
maintain goals, resolve evidence conflicts, and repair a recommendation.
