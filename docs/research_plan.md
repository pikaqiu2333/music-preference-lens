# Research Plan

## One-Sentence Thesis

LLM-based music recommendation explanations should be interpreted at two
levels: whether stated factors influence ranking behavior, and whether later
internal probes show matching preference/context sensitivity.

## Motivation

Music recommendation is inherently subjective. A recommendation can be good
because it fits a moment, a mood, a memory, or a taste profile that is hard to
reduce to fixed labels. This makes classic gold-label evaluation awkward.

The practical interpretability question is more tractable:

- Did the model respect explicit constraints?
- Did it use the current context?
- Did its explanation match the available item evidence?
- Did the ranking change when the cited preference changed?
- Can an activation/SAE probe later observe the same factor movement?

## Stage 1: Behavioral Faithfulness Probe

Stage 1 does not require model activations or large-scale annotation.

### Input

Each case contains:

- `user_profile`: stable user taste and dislikes.
- `current_context`: momentary listening need.
- `candidates`: structured track cards with tags, mood, energy, language,
  evidence summary, and optional editorial notes.
- `expected_sensitive_factors`: factors that a faithful recommender should be
  sensitive to.
- `counterfactuals`: modified contexts or profiles that should change the
  ranking or explanation.

### LLM Task

For each case, the LLM:

1. Scores each candidate from 1 to 5.
2. Ranks the candidates.
3. Provides concise reasons grounded in the provided track cards.
4. Names the user/context factors that most affected the decision.

### Probe Logic

For each counterfactual, compare the original output with the counterfactual
output:

- Did rankings change in the expected direction?
- Did reasons stop citing removed/negated factors?
- Did reasons remain grounded in track-card evidence?
- Did the model over-justify unknown or weak evidence?

## Stage 2: Small Human Casebook (Deferred)

This stage is deliberately deferred. The current project should not become a
music recommendation benchmark or a model leaderboard.

For now, the casebook is only a source of clean contrastive prompts for
mechanistic interpretability. Human judgment can be used later for qualitative
auditing, but not as the main contribution.

<!--
Original broader casebook plan retained below as a later optional path.

Create 20-50 high-quality cases from music product intuition:

- Night driving
- Rainy city walk
- Focus work
- Post-breakup workout
- Party warm-up
- Morning commute
- Low-key electronic discovery
- Chinese vocal nostalgia

Human annotation should be light:

- Mark hard constraints.
- Mark plausible recommendation factors.
- Mark obvious mismatch factors.
- Do not force one correct ranking when multiple rankings are plausible.
-->

## Stage 3: Mechanistic Interpretability Pilot

Use open-weight models and existing SAE resources. This is not an optional
appendix; it is the main research direction. The behavioral layer exists only
to create clean contrasts and generated reasons for the internal layer.

Candidate models:

- Gemma 3 1B/4B + Gemma Scope 2 for low-cost method validation.
- Qwen3-8B-Base + Qwen-Scope for Chinese music and local product scenarios.

Questions:

- Do context tokens such as night, rain, driving, lonely, energetic, or not too
  loud activate stable preference/context features?
- Do those activations change under counterfactual flips?
- Do cited explanation factors correspond to internal feature movement?
- Are there cases where the model cites a factor without a corresponding
  internal/context sensitivity signal?

Inputs:

- `runs/mechanistic_probe_pack.jsonl`
- `config/interpretability_dimensions.json`
- feature IDs from Gemma Scope, Qwen-Scope, Neuronpedia, or local SAE runs

Outputs:

- feature activation deltas for original versus counterfactual prompts
- explanation-feature alignment notes
- missing-signal and over-justification cases

## Evaluation Dimensions

### Constraint Sensitivity

Whether explicit constraints affect rankings and reasons.

Example: "not too loud" should penalize high-energy festival EDM.

### Context Sensitivity

Whether listening context changes recommendations.

Example: "rainy night drive" and "gym sprint" should prefer different tracks.

### Reason-Item Consistency

Whether recommendation reasons are supported by the track card.

Example: A reason should not call a high-BPM festival track "quiet and sparse"
unless that evidence is present.

### Explanation Faithfulness

Whether factors named in the explanation are causally relevant to the ranking.

Example: If the reason cites "Chinese lyrics", removing the Chinese-language
preference should reduce the advantage of Chinese vocal tracks.

### Over-Justification

Whether the model writes confident reasons despite weak item evidence.

Example: Claiming a track has "rainy city texture" from only a generic pop tag.

## Expected Output

The first public artifact should be a mechanistic research note:

> Music Preference Lens: Interpreting Generated Reasons in LLM-based Music
> Recommendation

It should include:

- Matched phrase-level probes
- Candidate SAE feature labels
- Generated-reason feature alignment traces
- Missing-signal or over-justification examples
- Identity/source features as negative controls
