# Mechanistic Pilot Plan

## Why A Hand-Curated Pilot

The generated probe pack is useful for scale, but the first interpretability
pilot should use clean contrasts. Preference ablations and dislike statements
can be ambiguous because keyword matching does not understand polarity.

For the first mechanistic run, use:

`data/mechanistic_pilot_specs.jsonl`

This file selects six probes with clear directional expectations.

## Pilot Probes

1. `pilot_night_drive_to_gym`
   - Rain/night/quiet/solitude should decrease.
   - High-energy/workout/rave features should increase.
   - EDM candidate should rise.

2. `pilot_night_drive_to_explosive`
   - Quietness should stop being a decisive constraint.
   - High-energy EDM should rise.

3. `pilot_study_to_emotional_lyrics`
   - Low-distraction/instrumental focus should decrease.
   - Lyric narrative and emotional vocal features should increase.

4. `pilot_study_to_no_vocals`
   - Chinese vocal preference should be overridden.
   - Instrumental focus should increase.

5. `pilot_party_warmup_to_rave`
   - Party warm-up groove should decrease.
   - Rave/hard-energy features should increase.

6. `pilot_party_warmup_to_low_presence`
   - Groove warm-up should lose weight.
   - Low-energy/low-presence features should increase.

## First Mechanistic Question

Do target dimension activations move in the expected direction when only the
listening context or constraint changes?

## Second Mechanistic Question

When the model's recommendation reason cites a factor, does the corresponding
target dimension show movement under the counterfactual?

## What We Should Not Claim Yet

- Do not claim to explain a production music recommender.
- Do not claim feature labels are final.
- Do not claim subjective recommendation quality from these probes alone.

## What We Can Claim If It Works

The project demonstrates a reproducible workflow for turning subjective music
recommendation explanations into contrastive interpretability probes.

## Complexity Follow-Up

This simple pilot may be too shallow for workspace/J-space effects. The next
stage should compare it with `data/j_space_complexity_tasks.jsonl`, which adds
planning, conflict handling, and repair tasks. If the simple pilot has weak
signals but the complex pilot has clearer movement, that supports the hypothesis
that recommendation interpretability needs complex, agentic tasks rather than
routine tag matching.
