# Long-Horizon Research Program

## Core Question

When an LLM recommends a complete music entity (`title + artist`), where does
the chain from user need to entity selection and explanation succeed or break?

The program separates five links that product systems often collapse:

1. The model represents the user's current constraint.
2. The constraint causally changes support for candidate entities.
3. The selected title-artist pair is a real catalog relation.
4. The generated reason reflects factors that affected selection.
5. The final system prevents a known mismatch from reaching the user.

No single metric is treated as a proxy for all five links.

## Completed Evidence

- Need counterfactuals changed complete title-artist likelihood for 10/12 saved
  recommendations.
- Component patching localized much of this need effect to middle and late
  layers, with exploratory attention-then-MLP behavior in two successful cases.
- `Space Oddity - David Bowie` received more support under active lyric
  listening than strict no-vocals, despite being freely generated for the
  no-vocals request with an instrumental-sounding reason.
- Reason-first generation changed every matched playlist. It improved catalog
  exactness only in the familiar emotional-vocal context and produced
  placeholders or concept-shaped names under strict no-vocals.
- Visible reason swaps showed strong broad-context steering (own reason beat an
  opposite-context reason for 18/20 pairs), but verified-exact pairs showed no
  stable own-reason advantage over same-context or neutral reasons (2/4 wins
  for each control).

These observations motivate a latent-knowledge/control-gap hypothesis. They do
not establish subjective awareness, deception, or a general failure rate.

## Stage 1: Visible Reason Causality

Fix each reason-first title-artist pair and replay it after four reason prefixes:

- its own generated reason;
- another reason from the same listening context;
- a reason from the opposite listening context;
- a neutral reason.

Measure complete-pair log probability and analyze verified exact,
catalog-conflict, unverified, and explicit-placeholder groups separately.

Decision: continue only if reason identity changes pair support beyond generic
context wording. A null result means reason-first changes generation trajectory
without encoding pair-specific planning.

Pilot decision: do not patch the free generated reasons more deeply. Their
effect is mainly broad semantic steering, while pair-specific faithfulness is
not stable for verified entities. Replace them with controlled, length-matched
attribute reasons before a mechanistic follow-up.

## Stage 2: Selection-Time Control Gap

Use matched real songs with known constraint attributes. Compare:

- free-generation inclusion;
- teacher-forced pair likelihood;
- reason claims;
- catalog and attribute evidence.

For mismatches like vocal songs in a no-vocals request, patch or steer the
validated need-sensitive layers during free generation. The outcome is whether
the intervention changes actual inclusion, not merely a fixed answer's score.

Decision: a useful control signal must reduce mismatch inclusion while
preserving valid recommendations in matched control contexts.

## Stage 3: Grounded Output Architectures

Compare three product architectures under identical needs:

- unconstrained title-artist text generation;
- retrieval-constrained generation over catalog entities;
- Song ID selection followed by a generated explanation.

Measure entity validity, constraint compliance, explanation faithfulness, and
abstention. Song IDs solve entity addressability but are not assumed to solve
reason faithfulness.

## Stage 4: Model and Domain Replication

Replicate only effects that survive Stages 1-3:

- at least one additional open base model;
- Chinese and English prompts;
- familiar and long-tail music needs;
- a small non-music recommendation domain.

The goal is mechanism replication, not a leaderboard.

## Reporting Rules

- Preserve raw generations before parser revisions.
- Pre-register technical and interpretation gates separately.
- Treat catalog misses as `unverified`, not proof of fabrication.
- Report explicit placeholders independently of catalog labels.
- Keep title and artist as one decision while retaining separate catalog and
  need-fit axes.
- Use causal language only for explicit interventions.
- Use `latent sensitivity` or `control gap`, not `self-awareness`, unless a
  future experiment directly supports metacognitive access.

## Publishable Unit

The smallest coherent paper or open-source report is:

> A causal study of when generated recommendation reasons guide entity
> selection, rationalize it after the fact, or induce concept-shaped
> fabrication, with catalog grounding and layer-level interventions.

The accompanying repository should publish prompts, raw generations, parsers,
catalog evidence, counterfactual bundles, intervention code, and negative
results.

## 2026-07-11 Milestone Result

The controlled vocality sequence localized two distinct failures: whole-entity
misdirection (`Space Oddity`) and strong title-versus-artist disagreement
(`Awake`). Frozen layer-18 choice heads did not disappear in failures; they
could carry the wrong entity-specific direction.

A pre-registered transfer to six double-source exact and six deterministic
free-generated catalog-conflict events did not validate a standalone same-model
warning rule. Two-order choice verification achieved perfect exact specificity
but only 3/6 conflict sensitivity. Factual first-token continuation detected
2/6 conflicts. The result is retained as a negative validation outcome.

Two conflicts nevertheless showed causal context override: an independent
factual prefix preferred Bon Iver for `Skinny Love` and Yiruma for `River Flows
in You`, while the reconstructed recommendation prefixes strongly preferred the
emitted wrong artists. Layer-21 attention and layer-24 MLP patches moved those
generation states toward the factual relation state.

Stage decision: internal disagreement can route a subset of outputs to review,
but it cannot replace catalog grounding. Any complementary verifier rule must
be frozen and tested on newly generated holdout events before interpretation.
