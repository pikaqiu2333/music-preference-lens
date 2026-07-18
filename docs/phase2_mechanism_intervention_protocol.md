# Phase 2 Preregistered Mechanism-to-Intervention Protocol

## Status and scope

This protocol defines the only confirmatory extension of the Phase 1 Music
Preference Lens study. It is committed and externally timestamped before
inspecting any Granite music generation, catalog label, relation score, or
intervention outcome. A technical smoke uses a context and seed excluded from
all scientific batches and may inspect only model loading, output
non-emptiness, parser yield, memory use, and hook endpoint reproduction.

The Phase 1 archive commit is
`5d4b9ba8ad20ac5d3aa402b92abdabbe8bc20e06`. The machine-readable source of
truth for this protocol is
[`config/phase2_mechanism_intervention_protocol.json`](../config/phase2_mechanism_intervention_protocol.json).
The protocol file and submitted Job scripts must print their SHA-256 digests.

## Research question

Can a mechanism label assigned before correction predict which correction
method will repair a naturally generated title-artist relation conflict?

The study does not test subjective awareness. It distinguishes two operational
mechanisms:

- **prior-masked:** a title-conditioned state causally favors a
  catalog-supported artist, although the emitted relation is wrong;
- **wrong-binding:** the title-conditioned state causally favors the emitted
  but catalog-unsupported artist;
- **indeterminate:** the two diagnostic templates, layers, or controls do not
  support either stable label.

## Frozen hypotheses

### H1: mechanism stability and coverage

Each diagnostic template independently assigns one of three labels:
prior-masked, wrong-binding, or indeterminate. Across all selected normalized-
title conflict clusters, including indeterminate labels, the templates must
have raw agreement of at least 80% and unweighted three-class Cohen's kappa of
at least 0.60. At least 60% of all clusters must receive the same
non-indeterminate label from both templates. Other rows remain in the released
data but are excluded from H2.

### H2: mechanism predicts correction response

A candidate-free anti-prior recall prompt will improve correction over a
candidate-free naive self-check at least 25 percentage points more for prior-
masked relations than for wrong-binding relations. Both conditions receive
only the title and the model's previously emitted artist; neither receives the
catalog-supported reference or a candidate list. The normalized-title-cluster
bootstrap 95% confidence interval for this difference in improvement must have
a lower bound above zero.

Candidate-prior-subtracted closed-set scoring is retained only as a
manipulation check. It cannot confirm H2.

### Safety control: external grounding

A strict double-catalog constraint is reported as a product safety control, not
as a scientific hypothesis. Its retrieval coverage, unsupported replacement
rate, exact preservation, and abstention rate are reported separately.
Abstention is never counted as a corrected recommendation.

## Model and architecture

The confirmatory model is
[`ibm-granite/granite-4.1-3b-base`](https://huggingface.co/ibm-granite/granite-4.1-3b-base)
at revision `dacb9cb9157bec98e99b09f285c92a4d58405c96`. It is a 40-layer
decoder-only dense Transformer under Apache 2.0. It was selected before any
project-specific Granite output was observed.

If and only if the model cannot pass the technical smoke because its raw
completion format cannot be parsed under the primary and one exactly frozen
fallback format, the
registered backup is
[`HuggingFaceTB/SmolLM3-3B-Base`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base)
at revision `d78a42f79198603e614095753484a04c10c2b940`. Its diagnostic layers are
18, 25, and 32, corresponding to the same registered depth fractions. Switching
for scientific performance, catalog yield, or effect size is prohibited.

## Free generation

The primary batch contains 12 recommendation contexts and five new seeds,
yielding 60 playlists and at most 300 title-artist-reason events. Each prompt
requests exactly five real existing tracks without supplying a candidate list.
The fixed decoding parameters are temperature 0.7, top-p 0.9, and 384 maximum
new tokens.

An extension batch of four registered contexts and the same five seeds must run
when the primary batch yields fewer than 40 unique normalized-title strict
conflict clusters or fewer than 240 parsed events. The hard cap is 80 playlists
and 400 parsed events. No context, seed, or decoding parameter may be added
after catalog inspection.

The technical smoke uses only the registered `format_only_smoke` context and
seed 997, neither of which appears in primary or extension data. It first uses
the frozen primary prompt. If the parser gate fails, a second smoke may use the
single frozen inline fallback prompt. No other prompt repair is allowed. Smoke
artifacts contain no completion text and are never uploaded as raw data.

All raw completions are archived before parsing. The selection and bootstrap
unit is normalized title. Within each title cluster, one relation record is
selected by its registered row hash. If more than 40 strict conflict title
clusters or 40 strict exact title clusters qualify, selection uses the
lexicographically smallest SHA-256 values under the frozen salts in the JSON
protocol, never model scores.

## Catalog labels

MusicBrainz and Apple are queried independently. A row qualifies as strict
exact only when the normalized emitted title-artist relation is supported by
both sources.

A row qualifies as strict conflict only when:

1. both source queries succeed;
2. the emitted title-artist relation is unsupported by both sources;
3. the normalized title has exactly one non-emitted artist shared by both
   sources; and
4. that shared relation is retained as a catalog-supported reference, not
   described as the unique real-world answer.

Ambiguous, alias-only, failed-query, and one-source rows are released but
excluded from confirmatory analyses. Catalog evidence supplies the external
label; model activations do not define truth.

Every catalog request archives its complete raw response, normalized query
parameters or URL, UTC timestamp, source track or recording identifiers, and
HTTP success or failure status. Later catalog changes do not relabel the frozen
analysis silently.

## Mechanism diagnosis

Diagnosis uses full-residual causal patching at layers 20, 28, and 36, equal to
50%, 70%, and 90% of Granite depth. It does not perform a layer sweep. Two
frozen factual prompt templates are evaluated against three diagnostic neutral
titles selected by hash from the strict exact pool. Three additional neutral
titles are held out exclusively for the closed-set manipulation check.

For a conflict title `c` and exact-pool title `n`, neutral candidates are sorted
by `SHA256("phase2-v1-neutral-control:" + normalize(c) + ":" + normalize(n))`.
The title must differ from the conflict title; its strict exact artist must
differ from both emitted and reference artists; and neither catalog may list
the emitted or reference artist for the neutral title. The first three eligible
unique normalized titles are diagnostic controls and the next three are
manipulation-check controls. Fewer than six eligible controls makes that row
indeterminate. If fewer than six unique exact-pool titles exist globally, the
study stops before mechanism scoring.

For each candidate artist separately, the source is the factual-title prompt
plus the complete artist continuation and the target is the otherwise identical
neutral-title prompt plus the same artist continuation. Source and target must
have identical artist token IDs. Patching replaces the residual stream at every
prediction position immediately preceding the complete artist continuation,
not only its first token.

For each layer and template, the patch relation shift is:

`(patched_reference - patched_emitted) - (neutral_reference - neutral_emitted)`

where each term is the mean full-sequence artist log probability. The template
label is prior-masked when at least two layers have a median shift strictly
above +0.05 nats, wrong-binding when at least two layers have a median shift
strictly below -0.05 nats, and indeterminate otherwise. A relation receives a
confirmatory mechanism label only when both templates agree on the same non-
indeterminate class.

Endpoint reproduction, candidate token alignment, finite values, complete
interventions, and exact model revision are mandatory technical gates. A row
with a failed alignment, missing intervention, NaN, or infinity is labeled
indeterminate and counted in H1's denominator. If more than 10% of selected
conflict clusters have a technical failure, the confirmatory study fails.

## Frozen correction conditions

Each selected conflict is evaluated under the following frozen conditions:

1. **naive candidate-free self-check:** provide the title and previously emitted
   artist, ask for the correct artist or `ABSTAIN`, and reveal no candidate or
   catalog reference;
2. **candidate-free anti-prior recall:** provide the same information but
   explicitly instruct the model to ignore artist fame and generic association,
   then ask for the specific title relation or `ABSTAIN`;
3. **prior-subtracted closed-set score:** subtract the mean candidate score
   under the three held-out neutral titles from the factual-title candidate
   score; this is a manipulation check only;
4. **catalog constraint:** retain only double-catalog-supported relations;
   otherwise abstain; this is a safety control only.

The two candidate-free conditions each use two frozen paraphrases, greedy
decoding, and at most 32 new tokens. Each paraphrase is scored independently as
1 only when its parsed free-text artist equals the catalog-supported reference
after frozen normalization, and 0 otherwise. The per-title correction accuracy
for a condition is the mean of its two indicators, so it can be 0, 0.5, or 1.
`ABSTAIN`, invalid format, a different artist, and paraphrase disagreement are
also retained as separate descriptive outcome categories. A shuffled-title
placebo uses the same computation as condition 3 but replaces the factual title
with a hash-selected unrelated title. It must not be used to tune a threshold.

For exact controls, a false candidate is selected by hash from artists attached
to other strict exact rows and rejected if either catalog supports that false
pair. Exact-control preservation is a safety outcome, not part of the H2
interaction.

## Analysis and decision rules

H1 uses all selected normalized-title conflict clusters. Agreement is raw
three-class agreement; kappa is unweighted three-class Cohen's kappa; and
classifiable coverage is the fraction receiving the same non-indeterminate
label from both templates.

The unit of resampling for H2 is normalized title, not generation event. The
primary interaction is:

`(anti_prior_free_recall - naive_free_self_check)_prior_masked - (anti_prior_free_recall - naive_free_self_check)_wrong_binding`

computed on per-title correction accuracy averaged across the two frozen
paraphrases. Ten thousand percentile bootstrap replicates are sampled
separately within each mechanism class with replacement, using seed 20260713;
class sizes are therefore fixed and no replicate can omit a class. Continued
error, invalid output, corrected artist, paraphrase disagreement, and abstention
are separate outcomes. Non-finite aggregate values fail the technical gate.

H2 is tested only when each mechanism class contains at least 15 normalized-
title clusters. No additional model is added to rescue a failed or underpowered
result.

## Stop conditions

The confirmatory line stops and is reported as a boundary or negative result
when any of the following occurs:

- fewer than 30 unique strict conflict title clusters remain after the 400-
  event cap;
- either mechanism class has fewer than 15 normalized-title clusters;
- H1 raw agreement is below 80%, kappa is below 0.60, or classifiable coverage
  is below 60%;
- the H2 interaction is below 25 percentage points or its confidence interval
  crosses zero;
- the direction appears only after changing a threshold, prompt, layer, model,
  or sample rule; or
- Granite does not reproduce both operational mechanism classes.

Passing H1 and H2 supports the narrow claim that mechanism diagnosis predicts
correction response in this workflow. It does not establish a universal truth
detector, conscious error awareness, or a replacement for catalog grounding.
