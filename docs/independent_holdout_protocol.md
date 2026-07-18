# Independent Free-Generation Holdout Protocol

## Purpose

Confirm or reject the exploratory complementary warning rule without reusing
any generation event, seed, threshold, or sample selected in the discovery set.

## Registration Status

This protocol was written and frozen in the working session before holdout
scoring, but it was not deposited in an external timestamped registry or a
pre-result Git commit. The public report therefore calls it **pre-specified**,
not formally preregistered. The archived Jobs metadata records the later
generation and scoring times, but does not independently prove the protocol's
creation time.

## Frozen Generation

- model: `Qwen/Qwen3-1.7B-Base`;
- contexts: the same four original recommendation contexts;
- new seeds: `59`, `71`, `83`, `101`, `127`;
- 20 independent playlist generations;
- sampling: temperature `0.7`, top-p `0.9`, maximum 384 new tokens;
- request: five real existing title-artist-reason recommendations.

The original Job used the Hub default revision. For deterministic reruns, the
revision was retrospectively resolved to
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`; the model repository had not been
modified since 2025-07-26, before any archived Job. This revision pin was not
part of the original protocol freeze.

All raw completions are persisted before parsing or catalog inspection. If the
first batch does not yield at least eight qualifying examples in each label,
an additional seed batch must be registered before generation.

## Parsing and Catalog Grounding

Use the already-tested playlist parser without modification. Verify every
parsed title-artist pair independently against MusicBrainz and Apple.

Qualifying exact event:

- emitted title-artist relation is exact in both sources.

Qualifying conflict event:

- no emitted exact relation;
- both source queries succeed;
- the normalized title matches contain exactly one non-emitted artist shared
  by both sources.

The shared artist becomes the reference. This does not assume that all other
catalog conflicts are false; ambiguous rows are excluded from confirmation.

## Frozen Holdout Selection

Let `n = min(double_source_exact_count, unique_shared_conflict_count, 12)`.
Require `n >= 8`. Within each label, sort by SHA-256 of
`holdout_v1:<generation_id>:<rank>` and take the first `n`. Hash selection is
independent of model score, artist familiarity, and verifier outcome.

For exact events, construct the wrong reference by cycling through the
hash-sorted exact artists until finding a different artist.

## Frozen Warning Rule

Reuse the two discovery paths with no fitted threshold:

- two-order independent A/B emitted-minus-reference margin;
- factual-prefix complete-artist mean-log-prob emitted-minus-reference margin.

Predict catalog conflict when **either** margin is negative. Otherwise predict
exact. This OR rule is exploratory on the old 12 events and confirmatory only on
this new holdout.

Confirmation requires:

- balanced accuracy at least `0.75`;
- exact specificity at least `0.75`;
- conflict sensitivity at least `0.75`;
- at least eight events in each label;
- all technical gates and both choice orders.

Single-path results and first-token margins remain diagnostics. No threshold or
prompt may change after catalog labels are observed.

## Stop Conditions

- If the frozen rule passes, localize which path contributes unique detections
  and replicate the behavior on a second open model.
- If it fails, characterize the failure by prompt path and relation familiarity,
  then test the same frozen rule on a second model before concluding that the
  design is not robust.
