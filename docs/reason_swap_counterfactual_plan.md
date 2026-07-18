# Reason-Swap Counterfactual Plan

## Question

When a visible reason is generated before a music entity, does that specific
reason causally support the subsequent complete title-artist pair, or does any
plausible reason from the same listening context work equally well?

## Records

Use all 20 archived `reason_first` recommendations from the matched order
smoke. Keep title and artist as one scored decision. Preserve four entity
groups:

- `verified_exact`;
- `catalog_conflict`;
- `unverified`;
- `invalid_placeholder` for explicit artist placeholders, regardless of the
  catalog label returned for a coincidentally matching title.

## Conditions

For each fixed title-artist pair, replay the first playlist slot with:

- `own`: its originally generated reason;
- `same_context_swap`: a different reason string from the same context;
- `opposite_context`: the reason at the same seed and rank in the other
  listening context;
- `neutral`: a fixed generic recommendation reason.

Every original reason is checked for direct title leakage. A same-context swap
must differ as text, not merely come from a different record.

## Metric

Teacher-force every title and artist token and compute complete-pair mean log
probability. For each control condition:

`own_margin = own_pair_mean_logp - control_pair_mean_logp`

Aggregate margins and own-win rates overall and separately by entity group and
context.

## Scope

This experiment identifies whether the visible reason text is a causal prefix
for the fixed pair. It does not show that the reason is factually correct, that
the pair is a good recommendation, or that the model planned the reason before
writing it.

## Registered Technical Gate

- Exactly 20 records and 80 reason-pair conditions are scored.
- Every record has all four distinct condition labels.
- Complete title and artist token spans are nonempty.
- Same-context reason text differs from the original reason.
- Original reasons do not directly contain their title.
- All log probabilities and derived margins are finite.

No behavioral superiority threshold is registered. The pilot determines
whether pair-specific visible-reason causality is large enough for a later
mechanistic intervention.
