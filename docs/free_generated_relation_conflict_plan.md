# Free-Generated Title-Artist Conflict Transfer Plan

## Milestone Question

After the model freely emits a title and artist, can the same model's independent
relation-verification path flag catalog-conflict pairs without a trained
classifier or Song ID? If so, does causal patching show that correct relation
knowledge was available but overridden by the generation context?

## Frozen Transfer Set

Use the archived 55 freely generated title-artist-reason rows and their original
raw completions.

- Exact group: all six rows verified exact by both MusicBrainz and Apple.
- Conflict group: six rows selected before model scoring. A candidate must have
  both catalog sources available and exactly one normalized artist shared by
  both sources' title matches, excluding the emitted artist.
- Match the exact group's context counts. Within each context, rank conflict
  candidates by combined occurrence count of the shared artist in both catalog
  result lists, then generation ID and rank.

This deterministically yields four emotional-vocal, one peak-time-rave, and one
strict-no-vocals conflict. Duplicate exact relations generated in different
contexts/seeds remain separate output events.

For a conflict, the reference is the catalog-shared artist. For an exact event,
the reference is a deterministic different artist from the next exact event;
the emitted artist is the correct one.

## Three Frozen Verifiers

All margins are oriented as `emitted artist minus reference artist`:

1. **Catalog continuation:** factual `Title -> Artist` prefix, first-token and
   full-sequence mean-log-prob margins.
2. **Independent choice:** emitted and reference artists shown as A/B in both
   orders; average the order-corrected emitted-minus-reference letter margin.
3. **Original generation prefix:** reconstruct the exact playlist text up to
   the emitted artist and score the same first-token/sequence margin.

An exact event is predicted when a verifier margin is nonnegative; a conflict
is predicted when it is negative. No threshold is fitted.

The small pilot is `validated_small_pilot` only if both catalog first-token and
independent-choice verifiers reach balanced accuracy at least `0.75`, with at
least 4/6 exact and 4/6 conflict events correct in each. If only one verifier
meets that rule, status is `promising_single_path`; otherwise it is
`not_supported`.

## Internal Conflict Categories

For catalog-conflict outputs:

- `lower_probability_sample`: catalog and generation prefixes both prefer the
  reference, yet sampling emitted the other artist;
- `context_override`: catalog prefix prefers the reference but the original
  generation prefix prefers the emitted artist;
- `relation_not_recovered`: the catalog prefix does not prefer the reference.

These categories describe probability competition, not consciousness.

## Mechanistic Readout and Patching

Read emitted-minus-reference first-token margins at layers 14, 18, 21, 24, 27,
and 28 in both catalog and generation prefixes.

Patch the catalog-prefix final-position state into the original generation
prefix at the previously localized relation components:

- layer-21 attention output;
- layer-24 MLP output;
- layer-27 full residual;
- layer-28 full residual as an endpoint control.

Positive normalized recovery means the intervention moves the generation
margin toward the independent catalog relation state. Recovery is reported only
when catalog and generation margins differ by at least `0.10`.

## Technical Gates

- exactly six double-source exact and six deterministically selected conflict
  events;
- all selected original prefixes reconstructed from archived raw completions;
- emitted and reference first continuation tokens are distinct and identical
  across catalog/generation prefix formatting;
- both choice orders present for every event;
- final-layer logit-lens error and layer-28 endpoint-patch error at most `0.02`;
- all scores finite and all interventions present.

No detection result is required for technical success. A negative result still
completes the stage by ruling out this no-training warning design.
