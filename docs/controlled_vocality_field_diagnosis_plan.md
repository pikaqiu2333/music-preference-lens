# Controlled Vocality Title-vs-Artist Field Diagnosis Plan

## Stage Question

When a matched vocality reason lowers the probability of a real title-artist
pair, does the failure begin in the title tokens, the artist tokens, or the
binding between the two fields?

## Frozen Inputs

Reuse the four real title-artist sentinels, matched/flipped equal-length reasons,
layer 18, and the choice-consistent heads `0`, `1`, `8`, and `9` from run
`20260711T080524Z_pilot`. No Song ID, track, reason, layer, or head is added.

## Baseline Field Scores

For matched-reason source and flipped-reason target runs, score separately:

- mean title-token log probability;
- mean artist-token log probability;
- complete-pair token-weighted mean log probability.

The weighted title/artist effects must reconstruct the complete-pair effect.
A field effect with absolute magnitude below `0.03` is reported as weak and is
not assigned a normalized recovery.

## Causal Interventions

At the layer-18 input to `self_attn.o_proj`, patch source head output into the
target run under three scopes:

1. title prediction positions only;
2. artist prediction positions only;
3. both title and artist prediction positions.

Run every frozen head separately and all four frozen heads together. As a
technical control for the both-field scope, also patch all 16 heads and directly
patch the complete post-`o_proj` attention output; their field scores must match.

## Frozen Failure-Locus Rule

Use source-minus-target field effects:

- `title_only`: title is negative and artist is nonnegative;
- `artist_only`: artist is negative and title is nonnegative;
- `both_fields`: both are negative;
- `neither_field`: both are nonnegative;
- append `_weak_boundary` when either field has absolute effect below `0.03`.

This label describes the teacher-forced probability failure. It does not claim
that a field is semantically unknown or that the model is self-aware.

## Technical Gates

- four frozen sentinels and exactly heads `0/1/8/9`;
- nonempty title and artist token spans in both reason conditions;
- source and target use identical title and artist target token IDs;
- complete-pair effects reconstructed from field effects within `0.002`;
- all 12 single-head scope interventions and three frozen-head-group scopes;
- all-head versus direct-attention field-score error at most `0.002`;
- finite source, target, patched, and non-null recovery values.

There is no registered outcome requirement for a particular failure locus.
