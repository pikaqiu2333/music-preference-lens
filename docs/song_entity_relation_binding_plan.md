# Song Entity Relation-Binding Probe

## Question

The free-generation experiment found many fluent title-artist pairs that were
not supported by external catalogs. Its SAE knownness score did not pass the
registered control gate, so it could not distinguish relation knowledge from
famous titles, common artists, or playlist context.

This follow-up asks a narrower question:

> Does Qwen3 represent the correct relation between a fixed song title and its
> artist differently from a matched, incorrect relation?

It is a mechanism calibration experiment, not a recommendation benchmark.

## Matched Design

The control set contains 20 catalog-supported titles, split evenly between
English and Chinese. Every title has two teacher-forced variants:

- exact: the catalog-supported artist;
- mismatch: an artist taken from another title in the same language block.

Each language contains five independent two-title swap blocks. Inside a block,
the two correct artists are exchanged to create the mismatches, so each artist
appears once under each label. Cross-validation holds out complete swap blocks.
Title, prompt framing, punctuation, and suffix are held constant inside each
relation. This removes first-order title popularity, artist popularity,
language, recommendation-context shortcuts, and opposite-label artist leakage
across train/test folds.

## Signals

### Direct Conditional Score

For each artist, compute mean token log probability after the real title and
after a neutral title. The primary behavioral score is:

`[log P(correct | title) - log P(correct | neutral)] -`
`[log P(wrong | title) - log P(wrong | neutral)]`

Positive values favor the correct relation while subtracting each artist's
generic prior.

### Order-Flipped Choice Score

The model chooses between the same two artists using both A/B orders. The
order-averaged correct-minus-wrong margin is compared with the same options
under a neutral title. This is a secondary behavioral check without artist
string-length bias.

### SAE Paired Delta

Layer-24 Qwen-Scope activations are captured at:

1. the last artist token;
2. the fixed `Relation` marker after the complete title-artist pair.

A feature direction is fit only on training-pair exact-minus-mismatch deltas.
Five-fold evaluation holds out complete title pairs. The pair-end score is the
primary mechanistic metric; artist-end is secondary.

## Registered Gates

- Technical gate: all selected pairs produce both variants, all score fields,
  and valid A/B token IDs.
- Behavioral gate: direct PMI-style pair accuracy is at least `0.80`.
- Mechanistic gate: pair-end SAE paired cross-validation accuracy is at least
  `0.80`.
- Intervention gate: both behavioral and mechanistic gates pass.

The choice score and artist-end SAE result are diagnostic and cannot rescue a
failed primary gate. No generated-pair intervention is interpreted before the
intervention gate passes.

## Next Step If It Passes

For each freely generated playlist, compare every original title-artist pair
with an artist-shuffled counterfactual from the same playlist. This preserves
the generated title and artist marginals and tests whether the calibrated
relation direction tracks the original binding. Only then should feature
ablation or steering be attempted.
