# Song Entity Generation-Time Grounding Plan

## Question

When a base language model freely recommends a song, does its internal state
already distinguish a stable title-artist entity from an unverified or
mismatched pair, even when the final recommendation sounds confident?

This experiment upgrades the unit of analysis from a title to an exact
`title + artist` pair. A catalog miss is called `unverified`, not fictional.

## Design

- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Contexts: quiet night drive, peak-time rave, emotional vocal, and strict
  no-vocals writing
- Seeds: `17`, `29`, and `43`
- Output: five freely generated `Title / Artist / Reason` rows per run
- Controls: 20 known exact pairs, 10 real-title/wrong-artist pairs, and 10
  synthetic pairs

For each generated pair, capture SAE activations on title, artist, pair-end,
and reason spans. Immediately score the exact pair under a neutral catalog
prompt and a self-attributed prompt saying that the model generated it.

## Calibration

Verification uses two single-token answer letters for `Known exact pair` and
`Unknown or not a known exact pair`. Both A/B orders are scored. A matching
null prompt supplies the letter prior, which is subtracted before the two
orders are averaged.

The primary internal score is a held-out nearest-centroid probe over layer-24
SAE activations. Feature selection happens inside each fold. Interpretation is
allowed only if five-fold balanced accuracy on controls reaches 80 percent and
the direction is stable across contexts.

## External Reference

Generated pairs are checked against MusicBrainz and Apple Search. Labels are:

- `verified_exact`: at least one catalog returns the exact title-artist pair.
- `catalog_conflict`: the title is found but not with the generated artist.
- `unverified`: neither catalog supplies reliable evidence.
- `verification_error`: a request failed and the row is excluded.

## Run Gates

The smoke job must load the model and SAE, parse at least three complete pairs,
score both answer letters, calculate control metrics, and persist artifacts.

The full job requires at least 10 of 12 valid playlist generations and at
least 50 parsed pairs. If those gates fail, repair the generation protocol
before interpreting any mechanism result.
