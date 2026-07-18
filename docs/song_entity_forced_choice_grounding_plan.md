# Song Entity Forced-Choice Grounding Plan

## Motivation

The first song-entity grounding probe used open artist continuation. It showed
that free-generated titles often drift toward album-like metadata, but open
generation makes the result noisy because the model can ignore the requested
field.

This follow-up turns the same question into a forced-choice / logprob probe:

> Does the model assign more probability to a known artist, a distractor artist,
> a fabricated style artist, or `Unknown`?

The goal is still interpretability, not a music-catalog benchmark.

## Prompt

```text
Music catalog lookup.
If the title is not a known real music release, answer Unknown.
Song title: "{{title}}"
Artist:
```

Each candidate is scored as the continuation after `Artist:`.

## Groups

1. `known_real`: real songs with a correct artist candidate, distractors, and
   `Unknown`.
2. `invented_control`: deliberately invented titles with `Unknown`, real-artist
   distractors, and one fabricated style artist.
3. `free_generated`: titles produced by the free playlist generation probe.
   These are treated as unverified generated titles rather than as guaranteed
   non-existent songs.

## Measurements

1. Mean token logprob for each candidate continuation.
2. Whether the best candidate type matches the expected type:
   `correct_artist` for `known_real`, `unknown` for invented and free-generated
   controls.
3. Prompt-final SAE activation proximity to known-real versus invented-control
   prototypes.
4. Best-answer SAE activation proximity to correct-artist versus unknown-answer
   prototypes.

## Interpretation

If a free-generated title gives higher logprob to `Unknown` and its internal
state is closer to invented controls, the title is probably being treated as a
style-shaped placeholder. If it gives high probability to a specific artist and
has known-real-like internal state, it may be colliding with or recalling a real
song entity.
