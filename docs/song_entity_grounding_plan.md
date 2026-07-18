# Song Entity Grounding Plan

## Motivation

Free playlist generation can produce plausible song titles without a music
catalog. The key research question is:

> When a model outputs a song title, is it recalling a known song entity from
> pretraining, or generating a style-shaped title that only looks like a song?

This project treats that as an entity-grounding question, not a recommendation
benchmark.

## Probe Design

Use three title groups:

1. `known_real`: well-known songs with stable artist bindings.
2. `invented_control`: deliberately invented, music-like titles.
3. `free_generated`: titles produced by the previous free playlist generation
   probe.

For each title, render a short catalog-style continuation prompt:

```text
Music catalog lookup.
Song title: "{{title}}"
Artist:
```

Then measure:

1. Whether the model completes a stable known artist for `known_real`.
2. Whether `invented_control` and `free_generated` produce unstable or generic
   artist completions.
3. Whether Qwen-Scope SAE activations for `free_generated` titles are closer to
   the known-real or invented-control prototype.

## Interpretation

If free-generated titles behave like known real titles, the model may be
recalling parameterized world knowledge. If they behave like invented controls,
the playlist generator is likely producing style-grounded placeholder music
objects.

This does not prove subjective awareness. It tests grounding behavior and
internal feature proximity.

