# Song Entity Option-Letter Grounding Plan

## Motivation

The forced-choice candidate scoring probe removed open generation, but it still
scored full artist strings. That introduced length and copy bias: long artist
names and title-like fake artists could beat `Unknown` under mean token logprob.

This follow-up scores only answer letters.

## Prompt

```text
Music catalog lookup.
If the title is not a known real music release, choose Unknown.
Song title: "{{title}}"
Options:
A. ...
B. ...
C. ...
D. ...
Answer with one letter only.
Answer:
```

The model is scored on the next-token probability of `A`, `B`, `C`, or `D`.

## Design

- Reuse `data/qwen_scope_song_entity_forced_choice_specs.jsonl`.
- Generate three deterministic option-order permutations per title.
- Score only option letters, not artist strings.
- Track both variant-level and item-level majority decisions.
- Keep a title-only SAE probe prompt so internal title grounding can be
  compared separately from option text.

## Interpretation

If free-generated titles remain closer to invented controls in the title-only
SAE state, and option-letter scoring also shifts toward `Unknown`, the evidence
for style-shaped placeholders becomes stronger.

If the title-only SAE state remains invented-like but option-letter choices
prefer real or fake artists, that is evidence for a split between internal
grounding and output-completion bias.
