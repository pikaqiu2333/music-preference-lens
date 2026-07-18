# Controlled Vocality Layer-18 Attention-Head Plan

## Question

Which layer-18 attention heads carry the order-stable vocality effect in fixed
candidate choice, and do the same heads affect complete title-artist replay
differently for behavioral successes and failures?

## Frozen Inputs

Reuse the four real title-artist sentinels and two reversed eight-track
candidate orders from the component-path pilot. No Song ID, new track, prompt,
reason, or behavior label is introduced after observing the component result.

## Intervention

Qwen3-1.7B has 16 query-attention heads of width 128. At layer 18, capture the
input to `self_attn.o_proj`, where the 16 head outputs are concatenated.

- Pair path: patch one source head at a time into the flipped-reason run at all
  complete title and artist prediction positions.
- Choice path: patch one source head at a time at the final answer prediction
  position for both vocality directions and both candidate orders.
- Patch all 16 heads together and separately patch the post-`o_proj` attention
  output. These two interventions must reproduce one another.

Recovery is signed toward the matched-reason source, including the two pair
failures whose matched-reason behavior effect is negative.

## Descriptive Head Rule

A choice-consistent head has positive recovery in all four choice relations and
mean recovery of at least 0.05. This rule identifies follow-up candidates; it is
not a confirmatory success gate, and single-head effects are not assumed to add.

For every head, report choice mean and direction rate together with pair-path
means for the two behavioral successes and two failures.

## Technical Gates

- model architecture is 16 attention heads by 128 dimensions;
- four nontrivial pair source-target effects and four choice relations;
- all 16 single-head interventions plus all-head and direct-attention controls;
- maximum difference between all-head and direct-attention patched scores at
  most `0.002` in both paths;
- all source, target, patched, and recovery values are finite.

No number of selected heads, success-failure gap, or individual head identity is
registered as an outcome threshold.
