# Music Reason Component-Patching Pilot

## Question

When a listening need changes to its semantic opposite, which model layers and
components causally change support for the already-generated complete
title-artist recommendation?

This is an exploratory three-case mechanistic pilot, not a benchmark. Catalog
truth and need-fit faithfulness remain separate labels.

## Cases

- `All of Me - John Legend`: verified pair with a strong positive need effect.
- `One More Time - Dead or Alive`: catalog-conflict pair with a strong positive
  need effect.
- `Space Oddity - David Bowie`: verified pair whose score moves in the wrong
  direction for the strict no-vocals constraint.

The cases were selected after the behavioral smoke to represent distinct
failure modes. No population-level claim will be made from them.

## Intervention

For every case, teacher-force the same complete title and artist under the
original and opposite listening needs. At aligned prediction positions for all
title and artist tokens, patch opposite-condition activations into the original
condition at layers 14, 16, 18, 21, 24, 27, and 28.

Patch three outputs separately:

- self-attention output;
- MLP output;
- full decoder-layer residual output.

The primary metric is pair mean log probability. For a valid case:

`recovery = (original_logp - patched_logp) / (original_logp - opposite_logp)`

A recovery of 1 means the patch reproduces the opposite-condition effect; 0
means it leaves the original score unchanged. The signed definition also works
for the `Space Oddity` failure case.

## Registered Technical Gates

- Exactly three registered cases are scored.
- Every case has an absolute original-vs-opposite pair effect of at least 0.05.
- Archived and replayed baseline pair log probabilities differ by at most 0.02.
- Layer-28 full-residual patch endpoint error is at most 0.02.
- All attention, MLP, and full-residual patch values are finite.

Passing these gates validates the intervention. It does not establish that the
model knows a recommendation is false or hallucinatory.
