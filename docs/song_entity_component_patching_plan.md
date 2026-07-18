# Song Entity Attention-vs-MLP Component Patching

## Motivation

Classifier-free layerwise attribution showed that title-conditioned artist
information begins to concentrate at the prediction position around layers
14-17 and becomes stably output-readable from layer 24. Full-residual patching
cannot tell whether self-attention retrieves the title information or the MLP
transforms and amplifies it.

## Design

Reuse the same 12 relations from six independent title-artist swap blocks. At
selected layers `14, 16, 18, 21, 24, 27`, capture the neutral-title output at
the final `Artist:` prefix position for three components:

1. self-attention output;
2. MLP output;
3. complete decoder-layer output.

Replace one component at a time in the real-title forward pass, continue the
model, and measure the correct-first-token minus wrong-first-token margin. Layer
28 full-residual replacement is included only as an endpoint consistency check.

No classifier, SAE feature selection, or new recommendation labels are used.

## Metrics

For relations where the real-title and neutral-title first-token margins differ
by at least `0.05`:

- aligned effect: movement from the real margin toward the neutral margin;
- recovery fraction: fraction of the real-vs-neutral difference removed by the
  component patch;
- toward-neutral accuracy: fraction of relations moved closer to neutral.

Positive recovery means that the patched component carries title-conditioned
information. Attention and MLP effects are compared within the same layer, but
they are not assumed to add linearly.

## Registered Checks

- At least eight valid relations have an absolute real-vs-neutral first-token
  difference of `0.05` or more.
- Every selected neutral component output is captured.
- Layer-28 full-residual replacement reproduces the neutral first-token margin
  within `0.02` for every relation.
- Interpretation is allowed only if all checks pass.

The experiment localizes component contribution. It does not by itself prove
that the model recognizes a freely generated catalog conflict or fabrication.
