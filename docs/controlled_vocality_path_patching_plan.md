# Controlled Vocality Pair-vs-Choice Patching Plan

## Question

Why can the same model use vocality correctly when comparing real candidates
but fail to apply it consistently while generating a title-artist continuation?

## Sentinel Tracks

- vocal success: `Blinding Lights - The Weeknd`;
- vocal pair failure: `Space Oddity - David Bowie`;
- instrumental success: `River Flows in You - Yiruma`;
- instrumental pair failure: `Awake - Tycho`.

All four have an absolute matched-versus-flipped pair effect above 0.10.

## Pair Path

For each fixed title-artist pair, use the matching vocality reason as source and
the flipped reason as target. At every title and artist prediction position,
patch source attention, MLP, or full decoder residual output into the target
run. Measure recovery of complete-pair mean log probability toward the source.

The signed recovery remains valid for failure tracks whose source effect is
negative.

## Choice Path

For each vocality class and each of two reversed candidate orders, use the
matching class reason as source and the opposite reason as target. Patch the
final prompt prediction position and measure recovery of class-versus-other
candidate logit mass toward the source.

## Layers and Components

- layers: 14, 16, 18, 21, 24, 27, and endpoint layer 28;
- components: self-attention, MLP, and full residual.

## Registered Technical Gates

- four sentinel pair effects with absolute magnitude at least 0.10;
- four valid choice relations (two classes by two orders);
- complete component captures at all seven layers;
- layer-28 full-residual endpoint error at most 0.02 for both paths;
- finite source, target, patched, and recovery metrics.

This is an exploratory mechanism contrast. No average component dominance
threshold is registered.
