# Relation-Knowledge Recoverability Audit

## Status

This protocol was frozen on 2026-07-14 before running the audit. It is a new
exploratory measurement audit and does not modify or reopen the stopped Phase 2
confirmatory protocol.

## Why this audit is necessary

The seven Phase 2 strict catalog conflicts satisfy a conservative external
rule: both catalogs reject the emitted title-artist pair and share one other
catalog-supported artist for the title. That rule establishes a catalog
mismatch, but it does not establish that the title was a known song in the
model's generation process.

Several observed titles, including `Retro Disco`, `Hard Energy`, and
`Lunar Pulse`, are generic enough that a fabricated title could collide with an
unrelated catalog entry. Treating every such collision as "the model knew the
song but bound the wrong artist" would overstate the evidence.

## Frozen inputs

- Seven normalized-title conflict clusters from the Phase 2 final selection.
- Twenty-five generated strict-exact title controls from the same model and run.
- Eight manually frozen, high-familiarity positive controls used only to test
  whether the model and parser can recover ordinary title-artist relations.
- The exact Granite model revision used for Phase 2.

No title is added, removed, or relabeled after viewing this audit's outputs.

## Measurements

Each title is queried with three candidate-free templates. Greedy generation is
parsed using a frozen normalization rule. The audit also teacher-forces the
catalog-supported target artist and, for conflict rows, the originally emitted
artist under the same title prompt.

The primary score for a conflict is the median, across the three templates, of:

`mean_logp(catalog_reference | title_prompt) - mean_logp(emitted_artist | title_prompt)`

Mean token log probability is used to reduce artist-name length bias. Total
sequence log probability is retained as a secondary diagnostic.

## Frozen categories

- `reference_recoverable`: the catalog reference is generated in at least two
  templates and the primary margin is positive.
- `generation_only`: the reference is generated in at least two templates but
  the primary margin is non-positive.
- `margin_only`: the reference is generated fewer than two times but the primary
  margin is positive.
- `persistent_emitted_binding`: the original emitted artist is generated in at
  least two templates, the reference is not, and the primary margin is
  non-positive.
- `unrecovered_or_indeterminate`: every other pattern.

Only `reference_recoverable` rows can enter a later mechanistic discovery pool
as candidate known-relation errors. The remaining rows are still catalog
mismatches, but they carry no latent-knowledge interpretation.

## Validity gates

The audit is technically interpretable only when at least 95% of the requested
generations are nonempty and at least six of eight canonical controls recover
their target artist in two of three templates. Failure of the canonical gate
means the direct-recall assay is too weak for this model; it does not relabel the
seven conflicts as unknown.

## Claim boundary

This audit is descriptive and underpowered. It cannot establish subjective
awareness, a hidden-state mechanism, a necessary circuit, population
prevalence, or successful self-correction. A positive result only shows that a
catalog-supported relation is behaviorally accessible under an independent
direct query despite an earlier free-generation mismatch.
