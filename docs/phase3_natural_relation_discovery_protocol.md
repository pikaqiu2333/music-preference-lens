# Phase 3 Natural Relation-Conflict Discovery Protocol

## Purpose

Phase 3 replaces the failed assumption that every strict catalog mismatch is a
known title-artist relation error. A model can invent a generic title that
happens to match an unrelated catalog record. The new unit of analysis is
therefore a **recoverable relation conflict**:

1. the title and emitted artist come from an unconstrained model-generated
   playlist;
2. both frozen catalogs reject the emitted pair and share another supported
   artist for the title;
3. under independent candidate-free title queries, the same model generates
   that catalog-supported artist in at least two of three templates; and
4. the median target-versus-emitted mean-token log-probability margin is
   positive.

This definition does not prove subjective awareness. It establishes a narrower
behavioral fact: the relation is independently accessible to the model despite
the earlier mismatch.

## Design status

The protocol was frozen before the Granite relation-recoverability audit
returned. It is exploratory because the discovery set may be used to select a
layer and regularization. A later holdout is isolated from all such choices.

## Model choice

The discovery model is the pinned `Qwen/Qwen3-4B-Base` revision already used in
the project as a cross-model verifier. It is open-weight, small enough for full
residual extraction and patching, and distinct from the Granite run that
exposed the catalog-collision problem.

## Discovery pilot

Twelve relation-stress contexts cover long-tail synth-pop, city pop, post-punk,
ambient, underground hip-hop, alternative R&B collaborations, modern jazz,
folk deep cuts, IDM, soundtrack credits, Latin alternative, and metal
subgenres. Six frozen seeds produce 72 playlists and at most 360 track events.

The stress contexts intentionally increase factual difficulty. Their yield is
not an estimate of ordinary recommendation hallucination prevalence.

The pilot expands only when both conditions hold:

- at least 300 events parse under the frozen field order; and
- at least eight unique normalized titles pass catalog conflict and behavioral
  recoverability.

If either gate fails, the study stops before the registered expansion. Eight
recoverable conflicts are the minimum pilot yield consistent with reaching
roughly 30 at the 1,440-event discovery cap.

## Discovery expansion

If the pilot gate passes, eighteen new seeds are added to the same contexts.
The combined discovery maximum is 288 playlists. At least 30 unique recoverable
conflict titles and 30 matched strict-exact controls are required before probe
development.

One row is selected per normalized title. Exact controls are matched on title
and artist token counts, generation rank, context family, and catalog title
result-count bin. Unverified, ambiguous, source-error, and title-absent rows
remain visible but cannot enter the primary contrast.

## Position-wise probe

Residual states are extracted at prompt end, title end before artist generation,
the first artist token, and complete pair end. The primary position is title
end, accurately described as `pre-artist` or `pre-relation-completion`.

All 36 layers may be examined inside nested discovery-only cross-validation.
The hidden-state model must be compared with frozen surface baselines: entropy,
logit margin, sequence probability, entity token lengths, catalog result-count
bin, lexical title embedding, random labels, and random layers.

Discovery results may select a layer and regularization but cannot support the
final generalization claim.

## Sealed holdout

The holdout has eight different contexts, eight new seeds, and a differently
worded prompt with the same title-before-artist order. It must not be generated
until the verifier, recoverability assay, layer-selection rule, baselines, and
all hyperparameters are frozen in a committed pipeline receipt.

The holdout additionally removes every title and every emitted/reference artist
seen in discovery. At least ten recoverable conflict clusters must remain. If
not, the result is reported as underpowered and no intervention is run.

## Intervention gate

Patching is allowed only when the held-out pre-artist probe exceeds the best
surface baseline and the cluster-bootstrap lower confidence bound on that
difference is above zero. The primary intervention outcome is the sequence
log-probability margin of the catalog-supported artist over the emitted artist.

Random-layer, random-donor, entity-mismatched, and sham patches are mandatory,
along with non-target-token KL, format, and output-length side-effect checks.

## Exclusions and claims

Reason-first order, Song IDs, candidate lists, subjective recommendation
quality, and persuasive reasons are outside the primary Phase 3 analysis.
Product systems should still use retrieval, IDs, constrained decoding, or
catalog validation. Even a successful Phase 3 result would show a decodable and
causally usable relation-validity signal, not consciousness, deception, or a
universal hallucination detector.
