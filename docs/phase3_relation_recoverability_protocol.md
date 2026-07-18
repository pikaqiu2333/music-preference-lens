# Phase 3 Relation-Knowledge Recoverability Protocol

## Freeze point

This protocol was frozen while the Phase 3 natural-generation pilot was still
running and before any pilot catalog result was available. It operationalizes
the recoverability rule already registered in the parent Phase 3 protocol.

## Input and catalog gate

The input is the complete pilot catalog-verification table produced by the
unchanged `phase2_catalog_v2_complete_alias_audit` verifier. At least 300 rows
must be present. Every unique strict-conflict title is included; no ambiguous,
excluded, error, or title-absent row may enter the audit.

One representative is selected per normalized title using a salted SHA-256
rule. Strict-exact controls exclude every conflict title and use the same
one-row-per-title rule before a deterministic cap of 24 is applied.

## Candidate-free assay

Each title is queried with three fixed prompts that do not show either artist.
Greedy generation is parsed from the first line. For a strict conflict, the
catalog-supported artist must be generated in at least two prompts. The model
must also assign the reference artist a positive median mean-token log-
probability margin over its originally emitted artist.

A conflict satisfying both requirements is a
`recoverable_relation_conflict`. This is behavioral evidence that the relation
is independently accessible to the model. It is not evidence of consciousness,
deception, or a globally available confidence variable.

## Assay controls

Eight fixed high-familiarity title-artist relations test whether the assay can
recover obvious knowledge. At least 75% must be recovered in two of three
prompts. At least 95% of all candidate-free generations must be nonempty.

The natural strict-exact controls estimate assay behavior in the same stressed
catalog domain but do not alter either validity threshold.

## Pilot decision

The study continues to the previously registered expansion only when all of
the following hold:

1. at least 300 pilot catalog rows exist;
2. the technical and canonical-control gates pass; and
3. at least eight unique title clusters are recoverable relation conflicts.

Failure stops the expansion. Prompts, thresholds, controls, or the conflict
definition cannot be relaxed to rescue yield.
