# Phase 2 Catalog and Execution Integrity Amendment

- Amendment frozen at: `2026-07-13T12:37:59.9482667Z`
- Original protocol: `music_relation_mechanism_intervention_phase2_v1`
- Reason: blockers found by an independent read-only code review before any
  confirmatory mechanism or correction GPU batch was submitted.

## Information Seen Before This Amendment

Aggregate progress counters from the still-running primary catalog pass were
inspected. At `95/290`, the provisional legacy verifier counts included 3
strict conflicts and 8 strict exact rows. At `174/290`, they included 5 strict
conflicts, 21 strict exact rows, and 6 request errors. One unclassified raw
generation-input row (`quiet_night_drive__seed139__rank1`) was also displayed
while checking the provenance schema; it contained no catalog label or catalog
evidence. No item-level catalog output, catalog response, selected cluster,
mechanism score, or intervention result was inspected to design this amendment.

## Frozen Conservative Changes

1. A strict conflict now requires complete response windows from both catalog
   queries. A MusicBrainz response whose reported count exceeds the returned
   window, or an Apple response that fills its maximum result window, is
   excluded rather than treated as negative evidence.
2. The targeted MusicBrainz title-artist query is now an explicit alias audit.
   If it returns the shared canonical reference artist, the row is excluded as
   a possible alias instead of being promoted to a strict conflict.
3. Every final catalog row must carry the current verifier version, a terminal
   label, the frozen protocol/model provenance, a valid generation-row hash,
   and request IDs linked to the archived raw evidence. Strict rows additionally
   require complete evidence and a completed alias audit.
4. Catalog recovery preserves first-row evidence, validates the full immutable
   source-row identity, and rebuilds its cache from the append-only evidence
   archive.
5. Final scoring recomputes and binds the protocol, catalog rows, catalog raw
   evidence, verifier, model revision, experiment bundles, and submitted runner
   scripts by SHA-256.
6. H2 is not tested and no confirmatory bootstrap interval is computed when
   either non-indeterminate mechanism class has fewer than the frozen 15-title
   minimum.
7. GPU runners emit provenance receipts and incremental checkpoints and support
   deterministic record shards so interrupted work can be recovered without
   silently dropping records.

## What Did Not Change

The model, revision, generation prompts, contexts, seeds, sampling parameters,
sample caps, normalized-title clustering, hash salts, diagnostic layers,
patching method, correction prompts, hypothesis thresholds, and stop rules are
unchanged. No additional model or post-result rescue sample is allowed.

## Consequence for Confirmatory Use

The legacy in-progress pass is retained as an audit artifact only. Before
selection, every primary row will be reclassified by
`phase2_catalog_v2_complete_alias_audit`, reusing archived raw responses and
querying only missing or failed evidence. Any registered extension uses the
same amended verifier. The truncation and alias rules can only remove a legacy
strict-conflict candidate; they cannot create one. Formal mechanism and
correction jobs remain forbidden until all integrity gates and tests pass.
