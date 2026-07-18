# Public Data Policy

## Purpose

This repository is a public, anonymized research snapshot. It publishes the
project's prompts, model outputs, derived catalog-verification rows, analysis
scripts, reports, and integrity receipts.

It intentionally does not redistribute frozen raw response bodies returned by
third-party catalog services.

## Public Assets

The public package includes:

- generated title-artist-reason events;
- normalized MusicBrainz and Apple query provenance;
- derived catalog labels and source identifiers;
- frozen selection rows and analysis summaries;
- model-run scripts, result rows, and report figures;
- byte counts and SHA-256 hashes for omitted private evidence archives.

The public package does not include Apple artwork, audio previews, or complete
raw API response bodies.

## Omitted Evidence

Four frozen response archives are retained in non-public evidence storage.
Their original paths, byte counts, request counts, compressed hashes, and
available decompressed hashes are recorded in
[`runs/private_evidence_receipt.json`](../runs/private_evidence_receipt.json).

The receipt allows a holder of a private archive to verify that it is the exact
frozen artifact. It does not make byte-level raw-response replay a public
reproducibility claim.

## Reproduction Boundary

Public users can:

1. validate every released derived row and report asset;
2. reproduce the frozen analysis and stop decisions from those rows;
3. rerun catalog queries with the released verifier and current services;
4. compare a privately held raw archive against the published receipt.

Live catalog services can change. A new live query is a contemporary
replication, not a reconstruction of the July 2026 response bytes.

## Source Terms

- MusicBrainz core and supplementary data have different Creative Commons
  terms. See the [MusicBrainz data license](https://musicbrainz.org/doc/About/Data_License).
- Apple Search API output and promotional-content fields remain governed by
  Apple's terms. See the [Apple Search API overview](https://developer.apple.com/library/archive/documentation/AudioVideo/Conceptual/iTuneSearchAPI/).

No project license grants rights to third-party material beyond rights already
provided by its source.
