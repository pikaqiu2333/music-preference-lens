# Reproduce the Phase 2 Granite Catalog-Yield Decision

## Scope

This public procedure reproduces the frozen Phase 2 selection and stop decision
from released derived catalog rows. It does not redistribute or replay complete
raw MusicBrainz or Apple response bodies.

Run commands from the repository root with Python 3.10 or newer.

## 1. Run the Public Validation

```powershell
python scripts/validate_publication.py
```

The validator:

- loads all 385 released primary and extension rows;
- checks their frozen hashes and label counts;
- reruns deterministic title clustering and selection;
- reproduces 7 strict-conflict and 25 strict-exact title clusters;
- confirms `STOP_INSUFFICIENT_STRICT_CONFLICT_CLUSTERS`;
- verifies that the four raw third-party response archives are absent.

Expected decision fields:

```json
{
  "combined_row_count": 385,
  "selected_conflict_cluster_count": 7,
  "selected_exact_cluster_count": 25,
  "decision": "STOP_INSUFFICIENT_STRICT_CONFLICT_CLUSTERS",
  "formal_mechanism_run_allowed": false
}
```

## 2. Run Focused Tests

```powershell
python -m unittest tests.test_phase2_catalog `
  tests.test_phase2_mechanism_intervention `
  tests.test_finalize_phase2_catalog `
  tests.test_merge_phase2_catalog_evidence `
  tests.test_publication_assets
```

## 3. Inspect Frozen Public Assets

The public replay inputs and outputs are:

- `runs/phase2_granite_primary_catalog_verified_v2.jsonl`;
- `runs/phase2_granite_extension_catalog_verified_v2.jsonl`;
- `runs/phase2_granite_combined_catalog_verified_v2.jsonl`;
- `runs/phase2_granite_final_selected_conflicts_v2.jsonl`;
- `runs/phase2_granite_final_selected_exact_v2.jsonl`;
- `runs/phase2_granite_final_catalog_summary_v2.json`.

The original raw-archive sizes and SHA-256 digests remain in
[`runs/private_evidence_receipt.json`](../runs/private_evidence_receipt.json)
and the original Phase 2 receipt. A holder of a private archive can compare it
to those hashes, but byte-level raw-response replay is not part of the public
package.

## 4. Run a Contemporary Catalog Replication

The released verifier can query current MusicBrainz and Apple services:

```powershell
python scripts/verify_phase2_catalog.py --help
```

Catalog services can change. A live rerun is a contemporary replication, not a
reconstruction of the July 2026 response bytes. Do not overwrite the frozen
derived rows with live results.

The public protocol and amendment history remain in the
[external Phase 2 Gist](https://gist.github.com/a56668283b095f59f0eacf0527395b58).

Do not run the mechanism or correction exporter after a stop decision. A new
sample, model, threshold, or prompt belongs to a separately registered study.
