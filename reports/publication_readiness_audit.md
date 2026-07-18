# Publication Readiness Audit

**Audit date:** 2026-07-18
**Scope:** Music Preference Lens public-v1 title-artist relation hallucination snapshot

## Requirement Evidence

| Requirement | Authoritative evidence | Status |
| --- | --- | --- |
| Freeze the same-model relation-conflict rule before independent scoring | `docs/independent_holdout_protocol.md`; frozen seeds, hash selection, zero thresholds, and three gates | Complete as a pre-specified test; not externally preregistered |
| Generate a genuinely new free-recommendation holdout | `runs/independent_holdout_raw_generations.jsonl`; 20/20 generations from unseen seeds 59/71/83/101/127 | Complete |
| Parse and independently verify against two catalogs | `runs/independent_holdout_parse_summary.json`; `runs/independent_holdout_catalog_verified.jsonl`; MusicBrainz and Apple evidence | Complete: 19/20 parsed generations, 93 pairs, 9 strict exact and 21 strict conflict pool events |
| Test the frozen same-model rule | `runs/independent_holdout_verifier_rows.jsonl` and summary; Job `6a524112e4a4e82c0b58da32` | Complete negative result: BA 0.611, specificity 0.667, sensitivity 0.556 |
| Replicate the behavioral boundary on a second open model | `runs/qwen3_4b_cross_model_verifier_rows.jsonl` and summary; Job `6a52f77feffc02a91cbda1bb` | Complete: Qwen3-4B still missed the same three unique conflict relations |
| Diagnose why the rule fails | title-control rows/summary and full-sequence causal rows/summary | Complete: prior-masked correct relation and causally formed wrong binding are separable post hoc |
| Integrate mechanism interventions, field diagnosis, and positive/negative results | Section 2 and Sections 5-6 of the English report; reason, component, head, and field assets listed in `docs/reproduce_publication.md` | Complete; exploratory evidence remains separated from holdout confirmation |
| Produce public English and Chinese reports with figures, related work, limitations, and product implications | `reports/music_relation_hallucination_technical_report.md`; `reports/music_relation_hallucination_summary.zh.md`; `reports/related_work_positioning_2026_07.zh.md`; two reproducible SVG figures | Complete |
| Preserve reproducible experiment assets | `docs/reproduce_publication.md`; public manifest; bundles, scripts, derived rows, summaries, exact Job script payloads, terminal log snapshots, model revisions, and private-evidence hashes | Complete within the documented public-data boundary |
| Externally timestamp the Granite confirmatory protocol before generation | public Phase 2 Gist history; exact protocol and runner hashes; `runs/phase2_preregistration_receipt.json` | Complete before Granite smoke and scientific generation |
| Exhaust the frozen primary and extension generation cap | 60 primary plus 20 extension playlists; 385 parsed relations from the fixed Granite revision | Complete; three malformed completions were retained without repair |
| Reproduce the public Phase 2 stop decision without redistributing raw responses | released derived rows, deterministic cluster selection, `runs/private_evidence_receipt.json`, and `docs/public_data_policy.md` | Complete; 7 conflict and 25 exact clusters replay from 385 public rows |
| Apply the preregistered final stop gate | `runs/phase2_granite_final_catalog_summary_v2.json`; 7 unique strict conflicts versus a minimum of 30 | Complete boundary result; mechanism diagnosis, H1, H2, and correction were not run |

## Automated Audit

- `python scripts/validate_publication.py`: `ready`;
- 208 public publication files covered by SHA-256 in the public-v1 manifest;
- 5 completed Hugging Face Jobs archived with inspect metadata, exact submitted
  script hashes, and terminal-log hashes;
- 5 publication-stage and 6 integrated exploratory technical gates checked;
- all local report links resolve;
- full local suite: 229 tests run, with 228 passed and 1 NumPy-only test skipped
  by design.

## Claim Boundaries

1. The warning rule was pre-specified in the working session, but there is no
   external registry timestamp or pre-result Git commit.
2. The 4B experiment is a same-family cross-model critic test, not 4B
   self-verification on 4B-generated playlists.
3. Counterfactual and activation-patching results are post-hoc mechanism
   diagnostics, not a newly validated detector.
4. Catalog conflict is an operational double-catalog label, not proof that no
   legitimate release exists anywhere.
5. Behavioral and causal signals do not establish subjective awareness.
6. Phase 2 H1 and H2 were not tested. Seven strict conflict clusters were too
   few to pass the externally timestamped 30-cluster entry gate.
7. The 322 Phase 2 excluded rows are evidence-insufficient, not assumed correct
   or incorrect, so 7/304 is not a model hallucination-rate estimate.
8. Raw third-party response bodies are retained outside the public repository.
   The public package reproduces analysis from derived rows and preserves the
   omitted archive hashes, but it does not provide byte-level response replay.

## Public Distribution

The public snapshot is anonymized and contains no personal author record. It
uses the following distribution choices:

- code under Apache-2.0;
- original reports and figures under CC BY 4.0;
- third-party material excluded from both grants;
- complete raw catalog response bodies omitted from public Git history;
- a project-entity citation under `Music Preference Lens Contributors`.

These choices affect attribution, public reproducibility scope, and legal
clarity. They do not change the frozen experimental conclusions.
