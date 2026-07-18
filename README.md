# Music Preference Lens

[![Validate public snapshot](https://github.com/pikaqiu2333/music-preference-lens/actions/workflows/validate.yml/badge.svg)](https://github.com/pikaqiu2333/music-preference-lens/actions/workflows/validate.yml)
[![Code: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![Reports: CC BY 4.0](https://img.shields.io/badge/reports-CC%20BY%204.0-lightgrey.svg)](LICENSING.md)

Music Preference Lens is a public, anonymized, reproducible research snapshot
on title-artist hallucinations and explanation faithfulness in LLM-based music
recommendation.

The main question is deliberately narrow:

> Can a base LLM reliably detect that its own freely generated song title and
> artist do not form a catalog-supported relation?

## TL;DR

- A pre-specified Qwen3-1.7B self-check rule was **not confirmed** on an
  independent 9-exact/9-conflict holdout: balanced accuracy was `0.611`, below
  the frozen `0.75` gate.
- A Qwen3-4B cross-model verifier still missed the same three unique conflict
  relations; scaling within the family did not remove the boundary.
- Counterfactual and causal diagnostics separated at least two failure modes:
  a correct relation can exist but lose at readout, or the network can build a
  wrong relation state in middle and late layers.
- The product conclusion is not to stack another prompt. Resolve or retrieve a
  valid Song ID, use internal conflict signals only for routing, and generate
  reasons from grounded metadata.

This is a negative result about internal self-verification. It is not evidence
of subjective awareness and it is not a general hallucination detector.

## Main Results

| Stage | Frozen evidence | Result | Interpretation |
| --- | --- | --- | --- |
| Qwen3-1.7B confirmation | 9 exact + 9 conflict relations | BA `0.611`; rule not confirmed | Same-model self-check is not a reliable factual gate |
| Qwen3-4B cross-model check | Same 18 relations | BA `0.722`; same three unique misses | A larger same-family critic is insufficient |
| Counterfactual diagnosis | Real-title vs unrelated-title controls | Correct relative signal survives in some wrong outputs | Latent knowledge can lose to priors or readout |
| Causal trace | Full-sequence residual patching | Relation effects emerge around layers 16-18 and stabilize late | Some wrong bindings are causally constructed, not merely hidden |
| Granite Phase 2 boundary | 385 parsed relations | 7 unique strict conflicts vs required 30 | Stopped before mechanism/correction hypotheses |
| Qwen Phase 3 boundary | 11 conflict-title candidates | 0 candidate-free recoveries vs required 8 | Pairwise preference is not open-recall evidence |

![Independent holdout verification metrics](reports/figures/holdout_verifier_metrics.svg)

## Study Design

1. Generate playlists without candidate tracks.
2. Parse exact `title + artist + reason` events.
3. Build conservative operational labels using MusicBrainz and Apple queries.
4. Freeze an independent balanced holdout before model scoring.
5. Test independent-choice and full-name likelihood self-checks.
6. Use title counterfactuals and activation patching only as post-hoc mechanism
   diagnostics.
7. Stop confirmatory extensions when their preregistered entry gates fail.

The project separates four questions that are often collapsed into one:

- Is the generated entity catalog-supported?
- Does it fit the user's preference and context?
- Does it satisfy explicit hard constraints?
- Is the generated reason faithful to the selection process?

## Product Implication

A production music assistant should:

1. retrieve or resolve a valid Song ID before presenting a recommendation;
2. treat internal disagreement as a reason to verify, replace, or abstain;
3. avoid using model confidence as a truth label;
4. generate recommendation reasons from already grounded catalog metadata;
5. evaluate factual grounding separately from preference fit and explanation
   faithfulness.

Song IDs solve the open-vocabulary entity-validity problem. They do not by
themselves solve preference quality or rationale faithfulness.

## Reports

- [English technical report](reports/music_relation_hallucination_technical_report.md)
- [中文研究摘要](reports/music_relation_hallucination_summary.zh.md)
- [Independent holdout protocol](docs/independent_holdout_protocol.md)
- [Phase 2 Granite boundary](reports/phase2_granite_confirmatory_catalog_yield.md)
- [Phase 3 Qwen recoverability boundary](reports/phase3_qwen_relation_recoverability_pilot.zh.md)
- [Related-work positioning](reports/related_work_positioning_2026_07.zh.md)
- [Publication readiness audit](reports/publication_readiness_audit.md)

## Quick Validation

Requirements:

- Python 3.12 for the frozen publication runners;
- no GPU or third-party package is required for public integrity validation.

```powershell
git clone https://github.com/pikaqiu2333/music-preference-lens.git
cd music-preference-lens
python scripts/validate_publication.py
python -m unittest discover -s tests
```

Expected status:

- publication validator: `ready`;
- all available tests pass;
- one NumPy-only test may be skipped when NumPy is not installed.

Full GPU and Hugging Face Jobs reproduction steps are in the
[reproduction guide](docs/reproduce_publication.md). The public Phase 2
derived-row replay is documented separately in
[the Phase 2 guide](docs/reproduce_phase2_catalog_yield.md).

## Repository Map

- `reports/`: technical reports, Chinese summaries, and figures;
- `docs/`: frozen protocols, reproduction guides, and research boundaries;
- `scripts/`: generation, verification, intervention, and validation code;
- `runs/`: frozen public bundles, derived rows, summaries, and receipts;
- `data/`, `prompts/`, `schemas/`: experiment inputs and structured contracts;
- `tests/`: behavioral, integrity, and publication tests.

Earlier explanation-order, reason-swap, vocality, SAE, identity, and J-space
pilots remain in the repository as explicitly exploratory evidence. They are
not pooled with the independent holdout metrics.

## Public Data Boundary

Complete raw third-party catalog response bodies are intentionally not
redistributed. The public repository contains derived verification rows,
request provenance, frozen selections, and cryptographic receipts.

- [Public data policy](docs/public_data_policy.md)
- [Private-evidence hash receipt](runs/private_evidence_receipt.json)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

This means the public snapshot reproduces the released analysis and stop
decisions, but does not claim byte-level public replay of Apple or MusicBrainz
response bodies. Current catalog queries are contemporary replications because
the services can change.

## Scope and Limitations

- The independent holdout contains 18 events and is not an industry error-rate
  estimate.
- Catalog conflict is an operational double-source label, not proof that no
  legitimate release exists anywhere.
- Qwen3-4B is a cross-model critic on Qwen3-1.7B outputs, not 4B
  self-verification on its own generations.
- Counterfactual and activation-patching analyses are post-hoc diagnostics.
- The study uses base models; instruction tuning, retrieval training, and
  production recommendation stacks may behave differently.
- Behavioral and causal signals do not establish subjective awareness.

## Citation

See [`CITATION.cff`](CITATION.cff). Until a named author record is supplied,
cite the project entity as **Music Preference Lens Contributors**.

## Contributing

Reproduction reports, integrity fixes, catalog-verifier tests, and carefully
scoped replications are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md)
before opening a pull request. Do not submit raw third-party API responses,
personal data, credentials, or claims that exceed the documented evidence.

## License

- Code is licensed under [Apache-2.0](LICENSE).
- Original reports and figures are licensed under
  [CC BY 4.0](LICENSING.md).
- Third-party material is excluded from those grants and remains governed by
  its source terms. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
