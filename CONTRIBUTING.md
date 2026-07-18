# Contributing

Thank you for helping improve Music Preference Lens. Contributions should make
the public evidence easier to verify, reproduce, or interpret without widening
the claims beyond the released data.

## Useful Contributions

- reproduce a frozen public result and report the environment and outcome;
- add tests for parsers, catalog normalization, selection, or integrity checks;
- correct broken links, documentation, or provenance metadata;
- propose a separately frozen replication on another open model;
- improve accessibility or readability of reports and figures.

## Before Opening a Pull Request

1. Open an issue before starting a new GPU experiment or changing a scientific
   threshold.
2. Keep confirmatory and exploratory evidence separate.
3. Preserve negative results and failed continuation gates.
4. Run:

   ```powershell
   python scripts/validate_publication.py
   python -m unittest discover -s tests
   ```

5. Describe the files changed, the claim affected, and the verification run.

## Data and Security

Do not commit:

- raw third-party API response bodies;
- model or service credentials;
- private Hugging Face or GitHub URLs;
- personal names, emails, local paths, or other identifying data;
- copyrighted audio, artwork, paper PDFs, or model weights.

Derived catalog rows must include source provenance and must follow
[`docs/public_data_policy.md`](docs/public_data_policy.md).

## Claim Discipline

Do not describe catalog misses as proof of fiction, behavioral signals as
subjective awareness, or post-hoc interventions as a validated detector.
Replications that change models, prompts, thresholds, samples, or catalog dates
must be reported as new studies.

## License

By contributing, you agree that original code contributions are licensed under
Apache-2.0 and original documentation/report contributions are licensed under
CC BY 4.0, as described in [`LICENSING.md`](LICENSING.md).
