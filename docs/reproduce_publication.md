# Reproducing the Music Relation Hallucination Report

This guide rebuilds the independent holdout, confirmation runs, post-hoc
diagnostics, figures, and publication manifest used by the July 2026 technical
report.

The subsequent externally timestamped Granite catalog-yield boundary is
reproduced separately in
[`docs/reproduce_phase2_catalog_yield.md`](reproduce_phase2_catalog_yield.md).
Its final decision is included in publication manifest v3, but the stopped
mechanism and correction stages have no result artifacts.

## Requirements

- Python 3.12;
- `uv` for PEP 723 runner dependencies;
- network access for MusicBrainz, Apple, and public model downloads;
- an NVIDIA GPU for local model runs, or a Hugging Face account with Jobs;
- the current repository, including the archived discovery bundle.

The report uses no private API dataset and no manual annotation file. Catalog
labels are generated from the two public catalog queries. Catalog services can
change, so the released derived rows, frozen summaries, and SHA-256 receipts in
`runs/publication_manifest.json` are the authoritative public July 2026
snapshot. Complete raw third-party response bodies are retained outside the
public repository; see `docs/public_data_policy.md`.

Set the exact model revisions used for deterministic reruns:

```powershell
$QWEN17_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
$QWEN4_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
```

The original Jobs loaded the Hub default branch without an explicit revision.
These revisions were resolved retrospectively: both model repositories had
last been modified in July 2025, before every archived Job. This limitation is
recorded in `runs/hf_job_metadata.json` rather than hidden.

## 1. Independent Free Generation

Build the frozen new-seed generation bundle:

```powershell
python scripts/export_holdout_generation_probe.py
```

Run locally on a CUDA machine:

```powershell
uv run scripts/run_holdout_generation_probe.py `
  --bundle runs/independent_holdout_generation_bundle.json `
  --model-revision $QWEN17_REVISION `
  > runs/independent_holdout_generation.log
```

For Hugging Face Jobs, first build a standalone script containing the bundle:

```powershell
python scripts/build_embedded_hf_job.py `
  --runner scripts/run_holdout_generation_probe.py `
  --bundle runs/independent_holdout_generation_bundle.json `
  --placeholder __EXPERIMENT_BUNDLE_B64__ `
  --compression none `
  --output runs/jobs/run_holdout_generation_embedded.py

hf jobs uv run runs/jobs/run_holdout_generation_embedded.py `
  --flavor a10g-small --timeout 45m --python 3.12 `
  --env "MODEL_REVISION=$QWEN17_REVISION" --detach
```

Capture the completed job log and extract the records:

```powershell
hf jobs logs <JOB_ID> > runs/independent_holdout_generation.log

python scripts/extract_job_log_artifacts.py `
  --log runs/independent_holdout_generation.log `
  --row-marker HOLDOUT_RAW_GENERATION_JSON= `
  --summary-marker HOLDOUT_GENERATION_SUMMARY_JSON= `
  --rows runs/independent_holdout_raw_generations.jsonl `
  --summary runs/independent_holdout_generation_summary.json
```

## 2. Frozen Parsing and Catalog Verification

```powershell
python scripts/analyze_holdout_generations.py `
  --bundle runs/independent_holdout_generation_bundle.json `
  --raw runs/independent_holdout_raw_generations.jsonl `
  --rows runs/independent_holdout_generated_pairs.jsonl `
  --summary runs/independent_holdout_parse_summary.json

python scripts/verify_song_entity_catalog.py `
  --input runs/independent_holdout_generated_pairs.jsonl `
  --output runs/independent_holdout_catalog_verified.jsonl `
  --report reports/independent_holdout_catalog_verification.md `
  --all-pairs
```

The verifier checkpoints output rows. If a catalog request fails, rerun with
`--retry-errors-only`. Do not relabel ambiguous rows manually.

## 3. Qwen3-1.7B Confirmatory Verification

```powershell
python scripts/export_independent_holdout_verifier_probe.py

python scripts/build_embedded_hf_job.py `
  --runner scripts/run_independent_holdout_verifier_probe.py `
  --bundle runs/independent_holdout_verifier_bundle.json `
  --placeholder __EXPERIMENT_BUNDLE_ZLIB_B64__ `
  --compression zlib `
  --output runs/jobs/run_holdout_verifier_embedded.py

hf jobs uv run runs/jobs/run_holdout_verifier_embedded.py `
  --flavor a10g-small --timeout 45m --python 3.12 `
  --env "MODEL_REVISION=$QWEN17_REVISION" --detach
```

After fetching the log:

```powershell
hf jobs logs <JOB_ID> > runs/independent_holdout_verifier.log

python scripts/extract_job_log_artifacts.py `
  --log runs/independent_holdout_verifier.log `
  --row-marker HOLDOUT_VERIFY_ROW_JSON= `
  --summary-marker HOLDOUT_VERIFY_SUMMARY_JSON= `
  --rows runs/independent_holdout_verifier_rows.jsonl `
  --summary runs/independent_holdout_verifier_summary.json
```

The confirmatory rule and thresholds are stored in the bundle. Do not edit them
after catalog labels or model scores are available.

## 4. Qwen3-4B Cross-Model Verification

Create the same frozen events with a different verifier model:

```powershell
python scripts/export_independent_holdout_verifier_probe.py `
  --model-id Qwen/Qwen3-4B-Base `
  --bundle runs/qwen3_4b_cross_model_verifier_bundle.json `
  --encoded runs/qwen3_4b_cross_model_verifier_bundle.zlib.b64
```

Build, submit, fetch, and extract the 4B run explicitly:

```powershell
python scripts/build_embedded_hf_job.py `
  --runner scripts/run_independent_holdout_verifier_probe.py `
  --bundle runs/qwen3_4b_cross_model_verifier_bundle.json `
  --placeholder __EXPERIMENT_BUNDLE_ZLIB_B64__ `
  --compression zlib `
  --output runs/jobs/run_qwen3_4b_verifier_embedded.py

hf jobs uv run runs/jobs/run_qwen3_4b_verifier_embedded.py `
  --flavor a10g-small --timeout 45m --python 3.12 `
  --env "MODEL_REVISION=$QWEN4_REVISION" --detach

hf jobs logs <JOB_ID> > runs/qwen3_4b_cross_model_verifier.log

python scripts/extract_job_log_artifacts.py `
  --log runs/qwen3_4b_cross_model_verifier.log `
  --row-marker HOLDOUT_VERIFY_ROW_JSON= `
  --summary-marker HOLDOUT_VERIFY_SUMMARY_JSON= `
  --rows runs/qwen3_4b_cross_model_verifier_rows.jsonl `
  --summary runs/qwen3_4b_cross_model_verifier_summary.json
```

This is a cross-model critic test. It does not evaluate 4B self-verification on
4B-generated playlists.

## 5. Counterfactual Title Diagnosis

```powershell
python scripts/export_holdout_title_contrast_probe.py

python scripts/build_embedded_hf_job.py `
  --runner scripts/run_holdout_title_contrast_probe.py `
  --bundle runs/holdout_title_contrast_bundle.json `
  --placeholder __EXPERIMENT_BUNDLE_ZLIB_B64__ `
  --compression zlib `
  --output runs/jobs/run_title_contrast_embedded.py
```

Submit on `a10g-small`, fetch the log, and extract:

```powershell
hf jobs uv run runs/jobs/run_title_contrast_embedded.py `
  --flavor a10g-small --timeout 45m --python 3.12 `
  --env "MODEL_REVISION=$QWEN17_REVISION" --detach

hf jobs logs <JOB_ID> > runs/holdout_title_contrast.log

python scripts/extract_job_log_artifacts.py `
  --log runs/holdout_title_contrast.log `
  --row-marker TITLE_CONTRAST_ROW_JSON= `
  --summary-marker TITLE_CONTRAST_SUMMARY_JSON= `
  --rows runs/holdout_title_contrast_rows.jsonl `
  --summary runs/holdout_title_contrast_summary.json
```

This stage is explicitly post-hoc. Its contrast sign is a mechanism diagnostic,
not an additional confirmatory warning threshold.

## 6. Full-Sequence Causal Trace

```powershell
python scripts/export_holdout_sequence_causal_trace.py

python scripts/build_embedded_hf_job.py `
  --runner scripts/run_holdout_sequence_causal_trace.py `
  --bundle runs/holdout_sequence_causal_trace_bundle.json `
  --placeholder __EXPERIMENT_BUNDLE_ZLIB_B64__ `
  --compression zlib `
  --output runs/jobs/run_sequence_trace_embedded.py
```

The runner emits one compressed artifact in numbered log chunks because the
full intervention matrix exceeds common single-line log limits. After fetching
the log:

```powershell
hf jobs uv run runs/jobs/run_sequence_trace_embedded.py `
  --flavor a10g-small --timeout 60m --python 3.12 `
  --env "MODEL_REVISION=$QWEN17_REVISION" --detach

hf jobs logs <JOB_ID> > runs/holdout_sequence_causal_trace.log

python scripts/extract_job_log_artifacts.py `
  --log runs/holdout_sequence_causal_trace.log `
  --chunk-marker SEQ_CAUSAL_ARTIFACT_CHUNK_JSON= `
  --rows runs/holdout_sequence_causal_trace_rows.jsonl `
  --summary runs/holdout_sequence_causal_trace_summary.json
```

The final-layer endpoint must reproduce every source sequence score within the
registered tolerance. Recovery follows the factual-title state; inspect signed
effects before interpreting it as correction.

## 7. Publication Assets and Tests

```powershell
python scripts/render_technical_report_figures.py
python scripts/build_publication_manifest.py
python scripts/validate_publication.py
python -m unittest discover -s tests -v
```

Expected publication-stage checks:

- Qwen3-1.7B confirmation status: `not_confirmed`;
- Qwen3-4B cross-model status: `not_confirmed`;
- title counterfactual technical gate: `true`;
- causal trace technical gate: `true`;
- causal trace final endpoint error: `0`;
- publication validator status: `ready`, with no missing local links or stale
  artifact hashes;
- local test suite: all available tests pass, with the NumPy-only relation-
  binding test skipped when NumPy is absent.

## 8. Archived Job IDs

The exact July 2026 jobs are recorded in `reports/hf_jobs_run_log.md`,
`runs/hf_job_metadata.json`, and `runs/publication_manifest.json`.

For each publication-stage Job, the repository preserves:

- inspect metadata: image, hardware, timestamps, duration, status, and command
  hash;
- the exact submitted script bytes as `runs/job_scripts/<JOB_ID>.py.b64`;
- a readable LF-normalized copy as `runs/job_scripts/<JOB_ID>.py`;
- the terminal snapshot returned by the Jobs logs API under
  `runs/job_logs/<JOB_ID>.log`;
- hashes and byte counts for the exact script and terminal snapshot.

The terminal snapshots are explicitly the last 20 lines returned by the API,
not full lifecycle logs. The structured JSON/JSONL result files remain the
complete result artifacts. `python scripts/validate_publication.py` decodes each
exact script payload and verifies all archived hashes.

Reproduction jobs will receive new IDs and may experience different scheduling
times, but the model revisions, prompts, bundle contents, selection hashes, and
interpretation gates should remain unchanged.

## 9. Integrated Exploratory Evidence

Section 2 of the report summarizes earlier exploratory work. These results are
not part of the independent holdout confirmation, but their complete bundles,
rows, summaries, scripts, and reports are included in the publication manifest.

| Evidence | Bundle and runner | Archived result |
| --- | --- | --- |
| Output order | `runs/qwen_scope_music_reason_order_bundle.json`, `scripts/run_reason_order_probe.py` | `runs/qwen_scope_music_reason_order_rows.jsonl`, `runs/qwen_scope_music_reason_order_summary.json` |
| Reason swap | `runs/qwen_scope_music_reason_swap_bundle.json`, `scripts/run_reason_swap_probe.py` | `runs/qwen_scope_music_reason_swap_pilot_rows.jsonl`, `runs/qwen_scope_music_reason_swap_pilot_summary.json` |
| Controlled vocality behavior | `runs/controlled_vocality_reason_probe_bundle.json`, `scripts/run_controlled_vocality_reason_probe.py` | pair/choice rows and `runs/controlled_vocality_reason_summary.json` |
| Component patching | `runs/controlled_vocality_path_patching_bundle.json`, `scripts/run_controlled_vocality_path_patching_probe.py` | pair/choice rows and `runs/controlled_vocality_path_patching_summary.json` |
| Attention heads | `runs/controlled_vocality_attention_head_bundle.json`, `scripts/run_controlled_vocality_attention_head_probe.py` | pair/choice rows and `runs/controlled_vocality_attention_head_summary.json` |
| Title/artist fields | `runs/controlled_vocality_field_probe_bundle.json`, `scripts/run_controlled_vocality_field_probe.py` | `runs/controlled_vocality_field_probe_rows.jsonl`, `runs/controlled_vocality_field_probe_summary.json` |

The manifest records these as integrated exploratory evidence and preserves the
positive and negative sides separately. Do not pool them with the 18-event
holdout metrics or reinterpret their thresholds as confirmatory gates.
