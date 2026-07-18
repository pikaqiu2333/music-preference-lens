# Hugging Face Jobs Run Log

> **Public distribution note:** references below to archived raw catalog
> responses describe the original private evidence workflow. Complete
> third-party response bodies are omitted from the public repository; their
> frozen hashes are recorded in `runs/private_evidence_receipt.json`.

## 2026-07-08 Identity SAE Smoke Run

- Job ID: `6a4e4b4e1499512f23779bb6`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4e4b4e1499512f23779bb6
- Account: `REDACTED`
- Output dataset repo: https://huggingface.co/datasets/REDACTED/music-preference-lens-runs
- Workload: 3 identity/role-binding routine-vs-complex probes.
- Model: `google/gemma-3-270m-it`
- SAE: `gemma-scope-2-270m-it-resid_post`, `layer_12_width_16k_l0_small`
- Hardware: `t4-small`
- Timeout: `2h`
- Final status: `ERROR`
- Failure reason: the available HF token could run jobs but could not create a
  dataset repo under `REDACTED`, so result persistence failed with `403
  Forbidden`.

Expected outputs if the job completes:

- `runs/<run_id>/identity_sae_observations.csv`
- `runs/<run_id>/manifest.json`

If the job fails, it should upload:

- `runs/<run_id>/error.txt`

The error upload also depends on dataset write permission, so this first run did
not persist artifacts to the Hub.

## 2026-07-08 Identity SAE Logs Fallback

- Job ID: `6a4e4bc91499512f23779bb8`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4e4bc91499512f23779bb8
- Account: `REDACTED`
- Workload: same 3 identity/role-binding probes, compacted to top 5 features
  per probe.
- Model: `google/gemma-3-270m-it`
- SAE: `gemma-scope-2-270m-it-resid_post`, `layer_12_width_16k_l0_small`
- Hardware: `t4-small`
- Timeout: `2h`
- Persistence mode: compact `RESULT_JSON` printed to job logs.
- Status at submission: `RUNNING`

Later status: `ERROR`

- Failure reason: `google/gemma-3-270m-it` is a gated model and the current
  HF account/token was not authorized for that repo.

## 2026-07-08 Pythia Public SAE Logs Fallback

- Job ID: `6a4e4c401499512f23779bbc`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4e4c401499512f23779bbc
- Workload: 3 identity/role-binding probes, compact top 5 features.
- Model: `EleutherAI/pythia-70m-deduped`
- SAE: `pythia-70m-deduped-res-sm`, `blocks.5.hook_resid_post`
- Hardware: `t4-small`
- Persistence mode: compact `RESULT_JSON` printed to job logs.
- Final status: `COMPLETED`

## 2026-07-08 Qwen-Scope SAE Logs Fallback

- Job ID: `6a4e4cc51499512f23779bc0`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4e4cc51499512f23779bc0
- Workload: 3 identity/role-binding probes, compact top 5 features.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`, `layer5.sae.pt`
- Hardware: `t4-small`
- Persistence mode: compact `RESULT_JSON` printed to job logs.
- Final status: `COMPLETED`
- Run ID: `20260708T131259Z`
- Runtime: 67 seconds total, 62 seconds running.

## 2026-07-08 Qwen-Scope Identity-Control Layer Sweep

- Job ID: `6a4e4f0e1499512f23779bc9`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4e4f0e1499512f23779bc9
- Workload: 3 identity probes plus 3 non-identity controls.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layers: 5, 10, 18, 24
- Hardware: `t4-small`
- Persistence mode: `RESULT_ROW` and compact `RESULT_JSON` printed to job logs.
- Final status: `COMPLETED`
- Run ID: `20260708T132245Z`
- Runtime: 79 seconds total, 74 seconds running.

## 2026-07-08 Qwen-Scope Matched Identity-Control Sweep

- Job ID: `6a4e4feb1fba25b8ea3b200e`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4e4feb1fba25b8ea3b200e
- Workload: 3 matched identity-vs-control probes with similar structure and
  JSON schemas.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layers: 5, 10, 18, 24
- Hardware: `t4-small`
- Persistence mode: `MATCHED_ROW` and compact `MATCHED_RESULT_JSON` printed to
  job logs.
- Final status: `COMPLETED`
- Run ID: `20260708T132626Z`
- Runtime: 80 seconds total, 75 seconds running.

## 2026-07-08 Qwen-Scope Candidate Feature Activation Dataset

- Job ID: `6a4e516c1fba25b8ea3b2026`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4e516c1fba25b8ea3b2026
- Workload: score candidate layer 18 and layer 24 features across 35 prompt
  families.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Target features:
  - Layer 18: `16214`, `24817`, `18054`, `5837`, `29473`, `13004`
  - Layer 24: `19603`, `30224`, `24979`, `30584`, `27029`, `27497`,
    `19230`, `22232`, `18905`, `24916`
- Hardware: `t4-small`
- Persistence mode: `FEATURE_TOP` and compact `FEATURE_SUMMARY_JSON` printed
  to job logs.
- Final status: `COMPLETED`
- Run ID: `20260708T133250Z`
- Runtime: 71 seconds total, 65 seconds running.

## 2026-07-08 Qwen-Scope Music Recommendation Probe

- Job ID: `6a4e537b1fba25b8ea3b2043`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4e537b1fba25b8ea3b2043
- Workload: 6 music recommendation counterfactual probes from
  `data/mechanistic_pilot_specs.jsonl`.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layers: 18, 24
- Hardware: `t4-small`
- Persistence mode: `MUSIC_DELTA_ROW` and compact `MUSIC_SUMMARY_JSON`
  printed to job logs.
- Final status: `COMPLETED`
- Runtime: 71 seconds total, 66 seconds running.

## 2026-07-08 Qwen-Scope Matched Music Recommendation Probe

- Job ID: `6a4e54bb1499512f23779bf2`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4e54bb1499512f23779bf2
- Workload: 6 length/content-matched music recommendation counterfactual
  probes from `data/qwen_scope_music_matched_specs.jsonl`.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Hardware: `t4-small`
- Persistence mode: `MATCHED_MUSIC_ROW` and compact
  `MATCHED_MUSIC_SUMMARY_JSON` printed to job logs.
- Final status: `COMPLETED`
- Runtime: 189 seconds total, 68 seconds running.

## 2026-07-09 Qwen-Scope Music Span-Level Probe

- Job ID: `6a4ee2a01499512f23779ff4`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4ee2a01499512f23779ff4
- Workload: span-level activation analysis for matched music recommendation
  prompts, splitting prompts into profile, current_need, candidate_cards, and
  task spans.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Target features: `7884`, `30584`, `32078`, `8706`, `24916`, `24979`,
  `27497`, `27029`, `19603`, `30224`, `22232`
- Hardware: `t4-small`
- Persistence mode: `SPAN_DIFF` and compact `SPAN_SUMMARY` printed to job
  logs.
- Final status: `COMPLETED`
- Runtime: 189 seconds total, 66 seconds running.

## 2026-07-09 Qwen-Scope Music Phrase-Level Probe

- Job ID: `6a4f18571fba25b8ea3b2d48`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4f18571fba25b8ea3b2d48
- Workload: phrase-level activation analysis over 12 matched music
  recommendation probes from `data/qwen_scope_music_phrase_specs.jsonl`.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Target features: `7884`, `30584`, `32078`, `8706`, `24916`, `24979`,
  `27029`, `19603`, `30224`, `22232`
- Hardware: `t4-small`
- Persistence mode: `PHRASE_ROW_JSON` and compact `PHRASE_SUMMARY_JSON`
  printed to job logs.
- Final status: `COMPLETED`
- Runtime: 197 seconds total, 67 seconds running.

## 2026-07-09 Qwen-Scope Music Generated-Reason Probe

- Job ID: `6a4f1db41fba25b8ea3b2dae`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4f1db41fba25b8ea3b2dae
- Workload: generated-answer interpretability run over 8 original /
  counterfactual prompt rows from
  `runs/qwen_scope_music_generation_prompt_pack.jsonl`, measuring prompt and
  answer-token activations for candidate music-recommendation SAE features.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Target features: `7884`, `30584`, `32078`, `8706`, `24916`, `24979`,
  `27029`, `19603`, `30224`, `22232`
- Hardware: `t4-small`
- Persistence mode: `GEN_ROW_JSON` and compact `GEN_SUMMARY_JSON` printed to
  job logs.
- Final status: `COMPLETED`
- Runtime: 263 seconds total, 129 seconds running.
- Result note: the base model repeated prompt rule text instead of producing
  valid recommendation JSON, so answer-token feature alignment is not a valid
  recommendation-reason result for this run.

## 2026-07-09 Qwen-Scope Music Base-Completion Reason Probe

- Job ID: `6a4f273a1fba25b8ea3b2e17`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4f273a1fba25b8ea3b2e17
- Workload: base-model-friendly reason continuation run over 8 original /
  counterfactual prompt rows from
  `runs/qwen_scope_music_base_completion_prompt_pack.jsonl`, conditioning on
  preselected best tracks and measuring only the continuation after `Reason:`.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Target features: `7884`, `30584`, `32078`, `8706`, `24916`, `24979`,
  `27029`, `19603`, `30224`, `22232`
- Hardware: `t4-small`
- Persistence mode: `BASE_ROW_JSON` and compact `BASE_SUMMARY_JSON` printed to
  job logs.
- Final status: `COMPLETED`
- Runtime: 93 seconds total, 88 seconds running.
- Result note: base-completion prompt produced meaningful recommendation
  reasons, and all tracked answer-token feature deltas aligned with expected
  directions.

## 2026-07-09 Qwen-Scope Music Free Playlist Generation Probe

- Job ID: `6a4f4e881fba25b8ea3b308f`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4f4e881fba25b8ea3b308f
- Workload: free playlist generation run over 8 original / counterfactual
  prompt rows from `runs/qwen_scope_music_free_playlist_prompt_pack.jsonl`,
  with no candidate tracks provided.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Target features: `7884`, `30584`, `32078`, `8706`, `24916`, `24979`,
  `27029`, `19603`, `30224`, `22232`
- Hardware: `t4-small`
- Persistence mode: `FREE_ROW_JSON` and compact `FREE_SUMMARY_JSON` printed to
  job logs.
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: 120 seconds total, 115 seconds running.
- Result note: free playlist generation produced meaningful playlist shifts
  without candidate tracks or a preselected best track. Answer-token feature
  deltas aligned with 11 of 12 hypotheses; the only miss was the narrower
  `feature24916` instrumental/no-vocals signal, while the broader no-vocals /
  focus feature still aligned.

## 2026-07-09 Qwen-Scope Song Entity Grounding Probe

- Job ID: `6a4f59d71fba25b8ea3b3156`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4f59d71fba25b8ea3b3156
- Workload: song-title entity grounding probe over 20 catalog-style artist
  completion prompts from `runs/qwen_scope_song_entity_grounding_prompt_pack.jsonl`.
- Groups: `known_real`, `invented_control`, and `free_generated`.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Hardware: `t4-small`
- Persistence mode: `GROUND_ROW_JSON` and compact `GROUND_SUMMARY_JSON`
  printed to job logs.
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: 268 seconds total, 131 seconds running.
- Result note: known-real artist completion was weak at 3/6 hits, invented
  controls were clean at 6/6 prompt-proximity checks, and free-generated titles
  were mixed at prompt level but 7/8 answer spans were closer to invented
  controls. This supports treating most free-generated titles as style-shaped
  placeholders under this base-model probe, with the caveat that open artist
  generation drifted into album metadata.

## 2026-07-09 Qwen-Scope Song Entity Forced-Choice Grounding Probe

- Job ID: `6a4f60381499512f2377a20e`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4f60381499512f2377a20e
- Workload: forced-choice candidate artist logprob scoring over 20
  catalog-style title prompts from
  `runs/qwen_scope_song_entity_forced_choice_prompt_pack.jsonl`.
- Groups: `known_real`, `invented_control`, and `free_generated`.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Hardware: `t4-small`
- Persistence mode: `FORCED_ROW_JSON` and compact `FORCED_SUMMARY_JSON`
  printed to job logs.
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: 69 seconds total, 64 seconds running.
- Result note: prompt-final SAE proximity was clean: known-real controls were
  6/6 closer to known-real, invented controls were 6/6 closer to invented, and
  free-generated titles were 8/8 closer to invented. Candidate mean-logprob
  scoring was not reliable for `Unknown`: invented/free titles chose 0/14
  `Unknown`, apparently due length and copy biases, so the next fix should use
  option-letter scoring with randomized answer order.

## 2026-07-09 Qwen-Scope Song Entity Option-Letter Grounding Probe

- Job ID: `6a4f64bb1fba25b8ea3b320b`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4f64bb1fba25b8ea3b320b
- Workload: option-letter candidate scoring over 60 randomized prompt variants
  from `runs/qwen_scope_song_entity_option_letter_prompt_pack.jsonl`, plus
  title-only SAE proximity for 20 song titles.
- Groups: `known_real`, `invented_control`, and `free_generated`.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Hardware: `t4-small`
- Persistence mode: compact `LETTER_SUMMARY_JSON` printed to job logs.
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: 70 seconds total, 65 seconds running.
- Result note: option-letter scoring fixed the previous full-string length/copy
  bias for `Unknown`: invented + free-generated titles chose `Unknown` for
  12/14 item-average decisions, compared with 0/14 in the prior full-string
  forced-choice run. Title-only SAE proximity stayed clean: known-real controls
  were 6/6 closer to known-real, invented controls were 6/6 closer to invented,
  and free-generated titles were 8/8 closer to invented.

## 2026-07-10 Qwen-Scope Song Entity Generation-Time Smoke Probe

- Job ID: `6a506293a9bcc59cfbc4b7e7`
- Job URL: https://huggingface.co/jobs/REDACTED/6a506293a9bcc59cfbc4b7e7
- Workload: one-context generation-time smoke run with one free-playlist seed,
  18 exact/mismatched/synthetic title-artist controls, layer-24 SAE knownness,
  and neutral versus self-attributed calibrated pair verification.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Hardware: `t4-small`
- Persistence: private dataset `REDACTED/music-preference-lens-runs` plus compact
  `GEN_TIME_SUMMARY_JSON` in job logs.
- Status at submission: `RUNNING`
- Final status: `ERROR`
- Runtime: 68 seconds total, 63 seconds running.
- Failure note: the unconstrained `torch>=2.5` dependency resolved to a CUDA
  13/Triton stack that failed while compiling its driver helper on T4. No model
  probe ran. The retry pins the previously compatible CUDA 12 generation of
  PyTorch and the validated Transformers version.

## 2026-07-10 Qwen-Scope Song Entity Generation-Time Smoke Probe Retry

- Job ID: `6a506402a9bcc59cfbc4b7ed`
- Job URL: https://huggingface.co/jobs/REDACTED/6a506402a9bcc59cfbc4b7ed
- Workload: corrected retry of the generation-time smoke probe.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Hardware: `t4-small`
- Dependency fix: `torch==2.7.1`, `transformers==4.53.2`, CUDA 12 wheels.
- Persistence: private dataset `REDACTED/music-preference-lens-runs` plus compact
  `GEN_TIME_SUMMARY_JSON` in job logs.
- Status at submission: `RUNNING`
- Final status: `ERROR`
- Runtime: about 73 seconds running.
- Failure note: model generation, SAE extraction, control probing, and calibrated
  verification completed, but the final dataset creation returned `403` because
  the current token cannot create a dataset under `REDACTED`. The next retry keeps
  Hub upload as best effort and persists a recoverable chunked payload in logs.

## 2026-07-10 Qwen-Scope Song Entity Generation-Time Smoke Probe Log Fallback

- Job ID: `6a5065c6a9bcc59cfbc4b7f2`
- Job URL: https://huggingface.co/jobs/REDACTED/6a5065c6a9bcc59cfbc4b7f2
- Workload: generation-time smoke probe with best-effort Hub upload and
  recoverable `GEN_TIME_RESULT_CHUNK` log persistence.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Hardware: `t4-small`
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: 75 seconds total, 70 seconds running; script body about 33 seconds.
- Result note: dependency, model, SAE, control, verification, and chunked-log
  persistence paths all completed. The model generated five title-artist-reason
  items on one-line records, so the original line-oriented parser recovered
  zero rows and the technical gate failed. The parser was extended to support
  inline fields before the final smoke run.

## 2026-07-10 Qwen-Scope Song Entity Generation-Time Final Smoke Probe

- Job ID: `6a50679f84e0eddc25f12785`
- Job URL: https://huggingface.co/jobs/REDACTED/6a50679f84e0eddc25f12785
- Workload: final one-context smoke run with inline and multiline playlist
  parsing, calibrated verification, and chunked-log persistence.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Hardware: `t4-small`
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: about 75 seconds total; script body about 33 seconds.
- Result note: inline parsing recovered 5/5 generated pairs and the technical
  gate passed. Control balanced accuracy was only `0.50` on the reduced smoke
  controls, so no mechanism interpretation was made. The 20KB fallback chunks
  exceeded the log service's per-line retrieval limit; persistence was changed
  to one compact JSON line per playlist generation.

## 2026-07-10 Qwen-Scope Generation-Time Compact-Log Smoke Probe

- Job ID: `6a50691da9bcc59cfbc4b808`
- Job URL: https://huggingface.co/jobs/REDACTED/6a50691da9bcc59cfbc4b808
- Workload: persistence-only confirmation of one compact
  `GEN_TIME_GENERATION_JSON` line per generated playlist.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Hardware: `t4-small`
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: about 75 seconds total; script body about 33 seconds.
- Result note: one compact generation line was fully recoverable with five
  parsed title-artist-reason rows, calibrated neutral/self-attributed logits,
  generation knownness, and title/artist/pair-end/reason top features. The
  technical gate passed, so the full run can proceed.

## 2026-07-10 Qwen-Scope Song Entity Generation-Time Full Probe

- Job ID: `6a5069a2a9bcc59cfbc4b80a`
- Job URL: https://huggingface.co/jobs/REDACTED/6a5069a2a9bcc59cfbc4b80a
- Workload: 12 free-playlist generations across four contexts and three seeds,
  160 context-matched control examples, layer-24 SAE knownness, and neutral
  versus self-attributed calibrated exact-pair verification.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Hardware: `t4-small`
- Persistence: 12 compact `GEN_TIME_GENERATION_JSON` lines plus
  `GEN_TIME_SUMMARY_JSON` in logs; Hub upload is best effort.
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Script runtime: about 140 seconds; 12 compact generation rows and the summary
  were fully recoverable from logs.
- Result note: 7/12 playlist generations yielded 35 parsed entities. The
  technical gate failed because at least 10 valid generations and 50 entities
  were required. Control balanced accuracy was `0.725`, below the predeclared
  `0.80` interpretation gate, although all four context directions agreed.
  These rows are preserved as a partial artifact and are not treated as a
  mechanism conclusion.

## 2026-07-10 Song Entity Generation Format Diagnostic

- Job ID: `6a506af0a9bcc59cfbc4b812`
- Job URL: https://huggingface.co/jobs/REDACTED/6a506af0a9bcc59cfbc4b812
- Workload: reproduce the same 12 seeded free-playlist generations without SAE
  or controls and persist raw completions for parser diagnosis.
- Model: `Qwen/Qwen3-1.7B-Base`
- Hardware: `t4-small`
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Result note: all 12 deterministic raw completions were recovered. Two
  additional formats were identified: `Title by Artist`/`Title - Artist` with
  a separate reason, and title-artist lists followed by numbered
  recommendations. Local replay after parser repair recovers 55 entities from
  11/12 generations; the remaining generation omitted reasons entirely.

## 2026-07-10 Qwen-Scope Generation-Time Full Probe Parser Retry

- Job ID: `6a506c5284e0eddc25f127bd`
- Job URL: https://huggingface.co/jobs/REDACTED/6a506c5284e0eddc25f127bd
- Workload: complete 4-context, 3-seed generation-time probe after parser
  repair, with Hub upload disabled and compact log persistence enabled.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Hardware: `t4-small`
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Script runtime: about 144 seconds; 12 compact playlist records were fully
  recovered from logs.
- Result note: 11/12 generations and 55 title-artist-reason rows were valid,
  satisfying the registered technical coverage gate of at least 10 generations
  and 50 rows. The remote summary incorrectly required all 12 generations and
  reported `technical_gate=false`; the local summary preserves that remote
  value and records the corrected gate. Control balanced accuracy remained
  `0.725`, below the `0.80` interpretation gate, so generation knownness is
  reported descriptively rather than as a validated classifier.

## 2026-07-10 Exact-Pair Verification Control Calibration

- Job ID: `6a50a5076d2b10c09d6779eb`
- Job URL: https://huggingface.co/jobs/REDACTED/6a50a5076d2b10c09d6779eb
- Workload: neutral and self-attributed calibrated A/B verification over 20
  known-exact, 10 artist-mismatch, and 10 synthetic title-artist controls.
- Model: `Qwen/Qwen3-1.7B-Base`
- Hardware: `t4-small`
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Result note: neutral and self-attributed balanced accuracy were both `0.50`.
  None of the 20 mismatch/synthetic controls selected Unknown. Mean neutral
  Unknown logits were `-1.888` for known exact, `-1.531` for artist mismatch,
  and `-0.648` for synthetic pairs. The prompt carries graded uncertainty but
  is not a calibrated exact-pair classifier. Self-attribution raised Unknown
  logits by `+0.632` on controls, nearly identical to `+0.622` on generated
  pairs, so no generation-specific self-awareness effect was found.

## 2026-07-10 Matched Song Entity Relation-Binding Smoke

- Job ID: `6a50aacf2055b7ba2bc12cb4`
- Job URL: https://huggingface.co/jobs/REDACTED/6a50aacf2055b7ba2bc12cb4
- Workload: 15 matched title-artist relations in closed artist permutations;
  direct prior-corrected artist likelihood, order-flipped A/B relation score,
  and layer-24 SAE paired deltas at artist-end and pair-end.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Hardware: `t4-small`
- Registered primary gates: direct PMI-style accuracy at least `0.80` and
  pair-end SAE paired cross-validation accuracy at least `0.80`. Smoke is
  preliminary and cannot authorize intervention without a full 20-pair run.
- Catalog precheck: 20/20 exact controls verified, 20/20 mismatches confirmed
  as title-present artist conflicts, with no unverified or error rows.
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Script runtime: about 27 seconds; Hugging Face reported about 75 seconds of
  total GPU job runtime including dependency and model setup.
- Result note: the technical and behavioral gates passed. Direct PMI-style and
  order-flipped choice accuracy were both `0.867` (13/15). Artist-end SAE paired
  CV was `0.000`; pair-end SAE paired CV was `0.133`, below the `0.80` mechanism
  gate. Post-run audit found that relation-level folds split artist-permutation
  cycles, placing held-out artists in training under the opposite relation
  label and producing systematic sign inversion. The result is preserved, but
  the SAE scores are not interpreted. The next smoke uses independent two-way
  swap blocks and holds out complete blocks.

## 2026-07-10 Block-Held-Out Relation-Binding Smoke Retry

- Job ID: `6a50af642055b7ba2bc12ceb`
- Job URL: https://huggingface.co/jobs/REDACTED/6a50af642055b7ba2bc12ceb
- Workload: 12 relations from six independent two-title artist-swap blocks;
  three-fold CV holds out one complete English block and one complete Chinese
  block per fold.
- Model: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Hardware: `t4-small`
- Catalog delta precheck: all 10 newly introduced mismatch pairs were confirmed
  as title-present artist conflicts. Exact pairs and unchanged mismatches retain
  their previously verified labels.
- Registered gates remain unchanged: direct PMI-style accuracy and pair-end SAE
  paired CV accuracy must each reach `0.80`.
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: about 73 seconds total, including about 25 seconds inside the probe
  after model and SAE loading.
- Result note: the technical gate passed. Direct PMI-style accuracy was `0.833`
  (10/12), so the behavioral gate passed; order-flipped choice accuracy was
  `0.750`. Artist-end SAE paired CV recovered from the earlier inverted result
  to `0.500`, and pair-end SAE paired CV recovered to `0.583` (7/12), confirming
  that block-held-out splitting removed the systematic leakage. The pair-end
  score remained below the registered `0.80` mechanism gate. No full run or
  intervention was submitted. The next method should use layerwise logit-lens
  attribution or activation patching without supervised SAE feature selection.

## 2026-07-10 Classifier-Free Layerwise Attribution Smoke

- Job ID: `6a50b5906d2b10c09d677a6e`
- Job URL: https://huggingface.co/jobs/REDACTED/6a50b5906d2b10c09d677a6e
- Workload: 12 matched relations; observed-token target-logit lens over every
  hidden-state depth and batched neutral-title residual patching over every
  transformer layer.
- Model: `Qwen/Qwen3-1.7B-Base`
- Hardware: `t4-small`
- Trained probe or SAE feature selection: none.
- Registered checks: final sequence-PMI accuracy at least `0.80`, final target
  logit consistency within `0.02`, and final-layer patch endpoint consistency
  within `0.02`.
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: about 68 seconds total, including about 22 seconds inside the probe
  after model loading.
- Result note: all technical and behavioral gates passed. Final sequence-PMI
  accuracy was `0.833` (10/12), final target-logit consistency error was
  `0.0078`, and final-layer patch endpoint error was `0.0000`. The earliest
  three-depth sustained target-logit accuracy of at least `0.75` began at layer
  24. Neutral-state patching had little effect through the early layers, then
  remaining title-specific first-token signal fell from `0.897` at layer 14 to
  `0.504` at layer 17, `0.310` at layer 24, `0.010` at layer 27, and `0.000` at
  layer 28. The next mechanistic step is attention-versus-MLP component
  patching at selected late layers.

## 2026-07-11 Attention-vs-MLP Component Patching Smoke

- Job ID: `6a519130e4a4e82c0b58cb49`
- Job URL: https://huggingface.co/jobs/REDACTED/6a519130e4a4e82c0b58cb49
- Workload: patch neutral-title self-attention, MLP, and full decoder residual
  outputs into real-title prefixes at layers 14, 16, 18, 21, 24, 27, and 28.
- Model: `Qwen/Qwen3-1.7B-Base`
- Hardware: `t4-small`
- Trained probe or SAE feature selection: none.
- Registered checks: at least eight valid first-token title effects and
  layer-28 full-residual endpoint error within `0.02`.
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: about 69 seconds of GPU job runtime, including about 22 seconds
  inside the probe; scheduling took about 125 seconds.
- Result note: all registered checks passed. Ten of 12 relations had a valid
  title effect, and the layer-28 full-residual endpoint error was `0.0000`.
  Attention recovery peaked at layer 21 (`0.617` mean), while MLP recovery
  peaked at layer 24 (`0.396` mean). Full-residual recovery rose from `0.103`
  at layer 14 to `0.726` at layer 24 and `0.991` at layer 27. The validated
  method should next be applied to a small, source-agreed set of freely
  generated catalog conflicts, replayed at their original artist-generation
  positions; this result does not itself establish fabrication awareness.

## 2026-07-11 Music Recommendation Reason Faithfulness Smoke

- Initial Job ID: `6a51a244e4a4e82c0b58cc5c`
- Initial Job URL: https://huggingface.co/jobs/REDACTED/6a51a244e4a4e82c0b58cc5c
- Initial attempt: `FAILED` before model loading because terminal wrapping
  inserted newlines into the embedded Base64 bundle.
- Retry Job ID: `6a51a2e0e4a4e82c0b58cc62`
- Retry Job URL: https://huggingface.co/jobs/REDACTED/6a51a2e0e4a4e82c0b58cc62
- Retry fix: export a compact 12-record smoke-only bundle and strip all
  whitespace from its Base64 payload before submission. Experiment logic and
  registered gates are unchanged.
- Retry status: `CANCELED` after detecting that bundle injection had also
  replaced the runner's sentinel comparison. No experiment rows were scored.
- Final retry Job ID: `6a51a364e4a4e82c0b58cc66`
- Final retry Job URL: https://huggingface.co/jobs/REDACTED/6a51a364e4a4e82c0b58cc66
- Final retry fix: replace only the first bundle placeholder and leave the
  sentinel literal intact.
- Workload: teacher-forced complete title-artist sequence likelihood under
  original, paraphrased, opposite, and neutral listening needs.
- Records: 12 saved recommendation events, balanced between six
  `verified_exact` and six `catalog_conflict` pairs.
- Model: `Qwen/Qwen3-1.7B-Base`
- Hardware: `t4-small`
- Trained probe, SAE selection, or LLM judge: none.
- Registered follow-up gate: at least 8/12 opposite needs score below both
  semantically equivalent needs, with a positive median opposite margin.
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: 74 seconds total, including 69 seconds running and about 19 seconds
  inside the probe after model loading.
- Result note: both registered gates passed. Opposite needs scored below both
  semantically equivalent needs for 10/12 pairs (`0.833`), with median
  opposite margin `0.290`. Rates were `5/6` for both `verified_exact` and
  `catalog_conflict`. This warrants a small mechanistic follow-up, while the
  `Space Oddity - David Bowie` failure shows that entity reality and constraint
  faithfulness remain separate questions.

## 2026-07-11 Music Reason Component-Patching Pilot

- Job ID: `6a51a578e4a4e82c0b58cca2`
- Job URL: https://huggingface.co/jobs/REDACTED/6a51a578e4a4e82c0b58cca2
- Workload: patch opposite-need attention, MLP, and full-residual outputs into
  aligned prediction positions for the complete title-artist sequence.
- Cases: one verified need-sensitive pair, one catalog-conflict need-sensitive
  pair, and one verified constraint failure.
- Model: `Qwen/Qwen3-1.7B-Base`
- Hardware: `t4-small`
- Selected layers: 14, 16, 18, 21, 24, 27, and endpoint layer 28.
- Trained probe, SAE selection, or LLM judge: none.
- Registered checks: all three need effects at least `0.05` in magnitude,
  replay baseline error within `0.02`, and layer-28 full-residual endpoint
  error within `0.02`.
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: 74 seconds total, including 69 seconds running and about 21 seconds
  inside the probe after model loading.
- Result note: all three cases and both endpoint checks passed. Full-residual
  mean recovery rose from `0.184` at layer 14 to `0.651` at layer 16 and
  `1.129` at layer 18, remaining near `1.0` thereafter. The two need-sensitive
  successes showed exploratory attention effects around layer 18 followed by
  stronger MLP effects around layers 21-24. `Space Oddity - David Bowie` was
  more probable under active lyric listening than strict no-vocals, and the
  patch causally transferred that effect; this supports latent constraint
  sensitivity failing to control the generated recommendation, not subjective
  self-awareness.

## 2026-07-11 Music Recommendation Reason-Order Smoke

- Job ID: `6a51ae56effc02a91cbd9235`
- Job URL: https://huggingface.co/jobs/REDACTED/6a51ae56effc02a91cbd9235
- Workload: matched free generation under title-artist-reason (`pair_first`)
  and reason-title-artist (`reason_first`) output order.
- Contexts: emotional vocals and strict no-vocals.
- Seeds: 17 and 29; eight total generations, up to 40 pairs.
- Model: `Qwen/Qwen3-1.7B-Base`
- Hardware: `t4-small`
- Registered checks: eight valid generations, at least 32 complete rows, at
  least `0.80` actual field-order compliance, and no empty fields.
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: 346 seconds total, including 212 seconds scheduling and 133 seconds
  running; generation itself took about 93 seconds after model loading.
- Original parser result: `FAILED` the technical gate with 15 rows because the
  parser did not recognize multiple labeled fields on one line.
- Parser-only correction: archived all eight raw generations, added inline
  field and `Title by Artist` parsing, and recovered 40/40 complete rows with
  `1.00` actual order compliance. No model output was regenerated or rewritten.
- Catalog result: pair-first produced 1/20 verified exact pairs; reason-first
  produced 4/20. All verified gains occurred in the emotional-vocal context.
  Both orders produced 0/10 verified pairs for strict no-vocals, where
  reason-first additionally emitted five `[Artist Name]` placeholders.
- Interpretation: reason-first causally changes entity search but is not a
  uniform hallucination fix; it may scaffold familiar entities while inducing
  concept-shaped fabricated names under strong or long-tail constraints.

## 2026-07-11 Visible Reason-Swap Causality Pilot

- Job ID: `6a51e5f4e4a4e82c0b58d3f5`
- Job URL: https://huggingface.co/jobs/REDACTED/6a51e5f4e4a4e82c0b58d3f5
- Workload: teacher-force 20 fixed reason-first title-artist pairs after their
  own reason, a different same-context reason, the matched opposite-context
  reason, and a neutral reason.
- Conditions: 80 reason-pair replays.
- Entity groups: verified exact, catalog conflict, unverified, and explicit
  invalid placeholder.
- Model: `Qwen/Qwen3-1.7B-Base`
- Hardware: `t4-small`
- Trained probe, SAE selection, or LLM judge: none.
- Registered checks: 20 records, 80 conditions, all four labels, nonempty title
  and artist spans, different own/same-context reason text, no direct title
  leakage in own reasons, and finite metrics.
- Status at submission: `RUNNING`
- Initial status: `CANCELED` before allocation after an unusually long T4
  scheduling wait; no scoring ran and no experiment result was produced.
- L4 retry Job ID: `6a51e901e4a4e82c0b58d44b`
- L4 retry Job URL: https://huggingface.co/jobs/REDACTED/6a51e901e4a4e82c0b58d44b
- L4 retry hardware: `l4x1`; experiment bundle and gates unchanged.
- Retry status at submission: `RUNNING`
- L4 retry status: `CANCELED` before allocation after another extended
  scheduling wait; no experiment rows were scored.
- A10G retry Job ID: `6a51ea8aeffc02a91cbd96a3`
- A10G retry Job URL: https://huggingface.co/jobs/REDACTED/6a51ea8aeffc02a91cbd96a3
- A10G retry hardware: `a10g-small`; experiment bundle and gates unchanged.
- A10G retry status at submission: `RUNNING`
- First A10G retry status: `ERROR` before model loading because the repeated
  prompt bundle exceeded inline transport size and was truncated with a
  non-ASCII ellipsis.
- Packaging fix: store each reason once, reference counterfactual reason source
  record IDs, keep contexts once, and rebuild prompts in the runner. The
  minified bundle fell from roughly 19.4k to 15.7k Base64 characters;
  intervention logic and gates are unchanged.
- Final retry Job ID: `6a51eb4ee4a4e82c0b58d49f`
- Final retry Job URL: https://huggingface.co/jobs/REDACTED/6a51eb4ee4a4e82c0b58d49f
- Final retry status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: 68 seconds total, including 4 seconds scheduling and 63 seconds
  running; the probe took about 17 seconds after model loading.
- Result note: all technical gates passed for 20 records and 80 conditions.
  Own reasons beat opposite-context reasons for 18/20 pairs (`0.90`) and
  different same-context reasons for 14/20 (`0.70`), but beat a neutral reason
  for only 9/20 (`0.45`; median margin `-0.015`). For four verified-exact
  pairs, own reasons beat same-context and neutral controls only 2/4 times,
  with near-zero mean margins. Catalog conflicts showed stronger coupling to
  their own generated reasons. The result supports broad semantic steering and
  error co-construction, not stable pair-specific explanation faithfulness.
- Decision: stop before deeper patching of these free reasons. Next use
  controlled, length-matched attribute reasons and independently verified song
  attributes, then intervene only on cases that pass a behavioral gate.

## 2026-07-11 Controlled Vocality-Reason Causal Pilot

- Job ID: `6a51f201effc02a91cbd96d2`
- Job URL: https://huggingface.co/jobs/REDACTED/6a51f201effc02a91cbd96d2
- Workload: score eight real title-artist pairs after equal-length vocal,
  instrumental, and neutral reasons; also compare vocal versus instrumental
  candidate letter mass in two reversed eight-track candidate orders.
- Tracks: four independently evidenced instrumental recordings and four vocal
  recordings, all exact in MusicBrainz and Apple, with eight distinct artists.
- Entity unit: complete title plus artist; no Song ID.
- Model: `Qwen/Qwen3-1.7B-Base`
- Hardware: `a10g-small`
- Registered follow-up gate: matched reason wins at least 6/8 pair replays,
  both vocality groups have positive median margins, and both controlled reason
  directions shift fixed-candidate mass away from the neutral baseline.
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: 263 seconds total, including 194 seconds scheduling and 68 seconds
  running; the probe took about 18 seconds after model loading.
- Result note: all technical and behavioral gates passed. Complete-pair replay
  followed the independently verified attribute for 6/8 tracks (`0.75`), with
  positive medians for both vocal and instrumental groups. Fixed-candidate
  vocal-minus-instrumental mass averaged `1.330` under the vocal reason,
  `-0.651` under neutral, and `-1.233` under the instrumental reason; both
  shifts were correct in both candidate orders. `Space Oddity` and `Awake`
  failed isolated pair direction. `Space Oddity` nevertheless became a top
  vocal choice in the closed set, motivating a generation-versus-comparison
  mechanistic contrast rather than a generic deeper-reason probe.

## 2026-07-11 Controlled Vocality Pair-vs-Choice Path Patching

- Job ID: `6a51f5e5e4a4e82c0b58d532`
- Job URL: https://huggingface.co/jobs/REDACTED/6a51f5e5e4a4e82c0b58d532
- Workload: patch matched-reason activations into flipped-reason runs for the
  complete title-artist generation path and the fixed-candidate choice path.
- Sentinels: `Blinding Lights - The Weeknd` and `River Flows in You - Yiruma`
  as behavioral successes; `Space Oddity - David Bowie` and `Awake - Tycho`
  as behavioral failures.
- Model: `Qwen/Qwen3-1.7B-Base`
- Hardware: `a10g-small`
- Layers: 14, 16, 18, 21, 24, 27, and endpoint layer 28.
- Components: attention output, MLP output, and full residual output.
- Entity unit: complete real title plus artist; no Song ID.
- Registered technical checks: four nontrivial pair effects, four choice
  relations across reversed candidate orders, all captures and finite values,
  and final-layer full-residual endpoint error no greater than `0.02` for both
  paths.
- Interpretation scope: exploratory; no post-hoc effect-size success threshold.
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: 179 seconds total, including 110 seconds scheduling and 69 seconds
  running; the probe took about 21 seconds after model loading.
- Technical result: all checks passed, all values were finite, all captures
  were present, and both final-layer full-residual endpoint errors were `0`.
- Choice-path result: across both vocality directions and both reversed orders,
  layer-18 attention recovered `0.628` of the reason effect on average while
  layer-18 MLP recovered `0.016`; full-residual recovery reached `0.677` at
  layer 18 and `0.973` at layer 24.
- Pair-path result: behavioral-success sentinels averaged `0.784` and `0.922`
  full-residual recovery at layers 18 and 21, versus `0.247` and `0.348` for
  behavioral failures. All four reached the source endpoint by layer 27, so
  failure is associated with later and less stable reason integration rather
  than a globally absent source-target signal.
- Decision: localize the order-stable layer-18 choice effect to attention heads,
  then test the same heads on complete title-artist success and failure cases.

## 2026-07-11 Controlled Vocality Layer-18 Attention Heads

- Job ID: `6a51f91beffc02a91cbd9703`
- Job URL: https://huggingface.co/jobs/REDACTED/6a51f91beffc02a91cbd9703
- Workload: patch each of Qwen3-1.7B's 16 layer-18 attention heads from the
  matched-reason source into the flipped-reason pair and choice paths.
- Inputs: the same four real title-artist sentinels and two reversed candidate
  orders used by the completed component-path pilot; no Song ID.
- Intervention point: concatenated per-head output immediately before
  `self_attn.o_proj`, with all-head and direct post-`o_proj` attention controls.
- Model: `Qwen/Qwen3-1.7B-Base`
- Hardware: `a10g-small`
- Registered technical checks: 16 heads by 128 dimensions, four nontrivial pair
  effects, four choice relations, all interventions and finite values, and
  all-head versus direct-attention score error no greater than `0.002`.
- Descriptive candidate rule: positive recovery in all four choice relations
  and mean recovery at least `0.05`; not a confirmatory outcome gate.
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: 71 seconds total, including 5 seconds scheduling and 66 seconds
  running; the probe took about 20 seconds after model loading.
- Technical result: all gates passed; architecture was 16 heads by 128
  dimensions, every intervention was present and finite, and all-head versus
  direct-attention errors were `0` in both paths.
- Choice result: heads `1`, `0`, `8`, and `9` met the frozen consistency rule.
  Their mean recoveries were `0.245`, `0.163`, `0.065`, and `0.064`; every head
  moved toward the source in all four vocality-direction/order relations.
- Pair result: the same heads did not distinguish behavioral success from
  failure by being absent. Head 1 recovered `0.913` of the negative matched-
  reason effect for `Space Oddity`, while the full layer-18 attention output
  recovered `1.422`; the clean choice-routing heads can therefore participate
  in a wrong entity-specific direction. `Awake` showed a different mixed and
  opposing pattern.
- Decision: reject the simple missing-head explanation. Split pair effects into
  title and artist tokens using only the frozen heads `0/1/8/9`, then determine
  whether the failure is entity retrieval or title-artist relation binding.

## 2026-07-11 Controlled Vocality Title-vs-Artist Field Diagnosis

- Job ID: `6a5232f0effc02a91cbd9881`
- Job URL: https://huggingface.co/jobs/REDACTED/6a5232f0effc02a91cbd9881
- Workload: split matched-versus-flipped reason effects into title tokens,
  artist tokens, and the complete token-weighted pair; patch the frozen
  layer-18 heads `0/1/8/9` at title-only, artist-only, and both-field scopes.
- Inputs: the same four independently verified real title-artist sentinels; no
  Song ID or new post-hoc track selection.
- Model: `Qwen/Qwen3-1.7B-Base`
- Hardware: `a10g-small`
- Registered failure-locus labels: title only, artist only, both fields, or
  neither, with a weak-boundary suffix below absolute field effect `0.03`.
- Technical controls: field effects reconstruct the complete-pair effect;
  all-16-head pre-`o_proj` patching reproduces direct attention patching; all
  values finite and all 12 single-head scope interventions present.
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: 194 seconds total, including 126 seconds scheduling and 68 seconds
  running; the probe took about 20 seconds after model loading.
- Technical result: all gates passed. Maximum pair reconstruction error was
  `3.18e-7`, and all-head versus direct-attention error was `0`.
- Field result: `Space Oddity` was negative in both title (`-0.201`) and artist
  (`-0.065`); `Awake` had a strongly negative title effect (`-1.034`) but a
  positive artist effect (`+0.292`). `River Flows in You` was positive in both.
  `Blinding Lights` had a strong positive title and only a weak-boundary artist
  effect (`-0.027`).
- Head result: frozen heads `0/1/8/9` recovered `1.554` of the wrong
  `Space Oddity` title effect and, for `Awake`, recovered `0.313` of the wrong
  title effect alongside `0.236` of the correct artist effect.
- Decision: freeze both field-disagreement and generation-versus-independent-
  relation disagreement as migration signals; neither alone covers both
  failure types.

## 2026-07-11 Free-Generated Relation-Conflict Transfer Pilot

- Job ID: `6a5237c0effc02a91cbd9899`
- Job URL: https://huggingface.co/jobs/REDACTED/6a5237c0effc02a91cbd9899
- Workload: compare emitted versus reference artist support under a factual
  catalog prefix, the reconstructed original playlist-generation prefix, and
  a two-order independent A/B relation verifier; read layerwise margins and
  patch catalog states into generation states.
- Transfer set: all six free-generated pairs verified exact by both catalogs
  and six context-matched conflicts selected deterministically from a unique
  MusicBrainz/Apple shared artist, before model scoring.
- Entity unit: real title plus artist names only; no Song ID.
- Model: `Qwen/Qwen3-1.7B-Base`
- Hardware: `a10g-small`
- Frozen validation rule: both catalog first-token and independent-choice
  verifiers need balanced accuracy at least `0.75`, with at least 4/6 exact and
  4/6 conflict events correct; one passing path is only promising.
- Mechanistic checks: layers 14/18/21/24/27/28; catalog-to-generation patches
  at layer-21 attention, layer-24 MLP, layer-27 residual, and layer-28 endpoint.
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: 71 seconds total, including 5 seconds scheduling and 66 seconds
  running; the probe took about 21 seconds after model loading.
- Technical result: all gates passed; both final-layer readout and layer-28
  endpoint-patch maximum errors were `0`.
- Detection result: catalog first-token verification scored 5/6 exact and 2/6
  conflicts (`BA=0.583`); complete-artist catalog scoring scored 5/6 and 3/6
  (`BA=0.667`); two-order A/B verification scored 6/6 and 3/6 (`BA=0.750`).
  The A/B path missed the registered minimum of four conflicts, so the frozen
  validation status is `not_supported`.
- Generation result: the reconstructed original prefix favored the emitted
  artist for all 12 events, including all six catalog conflicts (`BA=0.500`).
- Conflict modes: two first-token `context_override` cases (`Skinny Love - Sam
  Smith` and `River Flows in You - Coldplay`) and four cases where the factual
  first-token relation was not recovered.
- Causal result: layer-21 attention patches moved all 6 conflicts toward the
  factual-prefix state (mean recovery `0.431`); layer-24 MLP did so for 5/6
  (`0.309` mean); layer-27 residual recovered the factual state almost fully.
  This is not always correction because the factual-prefix state itself was
  wrong on four first-token cases.
- Decision: same-model conflict monitoring is a useful routing signal for a
  subset, not a reliable standalone detector. Close the stage with a negative
  validation result and require external catalog grounding or a new independent
  holdout before testing the exploratory complementary rule.

## 2026-07-11 Independent New-Seed Holdout Generation

- Job ID: `6a523ac0effc02a91cbd98aa`
- Job URL: https://huggingface.co/jobs/REDACTED/6a523ac0effc02a91cbd98aa
- Workload: 20 raw free-playlist generations from the four frozen contexts and
  five unseen seeds `59/71/83/101/127`.
- Model: `Qwen/Qwen3-1.7B-Base`
- Sampling: temperature `0.7`, top-p `0.9`, maximum 384 new tokens.
- Persistence: one raw JSON log record per generation before parsing or catalog
  verification.
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: 271 seconds total, including 4 seconds scheduling and 266 seconds
  running.
- Generation result: all 20 raw completions were persisted and the generation
  technical gate passed.
- Frozen parser result: 19/20 generations and 93 title-artist-reason pairs were
  recovered, passing the registered minimum of 16 generations and 75 pairs.
- Catalog result: 79 catalog conflicts, 11 exact pairs, and 3 unverified rows.
  Under the stricter holdout criteria there are 9 double-source exact events
  and 21 conflicts with a unique non-emitted artist shared by both catalogs.
- Decision: no seed extension is needed. Hash-select 9 events per label and run
  the frozen complementary warning rule without changing prompts or thresholds.

## 2026-07-11 Independent Holdout Verifier Confirmation

- Job ID: `6a524112e4a4e82c0b58da32`
- Job URL: https://huggingface.co/jobs/REDACTED/6a524112e4a4e82c0b58da32
- Workload: test the preregistered OR warning rule on 9 double-source exact and
  9 unique-shared-reference catalog-conflict events selected only by the frozen
  `holdout_v1:<generation_id>:<rank>` SHA-256 order.
- Model: `Qwen/Qwen3-1.7B-Base`
- Hardware: `a10g-small`
- Frozen primary rule: predict conflict when either the two-order independent
  A/B emitted-minus-reference margin or factual-prefix complete-artist mean-log-
  probability margin is below zero.
- Confirmation gates: balanced accuracy, exact specificity, and conflict
  sensitivity each at least `0.75`; at least 8 events in each label; both A/B
  orders complete and all values finite.
- Diagnostics only: catalog and original-generation first-token margins,
  original-generation complete-artist margin, path overlap, order consistency,
  and unique title-artist relation counts.
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: 186 seconds total, including 115 seconds scheduling and 70 seconds
  running; scoring took about 22 seconds after model loading.
- Technical result: all gates passed; all 18 rows were finite, both choice
  orders were complete, and order signs agreed on 15/18 events.
- Frozen OR result: 6/9 exact and 5/9 conflict events were correct
  (`specificity=0.667`, `sensitivity=0.556`, `BA=0.611`), so the exploratory
  rule was not confirmed.
- Independent-choice result: 9/9 exact and 5/9 conflict events were correct
  (`BA=0.778`). It remained conservative and useful for routing but missed the
  registered 7/9 conflict-sensitivity gate.
- Complete-artist factual-prefix result: 6/9 exact and 2/9 conflict events were
  correct (`BA=0.444`). Its two conflict detections were already detected by
  choice, while it added three exact false positives; the discovery-set
  complementarity did not transfer.
- Original-generation result: first-token scoring detected 0/9 conflicts and
  complete-artist scoring detected 1/9, while preserving 9/9 exact events. The
  generation context therefore continued to support its own emitted relation.
- Decision: reject the frozen complementary OR rule for Qwen3-1.7B. Diagnose
  the four choice misses and three sequence false positives without tuning the
  confirmation threshold, then test the unchanged behavior probe on a second
  open model.

## 2026-07-12 Holdout Title-Counterfactual Diagnostic

- Job ID: `6a52f72fe4a4e82c0b58ea75`
- Job URL: https://huggingface.co/jobs/REDACTED/6a52f72fe4a4e82c0b58ea75
- Workload: keep each emitted/reference artist pair fixed and compare its
  factual title against two deterministic real-title controls whose known
  artist is neither candidate.
- Focus set: 17 unique candidate events after collapsing the duplicated
  `The Knife - The Knife` versus `Genesis` event; includes 3 unique choice
  misses, 3 sequence false positives, 5 choice hits, and 6 clean exact events.
- Model: `Qwen/Qwen3-1.7B-Base`
- Hardware: `a10g-small`
- Diagnostic: factual margin minus mean control-title margin, computed for the
  two-order independent choice path and factual-prefix complete-artist path.
  A correct contrast with a wrong absolute margin is interpreted as latent
  relation support masked by candidate prior; a wrong contrast means relation
  retrieval was not recovered under this probe.
- Scope: post-hoc causal diagnosis only; no threshold, warning rule, or
  confirmation claim is changed.
- Status at submission: `RUNNING`
- First job status: `ERROR` after all 17 row records were emitted. The summary
  helper divided by zero for label subgroups absent from a focus role; no GPU
  score was missing or invalid.
- First-job runtime: 255 seconds total, including 186 seconds scheduling and
  68 seconds running.
- Smallest fix: empty subgroups now return explicit zero counts and null means.
  The fp16 reference-reproduction tolerance was documented as `0.04` because
  the changed batch shape produced one `1/32` logit step (`0.03125`) while all
  score signs remained unchanged. This is a technical tolerance, not a
  behavior threshold.
- Retry job ID: `6a52f913e4a4e82c0b58ea8f`
- Retry URL: https://huggingface.co/jobs/REDACTED/6a52f913e4a4e82c0b58ea8f
- Retry status at submission: `RUNNING`
- Retry final status: `COMPLETED`
- Retry runtime: 71 seconds total, including 5 seconds scheduling and 66
  seconds running; all technical gates passed and the maximum fp16 reference
  reproduction error was `0.03125`.
- Sequence false-positive result: all 3/3 exact false positives had a correct
  factual-minus-control title direction. `All of Me - John Legend`,
  `Hallelujah - Jeff Buckley`, and `Watermelon Sugar - Harry Styles` therefore
  retained relation-specific support that was masked by absolute artist-name
  priors.
- Choice-miss result: 0/3 unique misses had a correct choice contrast.
  `The Knife - The Knife` versus `Genesis` and `Don't Stop the Music - DJ
  Snake` versus `Yarbrough & Peoples` also had wrong sequence contrasts, so
  neither probe recovered the catalog relation. `Halo - MØ` versus `Haley
  James Scott` had a correct sequence contrast despite both absolute decisions
  being wrong, indicating partial latent relation support with a failed
  decision readout.
- Controls: all 6 clean exact events were relation-specific in both paths.
  Four of five choice-detected conflicts had a correct choice contrast; the
  weak `Whispering Sounds` hit did not, showing that a correct warning can also
  arise from candidate prior rather than title-artist binding.
- Decision: separate relation knowledge from absolute candidate preference in
  all later claims. Localize the layerwise relation contrast for one latent-
  knowledge miss, two no-retrieval misses, and exact prior-masking controls.

## 2026-07-12 Full-Sequence Title-Effect Causal Trace

- Job ID: `6a52fbb7effc02a91cbda1c6`
- Job URL: https://huggingface.co/jobs/REDACTED/6a52fbb7effc02a91cbda1c6
- Workload: for all 17 title-contrast events and both controls, patch factual-
  title internal states into control-title sequences at every position that
  predicts a token of the complete emitted/reference artist name.
- Model: `Qwen/Qwen3-1.7B-Base`
- Hardware: `a10g-small`
- Sweep: full residual after every layer 1-28; attention and MLP outputs at
  layers 18, 21, 24, and 27; 68 aligned candidate sequences per condition.
- Causal metric: recovery from the control-title complete-artist margin toward
  the factual-title margin, computed separately for both controls before
  aggregation. Effects below absolute margin `0.10` are omitted from normalized
  recovery but retained as raw signed shifts.
- Technical controls: identical artist tokenization, duplicated factual scores,
  title-contrast reproduction within `0.05`, final-layer sequence endpoint
  reproduction within `0.02`, complete intervention matrix, and finite values.
- Scope: post-hoc mechanism localization and causal sufficiency, not a new
  warning classifier.
- Status at submission: `RUNNING`
- First job final status: `COMPLETED` in 73 seconds (5 scheduling, 68 running),
  but each raw JSON row exceeded the log connector's single-line retrieval
  limit, so the returned records could not be losslessly parsed locally.
- Output-only fix: rows and summary are now serialized together, zlib-
  compressed, base64-encoded, and split into 7,500-character numbered chunks.
  No model input, intervention, metric, or gate changed.
- Artifact-format retry job ID: `6a52fca6effc02a91cbda1c9`
- Retry URL: https://huggingface.co/jobs/REDACTED/6a52fca6effc02a91cbda1c9
- Retry status at submission: `RUNNING`
- Retry final status: `COMPLETED`
- Retry runtime: 73 seconds total, including 4 seconds scheduling and 68
  seconds running. All 10 artifact chunks were recovered and decoded into 17
  row records plus the full summary.
- Technical result: every gate passed. Final-layer complete-sequence endpoint
  error was `0`, maximum title-contrast reproduction error was `0.03146`, all
  68 artist continuations aligned, and all 36 interventions were finite.
- Residual result: factual-title effects were negligible through roughly layer
  12, became recoverable around layers 16-18, and were nearly complete by layer
  21. The earliest two-layer sustained recovery was layer 18 overall, layer 16
  for the three exact prior-masking cases, and layer 21 for clean exact controls.
- Component result: layer-21 attention had `0.358` mean recovery with `0.882`
  toward-source rate overall; layer-24 MLP had `0.261` mean recovery with
  `0.765` toward-source rate. The relation effect is distributed rather than
  owned by one component.
- Prior-masking mechanism: `Hallelujah` carried a strong correct relation effect
  by layer 12; `Watermelon Sugar` emerged around layers 16-18; weak `All of Me`
  support was present but close to the `0.10` effect boundary. Their final
  absolute artist scores were wrong despite these causally transmitted effects.
- Wrong-binding mechanism: `The Knife` and `Don't Stop the Music` encoded the
  wrong factual-title direction in the residual stream. Layer-21 attention
  contributed wrong-direction shifts in both; late MLP effects sometimes
  opposed the error (`+0.741` signed shift at layer 27 for `The Knife`,
  `+1.101` for `Don't Stop the Music`) but did not overturn it.
- Partial-knowledge mechanism: `Halo` had a correct factual-title effect already
  by layer 12, mediated mainly by MLP at layers 21 (`0.829` recovery), 24
  (`0.774`), and 27 (`0.538`), while attention-only patches were slightly
  opposing. The correct relation signal existed but lost to the absolute
  candidate preference and choice readout.
- Decision: the same surface hallucination has at least two mechanisms: latent
  correct relation knowledge masked by entity priors, and a causally formed
  wrong relation binding. Raw confidence or one self-check prompt cannot solve
  both; later recommendations must combine relation-contrast diagnostics with
  external catalog grounding.

## 2026-07-12 Qwen3-4B Cross-Model Holdout Verification

- Job ID: `6a52f77feffc02a91cbda1bb`
- Job URL: https://huggingface.co/jobs/REDACTED/6a52f77feffc02a91cbda1bb
- Workload: apply the unchanged 18-event verifier prompts, zero thresholds,
  two choice orders, and OR rule to the independent holdout generated by
  Qwen3-1.7B.
- Verifier model: `Qwen/Qwen3-4B-Base`
- Hardware: `a10g-small`
- Scope: cross-model critic transfer. This tests whether a larger open model
  can flag the 1.7B model's catalog conflicts; it is not evidence that the 4B
  model detects errors in its own free generations.
- Status at submission: `RUNNING`
- Final status: `COMPLETED`
- Runtime: 138 seconds total, including 31 seconds scheduling and 107 seconds
  running.
- Technical result: all gates passed; both choice orders were present, all
  values were finite, and order signs agreed on 16/18 events.
- Frozen OR result: 8/9 exact and 5/9 conflict events were correct
  (`specificity=0.889`, `sensitivity=0.556`, `BA=0.722`), so it was not
  confirmed.
- Independent-choice result: also 8/9 exact and 5/9 conflict (`BA=0.722`).
  Complete-artist factual-prefix scoring reached 9/9 exact and 3/9 conflict
  (`BA=0.667`); its conflict detections were a subset of choice detections.
- Stable boundary: the four missed events were exactly the same as for 1.7B,
  representing three unique relations: duplicated `The Knife - The Knife`
  versus `Genesis`, `Don't Stop the Music - DJ Snake` versus `Yarbrough &
  Peoples`, and `Halo - MØ` versus `Haley James Scott`.
- Decision: model scaling within Qwen3 did not repair relation retrieval. Use
  the running title-counterfactual diagnostic to distinguish prior masking
  from absent relation contrast before deciding on deeper interventions.

## 2026-07-12 Publication Provenance Archive

- The five publication-stage Jobs were re-inspected through the Hugging Face
  Jobs API. Structured image, hardware, timestamp, duration, status, command
  hash, script hash, and log hash records are stored in
  `runs/hf_job_metadata.json`.
- Every Job completed on `a10g-small` with the
  `ghcr.io/astral-sh/uv:python3.12-bookworm` image. Exact submitted script bytes
  are preserved as base64 under `runs/job_scripts/<JOB_ID>.py.b64`; readable
  LF-normalized copies are stored beside them.
- Terminal snapshots returned by the Jobs logs API are archived under
  `runs/job_logs`. The API surface returned the last 20 lines, so these files
  are not described as full lifecycle logs. Complete rows and summaries remain
  in the structured publication artifacts.
- The original scripts loaded model IDs without explicit revisions. Because
  both model repositories were last modified in July 2025, before all Jobs,
  the effective revisions were retrospectively resolved as
  `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` for Qwen3-1.7B-Base and
  `906bfd4b4dc7f14ee4320094d8b41684abff8539` for Qwen3-4B-Base. Publication
  runners now accept `--model-revision` or `MODEL_REVISION` and pin Python 3.12,
  Torch 2.7.1, Transformers 4.53.2, and Accelerate 1.8.1 for deterministic
  reruns.
- The holdout protocol was frozen in the working session before scoring but
  has no external registry timestamp or pre-result Git commit. Public-facing
  text therefore uses `pre-specified` rather than `preregistered`.

## 2026-07-13 Phase 2 External Preregistration

- The Phase 2 protocol was frozen locally at commit
  `2158739b282cea004f86d1beb22fe7e1f2c570ad` before any Granite generation or
  intervention result was observed.
- First publication Job: `6a54c867effc02a91cbdb46e` (`FAILED`). A large
  Base64 transport payload was truncated before decoding; no Hub commit and no
  scientific output were produced.
- Second publication Job: `6a54c91ee4a4e82c0b5912e4` (`FAILED`). The
  connected token can run Jobs but returned `403` when creating a Dataset
  repository; no Hub commit and no scientific output were produced.
- The exact frozen Markdown protocol, JSON protocol, smoke bundle, and embedded
  smoke script were then published as a public GitHub Gist at
  https://gist.github.com/a56668283b095f59f0eacf0527395b58.
- Public timestamp: `2026-07-13T11:17:59Z`; Gist commit:
  `5f0b778ddd6fbfdfbc07c1aa1b4b98c5d04fd195`.
- The local receipt and all four SHA-256 digests are stored in
  `runs/phase2_preregistration_receipt.json`.
- At the public timestamp, the Granite GPU smoke had not been submitted.

## 2026-07-13 Phase 2 Granite Technical Smoke

- Job ID: `6a54c9efeffc02a91cbdb4e4`
- Job URL: https://huggingface.co/jobs/REDACTED/6a54c9efeffc02a91cbdb4e4
- Exact script source: public preregistration Gist commit
  `5f0b778ddd6fbfdfbc07c1aa1b4b98c5d04fd195`.
- Hardware: `a10g-small`; timeout: 30 minutes.
- Final status: `COMPLETED`.
- Runtime: 109 seconds total, including 11 seconds scheduling and 98 seconds
  running.
- Provenance gates: both frozen protocol hashes and submitted script hash match
  the public preregistration record.
- Architecture gate: expected 40 Granite layers and observed 40.
- Format gate: one non-empty completion produced five parsed tracks, above the
  frozen minimum of four. The raw completion was not uploaded or inspected.
- Hook endpoint gate: maximum reproduced-logit error was `0.0`, below the
  frozen tolerance of `0.02`.
- Peak allocated GPU memory: `6,857,860,096` bytes.
- Technical gate: `PASS`. Decision: release the frozen 60-playlist primary
  generation; no catalog or scientific result was inspected in this smoke.

## 2026-07-13 Phase 2 Granite Primary Generation

- Job ID: `6a54cc73e4a4e82c0b591404`
- Job URL: https://huggingface.co/jobs/REDACTED/6a54cc73e4a4e82c0b591404
- Exact runner and primary bundle were published before submission in Gist
  commit `836343c10a34057fa9430b38e524d01a90f7ef0d`.
- Remote and local submitted-script SHA-256:
  `6672e8639822da0e1fa4f6d4308040cd5229a19e28e4ee35c9dbb236c19b0fdc`.
- Hardware: `a10g-small`; final status: `COMPLETED`.
- Runtime: 808 seconds total, including 4 seconds scheduling and 803 seconds
  running.
- All provenance, architecture, endpoint, generation-count, uniqueness, and
  nonempty-output gates passed. Endpoint error was `0.0`.
- The 60 registered playlists yielded 290 parsed events, above the frozen 240
  minimum. Two completions produced zero parser records and were retained
  without repair or selective rerun.
- The compressed output occupied three numbered log chunks. All three were
  recovered and decoded into 60 raw completion records.
- Parsed catalog input contains 223 unique normalized titles. The extension
  decision remains pending strict MusicBrainz-plus-Apple catalog labels because
  the protocol also requires at least 40 strict conflict title clusters.

## 2026-07-13 Phase 2 Primary Catalog v2

- The conservative verifier amendment and exact method code were published
  before replay in public Gist history
  `6444e8d73633e833bab3daf6568b24dd7c518891`.
- Verifier SHA-256:
  `c9659b932ac3f1064e75e0b5b03c8cad5aa34cf7ca84efa2946c8dd9a9bb9867`.
- All 290 primary relations reached a terminal v2 label with zero request
  errors: 5 strict conflicts, 37 strict exact, 9 ambiguous, and 239 excluded.
- After frozen normalized-title clustering there were 5 unique strict-conflict
  and 22 unique strict-exact title clusters.
- All 42 strict rows passed offline replay from only their linked raw catalog
  responses. The full 957-request evidence archive was frozen and compressed.
- Extension gate: `TRIGGERED`, because 5 primary strict-conflict title clusters
  are below the frozen minimum of 40. No threshold or sample rule changed.

## 2026-07-13 Phase 2 Granite Extension Generation

- Job ID: `6a54e481e4a4e82c0b591722`
- Job URL: https://huggingface.co/jobs/REDACTED/6a54e481e4a4e82c0b591722
- Exact public runner raw URL ends at Gist file revision
  `3713fb9b6e3cc2eff3e43f33c4f2950571118583`.
- Local and remote runner SHA-256:
  `ea617aef2f745c660701210a3f8cd4600ac338a68589f2f04930d28bc8423cd9`.
- Hardware: `a10g-small`; timeout: 1 hour.
- Final status: `COMPLETED`.
- Runtime: 338 seconds total, including 5 seconds scheduling and 333 seconds
  running.
- All protocol, submitted-script, model-revision, 40-layer architecture, and
  endpoint-reproduction gates passed; maximum endpoint logit error was `0.0`.
- All 20 registered generations returned non-empty completions. Nineteen
  completions parsed to five tracks each; `celebratory_dance__seed139` changed
  the frozen output format and parsed to zero without manual repair.
- The frozen extension therefore yielded 95 catalog-input events across 84
  unique normalized titles. Two numbered compressed artifact chunks were
  recovered without omission.

## 2026-07-13 Phase 2 Extension Catalog and Final Stop

- The same publicly frozen v2 verifier classified all 95 extension rows. Its
  SHA-256 remained
  `c9659b932ac3f1064e75e0b5b03c8cad5aa34cf7ca84efa2946c8dd9a9bb9867`.
- A checkpointed first pass was resumed with the primary request cache. It
  ended with 9 MusicBrainz network-error rows and no stderr output.
- Frozen `retry_errors_only` recovery passes reduced terminal errors from 9 to
  8 and then to 0. Failed attempts remain in the evidence archive; no non-error
  row was selectively rerun.
- Final extension event labels: 4 strict conflict, 4 strict exact, 4 ambiguous,
  and 83 excluded. After normalized-title clustering these became 2 unique
  strict conflicts and 4 unique strict exact titles.
- All 8 strict extension rows replayed exactly from their linked raw responses.
  The extension archive contains 394 request records and has raw SHA-256
  `03cb574543de3e86bbbd7a6436c5ed8a03ee3ce18f16ac1e53d5e82d471d6036`.
- Primary and extension were merged by request ID. Two reused records were
  byte-identical; the final archive contains 1,349 unique requests and has raw
  SHA-256
  `5c14f2e6ed519a6a3005801c9a0cb0989deeb21bd83edeb8cefef0c474bfe57f`.
- Final combined labels across 385 parsed events: 9 strict conflict, 41 strict
  exact, 13 ambiguous, and 322 excluded. All 50 strict rows passed offline raw
  response replay.
- Frozen title clustering yielded 7 unique strict-conflict and 25 unique
  strict-exact titles. The confirmatory minimum was 30 conflicts.
- Final decision: `STOP_INSUFFICIENT_STRICT_CONFLICT_CLUSTERS`. Formal Granite
  mechanism diagnosis, correction, H1, and H2 were not run; no model, prompt,
  threshold, layer, or third sample batch was used to rescue the result.

## 2026-07-14 Granite Relation-Knowledge Recoverability Audit

- Job ID: `6a55c267effc02a91cbdc2ce`
- Job URL: https://huggingface.co/jobs/REDACTED/6a55c267effc02a91cbdc2ce
- Workload: candidate-free behavioral recoverability audit over the seven
  Phase 2 catalog conflicts, 25 generated strict-exact controls, and eight
  canonical positive controls.
- Model: pinned `ibm-granite/granite-4.1-3b-base` revision
  `dacb9cb9157bec98e99b09f285c92a4d58405c96`.
- Hardware: `t4-small`; timeout: 1 hour.
- Frozen prompts per title: 3; total records: 40; total generations: 120.
- Primary conflict diagnostic: median catalog-reference-minus-emitted mean
  token log-probability, combined with candidate-free reference generation in
  at least two of three templates.
- Bundle SHA-256:
  `0fd7de60ca56ffd29c59dcdbf1d319755415a340b922a084cc7626ca812efc26`.
- Submitted embedded script SHA-256:
  `8ff8b98e4b0782379354bcf6118d55d4b5e69528eae41d1fa45d729a9c804139`.
- Persistence: one `REL_KNOWLEDGE_ROW_JSON` line per record plus numbered
  compressed `REL_KNOWLEDGE_ARTIFACT_CHUNK_JSON` records and a terminal
  `REL_KNOWLEDGE_SUMMARY_JSON`; no Hub dataset write is required.
- Final status: `ERROR` after 334 seconds (133 seconds scheduling, 200 seconds
  running).
- Failure: the inline `python -c` environment does not define `__file__`, so
  the runner failed while recording its own script hash. The model downloaded
  and loaded successfully, but no `REL_KNOWLEDGE_*` result record was emitted.
- Retry rule: change only this execution-metadata field. Prompts, samples,
  decoding, thresholds, categories, and all scientific decision rules remain
  frozen.
- Claim boundary: this audit cannot reopen Phase 2 or establish hidden-state
  knowledge, subjective awareness, deception, prevalence, or self-correction.

### Metadata-only retry

- Job ID: `6a55c4f5effc02a91cbdc2e4`
- Job URL: https://huggingface.co/jobs/REDACTED/6a55c4f5effc02a91cbdc2e4
- Hardware and timeout: unchanged (`t4-small`, 1 hour).
- Submitted embedded script SHA-256:
  `e42062c430d94b152ea23cfe8ddff68e7754cd11bc4080732ee80d23a553825d`.
- Only change: record `script_execution_mode=inline_python_c_no_file` and
  leave the unavailable in-process script hash as null.
- Final status: `COMPLETED` after 214 seconds (14 seconds scheduling, 199
  seconds running).
- Technical gate: `PASS`; 40 records, 120 prompts, and 118 nonempty
  generations (98.33%).
- Assay validity gate: `PASS`; 7/8 canonical controls recovered (87.5%).
- Generated strict-exact controls: 14/25 recovered (56%).
- Conflict result: 0/7 were `reference_recoverable`; all seven were
  `unrecovered_or_indeterminate`, and all seven median reference-minus-emitted
  mean token log-probability margins were negative.
- Decision: permanently exclude these seven old catalog conflicts from
  hidden-state discovery. Proceed only with the separately frozen Phase 3
  natural-generation and recoverability pipeline.
- Archived artifact SHA-256:
  `b91e39a789278b7d996c565f2a330a5f3c389d18637da25f90060c804526b4b8`.
- Archived full log SHA-256:
  `26fe6156219e31c63a2ce6d34fa0b06036e1fd30f60ae4e89d87c6d5c45baf40`.

## 2026-07-14 Phase 3 Natural Relation-Discovery Pilot

- Frozen baseline commit: `9c338e3`.
- Job ID: `6a55d4dfeffc02a91cbdc365`
- Job URL: https://huggingface.co/jobs/REDACTED/6a55d4dfeffc02a91cbdc365
- Workload: 12 frozen relation-stress contexts by 6 frozen pilot seeds,
  yielding 72 natural five-track playlist requests and at most 360 parsed
  title-artist events.
- Model: pinned `Qwen/Qwen3-4B-Base` revision
  `906bfd4b4dc7f14ee4320094d8b41684abff8539`.
- Hardware: `t4-small`; timeout: 2 hours.
- Generation bundle SHA-256:
  `523ef90ccf74fd547f39cdcdf16db04cbe7cc53ab437088e3ec6efaf3af4f549`.
- Submitted embedded script SHA-256:
  `26b6bf6e33b1426542ffe0c059a7eaee46459aa0bea568be951f411d27cad8f7`.
- First technical gate: all 72 completions are nonempty and at least 300
  tracks parse under the frozen title-artist-reason parser.
- Scientific continuation still additionally requires at least eight unique
  catalog-conflict title clusters that pass the separate relation-knowledge
  recoverability audit. Generation success alone cannot pass that gate.
- Final status: `COMPLETED` after 779 seconds (5 seconds scheduling, 774
  seconds running).
- Generation technical gate: `PASS`; 72/72 nonempty completions and 360/360
  parsed title-artist-reason events across 318 unique normalized titles.
- Full generation artifact SHA-256:
  `433c8694d1b0683cbcd828c7d46e597633386f9dfc8b2d69df4aba428f3e5015`.
- Full generation log SHA-256:
  `5857a4d3ddf0c26701316583a61d1f2a5a0ae507d305bf59dfdc58299a834aa1`.

### Local double-catalog handoff

- Verifier: `phase2_catalog_v2_complete_alias_audit`; source SHA-256
  `c9659b932ac3f1064e75e0b5b03c8cad5aa34cf7ca84efa2946c8dd9a9bb9867`.
- Runtime: 2,551.2 seconds; all 360 rows processed with no source-error row.
- Labels: 12 strict conflict, 26 strict exact, 56 ambiguous, 266 excluded.
- Evidence: 1,309 archived raw requests; decompressed evidence SHA-256
  `396816ee003e25ec8c976654f41410c01ee5dd002e6e7af766c7ef7c9bc9127d`.
- Frozen selection: 11 unique strict-conflict titles, 20 disjoint strict-exact
  controls, and eight canonical controls; 39 audit records and 117 prompts.
- Recoverability bundle SHA-256:
  `b2a9031e7e46505578ad6d783a7837954872236e79504eb9320f64dd5e87d8e2`.
- Recoverability embedded script SHA-256:
  `8c8c360746f70ea2890f3e3a5a42e8eeb3328666de8db219187b74057520eb75`.
- Status: `READY_FOR_FROZEN_AUDIT`; catalog conflict counts alone do not pass
  the scientific continuation gate.

### Frozen Qwen recoverability audit

- Frozen input commit: `202793a`.
- Job ID: `6a55e370e4a4e82c0b592efe`
- Job URL: https://huggingface.co/jobs/REDACTED/6a55e370e4a4e82c0b592efe
- Model and revision: unchanged `Qwen/Qwen3-4B-Base` at
  `906bfd4b4dc7f14ee4320094d8b41684abff8539`.
- Hardware: `t4-small`; timeout: 1 hour.
- Inputs: 11 unique strict-conflict titles, 20 disjoint strict-exact controls,
  and eight canonical controls; 39 records and 117 candidate-free prompts.
- Bundle SHA-256:
  `b2a9031e7e46505578ad6d783a7837954872236e79504eb9320f64dd5e87d8e2`.
- Submitted embedded script SHA-256:
  `8c8c360746f70ea2890f3e3a5a42e8eeb3328666de8db219187b74057520eb75`.
- Frozen continuation threshold: at least eight unique
  `recoverable_relation_conflict` records; no prompt, threshold, control, or
  selection change is allowed after this submission.
- Final status: `ERROR` after 120 seconds (13 seconds scheduling, 107
  seconds running), before any prompt or scientific output was produced.
- Failure mode: the Phase 3 runner still read the legacy Granite scoring key
  `target_generation_recovery_minimum_prompts`, while the frozen Phase 3
  bundle correctly stores the unchanged threshold as
  `minimum_reference_generations`.
- Retry rule: change only those two key lookups in the runner. The frozen
  bundle, prompts, samples, scoring threshold, controls, and continuation gate
  remain byte-for-byte unchanged.

#### Scoring-key compatibility retry

- Job ID: `6a55e4d9e4a4e82c0b592f12`
- Job URL: https://huggingface.co/jobs/REDACTED/6a55e4d9e4a4e82c0b592f12
- Hardware and timeout: unchanged (`t4-small`, 1 hour).
- Frozen bundle SHA-256: unchanged
  `b2a9031e7e46505578ad6d783a7837954872236e79504eb9320f64dd5e87d8e2`.
- Submitted embedded script SHA-256:
  `37fca67bef245eb837dc3768067397651518edf6cbe73900b12ba7e04406e4d2`.
- Only change: both runner lookups now read the frozen Phase 3 key
  `minimum_reference_generations`.
- Final status: `ERROR` after 411 seconds (150 seconds scheduling, 260
  seconds running), after computation but before any result marker was
  emitted.
- Failure mode: final artifact assembly referenced the legacy Granite-only
  metadata field `downstream_rule`, which is absent from the frozen Phase 3
  bundle and does not participate in generation, scoring, validity, or the
  continuation decision.
- Second retry rule: remove only that invalid artifact metadata lookup. A
  static audit confirms that every other top-level bundle key read by the
  runner exists in the frozen bundle.

#### Artifact-metadata compatibility retry

- Job ID: `6a55e748e4a4e82c0b592f5f`
- Job URL: https://huggingface.co/jobs/REDACTED/6a55e748e4a4e82c0b592f5f
- Hardware and timeout: unchanged (`t4-small`, 1 hour).
- Frozen bundle SHA-256: unchanged
  `b2a9031e7e46505578ad6d783a7837954872236e79504eb9320f64dd5e87d8e2`.
- Submitted embedded script SHA-256:
  `878f178efc2f319922531b4a47758a0854708493f7caabf4a68a8367d919e0f1`.
- Only scientific-code change: none. The invalid, output-only
  `downstream_rule` lookup was removed, and a local regression test now checks
  every literal top-level bundle lookup before submission.
- Final status: `COMPLETED` after 247 seconds (5 seconds scheduling, 241
  seconds running).
- Technical gate: `PASS`; 39/39 records, 117/117 prompt outputs, and 100%
  nonempty generations.
- Assay validity gate: `PASS`; 8/8 canonical controls recovered (100%).
- Generated strict-exact controls: 14/20 recovered (70%).
- Conflict result: 0/11 were `recoverable_relation_conflict`; four were
  `margin_only` and seven were `unrecovered_or_indeterminate`. Across the 33
  conflict prompts, the catalog reference and original emitted artist were
  each generated only once.
- Frozen continuation gate: `FAIL` (0 recoverable clusters versus the required
  minimum of 8). Decision: `stop_before_expansion_and_report_yield_failure`;
  no probe or intervention is permitted on these pilot rows.
- Full artifact SHA-256:
  `38f9adc8e8becfd51032b7e7edae91b33cf28a3fc87eac1db274464221068249`.
- Row archive SHA-256:
  `9e31a0d25968480e72af409ac4774c9a090fbce19c15c0318d63926c464e7384`.
- Summary SHA-256:
  `5ce381ca0e7a25f330e9e7bcb056caf057bfae87c2287dc21ebd58522c1256b5`.
- Full log SHA-256:
  `63d4265bf8869fa18231d954e774ea262fbb353c560bc8d0d662b68d51f07e3b`.
