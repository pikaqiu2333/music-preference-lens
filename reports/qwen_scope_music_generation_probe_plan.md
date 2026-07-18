# Qwen-Scope Music Generated-Reason Probe Plan

## Purpose

This prompt pack is for generated-answer interpretability, not a music
recommendation benchmark. The analysis target is whether answer reasons
align with candidate SAE features discovered in phrase-level probes.

## Summary

- Probe pairs: 4
- Prompt rows: 8
- Variants per pair: original and counterfactual

| Probe | Dimension | Expected Original Factors | Expected Counterfactual Factors | Feature Hypotheses |
|---|---|---|---|---|
| `gen_high_energy_night_to_rave` | `high_energy_rave` | `quiet night drive`, `restrained beat`, `not too loud`, `calm urban mood` | `high energy`, `rave pressure`, `hard techno drive`, `strong drums`, `fast activation` | layer24.feature32078, layer24.feature7884, layer24.feature30224 |
| `gen_low_presence_party_to_quiet` | `low_presence_quiet` | `harder energy`, `rave-like pressure`, `intense drums`, `maximum momentum` | `low presence`, `soft groove`, `not too loud`, `easy conversation` | layer24.feature32078, layer24.feature30584, layer24.feature19603 |
| `gen_emotional_vocal_study_to_lyrics` | `emotional_lyrics` | `low distraction`, `steady focus`, `gentle texture`, `quiet energy` | `heartfelt vocal`, `lyrical story`, `dramatic chorus`, `emotional release` | layer24.feature8706, layer24.feature24979, layer24.feature22232 |
| `gen_no_vocals_lyrics_to_instrumental` | `no_vocals_instrumental` | `heartfelt vocal`, `lyrical story`, `dramatic chorus`, `emotional release` | `no vocals`, `instrumental texture`, `no lyric attention`, `white noise focus` | layer24.feature8706, layer24.feature24979, layer24.feature24916 |

## Analysis Logic

For each generated answer, identify the spans where the model names
reason factors such as `high energy`, `emotional vocal`, or
`no vocals`. Then compare candidate feature activations on those
answer spans with the prompt-side feature direction.

A useful failure case is an answer that confidently cites a reason while
the expected feature movement is weak, missing, or contradicted.
