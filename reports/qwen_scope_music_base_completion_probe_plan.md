# Qwen-Scope Base Completion Reason Probe Plan

## Purpose

This prompt pack avoids instruction-following JSON output. It conditions
the base model on a selected best track and measures only the continuation
after `Reason:`.

This is not a ranking benchmark. It isolates generated reason text so
answer-token SAE features can be compared against prompt-side features.

## Summary

- Probe pairs: 4
- Prompt rows: 8
- Variants per pair: original and counterfactual

| Probe | Dimension | Original Best | Counterfactual Best | Key Features |
|---|---|---|---|---|
| `base_high_energy_night_to_rave` | `high_energy_rave` | `A` | `B` | layer24.feature32078, layer24.feature7884, layer24.feature30224 |
| `base_low_presence_party_to_quiet` | `low_presence_quiet` | `B` | `A` | layer24.feature32078, layer24.feature30584, layer24.feature19603 |
| `base_emotional_vocal_study_to_lyrics` | `emotional_lyrics` | `A` | `B` | layer24.feature8706, layer24.feature24979, layer24.feature22232 |
| `base_no_vocals_lyrics_to_instrumental` | `no_vocals_instrumental` | `B` | `C` | layer24.feature8706, layer24.feature24979, layer24.feature24916 |
