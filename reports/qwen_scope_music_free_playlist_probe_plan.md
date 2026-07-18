# Qwen-Scope Free Playlist Generation Probe Plan

## Purpose

This prompt pack removes candidate tracks and asks the base model to
generate a short playlist directly from the user profile and current
listening need.

The goal is not to judge whether the generated song titles are factual.
The goal is to test whether the generated playlist language and
answer-token SAE features move together under controlled preference
counterfactuals.

## Summary

- Probe pairs: 4
- Prompt rows: 8
- Variants per pair: original and counterfactual
- Candidate tracks: none
- Generated object: 5-track playlist with title, sound, and reason

| Probe | Dimension | Original Need | Counterfactual Need | Key Features |
|---|---|---|---|---|
| `free_night_drive_to_rave` | `high_energy_rave` | rainy night driving; wants quiet night drive, restrained beat, calm urban mood, and not too loud. | peak-time rave session; wants high energy, rave pressure, hard techno drive, strong drums, and fast activation. | layer24.feature32078, layer24.feature7884, layer24.feature30224 |
| `free_peak_party_to_soft_afterparty` | `low_presence_quiet` | 2am peak-time dance floor; wants harder energy, rave-like pressure, intense drums, and maximum momentum. | small apartment after-party; wants low presence, soft groove, not too loud, and easy conversation. | layer24.feature32078, layer24.feature30584, layer24.feature19603 |
| `free_study_to_emotional_vocal` | `emotional_lyrics` | paper writing background; wants low distraction, steady focus, gentle texture, and quiet energy. | late-night emotional listening; wants heartfelt vocal, lyrical story, dramatic chorus, and clear emotional release. | layer24.feature8706, layer24.feature24979, layer24.feature22232 |
| `free_emotional_vocal_to_no_vocals` | `no_vocals_instrumental` | late-night emotional listening; wants heartfelt vocal, lyrical story, dramatic chorus, and clear emotional release. | strict writing mode; wants no vocals, instrumental texture, no lyric attention, and white noise focus. | layer24.feature8706, layer24.feature24979, layer24.feature24916 |
