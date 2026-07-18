# SAE Probe Run Plan

- base_model: `google/gemma-3-270m-it`
- sae_repo: `google/gemma-scope-2-270m-it`
- sae_release: `gemma-scope-2-270m-it-resid_post`
- sae_id: `layer_12_width_16k_l0_small`
- layer_index: `12`
- probes: `6`

## Probes

### `pilot_night_drive_to_gym`
- added: high_energy_workout, rave_hard_energy
- removed: rain_night_drive, low_energy_quiet, solitude_melancholy
- expected: Gym sprint context should raise high-energy EDM and reduce rain/night/solitude explanations.

### `pilot_night_drive_to_explosive`
- added: high_energy_workout, rave_hard_energy
- removed: low_energy_quiet
- expected: Flipping not-too-loud to explosive/strong-beat should raise the EDM candidate and remove quietness as a negative reason.

### `pilot_study_to_emotional_lyrics`
- added: lyric_narrative
- removed: instrumental_focus, low_energy_quiet
- expected: Switching from study background to emotional listening should raise vocal-forward narrative lyrics and reduce low-distraction criteria.

### `pilot_study_to_no_vocals`
- added: instrumental_focus
- removed: chinese_vocal
- expected: A hard no-vocals constraint should override Chinese-vocal preference and raise the instrumental focus candidate.

### `pilot_party_warmup_to_rave`
- added: rave_hard_energy
- removed: party_warmup_groove
- expected: Moving from pre-party warm-up to 2am rave should raise hard techno and reduce warm-up groove as the main criterion.

### `pilot_party_warmup_to_low_presence`
- added: low_energy_quiet
- removed: party_warmup_groove
- expected: When everyone is tired and wants low-presence background music, soft lo-fi should rise and groove warm-up should lose weight.
