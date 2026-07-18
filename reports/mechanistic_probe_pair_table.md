# Mechanistic Probe Pair Table

| Probe | Added | Removed | Expected Effect |
|---|---|---|---|
| `pilot_night_drive_to_gym` | `high_energy_workout`, `rave_hard_energy` | `rain_night_drive`, `low_energy_quiet`, `solitude_melancholy` | Gym sprint context should raise high-energy EDM and reduce rain/night/solitude explanations. |
| `pilot_night_drive_to_explosive` | `high_energy_workout`, `rave_hard_energy` | `low_energy_quiet` | Flipping not-too-loud to explosive/strong-beat should raise the EDM candidate and remove quietness as a negative reason. |
| `pilot_study_to_emotional_lyrics` | `lyric_narrative` | `instrumental_focus`, `low_energy_quiet` | Switching from study background to emotional listening should raise vocal-forward narrative lyrics and reduce low-distraction criteria. |
| `pilot_study_to_no_vocals` | `instrumental_focus` | `chinese_vocal` | A hard no-vocals constraint should override Chinese-vocal preference and raise the instrumental focus candidate. |
| `pilot_party_warmup_to_rave` | `rave_hard_energy` | `party_warmup_groove` | Moving from pre-party warm-up to 2am rave should raise hard techno and reduce warm-up groove as the main criterion. |
| `pilot_party_warmup_to_low_presence` | `low_energy_quiet` | `party_warmup_groove` | When everyone is tired and wants low-presence background music, soft lo-fi should rise and groove warm-up should lose weight. |
