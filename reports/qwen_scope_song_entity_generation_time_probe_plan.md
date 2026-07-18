# Qwen-Scope Song Entity Generation-Time Probe Plan

## Summary

- Contexts: 4
- Seeds: 17, 29, 43
- Playlist generations: 12
- Requested tracks per generation: 5
- Known exact controls: 20
- Artist-mismatch controls: 10
- Synthetic controls: 10
- Primary internal signal: layer-24 pair-end SAE activation
- Verification conditions: neutral and self-attributed

## Contexts

| Context | Dimension | Current need |
|---|---|---|
| `quiet_night_drive` | `quiet_night_drive` | rainy night driving; wants quiet night drive, restrained beat, calm urban mood, and not too loud. |
| `peak_time_rave` | `high_energy_rave` | 2am peak-time dance floor; wants harder energy, rave-like pressure, intense drums, and maximum momentum. |
| `emotional_vocal` | `emotional_lyrics` | late-night emotional listening; wants heartfelt vocal, lyrical story, dramatic chorus, and clear emotional release. |
| `strict_no_vocals` | `no_vocals_instrumental` | strict writing mode; wants no vocals, instrumental texture, no lyric attention, and white noise focus. |

## Interpretation Gate

The full run is interpreted only if control balanced accuracy is at
least 0.80, at least 10 of 12 generations are valid, and at least
50 complete title-artist pairs are parsed.
