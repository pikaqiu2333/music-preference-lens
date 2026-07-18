# Qwen-Scope Song Entity Option-Letter Grounding Probe Plan

## Purpose

This probe scores only answer letters, avoiding the full-string length
and title-copy biases observed in the previous forced-choice run.

## Summary

- Total titles: 20
- Total prompt variants: 60
- Permutations per title: 3
- Known real titles: 6
- Invented controls: 6
- Free-generated titles: 8

## Variant Counts

| Group | Variants |
|---|---:|
| `free_generated` | 24 |
| `invented_control` | 18 |
| `known_real` | 18 |

## Titles

| Group | Item | Title | Expected type | Candidate types |
|---|---|---|---|---|
| `known_real` | `known_blinding_lights` | Blinding Lights | `correct_artist` | The Weeknd (correct_artist), Billie Eilish (distractor_artist), Queen (distractor_artist), Unknown (unknown) |
| `known_real` | `known_bad_guy` | bad guy | `correct_artist` | Billie Eilish (correct_artist), The Weeknd (distractor_artist), Ed Sheeran (distractor_artist), Unknown (unknown) |
| `known_real` | `known_shape_of_you` | Shape of You | `correct_artist` | Ed Sheeran (correct_artist), The Weeknd (distractor_artist), Queen (distractor_artist), Unknown (unknown) |
| `known_real` | `known_bohemian_rhapsody` | Bohemian Rhapsody | `correct_artist` | Queen (correct_artist), Ed Sheeran (distractor_artist), The Weeknd (distractor_artist), Unknown (unknown) |
| `known_real` | `known_dao_xiang` | 稻香 | `correct_artist` | 周杰伦 (correct_artist), Jay Chou (correct_artist), 陈奕迅 (distractor_artist), Unknown (unknown) |
| `known_real` | `known_qing_hua_ci` | 青花瓷 | `correct_artist` | 周杰伦 (correct_artist), Jay Chou (correct_artist), 林俊杰 (distractor_artist), Unknown (unknown) |
| `invented_control` | `invented_neon_rain_avenue` | Neon Rain Avenue | `unknown` | Unknown (unknown), The Weeknd (distractor_artist), Billie Eilish (distractor_artist), Neon Avenue Collective (fake_style_artist) |
| `invented_control` | `invented_soft_circuit_dream` | Soft Circuit Dream | `unknown` | Unknown (unknown), Ed Sheeran (distractor_artist), Queen (distractor_artist), Soft Circuit Ensemble (fake_style_artist) |
| `invented_control` | `invented_velvet_taxi_3am` | Velvet Taxi at 3AM | `unknown` | Unknown (unknown), Billie Eilish (distractor_artist), The Weeknd (distractor_artist), Velvet Taxi (fake_style_artist) |
| `invented_control` | `invented_paper_lantern_pulse` | Paper Lantern Pulse | `unknown` | Unknown (unknown), 周杰伦 (distractor_artist), 陈奕迅 (distractor_artist), Paper Lantern Project (fake_style_artist) |
| `invented_control` | `invented_white_noise_harbor` | White Noise Harbor | `unknown` | Unknown (unknown), Queen (distractor_artist), Ed Sheeran (distractor_artist), Harbor Tones (fake_style_artist) |
| `invented_control` | `invented_moonlit_circuit_waltz` | Moonlit Circuit Waltz | `unknown` | Unknown (unknown), The Weeknd (distractor_artist), 林俊杰 (distractor_artist), Moonlit Circuit (fake_style_artist) |
| `free_generated` | `free_city_lights` | City Lights | `unknown` | Unknown (unknown), The Weeknd (distractor_artist), Queen (distractor_artist), City Lights Collective (fake_style_artist) |
| `free_generated` | `free_rave_anthem` | Rave Anthem | `unknown` | Unknown (unknown), The Weeknd (distractor_artist), Billie Eilish (distractor_artist), Rave Anthem Crew (fake_style_artist) |
| `free_generated` | `free_hard_drive_groove` | Hard Drive Groove | `unknown` | Unknown (unknown), Ed Sheeran (distractor_artist), Queen (distractor_artist), Hard Drive Groove (fake_style_artist) |
| `free_generated` | `free_rave_revolution` | Rave Revolution | `unknown` | Unknown (unknown), The Weeknd (distractor_artist), 林俊杰 (distractor_artist), Rave Revolution (fake_style_artist) |
| `free_generated` | `free_sunset_glow` | Sunset Glow | `unknown` | Unknown (unknown), Ed Sheeran (distractor_artist), Billie Eilish (distractor_artist), Sunset Glow Project (fake_style_artist) |
| `free_generated` | `free_quiet_reflection` | Quiet Reflection | `unknown` | Unknown (unknown), Queen (distractor_artist), The Weeknd (distractor_artist), Quiet Reflection (fake_style_artist) |
| `free_generated` | `free_heartfelt_melody` | Heartfelt Melody | `unknown` | Unknown (unknown), 周杰伦 (distractor_artist), Ed Sheeran (distractor_artist), Heartfelt Melody (fake_style_artist) |
| `free_generated` | `free_whispers_of_the_moon` | Whispers of the Moon | `unknown` | Unknown (unknown), Billie Eilish (distractor_artist), 周杰伦 (distractor_artist), Moonlit Whispers (fake_style_artist) |
