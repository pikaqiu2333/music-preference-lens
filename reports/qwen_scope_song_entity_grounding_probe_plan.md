# Qwen-Scope Song Entity Grounding Probe Plan

## Purpose

This probe tests whether generated song titles behave more like known
song entities or deliberately invented title controls.

The prompt asks for a catalog-style artist continuation:

```text
Music catalog lookup.
Song title: "{{title}}"
Artist:
```

## Summary

- Total prompts: 20
- Known real controls: 6
- Invented controls: 6
- Free-generated titles: 8

| Group | Item | Title | Accepted Artists | Source |
|---|---|---|---|---|
| `known_real` | `known_blinding_lights` | Blinding Lights | The Weeknd | known_song_control |
| `known_real` | `known_bad_guy` | bad guy | Billie Eilish | known_song_control |
| `known_real` | `known_shape_of_you` | Shape of You | Ed Sheeran | known_song_control |
| `known_real` | `known_bohemian_rhapsody` | Bohemian Rhapsody | Queen | known_song_control |
| `known_real` | `known_dao_xiang` | 稻香 | 周杰伦, Jay Chou | known_song_control |
| `known_real` | `known_qing_hua_ci` | 青花瓷 | 周杰伦, Jay Chou | known_song_control |
| `invented_control` | `invented_neon_rain_avenue` | Neon Rain Avenue | - | hand_invented_control |
| `invented_control` | `invented_soft_circuit_dream` | Soft Circuit Dream | - | hand_invented_control |
| `invented_control` | `invented_velvet_taxi_3am` | Velvet Taxi at 3AM | - | hand_invented_control |
| `invented_control` | `invented_paper_lantern_pulse` | Paper Lantern Pulse | - | hand_invented_control |
| `invented_control` | `invented_white_noise_harbor` | White Noise Harbor | - | hand_invented_control |
| `invented_control` | `invented_moonlit_circuit_waltz` | Moonlit Circuit Waltz | - | hand_invented_control |
| `free_generated` | `free_city_lights` | City Lights | - | free_playlist_job_6a4f4e881fba25b8ea3b308f |
| `free_generated` | `free_rave_anthem` | Rave Anthem | - | free_playlist_job_6a4f4e881fba25b8ea3b308f |
| `free_generated` | `free_hard_drive_groove` | Hard Drive Groove | - | free_playlist_job_6a4f4e881fba25b8ea3b308f |
| `free_generated` | `free_rave_revolution` | Rave Revolution | - | free_playlist_job_6a4f4e881fba25b8ea3b308f |
| `free_generated` | `free_sunset_glow` | Sunset Glow | - | free_playlist_job_6a4f4e881fba25b8ea3b308f |
| `free_generated` | `free_quiet_reflection` | Quiet Reflection | - | free_playlist_job_6a4f4e881fba25b8ea3b308f |
| `free_generated` | `free_heartfelt_melody` | Heartfelt Melody | - | free_playlist_job_6a4f4e881fba25b8ea3b308f |
| `free_generated` | `free_whispers_of_the_moon` | Whispers of the Moon | - | free_playlist_job_6a4f4e881fba25b8ea3b308f |
