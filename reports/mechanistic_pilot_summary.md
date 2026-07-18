# Mechanistic Pilot Summary

The project is now interpretability-first.

The behavioral recommendation task is used to create controlled contrasts. The
main question is whether an open-weight model's internal signals move with the
same factors that appear in its recommendation explanations.

## Pilot Set

Use:

`data/mechanistic_pilot_specs.jsonl`

The pilot has six clean contrast probes:

| Probe | Added Dimensions | Removed Dimensions | Expected Candidate Movement |
|---|---|---|---|
| `pilot_night_drive_to_gym` | `high_energy_workout`, `rave_hard_energy` | `rain_night_drive`, `low_energy_quiet`, `solitude_melancholy` | EDM up, ambient/night-drive down |
| `pilot_night_drive_to_explosive` | `high_energy_workout`, `rave_hard_energy` | `low_energy_quiet` | high-energy EDM up |
| `pilot_study_to_emotional_lyrics` | `lyric_narrative` | `instrumental_focus`, `low_energy_quiet` | dramatic vocal track up |
| `pilot_study_to_no_vocals` | `instrumental_focus` | `chinese_vocal` | instrumental focus track up |
| `pilot_party_warmup_to_rave` | `rave_hard_energy` | `party_warmup_groove` | hard techno up |
| `pilot_party_warmup_to_low_presence` | `low_energy_quiet` | `party_warmup_groove` | soft lo-fi up |

## Mechanistic Readout

For each probe, compare original and counterfactual prompts.

Record:

- target SAE features or feature groups,
- activation deltas,
- top activated features,
- whether cited explanation factors align with moved features,
- whether candidate score movement aligns with feature movement.

## First Model Recommendation

Start with Gemma 3 270M-IT + Gemma Scope 2 as a smoke test, then Gemma 3 1B-IT
for a stronger pilot. Move to Qwen3-1.7B-Base or Qwen3-8B-Base + Qwen-Scope
after the measurement workflow is stable, because Qwen is more relevant for
Chinese music/product phrasing.

## Key Risk

Feature labels are not ground truth. The pilot should report feature evidence
as tentative and triangulate with behavior, counterfactuals, and human product
judgment.

## Good First Result

A good first result is not "the model understands music taste." A good first
result is:

> Context/constraint flips cause both behavior changes and interpretable feature
> movement in the expected direction for several clean music recommendation
> probes.
