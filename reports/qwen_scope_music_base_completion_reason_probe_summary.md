# Qwen-Scope Music Base-Completion Reason Probe Summary

## Run Metadata

- Job ID: `6a4f273a1fba25b8ea3b2e17`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4f273a1fba25b8ea3b2e17
- Status: `complete`
- Base model: `Qwen/Qwen3-1.7B-Base`
- SAE repo: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Runtime: 93 seconds total on `t4-small`
- Prompt rows: 8 base-completion prompts

## Purpose

This run replaced the failed instruction/JSON generated-reason prompt with a
base-model-friendly continuation protocol:

```text
Best track: B
Reason:
```

The selected best track is provided by the probe. The model only needs to
continue a short recommendation reason. This isolates reason-token generation
instead of testing ranking quality.

## Main Result

This run succeeded.

The model generated meaningful recommendation reasons in all four matched
original/counterfactual pairs. More importantly, the answer-token SAE feature
deltas moved in the expected direction for every tested hypothesis, including
identity/source controls staying near zero.

## Pair Results

| Probe | Reason Shift | Key Feature | Expected | Prompt Delta | Answer Delta | Aligned |
|---|---|---|---|---:|---:|---|
| `base_high_energy_night_to_rave` | quiet night drive -> high energy / rave pressure | `feature32078` energy/activation | positive | `+29.978` | `+69.192` | Yes |
| `base_high_energy_night_to_rave` | quiet night drive -> high energy / rave pressure | `feature7884` rave/intensity | positive | `+64.729` | `+12.782` | Yes |
| `base_high_energy_night_to_rave` | control | `feature30224` identity control | near zero | `-0.035` | `0.000` | Yes |
| `base_low_presence_party_to_quiet` | rave pressure -> soft groove / low presence | `feature32078` energy/activation | negative | `-13.345` | `-17.912` | Yes |
| `base_low_presence_party_to_quiet` | rave pressure -> soft groove / low presence | `feature30584` quiet/soft-context | positive | `+2.770` | `+15.387` | Yes |
| `base_low_presence_party_to_quiet` | control | `feature19603` identity control | near zero | `+0.044` | `0.000` | Yes |
| `base_emotional_vocal_study_to_lyrics` | focus background -> emotional vocal / story | `feature8706` emotional-vocal/lyrics | positive | `+25.827` | `+18.356` | Yes |
| `base_emotional_vocal_study_to_lyrics` | focus background -> emotional vocal / story | `feature24979` focus/no-vocals | negative | `-12.771` | `-10.482` | Yes |
| `base_emotional_vocal_study_to_lyrics` | control | `feature22232` identity control | near zero | `-0.011` | `+0.148` | Yes |
| `base_no_vocals_lyrics_to_instrumental` | emotional vocal -> no vocals / white noise focus | `feature8706` emotional-vocal/lyrics | negative | `-18.415` | `-28.091` | Yes |
| `base_no_vocals_lyrics_to_instrumental` | emotional vocal -> no vocals / white noise focus | `feature24979` no-vocals/focus | positive | `+11.048` | `+13.047` | Yes |
| `base_no_vocals_lyrics_to_instrumental` | emotional vocal -> no vocals / white noise focus | `feature24916` instrumental/no-vocals | positive | `+7.703` | `+5.549` | Yes |

## Example Generated Reasons

### High Energy / Rave

Original reason:

> Matches user's preference for quiet night drive and calm urban mood.

Counterfactual reason:

> Matches the user's preference for high energy and rave pressure.

Matched expected factors:

- Original: `quiet night drive`, `restrained beat`, `calm urban mood`
- Counterfactual: `high energy`, `rave pressure`, `strong drums`,
  `fast activation`

Answer-token movement:

- `feature32078` energy/activation: `+69.192`
- `feature7884` rave/intensity: `+12.782`
- `feature30224` identity control: `0.000`

### No Vocals / Instrumental

Original reason:

> Best emotional match: Drama Hook's emotional story and dramatic chorus align
> with user's need for heartfelt vocal and lyrical story.

Counterfactual reason:

> Best fit for white noise focus: C has a neutral mood, minimal energy, and
> almost no melody, making it ideal for white noise focus.

Matched expected factors:

- Original: `heartfelt vocal`, `lyrical story`, `dramatic chorus`
- Counterfactual: `no vocals`, `instrumental texture`,
  `no lyric attention`, `white noise focus`

Answer-token movement:

- `feature8706` emotional-vocal/lyrics: `-28.091`
- `feature24979` no-vocals/focus: `+13.047`
- `feature24916` instrumental/no-vocals: `+5.549`

## Interpretation

This is the first clean generated-reason interpretability result in the
project:

> When the model is conditioned to produce a short reason, answer-token SAE
> features move with the music factor named in that reason.

The result is still not causal. We are not yet intervening on features or
showing that changing a feature changes the reason. But it is a strong
correlational alignment result under matched prompt pairs.

## Why This Matters

The previous generated-reason run failed because an instruction/JSON prompt
made the base model repeat prompt rules. This run shows the practical method
fix:

- Do not ask the base model to solve an instruction-following JSON task.
- Condition it with a short natural continuation.
- Measure only the tokens after `Reason:`.

This creates a viable path for explanation-faithfulness probes without a
benchmark or large human labels.

## Next Experiment

The next step should make the analysis more local:

1. Split generated reasons into phrase spans such as `high energy`,
   `rave pressure`, `heartfelt vocal`, and `no vocals`.
2. Measure feature activations on those generated phrase spans, not only the
   whole answer.
3. Add Chinese-language versions of the same four probes.
4. Keep identity/source features as negative controls.

Current conclusion:

> The base-completion protocol turns Music Preference Lens into a real
> generated-reason interpretability experiment. We now have prompt-side feature
> movement, generated reason text, and answer-token feature movement aligned in
> the same matched probes.
