# Qwen-Scope Music Generated-Reason Probe Summary

## Run Metadata

- Job ID: `6a4f1db41fba25b8ea3b2dae`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4f1db41fba25b8ea3b2dae
- Status: `complete`
- Base model: `Qwen/Qwen3-1.7B-Base`
- SAE repo: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Runtime: 263 seconds total on `t4-small`
- Prompt rows: 8 generated-reason prompts

## Purpose

This run attempted to move from prompt-side phrase features to generated-answer
feature alignment. The intended question was:

> When the model writes a recommendation reason, do answer-token SAE features
> align with the music factors named in that reason?

## Main Result

This run is a useful negative control, not a valid generated-reason alignment
result.

`Qwen/Qwen3-1.7B-Base` did not follow the recommendation task. Across all eight
prompt rows, the generated text repeated the prompt rule:

> If evidence is strong, name the uncertainty instead of adding a vivid story.

No generated answer matched the expected music reason factors such as
`high energy`, `low presence`, `heartfelt vocal`, or `no vocals`.

Because the model did not generate recommendation reasons, answer-token feature
deltas mostly reflect prompt-copying artifacts rather than music-reason
alignment.

## What Still Worked

The prompt-side feature deltas still recover several expected directions:

| Probe | Feature | Expected | Prompt Delta | Note |
|---|---|---|---:|---|
| `gen_high_energy_night_to_rave` | `layer24.feature32078` | positive | `+12.866` | Energy/activation rises for rave need. |
| `gen_high_energy_night_to_rave` | `layer24.feature7884` | positive | `+26.914` | Rave/intensity rises for rave need. |
| `gen_low_presence_party_to_quiet` | `layer24.feature32078` | negative | `-3.966` | Energy feature falls for quieter context. |
| `gen_low_presence_party_to_quiet` | `layer24.feature30584` | positive | `+2.423` | Quiet/soft-context feature rises, but weakly. |
| `gen_emotional_vocal_study_to_lyrics` | `layer24.feature8706` | positive | `+12.802` | Emotional-vocal feature rises for lyric listening. |
| `gen_emotional_vocal_study_to_lyrics` | `layer24.feature24979` | negative | `-6.345` | Focus/no-vocals feature falls for emotional lyrics. |
| `gen_no_vocals_lyrics_to_instrumental` | `layer24.feature8706` | negative | `-12.934` | Emotional-vocal feature falls for no-vocals writing mode. |
| `gen_no_vocals_lyrics_to_instrumental` | `layer24.feature24979` | positive | `+6.536` | Focus/no-vocals feature rises for writing mode. |
| `gen_no_vocals_lyrics_to_instrumental` | `layer24.feature24916` | positive | `+3.738` | Instrumental/no-vocals feature rises, but weakly. |

This confirms the earlier phrase-level result: Qwen-Scope features are useful
for prompt-side music context contrasts.

## What Failed

Answer-side deltas do not support recommendation-reason alignment:

| Probe | Feature | Expected | Answer Delta | Aligned? |
|---|---|---|---:|---|
| `gen_high_energy_night_to_rave` | `layer24.feature32078` | positive | `+0.006` | No |
| `gen_high_energy_night_to_rave` | `layer24.feature7884` | positive | `-0.015` | No |
| `gen_low_presence_party_to_quiet` | `layer24.feature32078` | negative | `-0.104` | Below threshold |
| `gen_low_presence_party_to_quiet` | `layer24.feature30584` | positive | `-0.392` | No |
| `gen_emotional_vocal_study_to_lyrics` | `layer24.feature8706` | positive | `+0.200` | Below threshold |
| `gen_emotional_vocal_study_to_lyrics` | `layer24.feature24979` | negative | `-0.718` | Yes |
| `gen_no_vocals_lyrics_to_instrumental` | `layer24.feature8706` | negative | `-0.344` | Yes |
| `gen_no_vocals_lyrics_to_instrumental` | `layer24.feature24979` | positive | `+0.143` | Below threshold |
| `gen_no_vocals_lyrics_to_instrumental` | `layer24.feature24916` | positive | `+0.126` | Below threshold |

These values should not be interpreted as recommendation-reason evidence,
because the answer text was not a recommendation answer.

Top answer-token activations were dominated by copied or repeated tokens such
as `If`, `vivid`, `-`, punctuation, and common words. That is a generation
artifact, not a music semantic signal.

## Interpretation

The failure is methodological rather than fatal:

- Qwen-Scope gives us useful feature probes on `Qwen3-1.7B-Base`.
- But `Qwen3-1.7B-Base` is not instruction-tuned and is weak at following the
  generated-reason JSON task.
- Therefore, generated-answer interpretability needs a generation setup that
  produces genuine recommendation reasons before answer-token alignment can be
  meaningful.

## Next Method Change

The next run should not repeat this exact prompt. Better options:

1. Use a base-model continuation format instead of an instruction/JSON task.
   For example, end the prompt with `Recommendation reason:` and measure a short
   free-form continuation.
2. Remove the Markdown JSON schema block, because the base model continued the
   instruction text instead of solving the task.
3. Add one compact in-context example that shows the desired completion style.
4. Use short, phrase-rich answer prefixes such as:
   - `Best track:`
   - `Reason:`
   - `Because the user wants`
5. Keep generated text short, then measure only tokens after `Reason:`.

The research direction remains valid, but the generated-answer step must be
adapted to base-model continuation behavior.

Current conclusion:

> Prompt-side interpretability is working. Generated-reason interpretability
> requires a base-model-friendly completion protocol before we can make any
> claim about answer-token faithfulness.
