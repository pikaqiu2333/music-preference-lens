# Qwen-Scope Matched Music Recommendation Probe Summary

## Run Metadata

- Job ID: `6a4e54bb1499512f23779bf2`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4e54bb1499512f23779bf2
- Status: `complete`
- Base model: `Qwen/Qwen3-1.7B-Base`
- SAE repo: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Probes: 6 matched music recommendation counterfactual pairs
- Runtime: 189 seconds total on `t4-small`

## Why This Run Matters

The previous music smoke test was dominated by prompt-length and candidate-card
omission effects. This matched run repeats the same candidate cards on both
sides and changes only the user context or constraint.

Token counts are now close:

| Probe | Original Tokens | Counterfactual Tokens |
|---|---:|---:|
| `matched_night_drive_to_gym` | 185 | 185 |
| `matched_night_drive_to_explosive` | 191 | 196 |
| `matched_study_to_emotional_lyrics` | 194 | 196 |
| `matched_study_to_no_vocals` | 198 | 201 |
| `matched_party_warmup_to_rave` | 195 | 198 |
| `matched_party_warmup_to_low_presence` | 195 | 201 |

This makes the deltas more interpretable than the first smoke run.

## Main Finding

After matching, huge artifact deltas disappear. The remaining movement is
smaller but more meaningful: several layer 24 features react to changes in
listening context, energy, lyric/vocal constraints, and party intensity.

The previous identity/source-binding features still do not move:

| Feature | Mean Abs Delta | Interpretation |
|---|---:|---|
| `layer24.feature19603` | `0.006` | No music-recommendation movement. |
| `layer24.feature30224` | `0.064` | No meaningful music-recommendation movement. |
| `layer24.feature22232` | `0.000` | No music-recommendation movement. |

This strengthens the separation between source/role identity features and music
recommendation features.

## Candidate Music Features

| Feature | Mean Abs Delta | Notable Pattern | Current Interpretation |
|---|---:|---|---|
| `layer24.feature7884` | `10.837` | Large positive in party-warmup to rave (`+42.171`) and positive in night-drive to explosive (`+10.117`). | Candidate for stronger energy / harder dance / intensity context. |
| `layer24.feature30584` | `9.063` | Positive in study to emotional lyrics (`+22.007`), negative in night-drive to gym (`-12.262`) and study to no-vocals (`-6.151`). | Music-context feature, possibly soft/vocal/emotional listening rather than generic catalog. |
| `layer24.feature27029` | `12.515` | Still recurrent in 4 / 6 top-delta rows. | Broad prompt/task feature; keep as confound control, not a music label. |
| `layer24.feature32078` | `6.350` | Positive in night-drive to gym/explosive and party-rave, negative in low-presence. | Candidate for activation/energy shift, needs more controls. |
| `layer24.feature8706` | `6.776` | Positive in emotional-lyrics, negative in no-vocals and night-drive-to-gym. | Candidate for vocal/emotional or text-rich listening context. |
| `layer24.feature24916` | `2.856` | Strong positive in study to no-vocals (`+12.215`). | Candidate for hard constraint / instrumental-focus shift. |
| `layer24.feature24979` | `4.575` | Negative in emotional lyrics (`-12.363`), positive in no-vocals (`+8.879`). | Broad evidence/schema feature; useful but not a clean music dimension yet. |
| `layer24.feature27497` | `5.201` | Mostly negative except no-vocals. | Contextual prompt feature; needs token-position inspection. |

## What Changed From The Smoke Test

The first music run made `layer24.feature30584` look huge because the
counterfactual prompts were short and omitted candidate-card details.

In the matched run:

- `feature30584` remains active but no longer dominates every probe.
- `feature7884`, `feature32078`, `feature8706`, and `feature24916` become more
  interesting.
- Artifact features such as `feature27029` still show up, but at much smaller
  scale and can be treated as controls.

This is a healthier result: less spectacular, more believable.

## Research Implication

Music recommendation can connect to the interpretability work, but the useful
question should be narrow:

> When a music recommender changes its explanation from "rainy quiet night
> drive" to "rave energy", "emotional lyrics", or "no vocals", which internal
> features move with that explanation?

That is more defensible than trying to explain subjective recommendation
quality directly.

## Next Experiment

The next run should move from max activation to token-position analysis:

1. Track candidate features over token positions for the same matched prompts.
2. Mark spans for profile, current need, candidate tags, evidence, and task.
3. Check whether `feature7884` fires on rave/intensity words, `feature30584` on
   vocal/emotional/music-card words, and `feature24916` on no-vocal or
   instrumental constraints.
4. Add one or two Chinese-language matched probe pairs after fixing the current
   mojibake in `data/seed_cases.jsonl`.
5. Only after span-level evidence should we attempt intervention or causal
   claims.

Current conclusion:

> The matched music recommendation probe is viable. The first contribution can
> be a small, reproducible workflow for turning subjective music-recommendation
> explanation changes into matched SAE probes, with identity/source features as
> negative controls.

Follow-up: the span-level run in
`qwen_scope_music_span_probe_summary.md` shows that the strongest feature
movement is concentrated in the `current_need` span, especially for
`layer24.feature7884`, `layer24.feature30584`, `layer24.feature8706`,
`layer24.feature32078`, and `layer24.feature24916`.
