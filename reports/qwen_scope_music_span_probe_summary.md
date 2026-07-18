# Qwen-Scope Music Span-Level Probe Summary

## Run Metadata

- Job ID: `6a4ee2a01499512f23779ff4`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4ee2a01499512f23779ff4
- Status: `complete`
- Base model: `Qwen/Qwen3-1.7B-Base`
- SAE repo: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Runtime: 189 seconds total on `t4-small`

## Design

This run split each matched music recommendation prompt into four spans:

- `profile`
- `current_need`
- `candidate_cards`
- `task`

It then measured selected layer 24 feature activations by span. The goal was to
test whether candidate features moved on the user need itself, or merely on
candidate cards, task formatting, or source/identity controls.

## Main Finding

The useful movement is concentrated in `current_need`, not `profile`.

That is exactly what we wanted from a music recommendation faithfulness probe:
the profile and candidate cards are mostly held constant, while the listening
need changes.

## Strongest Span Deltas

| Feature | Span | Mean Abs Delta | Mean Signed Delta | Interpretation |
|---|---|---:|---:|---|
| `layer24.feature7884` | `current_need` | `55.763` | `+19.746` | Strongest candidate for context/energy/intensity shifts. |
| `layer24.feature30584` | `current_need` | `36.187` | `+0.403` | Responds strongly but direction depends on music context. |
| `layer24.feature8706` | `current_need` | `29.427` | `+4.008` | Candidate for vocal/emotional/text-rich listening context. |
| `layer24.feature32078` | `current_need` | `18.738` | `+5.794` | Candidate for activation/energy shifts. |
| `layer24.feature27029` | `current_need` | `18.216` | `-3.119` | Still a broad confound/control feature. |
| `layer24.feature24979` | `current_need` | `9.828` | `+1.960` | Broad schema/evidence feature. |
| `layer24.feature24916` | `current_need` | `5.342` | `+5.342` | Candidate for no-vocals / hard-constraint shifts. |

The `profile` span has zero delta across all features because it is held
constant. This is a useful sanity check.

## Feature Notes

### `layer24.feature7884`

Most promising for high-energy or intensity changes.

- `party_to_rave`: `current_need +144.999`, `candidate_cards +42.554`
- `night_to_explosive`: `current_need +16.352`, `candidate_cards +19.761`
- `night_to_gym`: `current_need -108.052`

Interpretation: this feature is sensitive to the need-level context shift, but
also has high baseline activations on candidate-card structure such as tags and
punctuation. It should be treated as a context/intensity candidate, not yet a
clean semantic label.

### `layer24.feature30584`

Still music-related, but not a single simple dimension.

- `study_to_lyrics`: `current_need +91.360`, `candidate_cards +18.541`
- `night_to_gym`: `current_need -96.139`
- `study_to_no_vocals`: `current_need +2.279`, `candidate_cards -20.084`

Interpretation: likely soft/vocal/emotional/music-context related. It moves
strongly when the prompt asks for emotional Chinese lyrics, and drops when
rainy-night listening becomes gym training.

### `layer24.feature8706`

Good secondary candidate for vocal/emotional listening.

- `study_to_lyrics`: `current_need +52.409`, `candidate_cards +27.083`
- `study_to_no_vocals`: `current_need -27.867`, `candidate_cards -19.240`
- `night_to_gym`: `current_need -48.211`

Interpretation: more aligned with emotional/vocal/text-rich music than with
instrumental or gym contexts.

### `layer24.feature32078`

Likely energy/activation related.

- `night_to_explosive`: `current_need +36.011`
- `party_to_rave`: `current_need +26.967`
- `party_to_low_presence`: `current_need -32.455`

Interpretation: promising for high-energy versus low-presence context.

### `layer24.feature24916`

Useful for hard constraints.

- `study_to_no_vocals`: `current_need +23.086`, `candidate_cards +12.463`

Interpretation: candidate for no-vocals / instrumental-focus constraint, but
we need more no-vocal and vocal-control pairs.

## Negative Controls

Identity/source-binding features remain essentially inactive:

| Feature | Largest Useful Span Movement |
|---|---:|
| `layer24.feature19603` | `current_need mean abs delta 0.296` |
| `layer24.feature30224` | `candidate_cards mean abs delta 0.039` |
| `layer24.feature22232` | `0.000` on `current_need` and `candidate_cards` |

This strengthens the earlier conclusion: source/role identity features and
music recommendation features are separable.

## Caution

Raw top-token activations are noisy. Several high activations occur on
punctuation, `tags`, or separators. This does not invalidate the span result,
but it means we should not label features from top tokens alone.

Span deltas are more useful here because they compare the same structural
region across matched prompts.

## Updated Research Direction

The strongest contribution path is now:

> Build matched music recommendation probes where only `current_need` changes,
> then use SAE span deltas to test whether explanation-relevant dimensions
> appear inside the prompt before the model writes its recommendation.

This avoids subjective human labels while still connecting to recommendation
faithfulness.

## Next Experiment

1. Add 2-3 more matched pairs for each candidate dimension:
   - rave / high energy
   - quiet / low presence
   - emotional lyrics
   - no vocals / instrumental
2. Normalize away punctuation and tag-layout artifacts by aggregating over
   phrase spans such as `high energy`, `rave`, `no vocals`, `narrative lyrics`.
3. Track candidate features during generated answers, not only prompts, to test
   whether internal feature movement predicts explanation text.
4. Use identity/source features as negative controls in every run.

## Follow-Up Completed

The phrase-level follow-up has been run and summarized in
`reports/qwen_scope_music_phrase_probe_summary.md`. It confirms cleaner feature
labels for high-energy/rave, low-presence/quiet, emotional-vocal, and
no-vocals/instrumental dimensions.

Current conclusion:

> The music recommendation project is viable. The clearest early claim is not
> "we explain music taste," but "we can localize feature movement for
> explanation-relevant music context changes under matched recommendation
> prompts."
