# Qwen-Scope Music Recommendation Probe Summary

## Run Metadata

- Job ID: `6a4e537b1fba25b8ea3b2043`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4e537b1fba25b8ea3b2043
- Status: `complete`
- Base model: `Qwen/Qwen3-1.7B-Base`
- SAE repo: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layers: 18, 24
- Probes: 6 music recommendation counterfactual pairs
- Runtime: 71 seconds total on `t4-small`

## Probe Design

This smoke test reused `data/mechanistic_pilot_specs.jsonl`:

- rainy night drive to gym/explosive energy
- study focus to emotional lyric listening/no vocals
- party warm-up to rave/low-presence background

The job measured SAE feature activation deltas between original and
counterfactual prompts, then compared those deltas with identity/source-binding
candidates from the previous run.

## Important Caveat

This first music run is not length matched.

The original prompts contain richer user profile, context, and candidate-card
details. The counterfactual prompts are much shorter and often say "candidate
cards stay the same" instead of repeating the cards. For example:

| Probe | Original Tokens | Counterfactual Tokens |
|---|---:|---:|
| `pilot_night_drive_to_gym` | 73 | 22 |
| `pilot_study_to_emotional_lyrics` | 55 | 28 |
| `pilot_party_warmup_to_rave` | 56 | 29 |

So the largest deltas should be treated as smoke-test signals, not final music
preference interpretations.

## Main Findings

### Identity/Source Features Do Not Carry Over

The three strongest source/role/app-context candidates from the identity run
were nearly inactive in music recommendation probes:

| Feature | Mean Abs Delta |
|---|---:|
| `layer24.feature19603` | `0.502` |
| `layer24.feature30224` | `0.000` |
| `layer24.feature22232` | `0.000` |

This is useful: the identity/source-binding features are not simply firing on
any complex prompt. They appear more specific to source hierarchy, app metadata,
and role identity than to music recommendation.

### Music/Card-Context Features Move Strongly

| Feature | Count In Top Deltas | Mean Signed Delta | Interpretation |
|---|---:|---:|---|
| `layer24.feature30584` | 6 / 6 | `-132.252` | Strong music/catalog/candidate-card feature. It drops when the counterfactual omits detailed candidate cards. |
| `layer24.feature27029` | 6 / 6 | `+216.970` | Short/generic prompt artifact; reject as music-specific. |
| `layer24.feature7884` | 5 / 6 | `-257.332` | Candidate for rich original recommendation-card context; needs matched controls. |
| `layer24.feature27497` | 4 / 6 | `-79.265` | Likely rich context/app/model-card-ish confound; not clean yet. |
| `layer24.feature24979` | target feature | `+23.409` | Broad schema/evidence integration; may help with explanation structure. |
| `layer18.feature29473` | 6 / 6 | `+125.363` | Previously identified short-prompt artifact; reject. |
| `layer18.feature13004` | 6 / 6 | `+84.741` | Same short-prompt artifact family; reject. |

## Interpretation

This run gives two useful signals:

1. Source/identity features and music recommendation features separate cleanly.
   `19603`, `30224`, and `22232` do not explain music preference movement.
2. `layer24.feature30584` is still the strongest music-related candidate, but
   this run suggests it may represent rich music/catalog/candidate-card context
   rather than a specific preference dimension like "night drive" or "rave".

The first result is stronger than the second. The second needs a better prompt
design before we can claim anything about preference faithfulness.

## Next Experiment

The next music run should be strictly matched:

1. Repeat the full candidate cards in both original and counterfactual prompts.
2. Keep token counts close by changing only the listening context or constraint.
3. Track target features across token positions, especially user context,
   candidate tags, evidence fields, and final instruction.
4. Add feature targets:
   - `layer24.feature30584`
   - `layer24.feature7884`
   - `layer24.feature24979`
   - `layer24.feature27497`
5. Keep source/identity controls:
   - `layer24.feature19603`
   - `layer24.feature30224`
   - `layer24.feature22232`

Current conclusion:

> The music recommendation direction is worth trying, but the first clean
> contribution should use matched candidate-card prompts. The strongest current
> music candidate is `layer24.feature30584`, while the previous identity/source
> candidates appear cleanly separated from music recommendation behavior.

Follow-up: the matched candidate-card version was run in
`qwen_scope_matched_music_recommendation_probe_summary.md`. It confirms that
the first smoke run was length-confounded and narrows the next candidates to
`layer24.feature7884`, `layer24.feature30584`, `layer24.feature32078`,
`layer24.feature8706`, and `layer24.feature24916`.
