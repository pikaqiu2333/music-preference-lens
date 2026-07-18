# Qwen-Scope Music Phrase-Level Probe Summary

## Run Metadata

- Job ID: `6a4f18571fba25b8ea3b2d48`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4f18571fba25b8ea3b2d48
- Status: `complete`
- Base model: `Qwen/Qwen3-1.7B-Base`
- SAE repo: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layer: 24
- Runtime: 197 seconds total on `t4-small`

## Design

This run extends the span-level music probe by measuring named phrase spans
inside matched recommendation prompts. The prompt profile and candidate cards
stay stable while the current listening need changes.

The probe set contains 12 matched prompts and 116 phrase spans across four
music recommendation dimensions:

- `high_energy_rave`
- `low_presence_quiet`
- `emotional_lyrics`
- `no_vocals_instrumental`

The goal is to reduce punctuation, tag-layout, and prompt-template artifacts by
aggregating directly over phrases such as `high energy`, `rave pressure`,
`quiet night drive`, `heartfelt vocal`, and `no vocals`.

## Main Finding

Phrase-level aggregation makes the candidate feature labels much cleaner than
token-level top activations alone.

The clearest music recommendation signals are:

| Dimension | Best Candidate Features | Interpretation |
|---|---|---|
| `high_energy_rave` | `layer24.feature7884`, `layer24.feature32078` | Strong positive movement on rave, hard techno, strong drums, fast activation, and peak-time dance-floor phrases. |
| `low_presence_quiet` | `layer24.feature32078` negative, `layer24.feature30584` positive | `feature32078` behaves like an energy/activation feature; `feature30584` rises for quiet, soft, low-presence contexts. |
| `emotional_lyrics` | `layer24.feature8706` | Strong positive movement on heartfelt vocal, emotional vocal, ballad, story-driven chorus, and dramatic chorus phrases. |
| `no_vocals_instrumental` | `layer24.feature24979`, `layer24.feature24916`, plus negative `feature8706` / `feature30584` | Focus/no-vocals constraints are visible, while vocal/emotional features move in the opposite direction. |

Identity/source-binding controls remain near zero, which supports the earlier
finding that model/app identity features are separable from music preference
features.

## Dimension Results

### High Energy / Rave

| Feature | Mean Delta | Mean Abs Delta | Note |
|---|---:|---:|---|
| `layer24.feature7884` | `+129.499` | `129.499` | Strongest and consistent positive movement across the three high-energy probes. |
| `layer24.feature32078` | `+94.143` | `94.143` | Also strong and consistent; likely a cleaner energy/activation candidate. |
| `layer24.feature27029` | `-8.756` | `37.936` | Moves, but direction is mixed; likely a broad confound. |
| `layer24.feature30584` | `-12.835` | `34.059` | Mixed movement; not a clean high-energy feature. |
| `layer24.feature8706` | `+4.731` | `29.327` | Mixed movement, with some overlap from dramatic or vocal phrasing. |

Visible phrase evidence for `feature7884` includes high activations on
`hard techno drive`, `strong drums`, `high energy`, `rave-like pressure`,
`fast activation`, and `peak-time dance floor`.

### Low Presence / Quiet

| Feature | Mean Delta | Mean Abs Delta | Note |
|---|---:|---:|---|
| `layer24.feature32078` | `-75.439` | `75.439` | Consistent negative movement; useful inverse energy/activation marker. |
| `layer24.feature30584` | `+63.669` | `63.669` | Strong positive movement for quiet, soft, low-presence needs. |
| `layer24.feature7884` | `-19.815` | `134.485` | Large but mixed; too broad to label as low-presence. |
| `layer24.feature27029` | `+38.444` | `38.444` | Moves broadly; keep as a confound candidate. |

This is the strongest evidence that the project should use paired feature
directions rather than one feature per product label. For example,
`feature32078` down plus `feature30584` up is more informative than either
feature alone.

### Emotional Lyrics

| Feature | Mean Delta | Mean Abs Delta | Note |
|---|---:|---:|---|
| `layer24.feature8706` | `+50.896` | `50.896` | Strong and consistent positive movement; best emotional/vocal candidate. |
| `layer24.feature24979` | `-25.447` | `25.447` | Consistent negative movement; likely more aligned with focus/no-vocals than emotional lyrics. |
| `layer24.feature30584` | `+12.433` | `14.296` | Modest positive movement, but less clean than `feature8706`. |

Visible phrase evidence for `feature8706` includes high activations on
`heartfelt vocal`, `emotional vocal`, `late-night ballad listening`,
`story-driven chorus`, and `dramatic chorus`.

### No Vocals / Instrumental

| Feature | Mean Delta | Mean Abs Delta | Note |
|---|---:|---:|---|
| `layer24.feature30584` | `-82.878` | `82.878` | Strong consistent negative movement, inverse of soft/vocal contexts. |
| `layer24.feature8706` | `-77.034` | `77.034` | Strong consistent negative movement, inverse of emotional/vocal contexts. |
| `layer24.feature24979` | `+25.381` | `25.381` | Positive movement for no-vocals and writing/focus constraints. |
| `layer24.feature24916` | `+6.802` | `6.802` | Smaller but consistent no-vocals/instrumental candidate. |

Visible phrase evidence for `feature24979` includes `strict writing mode`,
`no vocals`, `no lyric attention`, and `deep focus writing`. `feature24916`
also fires on `no vocals`, `white noise focus`, `instrumental texture`, and
`white-noise-like background`, but its magnitude is smaller.

## Updated Feature Labels

| Feature | Working Label | Confidence | Why |
|---|---|---|---|
| `layer24.feature7884` | high-energy / rave / intensity | Medium | Very strong on rave and hard techno phrases, but still broad in quiet probes. |
| `layer24.feature32078` | energy / activation | High | Positive for high-energy phrases and consistently negative for quiet/low-presence phrases. |
| `layer24.feature30584` | soft / quiet / vocal-context inverse no-vocals | Medium | Positive for low-presence and some emotional contexts; negative for no-vocals. |
| `layer24.feature8706` | emotional vocal / lyric salience | High | Strong positive on emotional-lyrics phrases and strong negative on no-vocals. |
| `layer24.feature24979` | no-vocals / focus / writing constraint | Medium | Positive for no-vocals and strict writing mode; negative for emotional lyrics. |
| `layer24.feature24916` | instrumental / no-vocals constraint | Low-Medium | Direction is useful but magnitude is smaller. |
| `layer24.feature27029` | broad prompt/music confound | Low | Moves across many phrase types without a clean direction. |
| `layer24.feature19603`, `layer24.feature30224`, `layer24.feature22232` | identity/source controls | High as controls | Near-zero movement on music phrase dimensions. |

## Caveats

This is still prompt-side evidence, not a causal claim about generated
recommendations. The current run shows that the base model plus SAE contains
separable internal signals for music context phrases before the model answers.

The prompts are English and hand-written. Chinese music recommendation probes
are needed before claiming fit to a Chinese music-product setting.

`Qwen/Qwen3-1.7B-Base` is a base model, not an instruction-tuned recommender.
That is acceptable for feature discovery, but generated-answer faithfulness
should be tested separately.

## Next Experiment

The next useful experiment is generated-answer probing:

1. Ask the model to rank candidates and write recommendation reasons.
2. Track whether `feature32078`, `feature8706`, `feature24979`, and
   `feature24916` move during the generated explanation itself.
3. Compare generated reason text with prompt-side feature movement:
   - If the reason says "high energy", do energy features move?
   - If the reason says "lyrics" or "vocal emotion", does `feature8706` move?
   - If the reason says "focus" or "no vocals", do `feature24979` and
     `feature24916` move while `feature8706` drops?
4. Keep identity/source features as negative controls.

Current conclusion:

> The clearest first contribution is a matched phrase-level SAE probe for
> subjective LLM4Rec explanations. It does not require large human labels, and
> it produces product-readable dimensions that can be connected to generated
> recommendation reasons in the next stage.
