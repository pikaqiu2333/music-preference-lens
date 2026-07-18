# Song Entity Layerwise Attribution Probe

## Motivation

The matched relation-binding experiment showed that Qwen3-1.7B-Base assigns
higher prior-corrected probability to the correct artist for 10 of 12 smoke
relations. A supervised layer-24 SAE direction reached only 7 of 12 under
block-held-out evaluation. The next experiment therefore removes the trained
readout and asks where the model's own output preference becomes visible.

## Questions

1. At which layer does the correct title-artist relation become decodable by
   the model's own final normalization and unembedding matrix?
2. At which layers can a neutral-title residual state replace the real-title
   state and causally reduce the correct-artist advantage?
3. Does relation information reappear after an early patch because later
   layers can retrieve it again from the title tokens?

## Layerwise Target-Logit Lens

For every real title and neutral title, teacher-force the correct and swapped
artist continuations. At each hidden-state depth, apply the model's final RMS
normalization and score only the observed continuation token through its
unembedding row. No classifier or feature selection is trained.

For layer `l`, the relation margin is:

`[score_l(correct | title) - score_l(correct | neutral)] -`
`[score_l(wrong | title) - score_l(wrong | neutral)]`

The final behavioral gate still uses full model log probabilities, not the
target-logit lens approximation.

## Activation Patching

Use a prefix ending immediately after `Artist:`. For each transformer layer,
replace the real-title residual at the final prefix position with the residual
from the neutral-title prefix at the corresponding layer. Continue the forward
pass and measure the correct-first-token minus wrong-first-token logit margin.

Early layers may recover after patching because later positions can still
attend to the untouched real title. Late-layer patches test whether relation
information has become concentrated at the prediction position.

## Registered Checks

- Technical coverage: all selected relations have complete layer and patch
  curves, and every correct/wrong artist has a different first continuation
  token.
- Final-logit consistency: the last target-logit lens value agrees with the
  model's actual target logit within `0.02`.
- Patch endpoint consistency: replacing the final transformer-layer state with
  the neutral state reproduces the neutral first-token margin within `0.02`.
- Behavioral gate: final prior-corrected sequence log-prob accuracy is at least
  `0.80`.
- Interpretation gate: all technical checks and the behavioral gate pass.

Layer localization is descriptive. The earliest sustained layer is the first
depth with at least `0.75` relation accuracy for three consecutive depths. It
is not used to rescue a failed interpretation gate.

## Scope

The smoke run reuses six independent swap blocks and 12 relations. No human
recommendation labels, benchmark expansion, or SAE intervention is involved.
