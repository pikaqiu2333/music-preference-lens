# First Findings: Manual Baseline

Run:

- Results: `runs/manual_current_model_results.jsonl`
- Analysis: `reports/manual_current_model_analysis.md`
- Model label: `codex_current_manual`

## What Worked

The first three seed cases are strong enough to show counterfactual movement.
Across nine counterfactual variants, six changed the top choice.

The clearest effects came from:

- `constraint_flip`: explicit hard constraints changed the winner.
- `context_flip`: listening situation changes changed the winner.

This supports the core method: we can avoid asking whether one song is
objectively correct and instead test whether the model is sensitive to factors
it claims to use.

## What Was More Interesting

All three `preference_ablation` variants kept the original top choice.

That is not automatically a failure. In these cases, the top choice often had
multiple supporting factors. For example, a rain-night ambient track can remain
the best choice even after removing the user's explicit electronic preference,
because the track still matches rain, night, low energy, and solitude.

This means top-choice change is too coarse. We need to track:

- score deltas,
- factor shifts,
- whether removed preferences disappear from reasons,
- whether the model replaces one rationale with another grounded rationale.

## Early Failure Modes To Watch

1. **Reason substitution without score movement**
   The model can keep the same ranking while swapping explanations. This may be
   faithful if the item has multiple valid supporting factors, but suspicious if
   scores do not move at all.

2. **Over-justification**
   A model may create a rich vibe from weak evidence. Track cards need to make
   evidence strength explicit.

3. **Hard-constraint masking**
   If cases are too easy, the model will pass by following obvious tags. We need
   close-call candidates where multiple tracks partially match.

4. **Preference ablation ambiguity**
   Removing a preference should not always force a ranking flip. Some variants
   should expect a smaller score gap rather than a new winner.

## Next Case Design Rules

- Add close-call pairs where two tracks share context but differ on one factor.
- Add weak-evidence candidates to test hallucinated explanations.
- Add preference-ablation cases where the removed preference is decisive.
- Add subjective-plausibility cases where two rankings are both acceptable but
  reasons must stay grounded.

## Updated Research Framing

The project should not claim to score music recommendation quality directly.
It should claim to probe explanation faithfulness:

> Does the model's recommendation explanation move with the user/context factor
> it cites?

