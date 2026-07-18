# Experiment Log Template

Use this file as a template for each model run.

## Run Metadata

- Date:
- Model:
- Provider:
- Temperature:
- Top-p:
- Prompt pack:
- Notes:

## Run Goal

Example:

Test whether the model changes rankings and reasons under constraint/context
counterfactuals for the first three seed cases.

## Observations

### Case: `case_id`

- Original top choice:
- Counterfactual top choice:
- Did ranking change as expected?
- Did cited factors change as expected?
- Any unsupported item claims?
- Any over-justification?

## Failure Taxonomy

Mark observed failures:

- `constraint_ignored`
- `context_ignored`
- `preference_ignored`
- `reason_ranking_mismatch`
- `unsupported_item_claim`
- `generic_reason`
- `over_safe_recommendation`
- `over_novel_recommendation`
- `unknown_song_hallucination`

## Takeaways

- What seems robust?
- What seems brittle?
- Which cases should be rewritten?
- Which cases deserve mechanistic probing later?

