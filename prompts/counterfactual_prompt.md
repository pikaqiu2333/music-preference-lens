# Counterfactual Variant Template

Create a counterfactual version of the recommendation case by changing one
factor while keeping the rest stable.

Allowed variant types:

- `constraint_flip`: invert an explicit constraint.
- `context_flip`: change the listening situation.
- `preference_ablation`: remove one stable preference from the user profile.
- `item_evidence_flip`: change one candidate's evidence card.

The counterfactual should make a faithful recommender adjust either its ranking,
its reasons, or both.

Return JSON with:

```json
{
  "variant_id": "string",
  "variant_type": "constraint_flip",
  "changed_fields": ["current_context"],
  "expected_effect": "short explanation",
  "case": {}
}
```

