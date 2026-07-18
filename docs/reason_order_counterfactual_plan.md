# Recommendation Reason-Order Counterfactual Plan

## Question

Does forcing the model to state a recommendation reason before the title and
artist change what it recommends, compared with the ordinary
title-artist-reason order?

This is an output-order intervention. It does not assume that visible reasons
are faithful internal plans.

## Matched Conditions

- `pair_first`: title, artist, then reason.
- `reason_first`: reason, then title and artist.

Everything else is held fixed: model, user profile, listening need, sampling
parameters, seed, requested list length, and requirement to name real tracks.

The smoke uses two contexts (`emotional_vocal` and `strict_no_vocals`) and two
seeds (17 and 29), for eight generations and up to 40 recommendation rows.

## Measurements

- Parse completion rate and actual field-order compliance.
- Exact title-artist overlap between matched lists.
- Catalog labels for every generated title-artist pair.
- Follow-up counterfactual need sensitivity for pairs from each order.

Catalog verification measures entity validity, not whether a real song has the
claimed musical attributes. The no-vocals constraint therefore remains a
separate analysis.

## Registered Technical Gate

- All eight matched generations return at least three complete items.
- At least 32 complete recommendation rows are parsed.
- At least 80% of parsed rows follow their assigned field order.
- Every saved row has nonempty reason, title, and artist fields.

No substantive superiority claim is registered for the smoke. Results decide
whether a larger causal follow-up is warranted.
