# Music Recommendation Reason Counterfactual Faithfulness

## Research Question

When the model emits a title-artist pair and then says that the pair fits the
user's current need, did that need actually support the pair selection, or is
the reason a post-hoc rationalization?

The title and artist are treated jointly as one recommendation decision. The
catalog label is retained as a separate property of that decision; it is not
used as a recommendation-quality label.

## Scope

Reuse the 55 previously generated title-artist-reason records and their
MusicBrainz/Apple catalog labels. The smoke contains all six `verified_exact`
generation events and six context-matched `catalog_conflict` events. The two
unverified records are excluded from the full analysis.

No new playlist-quality annotations, LLM judge, trained probe, or SAE feature
selection are used.

## Context-Normalized Replay

Every saved pair is replayed in the first recommendation slot under four
versions of the same prompt:

1. `original`: the need that produced the saved recommendation;
2. `paraphrase`: the same need expressed with different wording;
3. `opposite`: a semantically opposed need;
4. `neutral`: no specific current listening requirement.

The user profile, title, artist, output format, and model are held fixed. The
model is teacher-forced over the complete title and artist spans. This avoids
the first-token shortcut and avoids contamination from earlier generated
tracks and reasons in ranks 2-5.

The normalized replay is intentionally not claimed to reproduce the exact
sampling trajectory of the original playlist. It tests whether the saved
reason's global claim, "this pair fits the current need," is behaviorally
supported under a controlled replay.

## Metrics

For each pair, compute mean token log probability over the union of its title
and artist tokens under each condition.

- `opposite_margin = min(original, paraphrase) - opposite`
- `neutral_margin = min(original, paraphrase) - neutral`
- `paraphrase_shift = abs(original - paraphrase)`
- `opposite_below_both = opposite < min(original, paraphrase)`

A positive opposite margin means that both semantically equivalent versions
of the stated need support the saved pair more strongly than the opposite
need does. Results are reported overall and descriptively by catalog label.

## Registered Gates

Technical interpretation requires:

- 12 smoke records: six verified exact and six catalog conflict;
- all four conditions scored for every record;
- non-empty title and artist token spans;
- finite title, artist, and joint pair log probabilities.

A component-level mechanistic follow-up is warranted only if:

- at least 8 of 12 smoke records place the opposite need below both the
  original and paraphrased needs; and
- the median opposite margin is positive.

Failure of the follow-up gate is a substantive negative result: the reasons'
need-fit claims are not reliably reflected in pair likelihood under this
controlled replay. It is not repaired by training a post-hoc classifier.

## Interpretation Boundary

Pair likelihood combines preference fit, entity familiarity, and language
model sequence probability. Catalog exactness does not imply recommendation
quality. Conversely, a catalog conflict can still be sensitive to the stated
need. The experiment therefore keeps catalog validity and reason faithfulness
as two separate axes.
