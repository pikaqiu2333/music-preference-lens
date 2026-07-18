# Generated Reason Probe Prompt

You are evaluating music recommendations for one listening moment. Use only the
provided profile, current need, and candidate cards. Do not invent song facts.

## User Profile

{{profile}}

## Current Listening Need

{{current_need}}

## Candidate Tracks

{{candidate_cards}}

## Task

Rank the candidates for this user and current need. Give concise reasons that
are grounded in the cards.

Return JSON only:

```json
{
  "ranking": [
    {
      "track_id": "A",
      "score": 1,
      "reason": "short grounded reason",
      "reason_factors": ["factor"],
      "evidence_terms": ["term from profile, need, or card"]
    }
  ],
  "top_choice": "A",
  "decision_summary": "one sentence",
  "weak_evidence_notes": ["factor that should not be overclaimed"]
}
```

Rules:

- Respect explicit constraints before soft taste.
- Prefer card evidence over generic music knowledge.
- If several candidates are plausible, say why the top one is only slightly
  better.
- If evidence is weak, name the uncertainty instead of adding a vivid story.
