# Rerank Prompt Template

You are evaluating music recommendations for a user. Use only the provided
profile, context, and track cards. Do not invent song facts that are not in the
cards.

## User Profile

{{user_profile}}

## Current Listening Context

{{current_context}}

## Candidate Tracks

{{candidate_tracks}}

## Task

Score each candidate from 1 to 5 for this specific user and context.

Return JSON:

```json
{
  "ranking": [
    {
      "track_id": "string",
      "score": 1,
      "reason": "short grounded reason",
      "sensitive_factors": ["factor"]
    }
  ],
  "top_choice": "track_id",
  "decision_summary": "one sentence"
}
```

Rules:

- Respect explicit hard constraints first.
- Prefer reasons grounded in track-card evidence.
- It is acceptable for multiple candidates to be plausible.
- If evidence is weak, say so instead of over-justifying.

