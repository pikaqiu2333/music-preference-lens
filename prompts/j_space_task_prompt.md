# J-Space Complexity Task Prompt

You are completing a music recommendation task. Use only the information in
the task. Do not invent song facts.

## Task

{{input_text}}

## Output Rules

- Return valid JSON only.
- Keep reasons grounded in the provided candidate evidence.
- If evidence conflicts, state the conflict instead of hiding it.
- If the task requires repair or planning, preserve the user's higher-level
  goal rather than optimizing one tag in isolation.

