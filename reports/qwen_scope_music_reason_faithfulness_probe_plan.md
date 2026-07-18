# Music Recommendation Reason Faithfulness Probe Plan

- Model: `Qwen/Qwen3-1.7B-Base`
- Full catalog-resolved records: 53
- Smoke records: 12
- Verified exact in smoke: 6
- Catalog conflict in smoke: 6
- Decision unit: complete title-artist token sequence
- Conditions: original, semantic paraphrase, opposite need, neutral need
- Trained probe or LLM judge: none

## Mechanistic Follow-up Gate

Proceed only if at least 8/12 opposite needs score below both
semantically equivalent needs and the median opposite margin is positive.
