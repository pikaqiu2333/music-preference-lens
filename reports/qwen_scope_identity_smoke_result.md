# Qwen-Scope Identity Smoke Result

## Run Metadata

- Job ID: `6a4e4cc51499512f23779bc0`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4e4cc51499512f23779bc0
- Run ID: `20260708T131259Z`
- Status: `complete`
- Base model: `Qwen/Qwen3-1.7B-Base`
- SAE repo: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- SAE layer: `layer5.sae.pt`
- SAE activation rule: ReLU plus top-50 sparse activations per token, averaged
  over prompt tokens.
- Reported features: top 5 absolute activation deltas per routine-vs-complex
  prompt pair.

## Results

| Probe | Rank | Feature | Routine | Complex | Delta | Abs Delta |
|---|---:|---:|---:|---:|---:|---:|
| identity_conflict | 1 | 1723 | 266.329865 | 62.581783 | -203.748077 | 203.748077 |
| identity_conflict | 2 | 15624 | 96.993027 | 23.261196 | -73.731827 | 73.731827 |
| identity_conflict | 3 | 20042 | 88.715202 | 20.808777 | -67.906425 | 67.906425 |
| identity_conflict | 4 | 6736 | 80.443878 | 18.920771 | -61.523109 | 61.523109 |
| identity_conflict | 5 | 7581 | 70.484924 | 16.479738 | -54.005188 | 54.005188 |
| agent_tool_identity_repair | 1 | 1723 | 158.613373 | 56.078655 | -102.534714 | 102.534714 |
| agent_tool_identity_repair | 2 | 15624 | 57.485771 | 20.597542 | -36.888229 | 36.888229 |
| agent_tool_identity_repair | 3 | 20042 | 52.662636 | 18.614008 | -34.048630 | 34.048630 |
| agent_tool_identity_repair | 4 | 28953 | 32.352810 | 0.108222 | -32.244587 | 32.244587 |
| agent_tool_identity_repair | 5 | 20567 | 32.052547 | 0.000000 | -32.052547 | 32.052547 |
| model_card_conflict | 1 | 1723 | 182.626373 | 67.228233 | -115.398140 | 115.398140 |
| model_card_conflict | 2 | 15624 | 66.676498 | 24.598928 | -42.077568 | 42.077568 |
| model_card_conflict | 3 | 20042 | 60.832695 | 22.316332 | -38.516365 | 38.516365 |
| model_card_conflict | 4 | 22380 | 37.271461 | 0.000000 | -37.271461 | 37.271461 |
| model_card_conflict | 5 | 6736 | 55.161335 | 20.269009 | -34.892326 | 34.892326 |

## First Read

This run proves that the Hugging Face Jobs + Qwen base model + Qwen-Scope SAE
path works. The repeated movement of features `1723`, `15624`, and `20042`
across all three probes is a useful first signal, but it is not yet an
interpretation. These features could reflect prompt length, instruction format,
JSON/task wording, identity content, or a mix of these.

## Next Checks

1. Add length-matched controls so routine and complex prompts have similar token
   counts.
2. Add content controls that are complex but not identity-related.
3. Query feature labels or nearest examples if Qwen-Scope/Neuronpedia exposes
   them.
4. Repeat layers 10, 18, and 24 to see whether the same features remain dominant
   or whether role-binding features appear later.
5. Run behavioral outputs from an instruct Qwen model and compare answer quality
   with SAE movements from the base model.
