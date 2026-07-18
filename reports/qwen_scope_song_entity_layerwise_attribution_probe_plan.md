# Qwen Song Entity Layerwise Attribution Run Plan

- Full controls available: 20
- Smoke relations: 12
- Smoke English: 6
- Smoke Chinese: 6
- Readout: final RMS norm plus observed-token unembedding row
- Causal test: neutral-title residual patch at the Artist prediction position
- Trained probes: none
- Behavioral gate: 0.80 final sequence-PMI accuracy
- Consistency tolerance: 0.02 logit
- Hardware target: Hugging Face `t4-small`
