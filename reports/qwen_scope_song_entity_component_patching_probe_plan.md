# Qwen Song Entity Component Patching Run Plan

- Smoke relations: 12
- Analysis layers: 14, 16, 18, 21, 24, 27
- Endpoint check layer: 28
- Components: self-attention, MLP, full residual
- Minimum valid relations: 8
- Minimum real-vs-neutral first-token effect: 0.05
- Layer-28 endpoint tolerance: 0.02
- Trained probes: none
- Hardware target: Hugging Face `t4-small`
