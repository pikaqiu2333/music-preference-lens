# Music Reason Component-Patching HF Plan

- Cases: 3
- Layers: 14, 16, 18, 21, 24, 27; endpoint layer 28
- Components: attention, MLP, full residual
- Target: complete title-artist token sequence
- Source condition: opposite need
- Target condition: original need
- Minimum absolute need effect: 0.05
- Baseline and endpoint tolerance: 0.02
- Hardware target: Hugging Face `t4-small`
- Interpretation: exploratory three-case mechanism pilot
