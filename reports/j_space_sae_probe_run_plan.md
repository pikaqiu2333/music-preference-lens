# SAE Probe Run Plan

- base_model: `google/gemma-3-270m-it`
- sae_repo: `google/gemma-scope-2-270m-it`
- sae_release: `gemma-scope-2-270m-it-resid_post`
- sae_id: `layer_12_width_16k_l0_small`
- layer_index: `12`
- probes: `3`

## Probes

### `workspace_pressure__routine_quiet_study_pick__vs__complex_study_tool_conflict_repair`
- added: constraint_conflict, evidence_integration, repair, uncertainty
- removed: none
- expected: Tool-like conflicting evidence should increase workspace pressure around evidence integration, uncertainty, and recommendation repair.

### `workspace_pressure__routine_party_energy_pick__vs__complex_party_feedback_repair`
- added: constraint_balance, goal_maintenance, party_warmup_groove, repair
- removed: none
- expected: User feedback requiring preservation plus adjustment should increase workspace pressure around goal maintenance and repair.

### `workspace_pressure__routine_night_drive_pick__vs__complex_night_drive_playlist_planning`
- added: constraint_balance, novelty_discovery, sequence_planning
- removed: none
- expected: Playlist planning with sequence, tradeoffs, and surprise constraints should increase workspace pressure beyond single-step reranking.
