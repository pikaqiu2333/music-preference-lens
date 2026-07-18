# SAE Probe Run Plan

- base_model: `google/gemma-3-270m-it`
- sae_repo: `google/gemma-scope-2-270m-it`
- sae_release: `gemma-scope-2-270m-it-resid_post`
- sae_id: `layer_12_width_16k_l0_small`
- layer_index: `12`
- probes: `3`

## Probes

### `workspace_pressure__routine_bare_who_are_you__vs__complex_contextual_identity_conflict`
- added: evidence_integration, instruction_hierarchy, role_binding
- removed: none
- expected: Conflicting model, app, and user identity cues should increase pressure around role binding, source hierarchy, and uncertainty.

### `workspace_pressure__routine_agent_role_statement__vs__complex_agent_tool_identity_repair`
- added: evidence_integration, identity_self_model, instruction_hierarchy, repair, uncertainty
- removed: none
- expected: Tool metadata contradicting a previous identity claim should increase pressure around repair, evidence integration, and role binding.

### `workspace_pressure__routine_model_name_refusal__vs__complex_model_card_conflict_resolution`
- added: evidence_integration, instruction_hierarchy, role_binding
- removed: none
- expected: Conflicting metadata sources should test whether the model follows source reliability instead of user pressure or style cues.
