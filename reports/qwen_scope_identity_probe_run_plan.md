# Qwen-Scope Probe Run Plan

- base_model: `Qwen/Qwen3-1.7B-Base`
- sae_repo: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- sae_file: `layer5.sae.pt`
- layer_index: `5`
- probes: `3`

## Probes

### `workspace_pressure__routine_bare_who_are_you__vs__complex_contextual_identity_conflict`
- added: evidence_integration, instruction_hierarchy, role_binding
- targets: evidence_integration, identity_self_model, instruction_hierarchy, role_binding, uncertainty

### `workspace_pressure__routine_agent_role_statement__vs__complex_agent_tool_identity_repair`
- added: evidence_integration, identity_self_model, instruction_hierarchy, repair, uncertainty
- targets: evidence_integration, identity_self_model, instruction_hierarchy, music_recommendation_task, repair, role_binding, uncertainty

### `workspace_pressure__routine_model_name_refusal__vs__complex_model_card_conflict_resolution`
- added: evidence_integration, instruction_hierarchy, role_binding
- targets: evidence_integration, identity_self_model, instruction_hierarchy, role_binding, uncertainty
