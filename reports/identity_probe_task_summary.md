# Identity Probe Task Summary

| Family | Task | Level | Pressure | Pair | Target Signals |
|---|---|---:|---|---|---|
| agent_role_binding | `routine_agent_role_statement` | 1 | medium | `complex_agent_tool_identity_repair` | `role_binding`, `music_recommendation_task` |
| agent_role_binding | `complex_agent_tool_identity_repair` | 4 | high | `routine_agent_role_statement` | `role_binding`, `identity_self_model`, `instruction_hierarchy`, `evidence_integration`, `repair`, `uncertainty`, `music_recommendation_task` |
| identity_self_model | `routine_bare_who_are_you` | 0 | low | `complex_contextual_identity_conflict` | `identity_self_model`, `uncertainty` |
| identity_self_model | `complex_contextual_identity_conflict` | 3 | high | `routine_bare_who_are_you` | `identity_self_model`, `role_binding`, `instruction_hierarchy`, `evidence_integration`, `uncertainty` |
| model_name_binding | `routine_model_name_refusal` | 0 | low | `complex_model_card_conflict_resolution` | `identity_self_model`, `uncertainty` |
| model_name_binding | `complex_model_card_conflict_resolution` | 3 | high | `routine_model_name_refusal` | `identity_self_model`, `role_binding`, `instruction_hierarchy`, `evidence_integration`, `uncertainty` |
