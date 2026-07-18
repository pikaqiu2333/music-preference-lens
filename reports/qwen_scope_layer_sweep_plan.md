# Qwen-Scope Layer Sweep Plan

## Goal

Test whether the identity-smoke features from layer 5 are stable across layers
and whether they also appear in non-identity controls.

## Model And SAE

- Base model: `Qwen/Qwen3-1.7B-Base`
- SAE repo: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layers: 5, 10, 18, 24
- SAE activation rule: ReLU plus top-50 sparse activations per token, averaged
  over prompt tokens.
- Reported features: top 5 absolute routine-vs-complex deltas per probe.

## Probe Families

Identity probes:

- `identity_conflict`
- `agent_tool_identity_repair`
- `model_card_conflict`

Non-identity controls:

- `control_music_metadata_repair`
- `control_json_schema_complexity`
- `control_app_metadata_no_identity`

## Interpretation Rules

Useful signal:

- A feature or feature family moves strongly in identity probes but weakly in
  non-identity controls.
- The same conceptual signal appears in later layers after length/schema
  controls are added.

Likely confound:

- The same features dominate all identity and non-identity controls.
- Feature movement tracks prompt length more than task family.
- Feature movement mostly reflects JSON formatting or metadata field names.

## Output

The next Hugging Face job prints compact JSON to logs because the current HF
token can run jobs but cannot create/push dataset repos. Once a write-enabled
token is available, rerun the same sweep and persist CSV artifacts to a dataset
repo.
