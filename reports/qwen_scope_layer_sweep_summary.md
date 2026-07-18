# Qwen-Scope Identity-Control Layer Sweep Summary

## Run Metadata

- Job ID: `6a4e4f0e1499512f23779bc9`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4e4f0e1499512f23779bc9
- Run ID: `20260708T132245Z`
- Status: `complete`
- Base model: `Qwen/Qwen3-1.7B-Base`
- SAE repo: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layers: 5, 10, 18, 24
- Probes: 3 identity probes plus 3 non-identity controls.
- Runtime: 79 seconds total on `t4-small`.

## Token Counts

| Probe | Family | Routine Tokens | Complex Tokens |
|---|---|---:|---:|
| identity_conflict | identity | 24 | 102 |
| agent_tool_identity_repair | identity | 39 | 109 |
| model_card_conflict | identity | 35 | 92 |
| control_music_metadata_repair | control | 59 | 157 |
| control_json_schema_complexity | control | 29 | 95 |
| control_app_metadata_no_identity | control | 34 | 116 |

The complex prompts are much longer than the routine prompts in every pair.
This is a major confound.

## Dominant Features By Layer

| Layer | Repeated Top Features | What It Means Now |
|---:|---|---|
| 5 | `1723`, `15624`, `20042`, `6736`, `7581` | These dominate identity probes and also dominate all three controls. |
| 10 | `25709`, `19873`, `13648`, `21006` | Same pattern: strong in identity, but also strong in controls. |
| 18 | `29473`, `13004`, `25143`, `8107`, `23043` | Same pattern across identity and controls. |
| 24 | `27029`, `19230`, `27497`, `30224`, `25409` | More mixed signs, but still broad control overlap. |

## Main Finding

The first smoke-run features are not identity-specific. They appear strongly in
non-identity controls that also contain longer instructions, JSON fields,
metadata lookup language, app labels, or repair-style tasks.

This is useful negative evidence: the current routine-vs-complex contrast mostly
measures prompt complexity, length, and formatting pressure. It does not yet
isolate identity self-model or role binding.

## Still Interesting

Layer 24 shows a few less-universal movements:

- `19603` appears as a large positive feature in `agent_tool_identity_repair`.
- `21011` appears in `model_card_conflict`.
- `27497` changes sign depending on app/model-card/control context.

These are candidates for follow-up, not interpretations. They need
length-matched and content-matched controls.

## Next Experiment

The next run should use matched prompt pairs:

1. Identity conflict versus same-length non-identity conflict.
2. Identity conflict versus same-length metadata/app-label control.
3. Identity repair versus same-length music recommendation repair.
4. Exact same JSON field count across identity and control tasks.

Success criterion:

- A candidate feature should be larger in identity tasks than in matched
  controls at the same layer and similar token count.
- Features should not merely track complex prompt length.

Current conclusion:

> Qwen-Scope runs work on Hugging Face Jobs, but the identity probe needs
> stricter controls before we can claim identity/role-binding features.
