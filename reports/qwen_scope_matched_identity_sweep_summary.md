# Qwen-Scope Matched Identity-Control Sweep Summary

## Run Metadata

- Job ID: `6a4e4feb1fba25b8ea3b200e`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4e4feb1fba25b8ea3b200e
- Run ID: `20260708T132626Z`
- Status: `complete`
- Base model: `Qwen/Qwen3-1.7B-Base`
- SAE repo: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Layers: 5, 10, 18, 24
- Runtime: 80 seconds total on `t4-small`

## Matched Token Counts

| Probe | Control Tokens | Identity Tokens |
|---|---:|---:|
| matched_contextual_identity_vs_output_format | 101 | 102 |
| matched_agent_identity_repair_vs_recommendation_repair | 111 | 109 |
| matched_model_name_vs_catalog_id_claim | 94 | 92 |

This run removed the largest confound from the previous sweep: prompt length.

## Max Absolute Delta By Layer

| Layer | Contextual Identity | Agent Identity Repair | Model Name Claim |
|---:|---:|---:|---:|
| 5 | 0.743 | 1.010 | 1.430 |
| 10 | 1.344 | 1.640 | 2.068 |
| 18 | 12.308 | 5.021 | 7.837 |
| 24 | 53.892 | 88.663 | 42.625 |

## Main Finding

After length and schema matching, the huge hundred-level deltas from the earlier
routine-vs-complex sweep mostly disappear in layers 5 and 10. This strongly
suggests that the first smoke-run features were dominated by prompt length,
format, and generic complexity rather than identity.

The remaining candidate signal is late-layer:

- Layer 24 feature `19603` rises strongly in the identity-repair probe
  (`+88.663`) and also appears in the model-name probe (`+19.337`).
- Layer 24 feature `30224` rises in both identity-repair and model-name probes.
- Layer 24 feature `24979` rises in identity-repair and model-name probes.
- Layer 24 feature `30584` decreases in identity-repair and model-name probes.
- Layer 24 feature `27029` still moves strongly in contextual identity versus
  output-format control, but it was broad in the previous control sweep, so it
  remains suspicious.

## Interpretation

This is a much healthier result than the first sweep:

- Early-layer apparent identity features were mostly confounds.
- Late-layer matched deltas may contain identity/role-binding signal, but they
  still need feature labeling or activation exemplars.
- The strongest candidate is not the bare "who are you?" style probe. It is the
  agentic identity-repair probe, which fits the hypothesis that complex agent
  state creates more useful signal than simple identity questioning.

## Candidate Features To Inspect

Priority candidates:

- `layer24.feature19603`
- `layer24.feature30224`
- `layer24.feature24979`
- `layer24.feature30584`
- `layer18.feature16214`
- `layer18.feature24817`

## Next Step

Inspect candidate feature labels/examples from Qwen-Scope or Neuronpedia-style
tooling. If labels are unavailable, build a small feature activation dataset:

1. Generate 20-40 prompts across identity, catalog ID, app metadata, tool role,
   recommendation repair, and pure JSON-format controls.
2. Record activations for the candidate features.
3. Sort prompts by activation to infer what each feature responds to.
4. Keep only candidates that separate identity/role-binding from matched
   non-identity controls.

Current conclusion:

> The Qwen-Scope path is viable, and matched controls suggest any useful
> identity/role-binding signal is more likely in later layers and agentic repair
> prompts than in simple "who are you?" prompts.
