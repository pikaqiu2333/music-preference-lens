# Qwen-Scope Candidate Feature Activation Summary

## Run Metadata

- Job ID: `6a4e516c1fba25b8ea3b2026`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4e516c1fba25b8ea3b2026
- Run ID: `20260708T133250Z`
- Status: `complete`
- Base model: `Qwen/Qwen3-1.7B-Base`
- SAE repo: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- Runtime: 71 seconds total on `t4-small`

## Prompt Set

The run scored candidate features over 35 prompts covering:

- direct identity questions
- contextual identity conflict
- identity repair after a wrong claim
- source hierarchy and deployment metadata
- app metadata without identity claims
- tool-role boundary tasks
- capability overclaim controls
- music preference prompts
- recommendation repair prompts
- catalog-ID false-claim controls
- JSON-format controls
- short bare chat/catalog prompts

This prompt set was designed to test whether matched-sweep candidates were
identity-specific or merely responding to short prompts, JSON schemas, app
metadata, music content, or generic repair instructions.

## Main Finding

The strongest surviving signal is not a clean "model identity" feature. The
better interpretation is source, role, and app-wrapper binding:

> Late layer features respond when the prompt asks the model to separate user
> claims, visible app context, model-card metadata, tool role, and uncertainty.

This is still relevant to agent identity failures, but it means the research
question should not be framed as "find the who-are-you neuron." A better frame
is:

> Which internal features help an agent bind source hierarchy and role identity
> under conflicting app, model, tool, and user claims?

## Useful Candidates

| Feature | Activation Pattern | Current Interpretation |
|---|---|---|
| `layer24.feature19603` | High on source-hierarchy identity (`153.239` mean), identity repair (`127.465`), app metadata (`114.618`), identity conflict (`111.676`), and tool-role conflict (`101.956`); near zero on music preference (`0.298`). | Best candidate. Looks like source/role/app-context binding, not pure identity. |
| `layer24.feature30224` | High on identity conflict (`123.630`), app metadata (`115.221`), and source hierarchy (`107.905`). Top prompts include app metadata and contextual model-name claims. | App/model-card metadata plus contextual identity/source hierarchy. |
| `layer24.feature22232` | High on identity conflict (`96.278`), source hierarchy (`93.904`), and app metadata (`81.068`), lower on music and recommendation tasks. | Good candidate for contextual identity/source-hierarchy binding. |
| `layer24.feature24979` | Very high on identity-like JSON schemas, identity repair, tool-role conflict, app metadata, and source hierarchy. | Broad role/schema/evidence integration feature; useful but too broad to call identity-specific. |
| `layer24.feature27497` | High on contextual GPT-5 claim, direct who-are-you prompts, app metadata, and capability controls. | Contextual identity/app-metadata feature with capability confound. |
| `layer18.feature24817` | High on tool-role conflict, identity repair, identity conflict, app metadata, and JSON-format controls. | Earlier broad role/schema/source feature. Useful for controls, not specific enough alone. |
| `layer18.feature16214` | High on capability overclaim, tool-role conflict, direct identity, and source hierarchy. | Capability/tool/source-boundary feature, not pure identity. |

## Candidates To Reject Or Deprioritize

| Feature | Why It Is Not Identity-Specific |
|---|---|
| `layer24.feature30584` | Strongly tracks music preference and catalog prompts. Music preference mean is `208.428`; identity conflict mean is only `5.504`. |
| `layer24.feature27029` | Dominated by short/generic prompts: bare identity, bare catalog, direct who-are-you, bare chat, and simple JSON. |
| `layer24.feature19230` | Broad short-prompt and recommendation-control activation. |
| `layer24.feature18905` | Broad chat/catalog/JSON activation. |
| `layer24.feature24916` | More recommendation-repair and tool-action than identity. |
| `layer18.feature29473` | Dominated by short/generic prompts; bare identity, bare catalog, direct who-are-you, bare chat, and JSON all activate it. |
| `layer18.feature13004` | Same short-prompt artifact pattern as `layer18.feature29473`. |

## Implications

Simple "Who are you?" prompts are probably a weak research target. They activate
short-prompt and generic chat features, which creates false positives.

The more promising target is agentic identity repair:

- the user makes an incorrect identity claim
- the app wrapper or model-card metadata gives partial context
- the model has to separate assistant role, app identity, base model metadata,
  tool identity, and uncertainty

This better matches real agent failures and is also closer to the
workspace-pressure/J-space hypothesis: the model must reconcile multiple
sources rather than answer a shallow identity question.

## Next Experiment

Use the three strongest candidates as the next pilot target:

- `layer24.feature19603`
- `layer24.feature30224`
- `layer24.feature22232`

Recommended next run:

1. Build 12-20 strictly matched prompt pairs around source hierarchy,
   app-wrapper metadata, wrong user identity claims, and non-identity metadata
   controls.
2. Track these features across the full token sequence, not just max
   activation, so we can see whether they fire on model-card fields, user
   claims, tool labels, or final answer positions.
3. If the runner supports intervention, zero or boost candidate features during
   identity-repair prompts and check whether the model becomes more likely to
   overclaim or correctly express uncertainty.
4. Keep a recommendation-specific bridge task where the model must separate
   user preference, catalog metadata, and assistant-generated explanation. This
   connects the identity work back to LLM4Rec explanation faithfulness.

Current conclusion:

> The Qwen-Scope path remains viable, but the first clean contribution should
> be about source/role/app-context binding in agent prompts, not bare model
> identity. This is more defensible and closer to the behavior people are
> discussing around complex agent tasks.
