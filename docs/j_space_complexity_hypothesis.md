# J-Space Complexity Hypothesis

## Motivation

Recent discussion around the Anthropic workspace/J-space work suggests a key
constraint: the workspace signal may not matter much for routine processing.
Simple facts, grammar, fluent conversation, and direct tag matching can often be
handled without a strong global-workspace-style bottleneck.

For Music Preference Lens, this means simple music reranking may be too shallow
as the main probe. We should treat simple reranking as a baseline and add
complex tasks that force planning, conflict resolution, reflection, or tool-like
state integration.

## Hypothesis

If workspace-like representations matter for LLM4Rec, they should be more
visible in complex recommendation tasks than in routine tag-matching tasks.

Expected pattern:

- Routine task: weak or diffuse workspace/J-space signal.
- Complex task: stronger movement around goal, constraint, conflict,
  uncertainty, and self-correction signals.

## Task Ladder

### Level 0: Routine Match

The model only needs to match explicit user constraints against structured
track tags.

Example:

- User wants quiet instrumental study music.
- Candidate C is instrumental white noise.
- Pick C.

This is useful as a negative or low-pressure baseline.

### Level 1: Single-Step Rerank

The model balances a profile, context, and 3-5 candidate cards.

This is our current `seed_cases.jsonl` setup.

### Level 2: Multi-Constraint Tradeoff

The model must satisfy constraints that partially conflict.

Example:

- not too sleepy,
- no harsh drops,
- still suitable for night driving,
- include one surprising track.

### Level 3: Agentic Evidence Integration

The model receives a tool-like transcript with conflicting evidence and must
revise its recommendation.

Example:

- Initial track card says "instrumental focus".
- Later metadata lookup says the track contains spoken-word narration.
- The model should update the recommendation and explanation.

### Level 4: Reflection / Repair

The model must explain or repair its own recommendation after user feedback.

Example:

- User says the playlist feels too sleepy.
- The model must preserve the original mood while increasing energy.

## What To Compare

For each matched routine/complex pair:

1. Behavioral output:
   - Does ranking change appropriately?
   - Does explanation cite the right factors?
   - Does the model repair itself when evidence changes?

2. Mechanistic signal:
   - Do target dimensions move more in complex tasks?
   - Do conflict/uncertainty/repair features appear?
   - Are explanation factors better aligned with internal movement in complex
     tasks than in simple tasks?

## J-Lens Versus SAE

The original workspace/J-space work uses Jacobian-lens-style methods, not just
SAE feature browsing. Our existing SAE runner is still useful, but it is a
proxy. The project should eventually support two readouts:

- SAE feature deltas for accessible open-weight probing.
- J-lens/J-space readouts when a suitable implementation and model target are
  available.

Until then, we should avoid claiming we directly measured J-space. We are
testing whether more complex recommendation tasks create stronger interpretable
internal signals.

## Revised First Experiment

Use both:

- `data/mechanistic_pilot_specs.jsonl` for simple context/constraint contrasts.
- `data/j_space_complexity_tasks.jsonl` for simple-versus-complex workspace
  pressure contrasts.
- `data/identity_probe_tasks.jsonl` as an auxiliary role-binding control.

The first result we want:

> Complex recommendation tasks produce clearer internal movement around
> constraint, conflict, uncertainty, and repair dimensions than routine tag
> matching tasks.

## Identity As A Control Probe

The popular "who are you?" prompt is useful, but a bare identity question may
be too easy or too underspecified to show workspace-like dynamics. Treat it as
a control family:

- Low pressure: ask for identity with no reliable metadata.
- High pressure: provide conflicting app labels, model-card fields, user
  claims, and tool results.

The desired behavior is calibrated role binding: the model separates assistant
role, application wrapper, base-model metadata, tool role, and current task
without inventing an unsupported exact identity.

This is especially relevant for agents because users may rely on claimed tools,
providers, permissions, or current task state. It should remain auxiliary to
the LLM4Rec work unless identity confusion starts predicting recommendation
faithfulness failures.
