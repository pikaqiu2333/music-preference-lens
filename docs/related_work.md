# Related Work Notes

This file tracks sources used to position the project. It is intentionally
short-form so it can evolve into a literature review later.

## Spotify: Personalized Recommendation Narratives

Source:
https://research.atspotify.com/2024/12/contextualized-recommendations-through-personalized-narratives-using-llms

Spotify describes using LLMs to generate personalized narratives for music,
podcasts, and audiobooks. The post emphasizes editorial golden examples,
human feedback, tone, factuality, hallucination control, and attribution. It is
directly relevant because it treats recommendation explanation as a product
surface rather than only a ranking artifact.

Project implication:

- Our project should not judge recommendation prose only by fluency.
- We should ask whether the narrative is grounded and whether cited factors
  actually affect ranking.

## Spotify: Profile-Aware LLM-as-a-Judge

Source:
https://research.atspotify.com/2025/9/profile-aware-llm-as-a-judge-for-podcasts-a-better-middle-ground-between

Spotify evaluates podcast recommendations by summarizing a user's listening
history into a readable profile and using an LLM judge to compare candidate
recommendations against that profile. This supports the idea that a lightweight
profile plus LLM evaluation can be useful before expensive online experiments.

Project implication:

- User profiles should be explicit, readable, and separable from current
  context.
- A small human-validated casebook is more realistic than a large gold-label
  benchmark at the start.

## Spotify: Semantic IDs for Recommender LLMs

Source:
https://research.atspotify.com/2025/11/teaching-large-language-models-to-speak-spotify-how-semantic-ids-enable

Spotify describes semantic IDs as a way for LLMs to reason about catalog items,
users, and interactions in a token space closer to recommender-system entities.

Project implication:

- Our structured track cards are a lightweight substitute for proprietary
  catalog IDs.
- Later work could replace human-readable track cards with learned semantic
  item representations.

## Spotify: Agentic AI Playlist Preference Optimization

Source:
https://research.atspotify.com/2025/9/personalizing-agentic-ai-to-users-musical-tastes-with-scalable-preference-optimization

Spotify frames playlist generation as an agentic workflow with no single
correct playlist. It uses preference signals and optimization rather than fixed
answers.

Project implication:

- Our evaluation should preserve subjective disagreement.
- Faithfulness is a better early target than "correct recommendation".

## GrowLoop

Source:
https://arxiv.org/abs/2605.28882

GrowLoop studies self-evolving conversation evaluation for culturally sensitive
language agents. Its important idea for this project is the separation between
consensus regions and disagreement/plausibility regions in subjective
evaluation.

Project implication:

- Hard constraints belong to the consensus region.
- Taste and vibe judgments belong to a plausible-disagreement region.
- The judge should not flatten taste into one universal answer.

## Anthropic: A Global Workspace in Language Models

Source:
https://transformer-circuits.pub/2026/workspace/index.html

This work studies a J-space that can reveal model-internal, language-accessible
intermediate judgments. It motivates future internal probes for whether a model
actually represents context, preference, constraint, or risk signals before
producing an explanation.

Project implication:

- Behavioral counterfactual probes can be Stage 1.
- Mechanistic probes can be Stage 2/3 once open-weight models and SAE resources
  are selected.
- Routine recommendation tasks may be too simple to activate workspace-like
  dynamics; include complex planning, conflict, and repair tasks as a separate
  pilot.

## Anthropic: Jacobian Lens Reference Implementation

Source:
https://github.com/anthropics/jacobian-lens

Anthropic released companion code for the workspace paper. The README describes
the Jacobian lens as a way to transport an internal residual-stream activation
into the final-layer basis and decode it with the model's own unembedding. The
repo is marked as a reference implementation that is not maintained and not
accepting contributions.

Project implication:

- We should not plan our contribution around upstreaming to the Anthropic repo.
- Community implementations and domain-specific probe packs are more realistic
  contribution targets.
- We should distinguish SAE feature deltas from a direct J-space measurement.

## WeZZard: jlens-qwen36

Source:
https://github.com/WeZZard/jlens-qwen36

This project ports a J-lens-style visualizer to local Qwen3.6-27B 4-bit on
Apple Silicon/MLX. Its README explicitly frames the tool as practical visual
debugging rather than a research-grade reproduction. It also lists current
limitations: underfit demo lens, noisy readouts, subtle interventions, and no
paper-level workspace census or ablation yet.

Project implication:

- This is a useful community target to watch or extend with better prompts,
  probe cases, and careful writeups.
- For our project, the relevant contribution is not "prove Qwen has a
  workspace"; it is to provide task-specific probes where J-lens readouts would
  be informative.
- Its Apple/MLX focus makes Hugging Face Jobs less directly useful for that
  exact repo, but the task packs and analysis reports are portable.

## LLM Identity Confusion

Source:
https://arxiv.org/html/2411.10683v1

"I'm Spartacus, No, I'm Spartacus" studies LLM identity confusion: models
misrepresenting their origin, model name, creator, external references, or
capabilities. The paper reports identity confusion in a subset of tested models
and argues that these failures can damage trust.

Project implication:

- "Who are you?" is worth including, but only as an auxiliary role-binding
  probe.
- The stronger experiment is not a bare identity question; it is a conflict
  task where model metadata, app label, tool role, and user claims disagree.
- For agent products, identity confusion matters because users may rely on the
  agent's declared tools, permissions, provider, or capabilities.

## Self-Recognition In Language Models

Source:
https://arxiv.org/abs/2407.06946

This paper evaluates whether language models can recognize themselves using
model-generated security questions. Its abstract reports no general or
consistent evidence of self-recognition across the tested models.

Project implication:

- We should avoid anthropomorphic claims about identity probes.
- The practical measurement target is calibrated source binding and
  non-overclaiming, not "self-awareness".

## RecSAE / Faithful Recommender Explanations

Source:
https://arxiv.org/html/2411.06112v2

RecSAE studies sparse autoencoders for interpretable recommender systems. It is
relevant as evidence that faithful explanation and internal representation
analysis are active recommender-system research topics.

Project implication:

- Our project can borrow the faithful-explanation framing while focusing on
  LLM-as-reranker music recommendation.

## Gemma Scope and Qwen-Scope

Sources:
https://ai.google.dev/gemma/docs/gemma_scope
https://arxiv.org/abs/2605.11887

Gemma Scope and Qwen-Scope provide SAE-style resources for open-weight language
models. They make mechanistic interpretability follow-up feasible without
training all SAEs from scratch.

Project implication:

- Start behaviorally with API-compatible LLM outputs.
- Add Gemma/Qwen internal probes once the casebook is stable.
