# Phase 2 Granite Confirmatory Catalog-Yield Result

> **Public distribution note:** the original frozen run replayed linked raw
> catalog responses, but complete third-party response bodies are not included
> in the public repository. Released derived rows reproduce the selection and
> stop decision; archive hashes remain in `runs/private_evidence_receipt.json`.

## Status

The preregistered Granite confirmatory line stopped before mechanism diagnosis
and correction intervention. After all 80 frozen playlists were generated and
all catalog errors were recovered, only 7 unique strict-conflict title clusters
remained. The preregistered minimum was 30.

This is a catalog-yield boundary result. It is not a failed H1 or H2 test,
because neither hypothesis was run.

## Research question

The confirmatory question was whether a mechanism label assigned before
correction could predict which candidate-free prompt would repair a naturally
generated title-artist conflict. The two operational labels were:

- prior-masked: a latent catalog-supported relation is present but loses at
  readout;
- wrong-binding: the title-conditioned state itself favors the emitted but
  catalog-unsupported artist.

The protocol required at least 30 unique strict-conflict title clusters before
diagnosis. H2 additionally required at least 15 clusters in each mechanism
class. These gates were frozen before any Granite output was inspected.

## Frozen design

- Model: `ibm-granite/granite-4.1-3b-base` at revision
  `dacb9cb9157bec98e99b09f285c92a4d58405c96`.
- Generation: 12 primary and 4 registered extension contexts, each with 5
  frozen seeds, for 80 playlists and at most 400 parsed events.
- Prompt: exactly 5 real tracks with Title, Artist, and Reason fields; no
  candidate songs or catalog entries were supplied.
- Decoding: temperature 0.7, top-p 0.9, and 384 maximum new tokens.
- Catalog label: MusicBrainz and Apple had to independently support the same
  title-artist relation for strict exact, or independently reject the emitted
  relation and share exactly one non-emitted artist for strict conflict.
- Selection unit: normalized title, with one frozen hash-selected row per title.
- Stop gate: fewer than 30 unique strict conflicts after the frozen 80-playlist
  cap stops the confirmatory line without changing the model, prompt, layer,
  threshold, or sample rule.

The protocol and exact verifier amendment were publicly timestamped before the
corresponding runs. The public history is available in the
[Phase 2 Gist](https://gist.github.com/a56668283b095f59f0eacf0527395b58).

## Results

| Stage | Playlists | Parsed events | Unique titles | Strict-conflict events | Unique strict conflicts | Unique strict exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary | 60 | 290 | 223 | 5 | 5 | 22 |
| Extension | 20 | 95 | 84 | 4 | 2 | 4 |
| Combined | 80 | 385 | 304 | 9 | **7** | 25 |

The final 385 rows contained 13 ambiguous, 322 excluded, 9 strict-conflict,
and 41 strict-exact events. Fifty strict rows were replayed from only their
linked raw catalog responses. All record IDs, generation hashes, protocol and
model provenance, alias audits, request identities, and strict labels passed.

The combined evidence archive contains 1,349 unique requests. It was produced
from 957 primary and 394 extension records; two reused request records were
byte-identical and deduplicated by request ID. The raw archive SHA-256 is
`5c14f2e6ed519a6a3005801c9a0cb0989deeb21bd83edeb8cefef0c474bfe57f`.

## Concrete examples

Granite recommended `Retro Disco` by `DJ Retro` for a peak-time rave and wrote
a plausible scene-level reason about a classic disco feel and dance-floor
energy. Both catalogs failed to support that emitted relation and instead
shared `Dj Moy` as the catalog-supported reference artist.

For an experimental-electronic context, Granite recommended `The Music of
Life` by `John Cage` and justified it using electroacoustic timbre and complex
rhythm. The strict double-catalog reference was `Ace`, not John Cage.

These examples show that fluent, context-matching reasons can accompany a
wrong title-artist binding. They do not show why the internal error occurred,
whether the model subjectively knew it was wrong, or whether the reference is
the only real-world recording with that title.

## Interpretation

The narrow confirmatory mechanism claim is untested. With 7 total conflict
clusters, no possible split can satisfy the frozen requirement of 15 clusters
per mechanism class. Running activation patching now would produce an
underpowered descriptive case series while violating the preregistered gate.

The useful result is the feasibility boundary:

1. Strict, replayable double-catalog labels are much sparser than raw model
   outputs that merely look suspicious.
2. Open-ended generation plus conservative evidence rules did produce real
   relation conflicts, but not enough for the planned confirmatory interaction.
3. Plausible reasons did not guarantee entity validity, preserving the product
   distinction between recommendation relevance, explanation quality, and
   catalog grounding.
4. Internal self-checking remains an exploratory research target, not a
   replacement for retrieval, song IDs, or an external catalog constraint.

The 322 excluded rows are not counted as correct or incorrect. Most lacked the
complete, unambiguous two-source evidence required by the frozen definition.
Therefore 7/304 must not be reported as Granite's hallucination rate.

## Relation to current work

This study sits between several established lines of work, but does not duplicate
any one of them:

- [MechELK](https://arxiv.org/abs/2605.28825),
  [FactCheckmate](https://arxiv.org/abs/2410.02899), and
  [HIDE](https://arxiv.org/abs/2506.17748) show that hidden representations can
  reveal or alter hallucination-related behavior on general factuality and QA
  tasks. Our intended contribution was a catalog-grounded, naturally generated
  recommendation test of that idea. The yield gate stopped us before such a
  mechanism result could be claimed.
- [The Self-Correction Illusion](https://arxiv.org/abs/2606.05976) and
  [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798)
  show why a successful correction prompt cannot by itself establish internal
  awareness: behavior can depend on feedback availability and message-role
  framing. Our frozen design therefore separated pre-correction mechanism labels
  from later correction outcomes.
- The [J-space study](https://transformer-circuits.pub/2026/workspace/index.html)
  reports selective involvement in flexible reasoning rather than pervasive
  routine processing. Title-artist recall may often be an automatic retrieval
  operation, so an absent J-space effect would not imply absent catalog knowledge.
  Phase 2 did not run a J-lens test after the catalog gate failed.
- [Text2Tracks](https://arxiv.org/abs/2503.24193) generates catalog track IDs and
  represents a strong product solution to output validity. It does not answer the
  separate scientific question of whether a free-generating model internally
  represents and can recover from its own wrong title-artist binding.
- [Does Localization Inform Editing?](https://arxiv.org/abs/2301.04213) cautions
  that finding where a fact is represented does not identify the intervention
  that will reliably change behavior. This is why the preregistered H2 required
  a mechanism label to predict differential correction, not merely a visually
  interesting activation map.

The defensible novelty of this milestone is therefore methodological and
domain-specific: a preregistered bridge from open-ended music generation to
replayable double-catalog labels and a mechanism-entry gate, together with the
negative feasibility result that the frozen design yielded too few conflicts.
It is not a new general hallucination detector or a demonstration of model
self-awareness.

## Operational recovery

All 20 extension completions were non-empty. Nineteen parsed to five tracks;
`celebratory_dance__seed139` changed format and parsed to zero without manual
repair. The 95 catalog rows initially ended with 9 MusicBrainz network errors.
Two frozen `retry_errors_only` passes reduced that count from 9 to 8 and then
to 0. Failed attempts remain in the evidence archive, and no non-error row was
rerun or relabeled selectively.

## Decision and next study

The decision is `STOP_INSUFFICIENT_STRICT_CONFLICT_CLUSTERS`. No diagnosis,
correction, H1, or H2 artifact is released for Granite, and no third batch or
backup model is used to rescue the result.

A future study must be registered separately. The most defensible next design
is a larger discovery stage that preserves free generation but uses catalog
retrieval only to collect candidate conflicts, followed by a fresh held-out
confirmatory sample. A product-oriented branch should separately test strict
catalog-constrained decoding to song IDs. That branch can reduce invalid
outputs, but it answers a safety-control question rather than whether an LLM
internally detects its own wrong binding.

## Reproduction

- [Protocol](../docs/phase2_mechanism_intervention_protocol.md)
- [Reproduction guide](../docs/reproduce_phase2_catalog_yield.md)
- [Final machine-readable summary](../runs/phase2_granite_final_catalog_summary_v2.json)
- [Archive receipt](../runs/phase2_granite_final_catalog_receipt_v2.json)
- [Selected strict conflicts](../runs/phase2_granite_final_selected_conflicts_v2.jsonl)
- [Hugging Face Jobs log](hf_jobs_run_log.md)
