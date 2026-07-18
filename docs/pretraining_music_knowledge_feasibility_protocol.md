# Pretraining Music Knowledge Feasibility Protocol

Status: design freeze before data inspection

Date: 2026-07-14

## Objective

Test whether an exact open pretraining corpus and checkpoint sequence can support a paper-scale study of how music relation evidence becomes candidate-free model knowledge.

The feasibility run does not test a final scientific hypothesis. It decides whether the natural-data sample, relation annotation, checkpoint behavior, and compute path are adequate for a confirmatory study.

## Primary Question

At matched title-artist exposure frequency, do independent-source diversity, expression diversity, duplicate concentration, and conflicting relations predict when a model first achieves stable candidate-free relation recall?

## Non-Claims

The run must not claim that:

- corpus occurrence proves exact model membership unless the model-specific order is reconstructed;
- co-occurrence is a semantic relation assertion;
- positive forced-choice margin is recoverable knowledge;
- linear decodability is conscious awareness or causal model use;
- one influential document uniquely taught a relation;
- the experiment estimates the hallucination rate of production recommenders.

## Models

### Feasibility primary

- `EleutherAI/pythia-1b-deduped`
- Exact pretokenized training order and 154 checkpoints are available.
- Evaluate a frozen set of 12 checkpoints first; refine around exposure events only after the smoke passes.

Initial checkpoint schedule:

```text
step0, step1000, step2000, step4000, step8000, step16000,
step32000, step48000, step64000, step96000, step128000, step143000
```

### Confirmatory candidate

- `allenai/OLMo-1B`
- Use released Dolma order files, original checkpoints, optimizer state, config, and logs.
- Do not substitute a regenerated trajectory without recording that it is a distinct run.

### External behavior only

- OLMo 2/3, Qwen, and DeepSeek may be used after the main analysis for final-model behavior checks.
- They do not support the primary source-to-parameter claim unless exact corpus/order/checkpoint linkage is independently verified.

## Natural Relation Sample

Create a deterministic MusicBrainz snapshot sample with 120 English-language relations:

| Stratum | Target n | Definition |
|---|---:|---|
| single-credit | 40 | one canonical recording-level artist credit and no known same-title competitor in the frozen snapshot |
| legitimate-multivalue | 40 | cover, featured, remix, collaboration, or multiple recording-level artist relations |
| competing-title | 40 | same normalized title has at least two distinct artist relations |

Sampling must freeze MBIDs, aliases, relation type, release date, language, and snapshot hash before corpus result inspection. Existing project-generated conflict rows must not be used as the confirmatory sample; they may only test parsers.

## Exposure Record Schema

Each candidate corpus hit must produce one record with at least:

```json
{
  "title_mbid": "...",
  "artist_mbid": "...",
  "relation_type": "primary_artist|featured|cover|remix|writer|other",
  "corpus": "pythia_deduped_pile",
  "source_component": "...",
  "source_id": "...",
  "document_hash": "...",
  "duplicate_cluster_id": "...",
  "token_start": 0,
  "training_step_first_seen": 0,
  "match_type": "exact|alias|normalized",
  "assertion_status": "supports|contradicts|negates|list_only|unrelated|uncertain",
  "expression_type": "metadata|encyclopedic|review|interview|chart|lyrics_context|forum|other",
  "evidence_excerpt_hash": "..."
}
```

Do not publish full lyrics or full copyrighted documents. Persist only shard/row identifiers, hashes, token offsets, labels, and the shortest excerpt needed for audit where licensing permits.

## Retrieval and Annotation

1. Generate title and artist alias sets from the frozen MusicBrainz snapshot.
2. Search exact phrases, both entity orders, bounded-window co-occurrence, and common relation patterns.
3. Cluster exact and near-duplicate documents before calculating diversity.
4. Apply deterministic high-precision rules first.
5. Use an LLM labeler only to propose `assertion_status` and `expression_type`; it is not ground truth.
6. Human-audit 200 stratified hits, including all rare relation types and all predicted contradictions.

Annotation precision gate: lower Wilson 95% confidence bound for `supports` precision must be at least `0.90`. Recall is reported on a separately exhaustively reviewed small subset and is not inferred from precision.

## Behavioral Measures

Every relation is evaluated with frozen, candidate-free prompts that never display artist options.

Primary behavioral endpoint:

- stable exact or alias-matched artist completion on at least 2 of 3 held-out prompt templates across 3 consecutive evaluated checkpoints.

Secondary endpoints:

- mean target artist token log probability;
- greedy completion exact/alias accuracy per template;
- prompt stability;
- reverse artist-to-title recall;
- reference-versus-matched-distractor margin;
- acquisition checkpoint and later forgetting events.

Forced-choice and pairwise margin are secondary only. A relation with positive margin but no candidate-free completion is classified as `relative_preference_without_access`.

Positive controls must include at least 20 globally prominent, unambiguous music relations. Candidate-free completion must reach at least `0.80` on 2-of-3 prompt agreement, or the behavioral instrument fails.

## Natural-Data Analysis

Pre-register a hierarchical model after the unlabeled retrieval smoke but before joining labels to model outputs. Candidate predictors are:

```text
log(valid_exposure_count + 1)
independent_source_count
expression_diversity
duplicate_cluster_concentration
conflict_exposure_ratio
relation_cardinality
tokens_since_last_valid_exposure
title and artist tokenization length
subject/object corpus popularity
```

The key comparison is the incremental predictive value of source/expression diversity and conflict after controlling for valid exposure count and entity popularity. Report uncertainty and continuous effects; do not search for a universal count threshold.

## Mechanistic Gate

No hidden-state classifier, SAE, LRE, or patching analysis may start unless:

1. at least 80 sampled relations have one or more audited valid exposures;
2. the sample contains at least 20 usable relations in each of the repeat-dominated, source-diverse, and conflict/multivalue analysis groups;
3. the positive-control behavioral gate passes;
4. at least one pre-registered exposure variable predicts candidate-free behavior out of title- and artist-disjoint validation.

If the gate passes, the first internal analysis is layerwise target-logit trajectory plus a relation representation estimate on title/artist/template-disjoint splits. Causal intervention remains a later gate.

## Controlled Continuation Handoff

Natural correlations must be followed by a controlled continuation experiment before a causal data-composition claim.

Provisional factorial design:

- exposure dose: low/high;
- document structure: one repeated template/multiple independent expressions;
- relation environment: clean/fixed conflict ratio;
- at least 3 training seeds;
- anonymous entities preserving a sampled MusicBrainz relation graph;
- held-out document genres and prompt templates;
- unrelated-data washout after injection.

Primary continuation endpoint: held-out-template stable candidate-free exact accuracy. A diversity advantage of at least 10 absolute percentage points is the provisional continuation gate, subject to a frozen power analysis before outcome inspection.

## Stop Rules

Stop before GPU evaluation if the exact training order cannot be reconstructed or the annotation precision gate fails.

Stop before mechanistic work if natural behavior is absent, limited to pairwise scores, or fails entity-disjoint validation.

Stop the causal claim if controlled continuation fails on held-out prompts or after washout, even if training-template log probability improves.

Treat disagreement between natural and controlled results as unresolved, not as permission to select the preferred result.

## Required Artifacts

- frozen MusicBrainz sample and snapshot receipt;
- query manifest and alias rules;
- corpus shard/row/token mapping with hashes;
- duplicate clusters and audited labels;
- prompt pack and checkpoint manifest;
- row-level behavior outputs and summary;
- environment lock, job script, logs, runtime, and cost;
- analysis plan hash frozen before labels and outputs are joined.
