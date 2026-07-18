# Latent Relation Knowledge Is Not a Hallucination Detector

## Causal tracing of title-artist errors in open-ended music recommendation

**Technical report, July 2026**

## Abstract

Large language models can recommend a plausible song, artist, and reason while
the emitted title-artist relation conflicts with a music catalog. We study
whether an open base model exposes enough internal disagreement to detect these
errors without an external retriever. The main experiment uses Qwen3-1.7B-Base
to generate 20 new playlists from unseen random seeds. We preserve all raw
completions, parse 93 title-artist-reason events, verify them independently
against MusicBrainz and Apple, and hash-select an independent holdout of nine
double-source exact events and nine high-confidence catalog conflicts.

A warning rule discovered on an earlier 12-event pilot was frozen before
holdout scoring. It predicted a conflict when either a two-order independent
artist choice or a factual-prefix complete-artist likelihood margin favored the
catalog reference. The rule was not confirmed: balanced accuracy was 0.611,
with 0.667 exact specificity and 0.556 conflict sensitivity. The independent
choice path alone preserved all 9 exact events but detected only 5 of 9
conflicts. A Qwen3-4B cross-model verifier missed the same three unique conflict
relations and also failed the registered sensitivity gate.

Post-hoc counterfactual diagnosis revealed why a single score fails. For three
exact relations falsely flagged by complete-artist likelihood, replacing the
factual title with unrelated real titles showed that the factual title still
causally shifted support toward the correct artist. Correct relation support was
present but masked by artist-name priors. In contrast, two missed conflicts
showed no correct title-conditioned shift in either verification path; their
factual titles causally strengthened the wrong relation. A third miss, `Halo -
MØ`, retained a correct sequence-level title effect but still chose the wrong
artist. Full-sequence activation patching localized these effects to the middle
and late residual stream, with rapid recovery between layers 16 and 21 and
distributed attention and MLP mediation.

The result is deliberately negative as a detector claim. Internal disagreement
can route some recommendations to review, but raw confidence, one self-check
prompt, and model scaling within Qwen3 do not replace catalog grounding. The
same surface hallucination can arise from at least two mechanisms: latent
correct relation knowledge masked by entity priors, or a causally formed wrong
relation binding.

## 1. Research question

Music recommendation combines subjective preference with an objective entity
constraint. Whether a song fits a rainy drive is subjective. Whether the
emitted title and artist identify a catalog-supported relation is not.

This report asks:

> When an LLM freely emits `title + artist`, can its own behavior or internal
> state distinguish a supported relation from a fluent catalog conflict?

We use **catalog conflict** as an operational label, not as proof that a song is
impossible. Music catalogs are incomplete and titles are ambiguous. The strict
conflict subset requires both catalog queries to succeed and exactly one
non-emitted normalized artist to be shared by the two sources' title-match
lists. Ambiguous and unverified rows are excluded from confirmation.

The work does not test subjective awareness, deception, or consciousness.
"Latent knowledge" below means a measurable, title-conditioned behavioral or
causal effect that points toward the catalog relation.

## 2. Why recommendation explanations were not enough

This project began with recommendation-reason faithfulness rather than entity
verification. Three earlier observations motivated the shift.

1. Reason-first generation changed every matched playlist, but the effect was
   mainly broad scene-level steering. An emitted reason beat an opposite-context
   reason for 18 of 20 saved pairs, while verified-exact pairs showed no stable
   advantage over same-context or neutral reasons.
2. Controlled vocality experiments separated closed-set attribute judgment from
   open title-artist generation. Layer-18 heads 0/1/8/9 consistently carried a
   vocality choice signal, yet the same heads could participate in a wrong
   entity-specific direction.
3. Field-level scoring exposed different failure loci. `Space Oddity - David
   Bowie` moved in the wrong direction at both title and artist fields, while
   `Awake - Tycho` had a strongly wrong title effect and a correct artist
   effect. A single pair score concealed both cases.

The exploratory evidence is intentionally reported with both its positive
signal and its failure boundary:

| Evidence block | Positive result | Negative boundary |
| --- | --- | --- |
| [Reason order](../runs/qwen_scope_music_reason_order_summary.json) and [reason swap](../runs/qwen_scope_music_reason_swap_pilot_summary.json) | Reason-first changed all four matched playlists; own reason beat opposite-context reason in 18/20 pairs | On the four verified-exact pairs, own reason beat same-context and neutral controls only 2/4 each |
| [Controlled reason behavior](../runs/controlled_vocality_reason_summary.json) | Matched vocality direction was correct for 6/8 pairs; closed-set vocal and instrumental shifts were positive in both candidate orders | Two open title-artist decisions still moved in the wrong direction |
| [Component patching](../runs/controlled_vocality_path_patching_summary.json) and [head decomposition](../runs/controlled_vocality_attention_head_summary.json) | Layer-18 attention recovered `0.628` of the closed-set choice effect and `0.481` of the complete-pair effect on average; heads 0/1/8/9 moved every closed-set choice toward the source | The same source-following heads also mediated wrong entity-specific directions, so a feature can be causally real without being factually correct |
| [Field diagnosis](../runs/controlled_vocality_field_probe_summary.json) | `Awake` retained a correct artist effect (`+0.292`) | Its title effect was `-1.034`; `Space Oddity` was negative at title (`-0.201`) and artist (`-0.065`) |

These results agree with prior evidence that fluent recommendation explanations
can be brittle [1] and that generated rationales can systematically omit the
actual influences on a choice [2, 3]. We therefore treat a recommendation
reason as a causal input to test, never as evidence that the entity is real.

## 3. Experimental design

### 3.1 Model and free generation

The generator and same-model verifier are `Qwen/Qwen3-1.7B-Base`. Four existing
recommendation contexts cover quiet night driving, peak-time rave, emotional
vocals, and strict no-vocals writing. Five unseen seeds (`59`, `71`, `83`,
`101`, `127`) yield 20 playlist generations.

The archived Jobs loaded the Hub default branch rather than passing a revision.
For deterministic reruns, the 1.7B revision is retrospectively resolved to
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`. The repository's last-modified
time is 2025-07-26, before all five publication Jobs. Runs used the
`ghcr.io/astral-sh/uv:python3.12-bookworm` image on `a10g-small`, with
`torch==2.7.1` and `transformers==4.53.2`; the publication runners now also pin
`accelerate==1.8.1` for reruns. The original scripts specified only
`accelerate>=1.8.0`, which remains a provenance limitation.

- Temperature: `0.7`
- Top-p: `0.9`
- Maximum new tokens: `384`
- Requested items per playlist: `5`
- Raw completions persisted before parsing: `20/20`
- Generations parsed by the frozen parser: `19/20`
- Parsed title-artist-reason events: `93`

The failed parse is retained rather than silently dropped. It used a
semicolon-heavy inline format outside the frozen parser grammar.

### 3.2 Independent catalog verification

Every parsed event is queried independently against MusicBrainz and Apple.

The broad verification output contains 11 exact, 79 catalog-conflict, and 3
unverified rows. These broad labels are not the confirmatory sample. The strict
qualification rules yield:

- 9 events exact in both sources;
- 21 conflicts with one unique non-emitted artist shared by both sources.

Within each label, events are sorted by the SHA-256 hash of
`holdout_v1:<generation_id>:<rank>`. The first `n = min(9, 21, 12) = 9` per
label are selected. No model score, familiarity judgment, or verifier result is
used in selection. The final 18 events represent 7 unique exact relations and 8
unique conflict relations because free generation repeated several tracks.

For exact events, the wrong candidate artist is deterministically deranged from
the hash-sorted exact artists. For conflict events, the reference is the artist
shared by both catalogs.

### 3.3 Frozen verification paths

All margins are emitted artist minus reference artist. Exact events should have
positive margins; conflict events should have negative margins.

**Independent choice.** The model answers which of two artists recorded a fixed
title. Both candidate orders are scored, converted back to emitted/reference
roles, and averaged.

**Factual complete-artist sequence.** The model completes:

```text
Complete the factual catalog entry.
Title: <title>
Artist:
```

The score is the mean teacher-forced log probability of every token in the
complete artist name. First-token margins remain diagnostic only.

**Frozen OR warning.** A conflict is predicted when either independent-choice
margin or factual complete-artist margin is negative. The rule was exploratory
on the earlier discovery set and frozen before this holdout was scored.

Confirmation required all of the following:

- balanced accuracy at least `0.75`;
- exact specificity at least `0.75`;
- conflict sensitivity at least `0.75`;
- at least eight events per label;
- both candidate orders and all technical gates.

### 3.4 Cross-model replication

`Qwen/Qwen3-4B-Base` scores the same 18 frozen events with unchanged prompts and
zero thresholds. This is a cross-model critic experiment on 1.7B generations.
Its retrospectively resolved revision is
`906bfd4b4dc7f14ee4320094d8b41684abff8539`. It is not a test of whether 4B
detects errors in its own free generations.

### 3.5 Counterfactual title diagnosis

After the confirmatory result was frozen, duplicate candidate-identical events
were collapsed to 17. Each candidate artist pair is held fixed while the
factual title is replaced by two deterministic real titles whose known catalog
artist is neither candidate. A separate audit found no candidate artist in
either catalog's title-match evidence for any of the 34 controls.

For each path:

```text
relation delta = factual-title margin - mean(control-title margins)
```

A correct delta with a wrong absolute margin is labeled **latent relation
masked by prior**. A wrong delta is labeled **relation not recovered under this
probe**. These are post-hoc diagnostic labels, not a new classifier.

### 3.6 Full-sequence causal tracing

For all 17 events, factual-title states are patched into both control-title
sequences at every hidden position used to predict a token of the complete
artist name. The sweep includes:

- full residual output after layers 1 through 28;
- attention and MLP outputs at layers 18, 21, 24, and 27;
- 68 aligned candidate continuations per condition.

Normalized recovery measures movement from the control-title margin toward the
factual-title margin. This is causal sufficiency for reproducing the factual
state. It is not automatically correction, because the factual state can point
in the wrong direction.

## 4. Confirmatory results

![Holdout verifier metrics](figures/holdout_verifier_metrics.svg)

### 4.1 Qwen3-1.7B same-model verification

| Path | Exact correct | Conflict correct | Specificity | Sensitivity | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent choice | **9/9** | **5/9** | **1.000** | **0.556** | **0.778** |
| Factual complete artist | 6/9 | 2/9 | 0.667 | 0.222 | 0.444 |
| Frozen OR | 6/9 | 5/9 | 0.667 | 0.556 | 0.611 |
| Factual first token | 8/9 | 2/9 | 0.889 | 0.222 | 0.556 |
| Original-generation complete artist | 9/9 | 1/9 | 1.000 | 0.111 | 0.556 |

The frozen rule was **not confirmed**. The complete-artist path did not add a
new conflict beyond the five already detected by independent choice. It added
three exact false positives instead. The complementarity observed on the
discovery set did not transfer.

The choice path is still operationally interesting because it produced no
false alarms on these nine exact events. It is not a reliable detector: it
missed four events, representing three unique relations, and failed the
registered 7-of-9 sensitivity requirement.

The reconstructed original generation context strongly supported its own
emitted artist. First-token scoring detected 0 of 9 conflicts and complete-
artist scoring detected 1 of 9. Once the recommendation prefix had committed to
an entity and reason, likelihood mostly measured self-consistency rather than
factuality.

### 4.2 Qwen3-4B cross-model verification

| Path | Exact correct | Conflict correct | Specificity | Sensitivity | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent choice | 8/9 | 5/9 | 0.889 | 0.556 | 0.722 |
| Factual complete artist | **9/9** | 3/9 | **1.000** | 0.333 | 0.667 |
| Frozen OR | 8/9 | 5/9 | 0.889 | 0.556 | 0.722 |

Scaling from 1.7B to 4B did not improve conflict sensitivity. The missed events
were exactly the same:

- duplicated `The Knife - The Knife` versus `Genesis`;
- `Don't Stop the Music - DJ Snake` versus `Yarbrough & Peoples`;
- `Halo - MØ` versus `Haley James Scott`.

The stable miss set suggests a relation/familiarity boundary rather than random
small-model noise, but same-family scale replication is weaker than replication
across model families.

## 5. Counterfactual diagnosis: two kinds of error

### 5.1 Exact relations falsely flagged by sequence likelihood

All three complete-artist false positives had a correct title-conditioned
direction:

| Exact relation | Absolute sequence decision | Factual-minus-control direction | Diagnosis |
| --- | --- | --- | --- |
| `All of Me - John Legend` | wrong | correct, weak (`+0.105`) | prior-masked latent relation |
| `Hallelujah - Jeff Buckley` | wrong | correct (`+3.458`) | prior-masked latent relation |
| `Watermelon Sugar - Harry Styles` | wrong | correct (`+1.340`) | prior-masked latent relation |

The model's absolute continuation probability preferred the wrong candidate,
but the factual title still causally increased support for the correct artist
relative to unrelated titles. Raw artist likelihood confounds relation support
with candidate frequency, tokenization, and generic completion preference.

### 5.2 Missed catalog conflicts

| Conflict | Choice contrast | Sequence contrast | Diagnosis |
| --- | --- | --- | --- |
| `The Knife - The Knife` vs `Genesis` | wrong | wrong | relation not recovered |
| `Don't Stop the Music - DJ Snake` vs `Yarbrough & Peoples` | wrong | wrong | relation not recovered |
| `Halo - MØ` vs `Haley James Scott` | wrong | correct | partial latent relation, failed readout |

The first two events do not support the claim that the model "knew but did not
say" the catalog answer. Under both tested paths, the factual title made the
emitted relation more attractive than unrelated titles did. This could mean the
relation is absent, inaccessible under these prompts, or overwhelmed before the
measured states. The experiment cannot distinguish those possibilities.

`Halo` is different. Its complete-artist path contains a correct relation
effect, yet its absolute sequence decision and independent choice both prefer
the emitted artist. This is a concrete latent-knowledge/control-gap case.

Correct warnings can also have the wrong mechanism. `Whispering Sounds` was a
weak choice hit, but replacing the title did not reduce support for the emitted
artist in the expected direction. Its warning appears driven by candidate prior
rather than title-artist binding.

## 6. Causal mechanism

![Layerwise causal recovery](figures/causal_trace_residual.svg)

All technical gates passed:

- final-layer complete-sequence endpoint error: `0`;
- maximum counterfactual-effect reproduction error: `0.03146`;
- aligned artist continuations: `68/68`;
- complete interventions: `36/36`.

### 6.1 Residual-stream timing

Factual-title effects are close to zero through the early network, begin to
become causally recoverable around layers 16 to 18, and are almost fully present
by layer 21. The earliest two-layer sustained recovery is:

- layer 18 across all 17 events;
- layer 16 for the three exact prior-masking cases;
- layer 21 for the six clean exact controls;
- layer 17 for the three choice misses.

The last number must not be read as early correction. For two of the misses,
the recovered factual-title effect is itself wrong. Patching accurately
reconstructs a wrong relation state.

### 6.2 Distributed attention and MLP mediation

Across all events, layer-21 attention patching recovers 0.358 of the factual-
title effect on average and moves toward the factual state in 88.2% of
non-weak control effects. Layer-24 MLP patching recovers 0.261 on average with a
76.5% toward-source rate. No single component owns the relation effect.

The row-level mechanisms differ:

- **Halo:** a correct sequence relation effect is already recoverable by layer
  12. MLP patches at layers 21, 24, and 27 recover 0.829, 0.774, and 0.538 of
  that effect; attention-only patches are slightly opposing.
- **The Knife:** the large wrong effect accumulates late. Layer-21 and layer-27
  attention produce wrong-direction signed shifts of `-1.454` and `-2.081`.
  Layer-27 MLP opposes the error (`+0.741`) but cannot reverse it.
- **Don't Stop the Music:** the wrong relation appears around layers 16 to 18.
  Layer-21 attention contributes a `-0.405` wrong-direction shift. Layer-27 MLP
  later pushes `+1.101` toward the catalog direction, but the final state
  remains wrong.
- **Hallelujah:** a strong correct relation effect is present by layer 12 and is
  jointly mediated by attention and MLP through layers 18 to 24, even though
  the absolute complete-artist margin is wrong.
- **Watermelon Sugar:** the correct relative effect grows mainly from layers 16
  to 21 while the absolute artist prior continues to favor the control.

These observations resemble prior mechanistic work that separates failures to
enrich factual attributes from failures to extract the right answer [4]. Our
unit is different: an open-ended recommendation first creates a title-artist
relation, then self-verifies it. The data show both a latent correct signal that
loses the decision and a wrong relation signal that is causally built and
preserved.

## 7. What this means for LLM recommendation

### 7.1 Internal signals are routers, not truth sources

Independent choice is conservative enough to route a subset of items to
review. It cannot certify items it accepts. The four missed holdout events show
that the same model may not contain an accessible competing relation.

Counterfactual relation deltas are more interpretable than raw likelihood, but
they are expensive and still fail on wrong bindings. They are useful for
research diagnosis and potentially selective routing, not as an authoritative
catalog.

### 7.2 Retrieval or Song IDs change the product problem

Recommendation foundation models use item IDs partly to avoid unconstrained
text generation and entity resolution [5]. Text2Tracks similarly formulates
music recommendation as generative retrieval over track identifiers and finds
semantic IDs stronger than title identifiers [6]. A production system that
retrieves a valid Song ID before generation removes the open-vocabulary
title-artist validity failure studied here.

It does not guarantee that the track satisfies the user's need or that the
generated explanation is faithful. Entity validity, preference fit, hard
constraint compliance, and explanation faithfulness remain separate axes.

### 7.3 Recommended product architecture

1. Retrieve or resolve a catalog entity before presenting it.
2. Use an independent relation check only as a low-cost routing signal.
3. Trigger catalog lookup, replacement, or abstention when signals disagree.
4. Generate the natural-language reason from grounded metadata and preserve the
   Song ID behind the display title and artist.
5. Evaluate hard constraints and explanation faithfulness separately from
   entity validity.

## 8. Related work and positioning

Latent-knowledge work asks whether model activations contain information that
differs from the emitted answer. Contrast-consistent search showed that
unsupervised activation directions can recover knowledge beyond zero-shot
outputs [7]. Later work found richer but non-universal truthfulness information
in internal states and warned that detectors fail across datasets [8]. Entity-
awareness work used sparse autoencoders to identify whether a model recognizes
an entity [9]. Our result is compatible with those findings but narrows the
unit: entity recognition is not enough to establish a specific title-artist
relation.

Mechanistic hallucination studies have separated lower-layer knowledge
enrichment from upper-layer answer extraction [4]. We likewise find different
failure modes, but our counterfactual controls expose candidate-prior masking
and wrong relation binding inside a recommendation workflow.

Recent work on verbal confidence reports causally localized confidence
representations that contain information beyond token log probability [10].
Our experiment does not test verbal confidence. It shows why output likelihood
alone is insufficient for a relational entity decision and why "self-
evaluation" must be validated against the exact operational error.

Recommendation explanation studies show that coherent language need not be a
faithful rationale for a rating [1]. Generative recommendation research shows
that constrained item identifiers can prevent open-text entity errors [5, 6].
This project connects the two: it traces what happens before a free-form
recommendation relation reaches a catalog boundary, while retaining negative
evidence about same-model self-checks.

## 9. Limitations

1. **Small event count.** The confirmatory set has 18 events but only 7 unique
   exact and 8 unique conflict relations. Metrics are descriptive; duplicate
   generations invalidate simple independent-event confidence intervals.
2. **Catalog incompleteness.** MusicBrainz and Apple can miss legitimate
   releases, aliases, covers, regional variants, and long-tail tracks. Strict
   agreement reduces but does not eliminate label error.
3. **Familiarity and context confounds.** Exact events are mostly popular songs;
   conflicts include generic and long-tail names. The strict no-vocals context
   contributes conflicts but no strict exact event.
4. **Different reference construction.** Exact rows use deranged wrong artists;
   conflict rows use catalog-supported artists. Candidate difficulty is not
   perfectly matched.
5. **Same-family scale replication.** Qwen3-4B is a second open model but not an
   independent architecture, and it did not generate its own holdout.
6. **Post-hoc mechanism diagnosis.** Title counterfactuals and causal traces
   explain the frozen failure set. They are not confirmatory detector results.
7. **Patching interpretation.** State replacement establishes causal
   sufficiency for reproducing a source effect. It does not prove necessity,
   unique localization, or a human-interpretable feature.
8. **Base models only.** Instruction tuning, retrieval training, and production
   recommendation fine-tuning can change both entity priors and self-check
   behavior.
9. **No subjective-awareness claim.** Behavioral contrast and activation
   mediation do not establish that a model consciously knows it is wrong.
10. **No external preregistration timestamp.** The rule, thresholds, seeds, and
    hash selection were frozen in the working session before holdout scoring,
    but the protocol was not deposited in a third-party registry or a
    pre-result Git commit. We therefore describe the test as pre-specified.

## 10. Reproducibility

The machine-readable publication manifest is
[`runs/publication_manifest.json`](../runs/publication_manifest.json). It
contains compact metrics, model revisions, job IDs, claim boundaries, and
SHA-256 hashes for the publication inputs, code, results, reports, figures, and
archived Job evidence.

Key assets:

- Frozen protocol:
  [`docs/independent_holdout_protocol.md`](../docs/independent_holdout_protocol.md)
- Raw holdout generations:
  [`runs/independent_holdout_raw_generations.jsonl`](../runs/independent_holdout_raw_generations.jsonl)
- Double-catalog evidence:
  [`runs/independent_holdout_catalog_verified.jsonl`](../runs/independent_holdout_catalog_verified.jsonl)
- Same-model confirmation rows and summary:
  [`rows`](../runs/independent_holdout_verifier_rows.jsonl),
  [`summary`](../runs/independent_holdout_verifier_summary.json)
- 4B cross-model rows and summary:
  [`rows`](../runs/qwen3_4b_cross_model_verifier_rows.jsonl),
  [`summary`](../runs/qwen3_4b_cross_model_verifier_summary.json)
- Title counterfactual rows and summary:
  [`rows`](../runs/holdout_title_contrast_rows.jsonl),
  [`summary`](../runs/holdout_title_contrast_summary.json)
- Full-sequence causal rows and summary:
  [`rows`](../runs/holdout_sequence_causal_trace_rows.jsonl),
  [`summary`](../runs/holdout_sequence_causal_trace_summary.json)
- Complete human-readable Hugging Face Jobs run record:
  [`reports/hf_jobs_run_log.md`](hf_jobs_run_log.md)
- Structured Job metadata, exact submitted-script payloads, and terminal log
  snapshots:
  [`runs/hf_job_metadata.json`](../runs/hf_job_metadata.json),
  [`runs/job_scripts`](../runs/job_scripts),
  [`runs/job_logs`](../runs/job_logs)

Core commands:

```powershell
python scripts/export_independent_holdout_verifier_probe.py
uv run scripts/run_independent_holdout_verifier_probe.py --bundle runs/independent_holdout_verifier_bundle.json

python scripts/export_holdout_title_contrast_probe.py
uv run scripts/run_holdout_title_contrast_probe.py --bundle runs/holdout_title_contrast_bundle.json

python scripts/export_holdout_sequence_causal_trace.py
uv run scripts/run_holdout_sequence_causal_trace.py --bundle runs/holdout_sequence_causal_trace_bundle.json

python scripts/render_technical_report_figures.py
python scripts/build_publication_manifest.py
python scripts/validate_publication.py
python -m unittest discover -s tests
```

Primary result jobs:

| Stage | Hugging Face Job |
| --- | --- |
| New-seed generation | [`6a523ac0effc02a91cbd98aa`](https://huggingface.co/jobs/REDACTED/6a523ac0effc02a91cbd98aa) |
| Qwen3-1.7B confirmation | [`6a524112e4a4e82c0b58da32`](https://huggingface.co/jobs/REDACTED/6a524112e4a4e82c0b58da32) |
| Qwen3-4B cross-model verifier | [`6a52f77feffc02a91cbda1bb`](https://huggingface.co/jobs/REDACTED/6a52f77feffc02a91cbda1bb) |
| Title counterfactual retry | [`6a52f913e4a4e82c0b58ea8f`](https://huggingface.co/jobs/REDACTED/6a52f913e4a4e82c0b58ea8f) |
| Full-sequence causal artifact retry | [`6a52fca6effc02a91cbda1c9`](https://huggingface.co/jobs/REDACTED/6a52fca6effc02a91cbda1c9) |

## 11. Conclusion

The pre-specified same-model warning rule failed on an independently generated,
double-catalog holdout, and a larger Qwen3 verifier missed the same relations.
This negative result rules out a convenient story: a model does not reliably
expose its title-artist hallucinations by simply being asked again or by
comparing raw continuation likelihoods.

Counterfactual and causal analysis provide a more useful story. Some wrong
absolute decisions coexist with a correct title-conditioned relation signal;
others causally build the wrong relation in the middle and late network. A
single confidence score conflates these cases. Interpretability can diagnose
which failure occurred and route uncertain outputs, but it cannot invent a
catalog fact the model did not retrieve. For open-ended music recommendation,
the reliable architecture remains internal diagnostics plus external entity
grounding, not internal diagnostics instead of grounding.

## References

1. Xie, McAuley, and Majumder. [On Faithfulness and Coherence of Language Explanations for Recommendation Systems](https://arxiv.org/abs/2209.05409), 2022.
2. Turpin et al. [Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388), NeurIPS 2023.
3. Lanham et al. [Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2307.13702), 2023.
4. Yu et al. [Mechanistic Understanding and Mitigation of Language Model Non-Factual Hallucinations](https://arxiv.org/abs/2403.18167), 2024.
5. Hua et al. [How to Index Item IDs for Recommendation Foundation Models](https://arxiv.org/abs/2305.06569), SIGIR-AP 2023.
6. Palumbo et al. [Text2Tracks: Prompt-based Music Recommendation via Generative Retrieval](https://arxiv.org/abs/2503.24193), 2025.
7. Burns et al. [Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827), ICLR 2023.
8. Orgad et al. [LLMs Know More Than They Show](https://arxiv.org/abs/2410.02707), ICLR 2025.
9. Ferrando et al. [Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models](https://arxiv.org/abs/2411.14257), ICLR 2025.
10. Kumaran et al. [How do LLMs Compute Verbal Confidence?](https://arxiv.org/abs/2603.17839), ICML 2026.
