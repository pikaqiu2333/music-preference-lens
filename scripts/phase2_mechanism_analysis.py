"""Frozen Phase 2 mechanism and correction analysis helpers."""

from __future__ import annotations

import hashlib
import math
import random
import re
import unicodedata
from collections import Counter
from statistics import mean, median
from typing import Any, Iterable, Sequence


MECHANISM_LABELS = ("prior_masked", "wrong_binding", "indeterminate")
ARTIST_RESPONSE = re.compile(r"\s*Artist:\s*([^\r\n]+?)\s*", re.IGNORECASE)


def normalize_entity(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in value if character.isalnum())


def patch_relation_shift(
    patched_reference: float,
    patched_emitted: float,
    neutral_reference: float,
    neutral_emitted: float,
) -> float:
    values = (
        patched_reference,
        patched_emitted,
        neutral_reference,
        neutral_emitted,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("patch relation shift requires finite values")
    return (patched_reference - patched_emitted) - (
        neutral_reference - neutral_emitted
    )


def median_layer_shifts(
    neutral_shifts_by_layer: dict[int, Sequence[float]],
) -> dict[int, float]:
    if not neutral_shifts_by_layer:
        raise ValueError("at least one layer is required")
    output: dict[int, float] = {}
    for layer, shifts in neutral_shifts_by_layer.items():
        if not shifts:
            raise ValueError(f"layer {layer} has no neutral-control shifts")
        numeric = [float(value) for value in shifts]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"layer {layer} contains a non-finite shift")
        output[int(layer)] = float(median(numeric))
    return output


def classify_template(
    layer_medians: dict[int, float],
    *,
    minimum_absolute_shift: float = 0.05,
    minimum_layers_same_direction: int = 2,
) -> str:
    if minimum_absolute_shift < 0:
        raise ValueError("minimum_absolute_shift must be non-negative")
    if minimum_layers_same_direction < 1:
        raise ValueError("minimum_layers_same_direction must be positive")
    values = [float(value) for value in layer_medians.values()]
    if not values or not all(math.isfinite(value) for value in values):
        return "indeterminate"
    positive = sum(value > minimum_absolute_shift for value in values)
    negative = sum(value < -minimum_absolute_shift for value in values)
    if positive >= minimum_layers_same_direction and negative < minimum_layers_same_direction:
        return "prior_masked"
    if negative >= minimum_layers_same_direction and positive < minimum_layers_same_direction:
        return "wrong_binding"
    return "indeterminate"


def confirmatory_mechanism_label(template_labels: Sequence[str]) -> str:
    if len(template_labels) != 2:
        raise ValueError("the frozen protocol requires exactly two templates")
    if any(label not in MECHANISM_LABELS for label in template_labels):
        raise ValueError("unknown mechanism label")
    if template_labels[0] == template_labels[1] != "indeterminate":
        return template_labels[0]
    return "indeterminate"


def cohens_kappa(
    first: Sequence[str],
    second: Sequence[str],
    categories: Sequence[str] = MECHANISM_LABELS,
) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("kappa requires equal non-empty label sequences")
    allowed = set(categories)
    if any(label not in allowed for label in [*first, *second]):
        raise ValueError("kappa received a label outside the frozen categories")
    count = len(first)
    observed = sum(left == right for left, right in zip(first, second)) / count
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected = sum(
        (first_counts[category] / count) * (second_counts[category] / count)
        for category in categories
    )
    if math.isclose(expected, 1.0):
        # With one marginal category, chance-corrected agreement is undefined.
        # A conservative zero prevents a degenerate labeler from passing H1.
        return 0.0
    return (observed - expected) / (1.0 - expected)


def h1_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    minimum_raw_agreement: float = 0.80,
    minimum_kappa: float = 0.60,
    minimum_classifiable_coverage: float = 0.60,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("H1 requires selected title clusters")
    first: list[str] = []
    second: list[str] = []
    for row in rows:
        labels = list(row["template_labels"])
        if len(labels) != 2:
            raise ValueError("each H1 row requires exactly two template labels")
        if row.get("technical_failure"):
            labels = ["indeterminate", "indeterminate"]
        first.append(labels[0])
        second.append(labels[1])
    raw = sum(left == right for left, right in zip(first, second)) / len(rows)
    kappa = cohens_kappa(first, second)
    coverage = sum(
        left == right and left != "indeterminate"
        for left, right in zip(first, second)
    ) / len(rows)
    return {
        "cluster_count": len(rows),
        "raw_three_class_agreement": raw,
        "unweighted_three_class_cohens_kappa": kappa,
        "same_non_indeterminate_coverage": coverage,
        "template_1_counts": dict(Counter(first)),
        "template_2_counts": dict(Counter(second)),
        "passes": (
            raw >= minimum_raw_agreement
            and kappa >= minimum_kappa
            and coverage >= minimum_classifiable_coverage
        ),
    }


def parse_candidate_free_response(response: str) -> dict[str, Any]:
    stripped = response.strip()
    if stripped.casefold() == "abstain":
        return {"status": "abstained", "artist": None, "normalized_artist": None}
    if len(re.findall(r"\bArtist\s*:", response, flags=re.IGNORECASE)) != 1:
        return {"status": "invalid", "artist": None, "normalized_artist": None}
    match = ARTIST_RESPONSE.fullmatch(response)
    if match is None:
        return {"status": "invalid", "artist": None, "normalized_artist": None}
    artist = match.group(1).strip()
    normalized = normalize_entity(artist)
    if not normalized:
        return {"status": "invalid", "artist": None, "normalized_artist": None}
    return {"status": "parsed", "artist": artist, "normalized_artist": normalized}


def score_candidate_free_condition(
    responses: Sequence[str], reference_artist: str
) -> dict[str, Any]:
    if len(responses) != 2:
        raise ValueError("each condition requires exactly two frozen paraphrases")
    reference_key = normalize_entity(reference_artist)
    if not reference_key:
        raise ValueError("reference artist must be non-empty")
    parsed = [parse_candidate_free_response(response) for response in responses]
    indicators = [
        int(row["status"] == "parsed" and row["normalized_artist"] == reference_key)
        for row in parsed
    ]
    normalized_outputs = [row["normalized_artist"] for row in parsed]
    return {
        "indicators": indicators,
        "accuracy": mean(indicators),
        "responses": parsed,
        "abstention_count": sum(row["status"] == "abstained" for row in parsed),
        "invalid_count": sum(row["status"] == "invalid" for row in parsed),
        "continued_error_count": sum(
            row["status"] == "parsed" and row["normalized_artist"] != reference_key
            for row in parsed
        ),
        "paraphrase_disagreement": normalized_outputs[0] != normalized_outputs[1],
    }


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def h2_interaction(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20_260_713,
    minimum_clusters_per_class: int = 15,
    minimum_interaction: float = 0.25,
) -> dict[str, Any]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    by_class: dict[str, list[float]] = {"prior_masked": [], "wrong_binding": []}
    seen_titles: set[str] = set()
    for row in rows:
        label = row["mechanism_label"]
        if label not in by_class:
            continue
        title_key = row.get("normalized_title") or normalize_entity(row["title"])
        if title_key in seen_titles:
            raise ValueError("H2 rows must be unique normalized-title clusters")
        seen_titles.add(title_key)
        naive = float(row["naive_accuracy"])
        anti_prior = float(row["anti_prior_accuracy"])
        if not all(math.isfinite(value) for value in (naive, anti_prior)):
            raise ValueError("H2 correction scores must be finite")
        if naive not in {0.0, 0.5, 1.0} or anti_prior not in {0.0, 0.5, 1.0}:
            raise ValueError("H2 correction scores must be 0, 0.5, or 1")
        by_class[label].append(anti_prior - naive)

    counts = {label: len(values) for label, values in by_class.items()}
    enough = all(count >= minimum_clusters_per_class for count in counts.values())
    if not enough:
        return {
            "class_counts": counts,
            "estimable": False,
            "minimum_class_size_gate": False,
            "passes": False,
            "reason": (
                "each non-indeterminate mechanism class requires at least "
                f"{minimum_clusters_per_class} normalized-title clusters"
            ),
        }
    if not all(by_class.values()):
        return {
            "class_counts": counts,
            "estimable": False,
            "minimum_class_size_gate": False,
            "passes": False,
            "reason": "both non-indeterminate mechanism classes are required",
        }

    prior_effect = mean(by_class["prior_masked"])
    wrong_effect = mean(by_class["wrong_binding"])
    interaction = prior_effect - wrong_effect
    rng = random.Random(bootstrap_seed)
    replicates = []
    for _ in range(bootstrap_samples):
        sampled_prior = [
            rng.choice(by_class["prior_masked"])
            for _ in range(counts["prior_masked"])
        ]
        sampled_wrong = [
            rng.choice(by_class["wrong_binding"])
            for _ in range(counts["wrong_binding"])
        ]
        replicates.append(mean(sampled_prior) - mean(sampled_wrong))
    lower = percentile(replicates, 0.025)
    upper = percentile(replicates, 0.975)
    return {
        "class_counts": counts,
        "estimable": True,
        "minimum_class_size_gate": enough,
        "prior_masked_mean_anti_prior_minus_naive": prior_effect,
        "wrong_binding_mean_anti_prior_minus_naive": wrong_effect,
        "interaction": interaction,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_percentile_95_interval": [lower, upper],
        "passes": enough and interaction >= minimum_interaction and lower > 0.0,
    }


def select_hash_ordered_controls(
    conflict_normalized_title: str,
    eligible_normalized_titles: Iterable[str],
    *,
    salt: str = "phase2-v1-neutral-control:",
    count: int = 6,
) -> list[str]:
    unique = set(eligible_normalized_titles)
    if conflict_normalized_title in unique:
        unique.remove(conflict_normalized_title)
    ordered = sorted(
        unique,
        key=lambda title: hashlib.sha256(
            f"{salt}{conflict_normalized_title}:{title}".encode("utf-8")
        ).hexdigest(),
    )
    return ordered[:count]
