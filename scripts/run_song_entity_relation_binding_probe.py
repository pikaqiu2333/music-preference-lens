# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.8.0",
#   "huggingface-hub>=0.33.0,<1.0",
#   "numpy>=1.26.0,<2.3",
#   "safetensors>=0.5.0",
#   "torch==2.7.1",
#   "transformers==4.53.2",
# ]
# ///
"""Run matched title-artist relation-binding probes on Qwen-Scope."""

from __future__ import annotations

import argparse
import base64
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EMBEDDED_BUNDLE_B64 = "__EXPERIMENT_BUNDLE_B64__"


def load_bundle(path: Path | None = None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    if EMBEDDED_BUNDLE_B64 == "__EXPERIMENT_BUNDLE_B64__":
        raise ValueError("pass --bundle or embed the experiment bundle")
    return json.loads(base64.b64decode(EMBEDDED_BUNDLE_B64).decode("utf-8"))


def select_smoke_controls(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    smoke_blocks = {
        "en_swap_1",
        "en_swap_2",
        "en_swap_3",
        "zh_swap_1",
        "zh_swap_2",
        "zh_swap_3",
    }
    return [row for row in rows if row["block_id"] in smoke_blocks]


def make_relation_folds(
    rows: list[dict[str, Any]],
    requested_folds: int = 5,
) -> list[set[str]]:
    blocks: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        blocks.setdefault(row["block_id"], []).append(row)
    by_language: dict[str, list[str]] = {}
    for block_id, block_rows in blocks.items():
        languages = {row["language"] for row in block_rows}
        if len(languages) != 1:
            raise ValueError(f"swap block {block_id} crosses languages")
        by_language.setdefault(next(iter(languages)), []).append(block_id)
    fold_count = min(
        requested_folds,
        *(len(block_ids) for block_ids in by_language.values()),
    )
    if fold_count < 2:
        raise ValueError("at least two relations per language are required")
    folds = [set() for _ in range(fold_count)]
    rng = random.Random(20260710)
    for language in sorted(by_language):
        block_ids = sorted(by_language[language])
        rng.shuffle(block_ids)
        for index, block_id in enumerate(block_ids):
            folds[index % fold_count].update(
                row["relation_id"] for row in blocks[block_id]
            )
    return folds


def build_sae_text(
    template: str,
    title: str,
    artist: str,
) -> tuple[str, list[int], list[int]]:
    text = template.format(title=title, artist=artist)
    artist_marker_end = text.index("Artist: ") + len("Artist: ")
    artist_start = text.index(artist, artist_marker_end)
    relation_start = text.rindex("Relation")
    return (
        text,
        [artist_start, artist_start + len(artist)],
        [relation_start, relation_start + len("Relation")],
    )


def build_likelihood_prefix(template: str, title: str) -> str:
    return template.format(title=title)


def render_choice_prompt(
    template: str,
    title: str,
    artist_a: str,
    artist_b: str,
) -> str:
    return template.format(title=title, artist_a=artist_a, artist_b=artist_b)


def fit_paired_feature_model(
    rows: list[dict[str, Any]],
    vector_key: str,
    feature_count: int,
) -> dict[str, Any]:
    import numpy as np

    exact = np.stack([row[f"exact_{vector_key}"] for row in rows]).astype(np.float32)
    mismatch = np.stack([row[f"mismatch_{vector_key}"] for row in rows]).astype(
        np.float32
    )
    all_vectors = np.concatenate([exact, mismatch], axis=0)
    std = all_vectors.std(axis=0) + 1e-6
    standardized_delta = (exact - mismatch) / std
    active = (all_vectors > 0).mean(axis=0) >= 0.20
    effect = standardized_delta.mean(axis=0)
    effect[~active] = 0.0
    selected_count = min(feature_count, int(active.sum()))
    if selected_count == 0:
        raise ValueError("no SAE features passed the activation-rate filter")
    indices = np.argsort(np.abs(effect))[-selected_count:][::-1]
    weights = effect[indices]
    weight_norm = float(np.linalg.norm(weights))
    if weight_norm == 0:
        raise ValueError("paired SAE feature direction has zero norm")
    return {
        "indices": indices,
        "std": std[indices],
        "weights": weights / weight_norm,
        "effects": effect[indices],
    }


def paired_feature_margin(
    row: dict[str, Any],
    vector_key: str,
    feature_model: dict[str, Any],
) -> float:
    import numpy as np

    delta = (
        row[f"exact_{vector_key}"][feature_model["indices"]]
        - row[f"mismatch_{vector_key}"][feature_model["indices"]]
    ) / feature_model["std"]
    return float(np.dot(delta, feature_model["weights"]))


def cross_validate_paired_features(
    rows: list[dict[str, Any]],
    vector_key: str,
    feature_count: int,
    requested_folds: int,
) -> dict[str, Any]:
    margins: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_index, test_ids in enumerate(make_relation_folds(rows, requested_folds)):
        train = [row for row in rows if row["relation_id"] not in test_ids]
        test = [row for row in rows if row["relation_id"] in test_ids]
        feature_model = fit_paired_feature_model(train, vector_key, feature_count)
        fold_margins = [
            paired_feature_margin(row, vector_key, feature_model) for row in test
        ]
        margins.extend(fold_margins)
        fold_rows.append(
            {
                "fold": fold_index,
                "test_relation_ids": sorted(test_ids),
                "accuracy": sum(margin > 0 for margin in fold_margins)
                / len(fold_margins),
            }
        )
    return {
        "accuracy": sum(margin > 0 for margin in margins) / len(margins),
        "margins": margins,
        "folds": fold_rows,
    }


def top_delta_features(
    exact_vector: Any,
    mismatch_vector: Any,
    limit: int = 6,
) -> list[dict[str, Any]]:
    import numpy as np

    delta = exact_vector - mismatch_vector
    indices = np.argsort(np.abs(delta))[-limit:][::-1]
    return [
        {"feature_id": int(index), "exact_minus_mismatch": float(delta[index])}
        for index in indices
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + f"_{args.mode}"
    bundle = load_bundle(args.bundle)
    controls = bundle["controls"]
    if args.mode == "smoke":
        controls = select_smoke_controls(controls)

    if not torch.cuda.is_available():
        raise RuntimeError("this probe requires a CUDA GPU")
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(bundle["model_id"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        bundle["model_id"],
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    layer_index = int(bundle["sae_layer"])
    sae_path = hf_hub_download(
        repo_id=bundle["sae_repo"],
        filename=f"layer{layer_index}.sae.pt",
    )
    try:
        sae = torch.load(sae_path, map_location=device, weights_only=True)
    except TypeError:
        sae = torch.load(sae_path, map_location=device)
    w_enc = sae["W_enc"].to(device=device, dtype=torch.float32)
    b_enc = sae["b_enc"].to(device=device, dtype=torch.float32)
    del sae

    letter_ids: dict[str, int] = {}
    for letter in ("A", "B"):
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{letter!r} is not a single token: {ids}")
        letter_ids[letter] = ids[0]

    def token_positions(
        offsets: list[tuple[int, int]],
        span: list[int],
    ) -> list[int]:
        start, end = span
        positions = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > start and token_start < end and token_end > token_start
        ]
        if not positions:
            raise ValueError(f"no token positions overlap character span {span}")
        return positions

    def capture_sae_vectors(
        text: str,
        artist_span: list[int],
        relation_span: list[int],
    ) -> tuple[Any, Any]:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=256,
        )
        offsets = [tuple(item) for item in encoded.pop("offset_mapping")[0].tolist()]
        inputs = {key: value.to(device) for key, value in encoded.items()}
        captured: dict[str, Any] = {}

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = hidden.detach()[0]

        handle = model.model.layers[layer_index].register_forward_hook(hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        hidden = captured["hidden"]

        def encode(positions: list[int]) -> Any:
            selected = hidden[positions].to(dtype=torch.float32)
            activations = torch.relu(selected @ w_enc.T + b_enc)
            return activations.mean(dim=0).cpu().numpy().astype(np.float32)

        artist_positions = token_positions(offsets, artist_span)
        relation_positions = token_positions(offsets, relation_span)
        return encode([artist_positions[-1]]), encode(relation_positions)

    relation_rows: list[dict[str, Any]] = []
    for control in controls:
        row: dict[str, Any] = dict(control)
        for condition, artist in (
            ("exact", control["correct_artist"]),
            ("mismatch", control["wrong_artist"]),
        ):
            text, artist_span, relation_span = build_sae_text(
                bundle["sae_template"],
                control["title"],
                artist,
            )
            artist_end, pair_end = capture_sae_vectors(
                text,
                artist_span,
                relation_span,
            )
            row[f"{condition}_artist_end"] = artist_end
            row[f"{condition}_pair_end"] = pair_end
        relation_rows.append(row)

    likelihood_specs: list[dict[str, Any]] = []
    for row in relation_rows:
        for title_condition, title in (
            ("title", row["title"]),
            ("neutral", row["neutral_title"]),
        ):
            prefix = build_likelihood_prefix(bundle["likelihood_template"], title)
            for artist_condition, artist in (
                ("correct", row["correct_artist"]),
                ("wrong", row["wrong_artist"]),
            ):
                likelihood_specs.append(
                    {
                        "relation_id": row["relation_id"],
                        "key": f"{title_condition}_{artist_condition}",
                        "prefix": prefix,
                        "continuation": artist,
                    }
                )

    def batch_continuation_scores(
        specs: list[dict[str, Any]],
        batch_size: int = 8,
    ) -> None:
        for start in range(0, len(specs), batch_size):
            batch = specs[start : start + batch_size]
            texts = [spec["prefix"] + spec["continuation"] for spec in batch]
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                return_offsets_mapping=True,
                padding=True,
                truncation=True,
                max_length=256,
            )
            offsets_batch = encoded.pop("offset_mapping").tolist()
            inputs = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            for index, spec in enumerate(batch):
                span = [len(spec["prefix"]), len(texts[index])]
                offsets = [tuple(item) for item in offsets_batch[index]]
                positions = token_positions(offsets, span)
                token_logps: list[float] = []
                for position in positions:
                    if position == 0:
                        raise ValueError("continuation unexpectedly begins at token zero")
                    target_id = int(inputs["input_ids"][index, position].item())
                    step_logits = logits[index, position - 1].to(dtype=torch.float32)
                    token_logps.append(
                        float(
                            (
                                step_logits[target_id] - torch.logsumexp(step_logits, dim=-1)
                            ).item()
                        )
                    )
                spec["sum_logp"] = float(sum(token_logps))
                spec["mean_logp"] = float(sum(token_logps) / len(token_logps))
                spec["token_count"] = len(token_logps)

    batch_continuation_scores(likelihood_specs)
    likelihood_by_relation: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in likelihood_specs:
        likelihood_by_relation.setdefault(spec["relation_id"], {})[spec["key"]] = spec

    choice_specs: list[dict[str, Any]] = []
    for row in relation_rows:
        for title_condition, title in (
            ("title", row["title"]),
            ("neutral", row["neutral_title"]),
        ):
            for order, (artist_a, artist_b, correct_letter) in enumerate(
                (
                    (row["correct_artist"], row["wrong_artist"], "A"),
                    (row["wrong_artist"], row["correct_artist"], "B"),
                )
            ):
                choice_specs.append(
                    {
                        "relation_id": row["relation_id"],
                        "key": f"{title_condition}_order{order}",
                        "correct_letter": correct_letter,
                        "prompt": render_choice_prompt(
                            bundle["choice_template"],
                            title,
                            artist_a,
                            artist_b,
                        ),
                    }
                )

    def batch_choice_logits(specs: list[dict[str, Any]], batch_size: int = 8) -> None:
        for start in range(0, len(specs), batch_size):
            batch = specs[start : start + batch_size]
            encoded = tokenizer(
                [spec["prompt"] for spec in batch],
                return_tensors="pt",
                padding=True,
            ).to(device)
            with torch.no_grad():
                logits = model(**encoded).logits
            for index, spec in enumerate(batch):
                final_position = int(
                    torch.nonzero(encoded["attention_mask"][index], as_tuple=False)[-1]
                    .item()
                )
                final_logits = logits[index, final_position]
                correct_letter = spec["correct_letter"]
                wrong_letter = "B" if correct_letter == "A" else "A"
                spec["correct_minus_wrong"] = float(
                    (
                        final_logits[letter_ids[correct_letter]]
                        - final_logits[letter_ids[wrong_letter]]
                    ).item()
                )

    batch_choice_logits(choice_specs)
    choice_by_relation: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in choice_specs:
        choice_by_relation.setdefault(spec["relation_id"], {})[spec["key"]] = spec

    for row in relation_rows:
        likelihood = likelihood_by_relation[row["relation_id"]]
        title_correct = likelihood["title_correct"]["mean_logp"]
        title_wrong = likelihood["title_wrong"]["mean_logp"]
        neutral_correct = likelihood["neutral_correct"]["mean_logp"]
        neutral_wrong = likelihood["neutral_wrong"]["mean_logp"]
        row["direct_raw_margin"] = title_correct - title_wrong
        row["direct_pmi_margin"] = (
            (title_correct - neutral_correct) - (title_wrong - neutral_wrong)
        )
        row["likelihood_scores"] = {
            key: {
                "mean_logp": value["mean_logp"],
                "sum_logp": value["sum_logp"],
                "token_count": value["token_count"],
            }
            for key, value in likelihood.items()
        }
        choice = choice_by_relation[row["relation_id"]]
        title_choice = (
            choice["title_order0"]["correct_minus_wrong"]
            + choice["title_order1"]["correct_minus_wrong"]
        ) / 2.0
        neutral_choice = (
            choice["neutral_order0"]["correct_minus_wrong"]
            + choice["neutral_order1"]["correct_minus_wrong"]
        ) / 2.0
        row["choice_raw_margin"] = title_choice
        row["choice_prior_margin"] = neutral_choice
        row["choice_pmi_margin"] = title_choice - neutral_choice

    feature_count = int(bundle["feature_count"])
    fold_count = int(bundle["pair_folds"])
    artist_end_cv = cross_validate_paired_features(
        relation_rows,
        "artist_end",
        feature_count,
        fold_count,
    )
    pair_end_cv = cross_validate_paired_features(
        relation_rows,
        "pair_end",
        feature_count,
        fold_count,
    )
    final_artist_model = fit_paired_feature_model(
        relation_rows,
        "artist_end",
        feature_count,
    )
    final_pair_model = fit_paired_feature_model(
        relation_rows,
        "pair_end",
        feature_count,
    )
    for row in relation_rows:
        row["artist_end_in_sample_margin"] = paired_feature_margin(
            row,
            "artist_end",
            final_artist_model,
        )
        row["pair_end_in_sample_margin"] = paired_feature_margin(
            row,
            "pair_end",
            final_pair_model,
        )
        row["pair_end_top_delta_features"] = top_delta_features(
            row["exact_pair_end"],
            row["mismatch_pair_end"],
        )

    direct_accuracy = sum(row["direct_pmi_margin"] > 0 for row in relation_rows) / len(
        relation_rows
    )
    choice_accuracy = sum(row["choice_pmi_margin"] > 0 for row in relation_rows) / len(
        relation_rows
    )
    threshold = float(bundle["interpretation_threshold"])
    numeric_fields = (
        "direct_raw_margin",
        "direct_pmi_margin",
        "choice_raw_margin",
        "choice_prior_margin",
        "choice_pmi_margin",
        "artist_end_in_sample_margin",
        "pair_end_in_sample_margin",
    )
    technical_gate = len(relation_rows) == len(controls) and all(
        math.isfinite(float(row[field]))
        for row in relation_rows
        for field in numeric_fields
    )
    behavioral_gate = technical_gate and direct_accuracy >= threshold
    mechanistic_gate = technical_gate and pair_end_cv["accuracy"] >= threshold
    intervention_gate = (
        args.mode == "full" and behavioral_gate and mechanistic_gate
    )

    def serialize_feature_model(feature_model: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"feature_id": int(feature_id), "effect": float(effect)}
            for feature_id, effect in zip(
                feature_model["indices"][:20],
                feature_model["effects"][:20],
            )
        ]

    summary = {
        "run_id": run_id,
        "mode": args.mode,
        "model_id": bundle["model_id"],
        "sae_repo": bundle["sae_repo"],
        "sae_layer": layer_index,
        "relation_count": len(relation_rows),
        "language_counts": {
            language: sum(row["language"] == language for row in relation_rows)
            for language in ("en", "zh")
        },
        "letter_token_ids": letter_ids,
        "direct_pmi_accuracy": direct_accuracy,
        "choice_pmi_accuracy": choice_accuracy,
        "artist_end_sae_cv": artist_end_cv,
        "pair_end_sae_cv": pair_end_cv,
        "selected_artist_end_features": serialize_feature_model(final_artist_model),
        "selected_pair_end_features": serialize_feature_model(final_pair_model),
        "technical_gate": technical_gate,
        "behavioral_gate": behavioral_gate,
        "mechanistic_gate": mechanistic_gate,
        "intervention_gate": intervention_gate,
        "threshold": threshold,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }

    for row in relation_rows:
        compact = {
            key: value
            for key, value in row.items()
            if not key.startswith("exact_") and not key.startswith("mismatch_")
        }
        print(
            "REL_BIND_ROW_JSON="
            + json.dumps(compact, ensure_ascii=True, separators=(",", ":"))
        )
    print(
        "REL_BIND_SUMMARY_JSON="
        + json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
