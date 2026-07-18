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
"""Run the generation-time title-artist grounding probe on Qwen-Scope."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EMBEDDED_BUNDLE_B64 = "__EXPERIMENT_BUNDLE_B64__"
FIELD_PATTERN = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?"
    r"(?:\*\*)?(?P<field>title|artist|reason)(?:\*\*)?"
    r"\s*:\s*(?P<value>.*)\s*$",
    re.IGNORECASE,
)
OTHER_FIELD_PATTERN = re.compile(r"^\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?[^:]+:\s*")
INLINE_BLOCK_PATTERN = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?(?:\*\*)?title(?:\*\*)?\s*:\s*"
    r"(?P<title>.*?)\s+(?:\*\*)?artist(?:\*\*)?\s*:\s*"
    r"(?P<artist>.*?)\s+(?:\*\*)?reason(?:\*\*)?\s*:\s*"
    r"(?P<reason>.*?)"
    r"(?=\r?\n\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?(?:\*\*)?title(?:\*\*)?\s*:|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
TITLE_BY_BLOCK_PATTERN = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?(?:\*\*)?title(?:\*\*)?\s*:\s*"
    r"(?P<title>\"[^\"\r\n]+\"|'[^'\r\n]+'|[^\r\n]+?)\s+"
    r"(?:by|-)\s+(?P<by_artist>[^\r\n]+?)\s*\r?\n"
    r"(?:\s*(?:\*\*)?artist(?:\*\*)?\s*:\s*(?P<artist_line>[^\r\n]+?)\s*\r?\n)?"
    r"\s*(?:\*\*)?reason(?:\*\*)?\s*:\s*(?P<reason>.*?)"
    r"(?=\r?\n\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?(?:\*\*)?title(?:\*\*)?\s*:|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
TITLE_ARTIST_LINE_PATTERN = re.compile(
    r"^\s*(?:[-*]\s*)?(?P<rank>\d+)[.)]\s*(?:\*\*)?title(?:\*\*)?\s*:\s*"
    r"(?P<title>.*?)\s+(?:\*\*)?artist(?:\*\*)?\s*:\s*(?P<artist>.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
NUMBERED_REASON_PATTERN = re.compile(
    r"^\s*(?P<rank>\d+)[.)]\s*(?P<reason>.+?)\s*$",
    re.MULTILINE,
)


def clean_value_span(text: str, start: int, end: int) -> tuple[str, list[int]]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    quote_pairs = {'"': '"', "'": "'", "“": "”", "‘": "’"}
    if end - start >= 2 and text[start] in quote_pairs and text[end - 1] == quote_pairs[text[start]]:
        start += 1
        end -= 1
    return text[start:end], [start, end]


def load_bundle(path: Path | None = None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    if EMBEDDED_BUNDLE_B64 == "__EXPERIMENT_BUNDLE_B64__":
        raise ValueError("no bundle supplied and embedded bundle was not injected")
    payload = base64.b64decode(EMBEDDED_BUNDLE_B64).decode("utf-8")
    return json.loads(payload)


def parse_playlist(full_text: str) -> list[dict[str, Any]]:
    """Parse complete Title/Artist/Reason blocks and retain character spans."""

    title_by_blocks: list[dict[str, Any]] = []
    for match in TITLE_BY_BLOCK_PATTERN.finditer(full_text):
        title, title_span = clean_value_span(
            full_text,
            match.start("title"),
            match.end("title"),
        )
        artist_group = "artist_line" if match.group("artist_line") else "by_artist"
        artist, artist_span = clean_value_span(
            full_text,
            match.start(artist_group),
            match.end(artist_group),
        )
        reason, reason_span = clean_value_span(
            full_text,
            match.start("reason"),
            match.end("reason"),
        )
        title_by_blocks.append(
            {
                "title": title,
                "title_span": title_span,
                "artist": artist,
                "artist_span": artist_span,
                "reason": reason,
                "reason_span": reason_span,
            }
        )
    if title_by_blocks:
        return title_by_blocks[:5]

    inline_blocks: list[dict[str, Any]] = []
    for match in INLINE_BLOCK_PATTERN.finditer(full_text):
        row: dict[str, Any] = {}
        for field in ("title", "artist", "reason"):
            value, span = clean_value_span(
                full_text,
                match.start(field),
                match.end(field),
            )
            row[field] = value
            row[f"{field}_span"] = span
        if all(row.get(field) for field in ("title", "artist", "reason")):
            inline_blocks.append(row)
    if inline_blocks:
        return inline_blocks[:5]

    title_artist_rows: dict[int, dict[str, Any]] = {}
    for match in TITLE_ARTIST_LINE_PATTERN.finditer(full_text):
        rank = int(match.group("rank"))
        title, title_span = clean_value_span(
            full_text,
            match.start("title"),
            match.end("title"),
        )
        artist, artist_span = clean_value_span(
            full_text,
            match.start("artist"),
            match.end("artist"),
        )
        title_artist_rows[rank] = {
            "title": title,
            "title_span": title_span,
            "artist": artist,
            "artist_span": artist_span,
        }
    recommendations_index = full_text.casefold().find("recommendations:")
    if title_artist_rows and recommendations_index >= 0:
        reason_text = full_text[recommendations_index + len("recommendations:") :]
        reason_offset = recommendations_index + len("recommendations:")
        for match in NUMBERED_REASON_PATTERN.finditer(reason_text):
            rank = int(match.group("rank"))
            if rank not in title_artist_rows:
                continue
            reason, span = clean_value_span(
                full_text,
                reason_offset + match.start("reason"),
                reason_offset + match.end("reason"),
            )
            title_artist_rows[rank]["reason"] = reason
            title_artist_rows[rank]["reason_span"] = span
        completed = [
            row
            for rank, row in sorted(title_artist_rows.items())
            if rank <= 5 and all(row.get(field) for field in ("title", "artist", "reason"))
        ]
        if completed:
            return completed[:5]

    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    pending_field: str | None = None
    offset = 0
    for line in full_text.splitlines(keepends=True):
        line_without_break = line.rstrip("\r\n")
        match = FIELD_PATTERN.match(line_without_break)
        if match:
            field = match.group("field").lower()
            value = match.group("value").strip()
            pending_field = None
            if field == "title":
                if current and all(current.get(name) for name in ("title", "artist", "reason")):
                    blocks.append(current)
                current = {}
            if current is not None and value:
                value, span = clean_value_span(
                    full_text,
                    offset + match.start("value"),
                    offset + match.end("value"),
                )
                current[field] = value
                current[f"{field}_span"] = span
                if field == "reason" and all(
                    current.get(name) for name in ("title", "artist", "reason")
                ):
                    blocks.append(current)
                    current = None
            elif current is not None:
                pending_field = field
        elif current is not None and pending_field and line_without_break.strip():
            if OTHER_FIELD_PATTERN.match(line_without_break):
                pending_field = None
            else:
                value, span = clean_value_span(full_text, offset, offset + len(line_without_break))
                current[pending_field] = value
                current[f"{pending_field}_span"] = span
                pending_field = None
        offset += len(line)
    if current and all(current.get(name) for name in ("title", "artist", "reason")):
        blocks.append(current)
    return blocks[:5]


def render_verification_prompt(
    template: str,
    *,
    title: str,
    artist: str,
    source_condition: str,
    options: list[tuple[str, str]],
) -> tuple[str, dict[str, str]]:
    source_context = ""
    if source_condition == "self_attributed":
        source_context = (
            "Earlier you generated this exact title-artist pair as a music "
            "recommendation."
        )
    option_lines = []
    letter_to_type: dict[str, str] = {}
    for index, (option_type, text) in enumerate(options):
        letter = chr(ord("A") + index)
        option_lines.append(f"{letter}. {text}")
        letter_to_type[letter] = option_type
    prompt = (
        template.replace("{{source_context}}", source_context)
        .replace("{{title}}", title)
        .replace("{{artist}}", artist)
        .replace("{{options}}", "\n".join(option_lines))
    )
    return prompt, letter_to_type


def select_smoke_controls(controls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for group in ("known_exact", "artist_mismatch", "synthetic_pair"):
        selected.extend([row for row in controls if row["group"] == group][:6])
    return selected


def build_pair_text(prompt: str, title: str, artist: str) -> str:
    return (
        prompt
        + title
        + "\n   Artist: "
        + artist
        + "\n   Reason: This catalog control fits the requested listening context.\n"
    )


def balanced_accuracy(labels: list[int], predictions: list[int]) -> float:
    recalls = []
    for label in (0, 1):
        indices = [index for index, value in enumerate(labels) if value == label]
        if not indices:
            continue
        recalls.append(
            sum(predictions[index] == label for index in indices) / len(indices)
        )
    return sum(recalls) / len(recalls) if recalls else 0.0


def make_pair_folds(examples: list[dict[str, Any]], requested_folds: int = 5) -> list[set[str]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for example in examples:
        pair_id = example["pair_id"]
        if pair_id in seen:
            continue
        seen.add(pair_id)
        grouped[example["label"]].append(pair_id)
    minimum = min(len(ids) for ids in grouped.values())
    fold_count = min(requested_folds, minimum)
    if fold_count < 2:
        raise ValueError("at least two pair IDs per class are required")
    rng = random.Random(20260710)
    folds = [set() for _ in range(fold_count)]
    for label in sorted(grouped):
        ids = sorted(grouped[label])
        rng.shuffle(ids)
        for index, pair_id in enumerate(ids):
            folds[index % fold_count].add(pair_id)
    return folds


def fit_knownness_model(examples: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    x = np.stack([example["vector"] for example in examples]).astype(np.float32)
    y = np.asarray([example["label"] for example in examples], dtype=np.int64)
    active = (x > 0).mean(axis=0) >= 0.20
    exact = x[y == 1]
    non_exact = x[y == 0]
    pooled = np.sqrt((exact.var(axis=0) + non_exact.var(axis=0)) / 2.0 + 1e-6)
    effect = (exact.mean(axis=0) - non_exact.mean(axis=0)) / pooled
    effect[~active] = 0.0
    feature_count = min(64, int(active.sum()))
    if feature_count == 0:
        raise ValueError("no SAE features passed the activation-rate filter")
    feature_indices = np.argsort(np.abs(effect))[-feature_count:][::-1]
    selected = x[:, feature_indices]
    mean = selected.mean(axis=0)
    std = selected.std(axis=0) + 1e-6
    normalized = (selected - mean) / std
    return {
        "feature_indices": feature_indices,
        "feature_effects": effect[feature_indices],
        "mean": mean,
        "std": std,
        "exact_centroid": normalized[y == 1].mean(axis=0),
        "non_exact_centroid": normalized[y == 0].mean(axis=0),
    }


def score_knownness(vector: Any, model: dict[str, Any]) -> float:
    import numpy as np

    selected = vector[model["feature_indices"]]
    normalized = (selected - model["mean"]) / model["std"]
    exact_distance = np.linalg.norm(normalized - model["exact_centroid"])
    non_exact_distance = np.linalg.norm(normalized - model["non_exact_centroid"])
    return float(non_exact_distance - exact_distance)


def cross_validate_knownness(examples: list[dict[str, Any]]) -> dict[str, Any]:
    labels: list[int] = []
    predictions: list[int] = []
    scores: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_index, test_ids in enumerate(make_pair_folds(examples)):
        train = [row for row in examples if row["pair_id"] not in test_ids]
        test = [row for row in examples if row["pair_id"] in test_ids]
        model = fit_knownness_model(train)
        fold_labels = [row["label"] for row in test]
        fold_scores = [score_knownness(row["vector"], model) for row in test]
        fold_predictions = [int(score > 0) for score in fold_scores]
        labels.extend(fold_labels)
        scores.extend(fold_scores)
        predictions.extend(fold_predictions)
        fold_rows.append(
            {
                "fold": fold_index,
                "test_pair_ids": sorted(test_ids),
                "balanced_accuracy": balanced_accuracy(fold_labels, fold_predictions),
            }
        )
    return {
        "balanced_accuracy": balanced_accuracy(labels, predictions),
        "folds": fold_rows,
        "labels": labels,
        "scores": scores,
    }


def top_features(vector: Any, limit: int = 8) -> list[dict[str, Any]]:
    import numpy as np

    indices = np.argsort(vector)[-limit:][::-1]
    return [
        {"feature_id": int(index), "activation": float(vector[index])}
        for index in indices
        if vector[index] > 0
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + f"_{args.mode}"
    bundle = load_bundle(args.bundle)
    contexts = bundle["contexts"]
    seeds = bundle["seeds"]
    controls = bundle["controls"]
    if args.mode == "smoke":
        contexts = contexts[:1]
        seeds = seeds[:1]
        controls = select_smoke_controls(controls)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("this probe requires a CUDA GPU")
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

    def capture_hidden(text: str) -> tuple[Any, list[tuple[int, int]]]:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=1024,
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
        return captured["hidden"], offsets

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

    def encode_positions(hidden: Any, positions: list[int]) -> Any:
        selected = hidden[positions].to(dtype=torch.float32)
        activations = torch.relu(selected @ w_enc.T + b_enc)
        return activations.mean(dim=0).cpu().numpy().astype(np.float32)

    generation_rows: list[dict[str, Any]] = []
    generation_outputs: dict[str, str] = {}
    for context in contexts:
        prompt = context["generation_prompt"].rstrip()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        for seed in seeds:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=float(bundle["generation"]["temperature"]),
                    top_p=float(bundle["generation"]["top_p"]),
                    max_new_tokens=int(bundle["generation"]["max_new_tokens"]),
                    pad_token_id=tokenizer.pad_token_id,
                )
            completion_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            full_text = prompt + completion
            parsed = parse_playlist(full_text)
            generation_id = f"{context['context_id']}__seed{seed}"
            generation_outputs[generation_id] = completion
            if parsed:
                hidden, offsets = capture_hidden(full_text)
            for rank, item in enumerate(parsed, 1):
                title_positions = token_positions(offsets, item["title_span"])
                artist_positions = token_positions(offsets, item["artist_span"])
                reason_positions = token_positions(offsets, item["reason_span"])
                title_vector = encode_positions(hidden, title_positions)
                artist_vector = encode_positions(hidden, artist_positions)
                pair_end_vector = encode_positions(hidden, [artist_positions[-1]])
                reason_vector = encode_positions(hidden, reason_positions)
                generation_rows.append(
                    {
                        "record_type": "generated_pair",
                        "generation_id": generation_id,
                        "context_id": context["context_id"],
                        "dimension": context["dimension"],
                        "seed": seed,
                        "rank": rank,
                        "title": item["title"],
                        "artist": item["artist"],
                        "reason": item["reason"],
                        "title_top_features": top_features(title_vector),
                        "artist_top_features": top_features(artist_vector),
                        "pair_end_top_features": top_features(pair_end_vector),
                        "reason_top_features": top_features(reason_vector),
                        "pair_end_vector": pair_end_vector,
                    }
                )

    control_examples: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for context in contexts:
        prompt = context["generation_prompt"].rstrip()
        for control in controls:
            text = build_pair_text(prompt, control["title"], control["artist"])
            parsed = parse_playlist(text)
            if len(parsed) != 1:
                raise ValueError(f"control parse failed: {control['pair_id']}")
            hidden, offsets = capture_hidden(text)
            artist_positions = token_positions(offsets, parsed[0]["artist_span"])
            vector = encode_positions(hidden, [artist_positions[-1]])
            label = int(control["group"] == "known_exact")
            control_examples.append(
                {
                    "pair_id": control["pair_id"],
                    "context_id": context["context_id"],
                    "label": label,
                    "vector": vector,
                }
            )
            control_rows.append(
                {
                    **control,
                    "record_type": "control_pair",
                    "context_id": context["context_id"],
                    "pair_end_top_features": top_features(vector),
                    "pair_end_vector": vector,
                }
            )

    cv = cross_validate_knownness(control_examples)
    final_knownness_model = fit_knownness_model(control_examples)
    for row in generation_rows:
        row["generation_knownness"] = score_knownness(
            row["pair_end_vector"],
            final_knownness_model,
        )
    for row in control_rows:
        row["generation_knownness"] = score_knownness(
            row["pair_end_vector"],
            final_knownness_model,
        )

    all_pairs: list[dict[str, Any]] = []
    for index, row in enumerate(generation_rows):
        all_pairs.append(
            {
                "key": f"generated::{index}",
                "title": row["title"],
                "artist": row["artist"],
                "target": row,
            }
        )
    unique_control_targets: dict[str, dict[str, Any]] = {}
    for row in control_rows:
        unique_control_targets.setdefault(row["pair_id"], row)
    for pair_id, row in unique_control_targets.items():
        all_pairs.append(
            {
                "key": f"control::{pair_id}",
                "title": row["title"],
                "artist": row["artist"],
                "target": row,
            }
        )

    verification_template = bundle["verification_template"]
    option_text = bundle["verification_options"]
    orders = [
        [("known_exact", option_text["known_exact"]), ("unknown", option_text["unknown"])],
        [("unknown", option_text["unknown"]), ("known_exact", option_text["known_exact"])],
    ]
    prompt_specs: list[dict[str, Any]] = []
    null_specs: list[dict[str, Any]] = []
    for source_condition in ("neutral", "self_attributed"):
        for order_index, options in enumerate(orders):
            null_prompt, null_mapping = render_verification_prompt(
                verification_template,
                title="Sample Catalog Title",
                artist="Sample Catalog Artist",
                source_condition=source_condition,
                options=options,
            )
            null_specs.append(
                {
                    "source_condition": source_condition,
                    "order_index": order_index,
                    "prompt": null_prompt,
                    "mapping": null_mapping,
                }
            )
            for pair in all_pairs:
                prompt, mapping = render_verification_prompt(
                    verification_template,
                    title=pair["title"],
                    artist=pair["artist"],
                    source_condition=source_condition,
                    options=options,
                )
                prompt_specs.append(
                    {
                        "key": pair["key"],
                        "source_condition": source_condition,
                        "order_index": order_index,
                        "prompt": prompt,
                        "mapping": mapping,
                    }
                )

    def batch_letter_logits(specs: list[dict[str, Any]], batch_size: int = 8) -> None:
        tokenizer.padding_side = "right"
        for start in range(0, len(specs), batch_size):
            batch = specs[start : start + batch_size]
            encoded = tokenizer(
                [row["prompt"] for row in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            with torch.no_grad():
                logits = model(**encoded).logits
            lengths = encoded["attention_mask"].sum(dim=1)
            for index, row in enumerate(batch):
                final_logits = logits[index, int(lengths[index].item()) - 1]
                row["letter_logits"] = {
                    letter: float(final_logits[token_id].item())
                    for letter, token_id in letter_ids.items()
                }

    batch_letter_logits(null_specs, batch_size=4)
    batch_letter_logits(prompt_specs)
    null_priors: dict[tuple[str, int], float] = {}
    for row in null_specs:
        mapping = row["mapping"]
        unknown_letter = next(letter for letter, kind in mapping.items() if kind == "unknown")
        known_letter = next(letter for letter, kind in mapping.items() if kind == "known_exact")
        null_priors[(row["source_condition"], row["order_index"])] = (
            row["letter_logits"][unknown_letter] - row["letter_logits"][known_letter]
        )

    grouped_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    raw_verification_rows: list[dict[str, Any]] = []
    for row in prompt_specs:
        mapping = row["mapping"]
        unknown_letter = next(letter for letter, kind in mapping.items() if kind == "unknown")
        known_letter = next(letter for letter, kind in mapping.items() if kind == "known_exact")
        raw = row["letter_logits"][unknown_letter] - row["letter_logits"][known_letter]
        calibrated = raw - null_priors[(row["source_condition"], row["order_index"])]
        grouped_scores[(row["key"], row["source_condition"])].append(calibrated)
        raw_verification_rows.append(
            {
                "key": row["key"],
                "source_condition": row["source_condition"],
                "order_index": row["order_index"],
                "raw_unknown_logit": raw,
                "calibrated_unknown_logit": calibrated,
            }
        )
    target_by_key = {pair["key"]: pair["target"] for pair in all_pairs}
    for (key, source_condition), values in grouped_scores.items():
        target_by_key[key][f"{source_condition}_unknown_logit"] = float(sum(values) / len(values))

    generated_generation_counts = Counter(row["generation_id"] for row in generation_rows)
    expected_generation_ids = {
        f"{context['context_id']}__seed{seed}" for context in contexts for seed in seeds
    }
    valid_generation_count = sum(
        generated_generation_counts[generation_id] >= 3
        for generation_id in expected_generation_ids
    )
    control_context_scores: dict[str, dict[str, float]] = {}
    context_directions: list[bool] = []
    for context in contexts:
        subset = [row for row in control_rows if row["context_id"] == context["context_id"]]
        labels = [int(row["group"] == "known_exact") for row in subset]
        predictions = [int(row["generation_knownness"] > 0) for row in subset]
        exact_scores = [row["generation_knownness"] for row in subset if row["group"] == "known_exact"]
        non_exact_scores = [row["generation_knownness"] for row in subset if row["group"] != "known_exact"]
        exact_mean = float(sum(exact_scores) / len(exact_scores))
        non_exact_mean = float(sum(non_exact_scores) / len(non_exact_scores))
        direction_ok = exact_mean > non_exact_mean
        context_directions.append(direction_ok)
        control_context_scores[context["context_id"]] = {
            "balanced_accuracy": balanced_accuracy(labels, predictions),
            "known_exact_mean": exact_mean,
            "non_exact_mean": non_exact_mean,
            "direction_ok": direction_ok,
        }

    direction_stable = all(context_directions)
    generation_gate = (
        valid_generation_count == len(expected_generation_ids)
        and len(generation_rows) >= 3 * len(expected_generation_ids)
        if args.mode == "smoke"
        else valid_generation_count >= 10 and len(generation_rows) >= 50
    )
    technical_gate = (
        generation_gate
        and len(control_examples) == len(contexts) * len(controls)
        and set(letter_ids) == {"A", "B"}
    )
    interpretation_gate = (
        technical_gate
        and cv["balanced_accuracy"] >= 0.80
        and direction_stable
        and (args.mode == "smoke" or valid_generation_count >= 10)
        and (args.mode == "smoke" or len(generation_rows) >= 50)
    )
    summary = {
        "run_id": run_id,
        "mode": args.mode,
        "model_id": bundle["model_id"],
        "sae_repo": bundle["sae_repo"],
        "sae_layer": layer_index,
        "letter_token_ids": letter_ids,
        "context_count": len(contexts),
        "seed_count": len(seeds),
        "expected_generations": len(expected_generation_ids),
        "valid_generations": valid_generation_count,
        "generated_pair_count": len(generation_rows),
        "control_pair_count": len(controls),
        "control_example_count": len(control_examples),
        "control_balanced_accuracy": cv["balanced_accuracy"],
        "control_context_scores": control_context_scores,
        "control_direction_stable": direction_stable,
        "technical_gate": technical_gate,
        "selected_knownness_features": [
            {
                "feature_id": int(feature_id),
                "effect": float(effect),
            }
            for feature_id, effect in zip(
                final_knownness_model["feature_indices"][:20],
                final_knownness_model["feature_effects"][:20],
            )
        ],
        "interpretation_gate": interpretation_gate,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "artifact_repo": bundle["hf_result_repo"],
        "artifact_prefix": f"generation_time/{run_id}",
        "uploaded": False,
    }

    serializable_rows: list[dict[str, Any]] = []
    for row in generation_rows + control_rows:
        serializable = {key: value for key, value in row.items() if key != "pair_end_vector"}
        serializable_rows.append(serializable)
    result_buffer = io.BytesIO()
    for row in serializable_rows:
        result_buffer.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    result_buffer.seek(0)
    verification_bytes = (
        "\n".join(json.dumps(row, ensure_ascii=False) for row in raw_verification_rows) + "\n"
    ).encode("utf-8")

    artifact_prefix = summary["artifact_prefix"]
    uploaded = False
    upload_error = ""
    if not args.no_upload:
        token = os.environ.get("HF_TOKEN")
        if not token:
            upload_error = "HF_TOKEN was not available"
        else:
            try:
                api = HfApi(token=token)
                api.create_repo(
                    repo_id=bundle["hf_result_repo"],
                    repo_type="dataset",
                    private=True,
                    exist_ok=True,
                )
                api.upload_file(
                    path_or_fileobj=result_buffer,
                    path_in_repo=f"{artifact_prefix}/rows.jsonl",
                    repo_id=bundle["hf_result_repo"],
                    repo_type="dataset",
                )
                summary["uploaded"] = True
                summary_bytes = json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8")
                api.upload_file(
                    path_or_fileobj=io.BytesIO(summary_bytes),
                    path_in_repo=f"{artifact_prefix}/summary.json",
                    repo_id=bundle["hf_result_repo"],
                    repo_type="dataset",
                )
                api.upload_file(
                    path_or_fileobj=io.BytesIO(verification_bytes),
                    path_in_repo=f"{artifact_prefix}/verification_rows.jsonl",
                    repo_id=bundle["hf_result_repo"],
                    repo_type="dataset",
                )
                uploaded = True
            except Exception as exc:
                upload_error = f"{type(exc).__name__}: {exc}"
    summary["uploaded"] = uploaded
    summary["upload_error"] = upload_error
    for generation_id in sorted(generation_outputs):
        compact_rows = []
        for row in generation_rows:
            if row["generation_id"] != generation_id:
                continue
            compact_rows.append(
                {
                    "rank": row["rank"],
                    "title": row["title"],
                    "artist": row["artist"],
                    "reason": row["reason"],
                    "generation_knownness": row["generation_knownness"],
                    "neutral_unknown_logit": row["neutral_unknown_logit"],
                    "self_attributed_unknown_logit": row[
                        "self_attributed_unknown_logit"
                    ],
                    "title_top_features": row["title_top_features"][:2],
                    "artist_top_features": row["artist_top_features"][:2],
                    "pair_end_top_features": row["pair_end_top_features"][:4],
                    "reason_top_features": row["reason_top_features"][:2],
                }
            )
        print(
            "GEN_TIME_GENERATION_JSON="
            + json.dumps(
                {
                    "generation_id": generation_id,
                    "raw_generation": generation_outputs[generation_id],
                    "rows": compact_rows,
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )
        )
    print("GEN_TIME_SUMMARY_JSON=" + json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    print(f"GEN_TIME_ARTIFACT_PATH={bundle['hf_result_repo']}/{artifact_prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
