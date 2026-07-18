# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.8.0",
#   "torch==2.7.1",
#   "transformers==4.53.2",
# ]
# ///
"""Patch controlled vocality reasons through pair and choice paths."""

from __future__ import annotations

import argparse
import base64
import json
import math
import statistics
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


def render_pair_text(reason: str, title: str, artist: str) -> tuple[str, list[int], list[int]]:
    text = f"Music recommendation plan.\nReason: {reason}\nTitle: "
    title_span = [len(text), len(text) + len(title)]
    text += title
    text += "\nArtist: "
    artist_span = [len(text), len(text) + len(artist)]
    text += artist
    return text, title_span, artist_span


def overlapping_positions(offsets: list[tuple[int, int]], span: list[int]) -> list[int]:
    start, end = span
    positions = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > start and token_start < end and token_end > token_start
    ]
    if not positions:
        raise ValueError(f"no token positions overlap character span {span}")
    return positions


def render_choice_prompt(
    reason: str,
    order: list[str],
    records_by_key: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, str]]:
    lines = [
        "Choose the one real track that best matches the stated reason.",
        f"Reason: {reason}",
        "Candidates:",
    ]
    mapping = {}
    for index, track_key in enumerate(order):
        letter = chr(ord("A") + index)
        row = records_by_key[track_key]
        lines.append(f"{letter}. {row['title']} - {row['artist']}")
        mapping[letter] = track_key
    lines.extend(["Answer with one letter only.", "Answer:"])
    return "\n".join(lines), mapping


def recovery(source: float, target: float, patched: float) -> float:
    effect = source - target
    if abs(effect) < 1e-12:
        raise ValueError("cannot normalize zero source-target effect")
    return (patched - target) / effect


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + "_pilot"
    bundle = load_bundle(args.bundle)
    focus = bundle["focus_records"]
    all_records = bundle["all_records"]
    records_by_key = {row["track_key"]: row for row in all_records}
    if not torch.cuda.is_available():
        raise RuntimeError("this probe requires a CUDA GPU")
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(bundle["model_id"], trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        bundle["model_id"],
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    selected_layers = [int(layer) for layer in bundle["selected_layers"]]
    layer_count = len(model.model.layers)
    if any(layer < 1 or layer > layer_count for layer in selected_layers):
        raise ValueError(f"selected layers exceed model depth {layer_count}")
    layer_indices = [layer - 1 for layer in selected_layers]
    components = list(bundle["components"])

    def output_hidden(output: Any) -> Any:
        return output[0] if isinstance(output, tuple) else output

    def component_module(layer: Any, component: str) -> Any:
        if component == "attention":
            return layer.self_attn
        if component == "mlp":
            return layer.mlp
        if component == "full_residual":
            return layer
        raise ValueError(f"unknown component: {component}")

    def encode_pair_side(side: str) -> tuple[Any, list[list[int]], list[list[int]], list[list[int]]]:
        texts = []
        spans = []
        for row in focus:
            matched = row["vocality"]
            flipped = "instrumental" if matched == "vocal" else "vocal"
            condition = matched if side == "source" else flipped
            text, title_span, artist_span = render_pair_text(
                bundle["reasons"][condition],
                row["title"],
                row["artist"],
            )
            texts.append(text)
            spans.append((title_span, artist_span))
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
        )
        offsets_batch = encoded.pop("offset_mapping").tolist()
        targets = []
        predictions = []
        ids = []
        for row_index, raw_offsets in enumerate(offsets_batch):
            offsets = [tuple(item) for item in raw_offsets]
            title_span, artist_span = spans[row_index]
            positions = overlapping_positions(offsets, title_span)
            positions += overlapping_positions(offsets, artist_span)
            targets.append(positions)
            predictions.append([position - 1 for position in positions])
            ids.append([int(encoded["input_ids"][row_index, position]) for position in positions])
        return encoded.to(device), targets, predictions, ids

    source_pair, _, source_pair_predictions, source_pair_ids = encode_pair_side("source")
    target_pair, _, target_pair_predictions, target_pair_ids = encode_pair_side("target")
    if source_pair_ids != target_pair_ids:
        raise ValueError("pair target tokenization differs between reason conditions")

    pair_source_vectors: dict[tuple[int, str], list[Any]] = {}
    handles = []
    for layer_number, layer_index in zip(selected_layers, layer_indices):
        layer = model.model.layers[layer_index]
        for component in components:
            module = component_module(layer, component)

            def capture_hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                key: tuple[int, str] = (layer_number, component),
            ) -> None:
                hidden = output_hidden(output)
                pair_source_vectors[key] = [
                    hidden[row_index, positions].detach().clone()
                    for row_index, positions in enumerate(source_pair_predictions)
                ]

            handles.append(module.register_forward_hook(capture_hook))
    try:
        with torch.no_grad():
            source_pair_outputs = model(**source_pair, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    with torch.no_grad():
        target_pair_outputs = model(**target_pair, use_cache=False)

    def pair_mean_logps(logits: Any, predictions: list[list[int]]) -> list[float]:
        output = []
        for row_index, positions in enumerate(predictions):
            step_logits = logits[row_index, positions].to(dtype=torch.float32)
            ids = torch.tensor(source_pair_ids[row_index], device=device)
            token_logits = step_logits[torch.arange(len(positions), device=device), ids]
            token_logps = token_logits - torch.logsumexp(step_logits, dim=-1)
            output.append(float(token_logps.mean().item()))
        return output

    pair_source_scores = pair_mean_logps(source_pair_outputs.logits, source_pair_predictions)
    pair_target_scores = pair_mean_logps(target_pair_outputs.logits, target_pair_predictions)
    del source_pair_outputs
    del target_pair_outputs

    pair_patched_scores: dict[tuple[int, str], list[float]] = {}
    for layer_number, layer_index in zip(selected_layers, layer_indices):
        layer = model.model.layers[layer_index]
        for component in components:
            module = component_module(layer, component)
            replacements = pair_source_vectors[(layer_number, component)]

            def patch_hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                values: list[Any] = replacements,
            ) -> Any:
                hidden = output_hidden(output)
                patched = hidden.clone()
                for row_index, positions in enumerate(target_pair_predictions):
                    patched[row_index, positions] = values[row_index]
                if isinstance(output, tuple):
                    return (patched, *output[1:])
                return patched

            handle = module.register_forward_hook(patch_hook)
            try:
                with torch.no_grad():
                    outputs = model(**target_pair, use_cache=False)
            finally:
                handle.remove()
            pair_patched_scores[(layer_number, component)] = pair_mean_logps(
                outputs.logits,
                target_pair_predictions,
            )
            del outputs

    pair_rows = []
    for row_index, record in enumerate(focus):
        source = pair_source_scores[row_index]
        target = pair_target_scores[row_index]
        patches = {component: {} for component in components}
        recoveries = {component: {} for component in components}
        for layer_number in selected_layers:
            key = str(layer_number)
            for component in components:
                patched = pair_patched_scores[(layer_number, component)][row_index]
                patches[component][key] = patched
                recoveries[component][key] = recovery(source, target, patched)
        row = {
            **record,
            "source_pair_mean_logp": source,
            "target_pair_mean_logp": target,
            "source_target_effect": source - target,
            "patched_pair_mean_logps": patches,
            "component_recoveries": recoveries,
        }
        pair_rows.append(row)
        print(
            "VOCALITY_PATH_PAIR_ROW_JSON="
            + json.dumps(row, ensure_ascii=True, separators=(",", ":"))
        )

    choice_specs = []
    for class_label in ("vocal", "instrumental"):
        target_label = "instrumental" if class_label == "vocal" else "vocal"
        for order_index, order in enumerate(bundle["candidate_orders"]):
            source_prompt, mapping = render_choice_prompt(
                bundle["reasons"][class_label],
                order,
                records_by_key,
            )
            target_prompt, target_mapping = render_choice_prompt(
                bundle["reasons"][target_label],
                order,
                records_by_key,
            )
            if mapping != target_mapping:
                raise ValueError("choice mappings differ across reason conditions")
            choice_specs.append(
                {
                    "class_label": class_label,
                    "source_reason": class_label,
                    "target_reason": target_label,
                    "order_index": order_index,
                    "source_prompt": source_prompt,
                    "target_prompt": target_prompt,
                    "mapping": mapping,
                }
            )

    source_choice = tokenizer(
        [row["source_prompt"] for row in choice_specs],
        return_tensors="pt",
        padding=True,
    ).to(device)
    target_choice = tokenizer(
        [row["target_prompt"] for row in choice_specs],
        return_tensors="pt",
        padding=True,
    ).to(device)

    def final_positions(attention_mask: Any) -> Any:
        return torch.tensor(
            [int(torch.nonzero(row, as_tuple=False)[-1].item()) for row in attention_mask],
            device=device,
            dtype=torch.long,
        )

    source_choice_positions = final_positions(source_choice["attention_mask"])
    target_choice_positions = final_positions(target_choice["attention_mask"])
    batch_indices = torch.arange(len(choice_specs), device=device)

    choice_source_vectors: dict[tuple[int, str], Any] = {}
    handles = []
    for layer_number, layer_index in zip(selected_layers, layer_indices):
        layer = model.model.layers[layer_index]
        for component in components:
            module = component_module(layer, component)

            def capture_choice_hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                key: tuple[int, str] = (layer_number, component),
            ) -> None:
                choice_source_vectors[key] = output_hidden(output)[
                    batch_indices,
                    source_choice_positions,
                ].detach().clone()

            handles.append(module.register_forward_hook(capture_choice_hook))
    try:
        with torch.no_grad():
            source_choice_outputs = model(**source_choice, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    with torch.no_grad():
        target_choice_outputs = model(**target_choice, use_cache=False)

    def continuation_token_id(prompt: str, letter: str) -> int:
        text = prompt + " " + letter
        encoded = tokenizer(text, return_offsets_mapping=True)
        offsets = [tuple(item) for item in encoded["offset_mapping"]]
        positions = overlapping_positions(offsets, [len(prompt), len(text)])
        if len(positions) != 1:
            raise ValueError(f"choice {letter} is not one continuation token")
        return int(encoded["input_ids"][positions[0]])

    def class_margins(logits: Any, positions: Any, prompt_key: str) -> list[float]:
        margins = []
        for row_index, spec in enumerate(choice_specs):
            final_logits = logits[row_index, positions[row_index]].to(dtype=torch.float32)
            class_values = []
            other_values = []
            prompt = spec[prompt_key]
            for letter, track_key in spec["mapping"].items():
                token_id = continuation_token_id(prompt, letter)
                value = final_logits[token_id]
                if records_by_key[track_key]["vocality"] == spec["class_label"]:
                    class_values.append(value)
                else:
                    other_values.append(value)
            margins.append(
                float(
                    (
                        torch.logsumexp(torch.stack(class_values), dim=0)
                        - torch.logsumexp(torch.stack(other_values), dim=0)
                    ).item()
                )
            )
        return margins

    choice_source_scores = class_margins(
        source_choice_outputs.logits,
        source_choice_positions,
        "source_prompt",
    )
    choice_target_scores = class_margins(
        target_choice_outputs.logits,
        target_choice_positions,
        "target_prompt",
    )
    del source_choice_outputs
    del target_choice_outputs

    choice_patched_scores: dict[tuple[int, str], list[float]] = {}
    for layer_number, layer_index in zip(selected_layers, layer_indices):
        layer = model.model.layers[layer_index]
        for component in components:
            module = component_module(layer, component)
            replacement = choice_source_vectors[(layer_number, component)]

            def patch_choice_hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                value: Any = replacement,
            ) -> Any:
                hidden = output_hidden(output)
                patched = hidden.clone()
                patched[batch_indices, target_choice_positions] = value
                if isinstance(output, tuple):
                    return (patched, *output[1:])
                return patched

            handle = module.register_forward_hook(patch_choice_hook)
            try:
                with torch.no_grad():
                    outputs = model(**target_choice, use_cache=False)
            finally:
                handle.remove()
            choice_patched_scores[(layer_number, component)] = class_margins(
                outputs.logits,
                target_choice_positions,
                "target_prompt",
            )
            del outputs

    choice_rows = []
    for row_index, spec in enumerate(choice_specs):
        source = choice_source_scores[row_index]
        target = choice_target_scores[row_index]
        patches = {component: {} for component in components}
        recoveries = {component: {} for component in components}
        for layer_number in selected_layers:
            key = str(layer_number)
            for component in components:
                patched = choice_patched_scores[(layer_number, component)][row_index]
                patches[component][key] = patched
                recoveries[component][key] = recovery(source, target, patched)
        row = {
            **spec,
            "source_class_margin": source,
            "target_class_margin": target,
            "source_target_effect": source - target,
            "patched_class_margins": patches,
            "component_recoveries": recoveries,
        }
        choice_rows.append(row)
        print(
            "VOCALITY_PATH_CHOICE_ROW_JSON="
            + json.dumps(row, ensure_ascii=True, separators=(",", ":"))
        )

    def component_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        curve = []
        for layer_number in bundle["analysis_layers"]:
            key = str(layer_number)
            for component in components:
                values = [float(row["component_recoveries"][component][key]) for row in rows]
                curve.append(
                    {
                        "layer": layer_number,
                        "component": component,
                        "mean_recovery": sum(values) / len(values),
                        "median_recovery": statistics.median(values),
                        "toward_source_rate": sum(value > 0 for value in values) / len(values),
                    }
                )
        return curve

    endpoint = str(selected_layers[-1])
    pair_endpoint_errors = [
        abs(row["patched_pair_mean_logps"]["full_residual"][endpoint] - row["source_pair_mean_logp"])
        for row in pair_rows
    ]
    choice_endpoint_errors = [
        abs(row["patched_class_margins"]["full_residual"][endpoint] - row["source_class_margin"])
        for row in choice_rows
    ]
    baseline_values = [
        value
        for row in pair_rows
        for value in (row["source_pair_mean_logp"], row["target_pair_mean_logp"], row["source_target_effect"])
    ] + [
        value
        for row in choice_rows
        for value in (row["source_class_margin"], row["target_class_margin"], row["source_target_effect"])
    ]
    patched_values = [
        value
        for row in pair_rows
        for component in components
        for value in row["patched_pair_mean_logps"][component].values()
    ] + [
        value
        for row in choice_rows
        for component in components
        for value in row["patched_class_margins"][component].values()
    ]
    recovery_values = [
        value
        for row in pair_rows + choice_rows
        for component in components
        for value in row["component_recoveries"][component].values()
    ]
    all_values_finite = all(
        math.isfinite(float(value))
        for value in baseline_values + patched_values + recovery_values
    )
    expected_capture_count = len(selected_layers) * len(components)
    all_captures_present = (
        len(pair_source_vectors) == expected_capture_count
        and len(choice_source_vectors) == expected_capture_count
        and len(pair_patched_scores) == expected_capture_count
        and len(choice_patched_scores) == expected_capture_count
        and all(len(values) == len(pair_rows) for values in pair_patched_scores.values())
        and all(len(values) == len(choice_rows) for values in choice_patched_scores.values())
    )
    technical_gate = (
        len(pair_rows) == 4
        and len(choice_rows) == 4
        and all(abs(row["source_target_effect"]) >= float(bundle["minimum_pair_effect"]) for row in pair_rows)
        and max(pair_endpoint_errors) <= float(bundle["endpoint_tolerance"])
        and max(choice_endpoint_errors) <= float(bundle["endpoint_tolerance"])
        and all_values_finite
        and all_captures_present
    )
    summary = {
        "run_id": run_id,
        "model_id": bundle["model_id"],
        "pair_count": len(pair_rows),
        "choice_relation_count": len(choice_rows),
        "selected_layers": selected_layers,
        "pair_component_curve": component_curve(pair_rows),
        "choice_component_curve": component_curve(choice_rows),
        "max_pair_endpoint_error": max(pair_endpoint_errors),
        "max_choice_endpoint_error": max(choice_endpoint_errors),
        "all_values_finite": all_values_finite,
        "all_captures_present": all_captures_present,
        "technical_gate": technical_gate,
        "interpretation_scope": "exploratory_generation_vs_comparison_path",
        "behavior_reference": bundle["behavior_reference"],
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    print(
        "VOCALITY_PATH_SUMMARY_JSON="
        + json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
