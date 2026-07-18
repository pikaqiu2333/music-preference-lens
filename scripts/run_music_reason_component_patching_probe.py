# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.8.0",
#   "torch==2.7.1",
#   "transformers==4.53.2",
# ]
# ///
"""Patch opposite-need components into complete title-artist predictions."""

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


def render_pair_text(
    prompt: str,
    title: str,
    artist: str,
) -> tuple[str, list[int], list[int]]:
    text = prompt.rstrip() + " "
    title_span = [len(text), len(text) + len(title)]
    text += title
    text += "\n   Artist: "
    artist_span = [len(text), len(text) + len(artist)]
    text += artist
    return text, title_span, artist_span


def overlapping_positions(
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


def component_recovery(original: float, opposite: float, patched: float) -> float:
    denominator = original - opposite
    if abs(denominator) < 1e-12:
        raise ValueError("cannot normalize a zero need effect")
    return (original - patched) / denominator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + "_pilot"
    bundle = load_bundle(args.bundle)
    controls = bundle["controls"]
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

    texts: dict[str, list[str]] = {"original": [], "opposite": []}
    spans: dict[str, list[tuple[list[int], list[int]]]] = {
        "original": [],
        "opposite": [],
    }
    for control in controls:
        for condition in ("original", "opposite"):
            text, title_span, artist_span = render_pair_text(
                control[f"{condition}_prompt"],
                control["title"],
                control["artist"],
            )
            texts[condition].append(text)
            spans[condition].append((title_span, artist_span))

    encoded: dict[str, Any] = {}
    target_positions: dict[str, list[list[int]]] = {}
    prediction_positions: dict[str, list[list[int]]] = {}
    target_ids: dict[str, list[list[int]]] = {}
    for condition in ("original", "opposite"):
        inputs = tokenizer(
            texts[condition],
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
        )
        offsets_batch = inputs.pop("offset_mapping").tolist()
        condition_targets = []
        condition_predictions = []
        condition_ids = []
        for row_index, offsets_raw in enumerate(offsets_batch):
            offsets = [tuple(item) for item in offsets_raw]
            title_span, artist_span = spans[condition][row_index]
            positions = overlapping_positions(offsets, title_span)
            positions += overlapping_positions(offsets, artist_span)
            if min(positions) < 1:
                raise ValueError("target token has no causal prediction position")
            condition_targets.append(positions)
            condition_predictions.append([position - 1 for position in positions])
            condition_ids.append(
                [int(inputs["input_ids"][row_index, position]) for position in positions]
            )
        encoded[condition] = inputs.to(device)
        target_positions[condition] = condition_targets
        prediction_positions[condition] = condition_predictions
        target_ids[condition] = condition_ids

    if target_ids["original"] != target_ids["opposite"]:
        raise ValueError("title-artist target tokenization differs across conditions")

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

    source_vectors: dict[tuple[int, str], list[Any]] = {}
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
                source_vectors[key] = [
                    hidden[row_index, positions].detach().clone()
                    for row_index, positions in enumerate(
                        prediction_positions["opposite"]
                    )
                ]

            handles.append(module.register_forward_hook(capture_hook))
    try:
        with torch.no_grad():
            opposite_outputs = model(**encoded["opposite"], use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    expected_keys = {
        (layer, component)
        for layer in selected_layers
        for component in components
    }
    if set(source_vectors) != expected_keys:
        raise ValueError("not all opposite-condition component vectors were captured")

    with torch.no_grad():
        original_outputs = model(**encoded["original"], use_cache=False)

    def pair_mean_logps(logits: Any, condition: str) -> list[float]:
        scores = []
        for row_index, positions in enumerate(prediction_positions[condition]):
            step_logits = logits[row_index, positions].to(dtype=torch.float32)
            ids = torch.tensor(target_ids[condition][row_index], device=device)
            token_logits = step_logits[
                torch.arange(len(positions), device=device), ids
            ]
            token_logps = token_logits - torch.logsumexp(step_logits, dim=-1)
            scores.append(float(token_logps.mean().item()))
        return scores

    original_scores = pair_mean_logps(original_outputs.logits, "original")
    opposite_scores = pair_mean_logps(opposite_outputs.logits, "opposite")
    del original_outputs
    del opposite_outputs

    patched_scores: dict[tuple[int, str], list[float]] = {}
    for layer_number, layer_index in zip(selected_layers, layer_indices):
        layer = model.model.layers[layer_index]
        for component in components:
            module = component_module(layer, component)
            replacements = source_vectors[(layer_number, component)]

            def patch_hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                values: list[Any] = replacements,
            ) -> Any:
                hidden = output_hidden(output)
                patched = hidden.clone()
                for row_index, positions in enumerate(
                    prediction_positions["original"]
                ):
                    patched[row_index, positions] = values[row_index]
                if isinstance(output, tuple):
                    return (patched, *output[1:])
                return patched

            handle = module.register_forward_hook(patch_hook)
            try:
                with torch.no_grad():
                    patched_outputs = model(**encoded["original"], use_cache=False)
            finally:
                handle.remove()
            patched_scores[(layer_number, component)] = pair_mean_logps(
                patched_outputs.logits,
                "original",
            )
            del patched_outputs

    rows = []
    baseline_errors = []
    minimum_effect = float(bundle["minimum_need_effect"])
    for row_index, control in enumerate(controls):
        original = original_scores[row_index]
        opposite = opposite_scores[row_index]
        need_effect = original - opposite
        baseline_error = max(
            abs(original - float(control["archived_original_pair_mean_logp"])),
            abs(opposite - float(control["archived_opposite_pair_mean_logp"])),
        )
        baseline_errors.append(baseline_error)
        patches = {component: {} for component in components}
        recoveries = {component: {} for component in components}
        for layer_number in selected_layers:
            key = str(layer_number)
            for component in components:
                patched = patched_scores[(layer_number, component)][row_index]
                patches[component][key] = patched
                recoveries[component][key] = component_recovery(
                    original,
                    opposite,
                    patched,
                )
        rows.append(
            {
                **control,
                "pair_token_count": len(target_ids["original"][row_index]),
                "replayed_original_pair_mean_logp": original,
                "replayed_opposite_pair_mean_logp": opposite,
                "replayed_need_effect": need_effect,
                "valid_need_effect": abs(need_effect) >= minimum_effect,
                "baseline_error": baseline_error,
                "patched_pair_mean_logps": patches,
                "component_recoveries": recoveries,
            }
        )

    valid_rows = [row for row in rows if row["valid_need_effect"]]
    curves = []
    for layer_number in bundle["analysis_layers"]:
        key = str(layer_number)
        for component in components:
            values = [
                float(row["component_recoveries"][component][key])
                for row in valid_rows
            ]
            curves.append(
                {
                    "layer": layer_number,
                    "component": component,
                    "mean_recovery": sum(values) / len(values),
                    "median_recovery": statistics.median(values),
                    "toward_opposite_rate": sum(value > 0 for value in values)
                    / len(values),
                }
            )

    endpoint_layer = str(selected_layers[-1])
    endpoint_errors = [
        abs(
            row["patched_pair_mean_logps"]["full_residual"][endpoint_layer]
            - row["replayed_opposite_pair_mean_logp"]
        )
        for row in rows
    ]
    all_metrics = [
        value
        for row in rows
        for value in (
            row["replayed_original_pair_mean_logp"],
            row["replayed_opposite_pair_mean_logp"],
            row["replayed_need_effect"],
            *(
                patched
                for component in components
                for patched in row["patched_pair_mean_logps"][component].values()
            ),
        )
    ]
    technical_gate = (
        len(rows) == len(controls) == 3
        and len(valid_rows) >= int(bundle["minimum_valid_cases"])
        and max(baseline_errors) <= float(bundle["baseline_tolerance"])
        and max(endpoint_errors) <= float(bundle["endpoint_tolerance"])
        and all(math.isfinite(float(value)) for value in all_metrics)
    )
    summary = {
        "run_id": run_id,
        "model_id": bundle["model_id"],
        "case_count": len(rows),
        "valid_case_count": len(valid_rows),
        "selected_layers": selected_layers,
        "analysis_layers": bundle["analysis_layers"],
        "component_curve": curves,
        "max_baseline_error": max(baseline_errors),
        "max_layer28_full_residual_endpoint_error": max(endpoint_errors),
        "technical_gate": technical_gate,
        "interpretation_scope": "exploratory_three_case_pilot",
        "behavior_reference": bundle["behavior_reference"],
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }

    for row in rows:
        print(
            "REASON_COMP_PATCH_ROW_JSON="
            + json.dumps(row, ensure_ascii=True, separators=(",", ":"))
        )
    print(
        "REASON_COMP_PATCH_SUMMARY_JSON="
        + json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
