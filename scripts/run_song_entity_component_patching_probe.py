# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.8.0",
#   "huggingface-hub>=0.33.0,<1.0",
#   "safetensors>=0.5.0",
#   "torch==2.7.1",
#   "transformers==4.53.2",
# ]
# ///
"""Patch neutral attention, MLP, and residual outputs into real song prefixes."""

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
SMOKE_BLOCKS = {
    "en_swap_1",
    "en_swap_2",
    "en_swap_3",
    "zh_swap_1",
    "zh_swap_2",
    "zh_swap_3",
}


def load_bundle(path: Path | None = None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    if EMBEDDED_BUNDLE_B64 == "__EXPERIMENT_BUNDLE_B64__":
        raise ValueError("pass --bundle or embed the experiment bundle")
    return json.loads(base64.b64decode(EMBEDDED_BUNDLE_B64).decode("utf-8"))


def select_smoke_controls(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["block_id"] in SMOKE_BLOCKS]


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


def component_recovery(real: float, neutral: float, patched: float) -> float | None:
    denominator = real - neutral
    if abs(denominator) < 0.05:
        return None
    return (real - patched) / denominator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()

    import torch
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
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        bundle["model_id"],
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    layer_count = len(model.model.layers)
    selected_layers = [int(layer) for layer in bundle["selected_layers"]]
    if any(layer < 1 or layer > layer_count for layer in selected_layers):
        raise ValueError(f"selected layers exceed model depth {layer_count}")
    layer_indices = [layer - 1 for layer in selected_layers]
    components = list(bundle["components"])

    real_prefixes = [
        bundle["patch_prefix_template"].format(title=row["title"])
        for row in controls
    ]
    neutral_prefixes = [
        bundle["patch_prefix_template"].format(title=row["neutral_title"])
        for row in controls
    ]

    def first_continuation_token_id(prefix: str, artist: str) -> int:
        text = prefix + " " + artist
        encoded = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = [tuple(item) for item in encoded.pop("offset_mapping")[0].tolist()]
        positions = overlapping_positions(offsets, [len(prefix), len(text)])
        return int(encoded["input_ids"][0, positions[0]].item())

    correct_ids = [
        first_continuation_token_id(prefix, row["correct_artist"])
        for prefix, row in zip(real_prefixes, controls)
    ]
    wrong_ids = [
        first_continuation_token_id(prefix, row["wrong_artist"])
        for prefix, row in zip(real_prefixes, controls)
    ]
    if any(correct == wrong for correct, wrong in zip(correct_ids, wrong_ids)):
        raise ValueError("a correct/wrong artist pair shares its first continuation token")

    real_encoded = tokenizer(
        real_prefixes,
        return_tensors="pt",
        padding=True,
    ).to(device)
    neutral_encoded = tokenizer(
        neutral_prefixes,
        return_tensors="pt",
        padding=True,
    ).to(device)

    def final_nonpad_positions(attention_mask: Any) -> Any:
        return torch.tensor(
            [
                int(torch.nonzero(row, as_tuple=False)[-1].item())
                for row in attention_mask
            ],
            device=device,
            dtype=torch.long,
        )

    real_positions = final_nonpad_positions(real_encoded["attention_mask"])
    neutral_positions = final_nonpad_positions(neutral_encoded["attention_mask"])
    batch_indices = torch.arange(len(controls), device=device)
    correct_tensor = torch.tensor(correct_ids, device=device, dtype=torch.long)
    wrong_tensor = torch.tensor(wrong_ids, device=device, dtype=torch.long)

    def output_hidden(output: Any) -> Any:
        return output[0] if isinstance(output, tuple) else output

    def replace_output(output: Any, replacement: Any, positions: Any) -> Any:
        hidden = output_hidden(output)
        patched = hidden.clone()
        patched[batch_indices, positions] = replacement
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    def component_module(layer: Any, component: str) -> Any:
        if component == "attention":
            return layer.self_attn
        if component == "mlp":
            return layer.mlp
        if component == "full_residual":
            return layer
        raise ValueError(f"unknown component: {component}")

    neutral_vectors: dict[tuple[int, str], Any] = {}
    capture_handles = []
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
                neutral_vectors[key] = output_hidden(output)[
                    batch_indices,
                    neutral_positions,
                ].detach().clone()

            capture_handles.append(module.register_forward_hook(capture_hook))
    try:
        with torch.no_grad():
            neutral_outputs = model(**neutral_encoded, use_cache=False)
    finally:
        for handle in capture_handles:
            handle.remove()
    expected_keys = {
        (layer, component)
        for layer in selected_layers
        for component in components
    }
    if set(neutral_vectors) != expected_keys:
        missing = sorted(expected_keys - set(neutral_vectors))
        raise ValueError(f"missing neutral component captures: {missing}")

    with torch.no_grad():
        real_outputs = model(**real_encoded, use_cache=False)

    def first_token_margins(logits: Any, positions: Any) -> Any:
        final_logits = logits[batch_indices, positions]
        return (
            final_logits[batch_indices, correct_tensor]
            - final_logits[batch_indices, wrong_tensor]
        ).to(dtype=torch.float32)

    neutral_margins = first_token_margins(neutral_outputs.logits, neutral_positions)
    real_margins = first_token_margins(real_outputs.logits, real_positions)
    del neutral_outputs
    del real_outputs

    patched_results: dict[tuple[int, str], list[float]] = {}
    for layer_number, layer_index in zip(selected_layers, layer_indices):
        layer = model.model.layers[layer_index]
        for component in components:
            module = component_module(layer, component)
            replacement = neutral_vectors[(layer_number, component)]

            def patch_hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                value: Any = replacement,
            ) -> Any:
                return replace_output(output, value, real_positions)

            handle = module.register_forward_hook(patch_hook)
            try:
                with torch.no_grad():
                    patched_outputs = model(**real_encoded, use_cache=False)
            finally:
                handle.remove()
            margins = first_token_margins(patched_outputs.logits, real_positions)
            patched_results[(layer_number, component)] = [
                float(value) for value in margins.tolist()
            ]
            del patched_outputs

    relation_rows: list[dict[str, Any]] = []
    minimum_effect = float(bundle["minimum_title_effect"])
    for row_index, control in enumerate(controls):
        real_margin = float(real_margins[row_index].item())
        neutral_margin = float(neutral_margins[row_index].item())
        title_effect = real_margin - neutral_margin
        patches: dict[str, dict[str, float]] = {component: {} for component in components}
        recoveries: dict[str, dict[str, float | None]] = {
            component: {} for component in components
        }
        for layer_number in selected_layers:
            for component in components:
                patched = patched_results[(layer_number, component)][row_index]
                layer_key = str(layer_number)
                patches[component][layer_key] = patched
                recoveries[component][layer_key] = component_recovery(
                    real_margin,
                    neutral_margin,
                    patched,
                )
        relation_rows.append(
            {
                **control,
                "correct_first_token_id": correct_ids[row_index],
                "wrong_first_token_id": wrong_ids[row_index],
                "real_first_token_margin": real_margin,
                "neutral_first_token_margin": neutral_margin,
                "title_effect": title_effect,
                "valid_title_effect": abs(title_effect) >= minimum_effect,
                "patched_margins": patches,
                "component_recoveries": recoveries,
            }
        )

    valid_rows = [row for row in relation_rows if row["valid_title_effect"]]
    component_curve: list[dict[str, Any]] = []
    analysis_layers = [int(layer) for layer in bundle["analysis_layers"]]
    for layer_number in analysis_layers:
        for component in components:
            key = str(layer_number)
            recoveries = [
                float(row["component_recoveries"][component][key])
                for row in valid_rows
            ]
            remaining = [
                abs(row["patched_margins"][component][key] - row["neutral_first_token_margin"])
                / abs(row["title_effect"])
                for row in valid_rows
            ]
            toward_neutral = [
                abs(row["patched_margins"][component][key] - row["neutral_first_token_margin"])
                < abs(row["title_effect"])
                for row in valid_rows
            ]
            component_curve.append(
                {
                    "layer": layer_number,
                    "component": component,
                    "mean_recovery": sum(recoveries) / len(recoveries),
                    "median_recovery": statistics.median(recoveries),
                    "toward_neutral_accuracy": sum(toward_neutral) / len(toward_neutral),
                    "mean_remaining_fraction": sum(remaining) / len(remaining),
                }
            )

    dominance: list[dict[str, Any]] = []
    curve_by_key = {
        (row["layer"], row["component"]): row for row in component_curve
    }
    for layer_number in analysis_layers:
        attention = curve_by_key[(layer_number, "attention")]["mean_recovery"]
        mlp = curve_by_key[(layer_number, "mlp")]["mean_recovery"]
        dominance.append(
            {
                "layer": layer_number,
                "attention_mean_recovery": attention,
                "mlp_mean_recovery": mlp,
                "attention_minus_mlp": attention - mlp,
                "larger_component": "attention" if attention >= mlp else "mlp",
            }
        )

    endpoint_layer = selected_layers[-1]
    max_endpoint_error = max(
        abs(
            row["patched_margins"]["full_residual"][str(endpoint_layer)]
            - row["neutral_first_token_margin"]
        )
        for row in relation_rows
    )
    valid_count = len(valid_rows)
    technical_gate = (
        valid_count >= int(bundle["minimum_valid_relations"])
        and max_endpoint_error <= float(bundle["endpoint_tolerance"])
        and len(patched_results) == len(expected_keys)
    )
    summary = {
        "run_id": run_id,
        "mode": args.mode,
        "model_id": bundle["model_id"],
        "relation_count": len(relation_rows),
        "valid_relation_count": valid_count,
        "selected_layers": selected_layers,
        "analysis_layers": analysis_layers,
        "component_curve": component_curve,
        "component_dominance": dominance,
        "max_layer28_full_residual_endpoint_error": max_endpoint_error,
        "behavior_reference": bundle["behavior_reference"],
        "technical_gate": technical_gate,
        "interpretation_gate": technical_gate,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if not all(
        math.isfinite(float(value))
        for row in relation_rows
        for value in (
            row["real_first_token_margin"],
            row["neutral_first_token_margin"],
            row["title_effect"],
        )
    ):
        raise ValueError("non-finite component-patching metric")

    for row in relation_rows:
        print(
            "COMP_PATCH_ROW_JSON="
            + json.dumps(row, ensure_ascii=True, separators=(",", ":"))
        )
    print(
        "COMP_PATCH_SUMMARY_JSON="
        + json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
