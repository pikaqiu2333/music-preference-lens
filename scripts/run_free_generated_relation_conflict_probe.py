# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.8.0",
#   "torch==2.7.1",
#   "transformers==4.53.2",
# ]
# ///
"""Test no-training relation-conflict warnings on freely generated song pairs."""

from __future__ import annotations

import argparse
import base64
import json
import math
import statistics
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EMBEDDED_BUNDLE_B64 = "__EXPERIMENT_BUNDLE_ZLIB_B64__"


def load_bundle(path: Path | None = None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    if EMBEDDED_BUNDLE_B64 == "__EXPERIMENT_BUNDLE_ZLIB_B64__":
        raise ValueError("pass --bundle or embed the compressed experiment bundle")
    payload = zlib.decompress(base64.b64decode(EMBEDDED_BUNDLE_B64))
    return json.loads(payload.decode("utf-8"))


def append_continuation(prefix: str, continuation: str) -> tuple[str, list[int]]:
    clean_prefix = prefix.rstrip()
    text = clean_prefix + " " + continuation
    return text, [len(clean_prefix) + 1, len(text)]


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
    title: str,
    emitted_artist: str,
    reference_artist: str,
    order: list[str],
) -> tuple[str, dict[str, str]]:
    artists = {"emitted": emitted_artist, "reference": reference_artist}
    lines = [
        f'Which artist recorded the track titled "{title}"?',
        "Options:",
    ]
    mapping = {}
    for index, role in enumerate(order):
        letter = chr(ord("A") + index)
        lines.append(f"{letter}. {artists[role]}")
        mapping[letter] = role
    lines.extend(["Answer with one letter only.", "Answer:"])
    return "\n".join(lines), mapping


def normalized_recovery(
    source: float,
    target: float,
    patched: float,
    minimum_effect: float,
) -> float | None:
    effect = source - target
    if abs(effect) < minimum_effect:
        return None
    return (patched - target) / effect


def classify_conflict(catalog_margin: float, generation_margin: float) -> str:
    if catalog_margin >= 0:
        return "relation_not_recovered"
    if generation_margin < 0:
        return "lower_probability_sample"
    return "context_override"


def binary_metrics(rows: list[dict[str, Any]], margin_key: str) -> dict[str, Any]:
    exact = [row for row in rows if row["catalog_label"] == "verified_exact"]
    conflict = [row for row in rows if row["catalog_label"] == "catalog_conflict"]
    exact_correct = sum(float(row[margin_key]) >= 0 for row in exact)
    conflict_correct = sum(float(row[margin_key]) < 0 for row in conflict)
    exact_rate = exact_correct / len(exact)
    conflict_rate = conflict_correct / len(conflict)
    return {
        "exact_count": len(exact),
        "conflict_count": len(conflict),
        "exact_correct": exact_correct,
        "conflict_correct": conflict_correct,
        "exact_specificity": exact_rate,
        "conflict_sensitivity": conflict_rate,
        "balanced_accuracy": (exact_rate + conflict_rate) / 2,
    }


def verifier_passes(
    metrics: dict[str, Any],
    minimum_balanced_accuracy: float,
    minimum_group_correct: int,
) -> bool:
    return (
        float(metrics["balanced_accuracy"]) >= minimum_balanced_accuracy
        and int(metrics["exact_correct"]) >= minimum_group_correct
        and int(metrics["conflict_correct"]) >= minimum_group_correct
    )


def validation_status(
    catalog_metrics: dict[str, Any],
    choice_metrics: dict[str, Any],
    minimum_balanced_accuracy: float,
    minimum_group_correct: int,
) -> str:
    catalog_pass = verifier_passes(
        catalog_metrics, minimum_balanced_accuracy, minimum_group_correct
    )
    choice_pass = verifier_passes(
        choice_metrics, minimum_balanced_accuracy, minimum_group_correct
    )
    if catalog_pass and choice_pass:
        return "validated_small_pilot"
    if catalog_pass or choice_pass:
        return "promising_single_path"
    return "not_supported"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + "_pilot"
    bundle = load_bundle(args.bundle)
    records = bundle["records"]
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
    batch_indices = torch.arange(len(records), device=device)

    def final_positions(attention_mask: Any) -> Any:
        return torch.tensor(
            [int(torch.nonzero(row, as_tuple=False)[-1].item()) for row in attention_mask],
            device=device,
            dtype=torch.long,
        )

    def first_token_id(prefix: str, artist: str) -> int:
        text, span = append_continuation(prefix, artist)
        encoded = tokenizer(text, return_offsets_mapping=True)
        offsets = [tuple(item) for item in encoded["offset_mapping"]]
        positions = overlapping_positions(offsets, span)
        return int(encoded["input_ids"][positions[0]])

    token_ids = []
    for row in records:
        catalog_ids = {
            "emitted": first_token_id(row["catalog_prefix"], row["emitted_artist"]),
            "reference": first_token_id(
                row["catalog_prefix"], row["reference_artist"]
            ),
        }
        generation_ids = {
            "emitted": first_token_id(
                row["generation_prefix"], row["emitted_artist"]
            ),
            "reference": first_token_id(
                row["generation_prefix"], row["reference_artist"]
            ),
        }
        if catalog_ids != generation_ids:
            raise ValueError(f"prefix formatting changed artist token IDs: {row['record_id']}")
        if catalog_ids["emitted"] == catalog_ids["reference"]:
            raise ValueError(f"artists share first continuation token: {row['record_id']}")
        token_ids.append(catalog_ids)

    catalog_inputs = tokenizer(
        [row["catalog_prefix"].rstrip() for row in records],
        return_tensors="pt",
        padding=True,
    ).to(device)
    generation_inputs = tokenizer(
        [row["generation_prefix"].rstrip() for row in records],
        return_tensors="pt",
        padding=True,
    ).to(device)
    catalog_positions = final_positions(catalog_inputs["attention_mask"])
    generation_positions = final_positions(generation_inputs["attention_mask"])

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

    source_vectors: dict[tuple[int, str], Any] = {}
    handles = []
    for intervention in bundle["interventions"]:
        layer_number = int(intervention["layer"])
        component = intervention["component"]
        module = component_module(model.model.layers[layer_number - 1], component)

        def capture_hook(
            _module: Any,
            _inputs: Any,
            output: Any,
            key: tuple[int, str] = (layer_number, component),
        ) -> None:
            source_vectors[key] = output_hidden(output)[
                batch_indices, catalog_positions
            ].detach().clone()

        handles.append(module.register_forward_hook(capture_hook))
    try:
        with torch.no_grad():
            catalog_outputs = model(
                **catalog_inputs, use_cache=False, output_hidden_states=True
            )
    finally:
        for handle in handles:
            handle.remove()
    with torch.no_grad():
        generation_outputs = model(
            **generation_inputs, use_cache=False, output_hidden_states=True
        )

    def first_token_margins(logits: Any, positions: Any) -> list[float]:
        margins = []
        for row_index, ids in enumerate(token_ids):
            step_logits = logits[row_index, positions[row_index]].to(dtype=torch.float32)
            margins.append(
                float((step_logits[ids["emitted"]] - step_logits[ids["reference"]]).item())
            )
        return margins

    catalog_first_margins = first_token_margins(
        catalog_outputs.logits, catalog_positions
    )
    generation_first_margins = first_token_margins(
        generation_outputs.logits, generation_positions
    )

    def layerwise_margins(outputs: Any, positions: Any) -> dict[str, list[float]]:
        result = {}
        last_layer = len(model.model.layers)
        for layer_number in bundle["selected_layers"]:
            hidden = outputs.hidden_states[int(layer_number)]
            selected = hidden[batch_indices, positions]
            if int(layer_number) < last_layer:
                selected = model.model.norm(selected)
            logits = model.lm_head(selected).to(dtype=torch.float32)
            result[str(layer_number)] = [
                float(
                    (
                        logits[row_index, token_ids[row_index]["emitted"]]
                        - logits[row_index, token_ids[row_index]["reference"]]
                    ).item()
                )
                for row_index in range(len(records))
            ]
        return result

    catalog_layerwise = layerwise_margins(catalog_outputs, catalog_positions)
    generation_layerwise = layerwise_margins(generation_outputs, generation_positions)
    last_layer_key = str(len(model.model.layers))
    final_readout_errors = [
        abs(catalog_layerwise[last_layer_key][index] - catalog_first_margins[index])
        for index in range(len(records))
    ] + [
        abs(
            generation_layerwise[last_layer_key][index]
            - generation_first_margins[index]
        )
        for index in range(len(records))
    ]
    del catalog_outputs
    del generation_outputs

    def sequence_mean_logp(prefix: str, artist: str) -> float:
        text, span = append_continuation(prefix, artist)
        encoded = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = [tuple(item) for item in encoded.pop("offset_mapping")[0].tolist()]
        target_positions = overlapping_positions(offsets, span)
        prediction_positions = [position - 1 for position in target_positions]
        target_ids = encoded["input_ids"][0, target_positions].to(device)
        encoded = encoded.to(device)
        with torch.no_grad():
            outputs = model(**encoded, use_cache=False)
        step_logits = outputs.logits[0, prediction_positions].to(dtype=torch.float32)
        token_logits = step_logits[
            torch.arange(len(prediction_positions), device=device), target_ids
        ]
        token_logps = token_logits - torch.logsumexp(step_logits, dim=-1)
        score = float(token_logps.mean().item())
        del outputs
        return score

    catalog_sequence_margins = []
    generation_sequence_margins = []
    for row in records:
        catalog_sequence_margins.append(
            sequence_mean_logp(row["catalog_prefix"], row["emitted_artist"])
            - sequence_mean_logp(row["catalog_prefix"], row["reference_artist"])
        )
        generation_sequence_margins.append(
            sequence_mean_logp(row["generation_prefix"], row["emitted_artist"])
            - sequence_mean_logp(row["generation_prefix"], row["reference_artist"])
        )

    choice_specs = []
    for row_index, row in enumerate(records):
        for order_index, order in enumerate(
            (["emitted", "reference"], ["reference", "emitted"])
        ):
            prompt, mapping = render_choice_prompt(
                row["title"],
                row["emitted_artist"],
                row["reference_artist"],
                list(order),
            )
            choice_specs.append(
                {
                    "row_index": row_index,
                    "order_index": order_index,
                    "prompt": prompt,
                    "mapping": mapping,
                }
            )
    choice_inputs = tokenizer(
        [spec["prompt"] for spec in choice_specs],
        return_tensors="pt",
        padding=True,
    ).to(device)
    choice_positions = final_positions(choice_inputs["attention_mask"])

    choice_token_cache: dict[tuple[str, str], int] = {}

    def choice_token_id(prompt: str, letter: str) -> int:
        key = (prompt, letter)
        if key not in choice_token_cache:
            choice_token_cache[key] = first_token_id(prompt, letter)
        return choice_token_cache[key]

    with torch.no_grad():
        choice_outputs = model(**choice_inputs, use_cache=False)
    order_margins: list[list[float]] = [[] for _ in records]
    for spec_index, spec in enumerate(choice_specs):
        logits = choice_outputs.logits[
            spec_index, choice_positions[spec_index]
        ].to(dtype=torch.float32)
        role_to_letter = {role: letter for letter, role in spec["mapping"].items()}
        emitted_id = choice_token_id(spec["prompt"], role_to_letter["emitted"])
        reference_id = choice_token_id(spec["prompt"], role_to_letter["reference"])
        order_margins[spec["row_index"]].append(
            float((logits[emitted_id] - logits[reference_id]).item())
        )
    del choice_outputs
    choice_margins = [sum(values) / len(values) for values in order_margins]

    patched_margins: dict[tuple[int, str], list[float]] = {}
    for intervention in bundle["interventions"]:
        layer_number = int(intervention["layer"])
        component = intervention["component"]
        key = (layer_number, component)
        module = component_module(model.model.layers[layer_number - 1], component)
        source_value = source_vectors[key]

        def patch_hook(
            _module: Any,
            _inputs: Any,
            output: Any,
            value: Any = source_value,
        ) -> Any:
            hidden = output_hidden(output)
            patched = hidden.clone()
            patched[batch_indices, generation_positions] = value
            if isinstance(output, tuple):
                return (patched, *output[1:])
            return patched

        handle = module.register_forward_hook(patch_hook)
        try:
            with torch.no_grad():
                outputs = model(**generation_inputs, use_cache=False)
        finally:
            handle.remove()
        patched_margins[key] = first_token_margins(
            outputs.logits, generation_positions
        )
        del outputs

    endpoint_key = (len(model.model.layers), "full_residual")
    endpoint_errors = [
        abs(patched_margins[endpoint_key][index] - catalog_first_margins[index])
        for index in range(len(records))
    ]

    rows = []
    for index, record in enumerate(records):
        interventions = {}
        for intervention in bundle["interventions"]:
            key = (int(intervention["layer"]), intervention["component"])
            label = f"layer{key[0]}_{key[1]}"
            patched = patched_margins[key][index]
            interventions[label] = {
                "patched_emitted_margin": patched,
                "raw_shift_from_generation": patched - generation_first_margins[index],
                "recovery_toward_catalog": normalized_recovery(
                    catalog_first_margins[index],
                    generation_first_margins[index],
                    patched,
                    float(bundle["minimum_patch_effect"]),
                ),
            }
        row = {
            **{key: value for key, value in record.items() if not key.endswith("_prefix")},
            "catalog_first_token_emitted_margin": catalog_first_margins[index],
            "generation_first_token_emitted_margin": generation_first_margins[index],
            "catalog_sequence_emitted_margin": catalog_sequence_margins[index],
            "generation_sequence_emitted_margin": generation_sequence_margins[index],
            "choice_order_emitted_margins": order_margins[index],
            "choice_emitted_margin": choice_margins[index],
            "catalog_layerwise_emitted_margins": {
                layer: values[index] for layer, values in catalog_layerwise.items()
            },
            "generation_layerwise_emitted_margins": {
                layer: values[index] for layer, values in generation_layerwise.items()
            },
            "conflict_category": (
                classify_conflict(
                    catalog_first_margins[index], generation_first_margins[index]
                )
                if record["catalog_label"] == "catalog_conflict"
                else "exact_control"
            ),
            "interventions": interventions,
            "final_readout_error": max(
                abs(
                    catalog_layerwise[last_layer_key][index]
                    - catalog_first_margins[index]
                ),
                abs(
                    generation_layerwise[last_layer_key][index]
                    - generation_first_margins[index]
                ),
            ),
            "endpoint_patch_error": endpoint_errors[index],
        }
        rows.append(row)
        print(
            "FREE_REL_CONFLICT_ROW_JSON="
            + json.dumps(row, ensure_ascii=True, separators=(",", ":"))
        )

    catalog_metrics = binary_metrics(rows, "catalog_first_token_emitted_margin")
    sequence_metrics = binary_metrics(rows, "catalog_sequence_emitted_margin")
    choice_metrics = binary_metrics(rows, "choice_emitted_margin")
    generation_metrics = binary_metrics(rows, "generation_first_token_emitted_margin")
    status = validation_status(
        catalog_metrics,
        choice_metrics,
        float(bundle["minimum_balanced_accuracy"]),
        int(bundle["minimum_group_correct"]),
    )

    patch_summary = []
    conflict_rows = [row for row in rows if row["catalog_label"] == "catalog_conflict"]
    for intervention in bundle["interventions"]:
        label = f"layer{intervention['layer']}_{intervention['component']}"
        values = [
            row["interventions"][label]["recovery_toward_catalog"]
            for row in conflict_rows
            if row["interventions"][label]["recovery_toward_catalog"] is not None
        ]
        patch_summary.append(
            {
                "intervention": label,
                "valid_count": len(values),
                "mean_recovery": sum(values) / len(values) if values else None,
                "median_recovery": statistics.median(values) if values else None,
                "toward_catalog_rate": (
                    sum(value > 0 for value in values) / len(values) if values else None
                ),
            }
        )

    def finite_numbers(value: Any) -> list[float]:
        if value is None or isinstance(value, bool):
            return []
        if isinstance(value, (int, float)):
            return [float(value)]
        if isinstance(value, dict):
            return [number for item in value.values() for number in finite_numbers(item)]
        if isinstance(value, list):
            return [number for item in value for number in finite_numbers(item)]
        return []

    all_values_finite = all(
        math.isfinite(number) for row in rows for number in finite_numbers(row)
    )
    labels = Counter(row["catalog_label"] for row in rows)
    all_interventions_present = all(
        len(row["interventions"]) == len(bundle["interventions"])
        and len(row["choice_order_emitted_margins"]) == 2
        for row in rows
    )
    technical_gate = (
        labels == Counter({"verified_exact": 6, "catalog_conflict": 6})
        and all_interventions_present
        and max(final_readout_errors) <= float(bundle["endpoint_tolerance"])
        and max(endpoint_errors) <= float(bundle["endpoint_tolerance"])
        and all_values_finite
        and len(source_vectors) == len(bundle["interventions"])
    )
    summary = {
        "run_id": run_id,
        "model_id": bundle["model_id"],
        "row_count": len(rows),
        "label_counts": dict(labels),
        "catalog_first_token_metrics": catalog_metrics,
        "catalog_sequence_metrics": sequence_metrics,
        "choice_metrics": choice_metrics,
        "generation_first_token_metrics": generation_metrics,
        "validation_status": status,
        "conflict_categories": dict(
            Counter(row["conflict_category"] for row in conflict_rows)
        ),
        "patch_summary": patch_summary,
        "maximum_final_readout_error": max(final_readout_errors),
        "maximum_endpoint_patch_error": max(endpoint_errors),
        "all_interventions_present": all_interventions_present,
        "all_values_finite": all_values_finite,
        "technical_gate": technical_gate,
        "interpretation_scope": "small_frozen_free_generation_transfer_pilot",
        "field_reference_run": bundle["field_reference_run"],
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    print(
        "FREE_REL_CONFLICT_SUMMARY_JSON="
        + json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
