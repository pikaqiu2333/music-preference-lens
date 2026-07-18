# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "accelerate==1.8.1",
#   "torch==2.7.1",
#   "transformers==4.53.2",
# ]
# ///
"""Trace causal mediation of factual-title effects on complete artist scores."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import statistics
import zlib
from collections import Counter, defaultdict
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


def overlapping_positions(
    offsets: list[tuple[int, int]], span: list[int]
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


def catalog_prefix(title: str) -> str:
    return f"Complete the factual catalog entry.\nTitle: {title}\nArtist:"


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


def expected_sign(label: str) -> int:
    if label == "verified_exact":
        return 1
    if label == "catalog_conflict":
        return -1
    raise ValueError(f"unknown catalog label: {label}")


def earliest_sustained_layer(
    points: list[dict[str, Any]],
    minimum_mean_recovery: float = 0.5,
    minimum_toward_source_rate: float = 0.75,
    consecutive: int = 2,
) -> int | None:
    ordered = sorted(points, key=lambda item: int(item["layer"]))
    for start in range(len(ordered) - consecutive + 1):
        window = ordered[start : start + consecutive]
        if all(
            item["mean_recovery"] is not None
            and float(item["mean_recovery"]) >= minimum_mean_recovery
            and item["toward_source_rate"] is not None
            and float(item["toward_source_rate"]) >= minimum_toward_source_rate
            for item in window
        ):
            return int(window[0]["layer"])
    return None


def intervention_label(layer: int, component: str) -> str:
    return f"layer{layer}_{component}"


def encode_artifact_chunks(value: Any, maximum_chars: int = 7500) -> list[str]:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    encoded = base64.b64encode(zlib.compress(payload, level=9)).decode("ascii")
    return [
        encoded[start : start + maximum_chars]
        for start in range(0, len(encoded), maximum_chars)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--model-revision")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + "_sequence_trace"
    bundle = load_bundle(args.bundle)
    model_revision = (
        args.model_revision
        or os.environ.get("MODEL_REVISION")
        or bundle.get("model_revision")
    )
    model_source = {"revision": model_revision} if model_revision else {}
    records = bundle["records"]
    if not torch.cuda.is_available():
        raise RuntimeError("this probe requires a CUDA GPU")
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(
        bundle["model_id"], trust_remote_code=True, **model_source
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        bundle["model_id"],
        torch_dtype=torch.float16,
        trust_remote_code=True,
        **model_source,
    ).to(device)
    model.eval()
    num_layers = len(model.model.layers)
    architecture_gate = num_layers == int(bundle["expected_num_layers"])
    if not architecture_gate:
        raise ValueError(
            f"expected {bundle['expected_num_layers']} layers, found {num_layers}"
        )

    example_specs = []
    source_texts = []
    target_texts = []
    source_spans = []
    target_spans = []
    for row_index, row in enumerate(records):
        for control_index, control in enumerate(row["control_titles"]):
            for candidate_role in ("emitted", "reference"):
                artist = row[f"{candidate_role}_artist"]
                source_text, source_span = append_continuation(
                    catalog_prefix(row["title"]), artist
                )
                target_text, target_span = append_continuation(
                    catalog_prefix(control["title"]), artist
                )
                example_specs.append(
                    {
                        "row_index": row_index,
                        "control_index": control_index,
                        "candidate_role": candidate_role,
                    }
                )
                source_texts.append(source_text)
                target_texts.append(target_text)
                source_spans.append(source_span)
                target_spans.append(target_span)

    def encode_batch(
        texts: list[str], spans: list[list[int]]
    ) -> tuple[Any, list[list[int]], list[list[int]]]:
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
        )
        offsets_batch = encoded.pop("offset_mapping").tolist()
        prediction_positions = []
        target_ids = []
        for row_index, raw_offsets in enumerate(offsets_batch):
            offsets = [tuple(item) for item in raw_offsets]
            token_positions = overlapping_positions(offsets, spans[row_index])
            prediction_positions.append([position - 1 for position in token_positions])
            target_ids.append(
                [int(encoded["input_ids"][row_index, position]) for position in token_positions]
            )
        return encoded.to(device), prediction_positions, target_ids

    source_inputs, source_positions, source_ids = encode_batch(
        source_texts, source_spans
    )
    target_inputs, target_positions, target_ids = encode_batch(
        target_texts, target_spans
    )
    token_alignment_gate = source_ids == target_ids and all(
        len(source_positions[index]) == len(target_positions[index])
        for index in range(len(example_specs))
    )
    if not token_alignment_gate:
        raise ValueError("artist continuation tokenization differs across title conditions")

    interventions = [
        {"layer": int(layer), "component": "full_residual"}
        for layer in bundle["full_residual_layers"]
    ] + [
        {"layer": int(item["layer"]), "component": item["component"]}
        for item in bundle["component_interventions"]
    ]

    def output_hidden(output: Any) -> Any:
        return output[0] if isinstance(output, tuple) else output

    def component_module(layer_number: int, component: str) -> Any:
        layer = model.model.layers[layer_number - 1]
        if component == "full_residual":
            return layer
        if component == "attention":
            return layer.self_attn
        if component == "mlp":
            return layer.mlp
        raise ValueError(f"unknown component: {component}")

    source_vectors: dict[tuple[int, str], list[Any]] = {}
    handles = []
    for intervention in interventions:
        layer_number = intervention["layer"]
        component = intervention["component"]
        key = (layer_number, component)
        module = component_module(layer_number, component)

        def capture_hook(
            _module: Any,
            _inputs: Any,
            output: Any,
            capture_key: tuple[int, str] = key,
        ) -> None:
            hidden = output_hidden(output)
            source_vectors[capture_key] = [
                hidden[index, positions].detach().clone()
                for index, positions in enumerate(source_positions)
            ]

        handles.append(module.register_forward_hook(capture_hook))
    try:
        with torch.no_grad():
            source_outputs = model(**source_inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    def sequence_scores(logits: Any, positions: list[list[int]]) -> list[float]:
        scores = []
        for index, prediction_positions in enumerate(positions):
            step_logits = logits[index, prediction_positions].to(dtype=torch.float32)
            ids = torch.tensor(source_ids[index], device=device)
            token_logits = step_logits[
                torch.arange(len(prediction_positions), device=device), ids
            ]
            token_logps = token_logits - torch.logsumexp(step_logits, dim=-1)
            scores.append(float(token_logps.mean().item()))
        return scores

    source_scores = sequence_scores(source_outputs.logits, source_positions)
    del source_outputs
    with torch.no_grad():
        target_outputs = model(**target_inputs, use_cache=False)
    target_scores = sequence_scores(target_outputs.logits, target_positions)
    del target_outputs

    patched_scores: dict[tuple[int, str], list[float]] = {}
    for intervention in interventions:
        layer_number = intervention["layer"]
        component = intervention["component"]
        key = (layer_number, component)
        module = component_module(layer_number, component)
        values = source_vectors[key]

        def patch_hook(
            _module: Any,
            _inputs: Any,
            output: Any,
            source_values: list[Any] = values,
        ) -> Any:
            hidden = output_hidden(output)
            patched = hidden.clone()
            for index, positions in enumerate(target_positions):
                patched[index, positions] = source_values[index]
            if isinstance(output, tuple):
                return (patched, *output[1:])
            return patched

        handle = module.register_forward_hook(patch_hook)
        try:
            with torch.no_grad():
                outputs = model(**target_inputs, use_cache=False)
        finally:
            handle.remove()
        patched_scores[key] = sequence_scores(outputs.logits, target_positions)
        del outputs

    example_index = {
        (spec["row_index"], spec["control_index"], spec["candidate_role"]): index
        for index, spec in enumerate(example_specs)
    }
    rows = []
    contrast_reproduction_errors = []
    source_duplicate_errors = []
    for row_index, record in enumerate(records):
        controls = []
        source_margins = []
        target_margins = []
        for control_index, control in enumerate(record["control_titles"]):
            emitted_index = example_index[(row_index, control_index, "emitted")]
            reference_index = example_index[(row_index, control_index, "reference")]
            source_margin = source_scores[emitted_index] - source_scores[reference_index]
            target_margin = target_scores[emitted_index] - target_scores[reference_index]
            source_margins.append(source_margin)
            target_margins.append(target_margin)
            controls.append(
                {
                    **control,
                    "source_factual_margin": source_margin,
                    "target_control_margin": target_margin,
                    "relation_effect": source_margin - target_margin,
                    "signed_relation_effect": expected_sign(record["catalog_label"])
                    * (source_margin - target_margin),
                }
            )
        source_duplicate_errors.append(max(source_margins) - min(source_margins))
        observed_delta = sum(
            control["relation_effect"] for control in controls
        ) / len(controls)
        contrast_reproduction_errors.append(
            abs(observed_delta - float(record["observed_sequence_relation_delta"]))
        )

        intervention_results = {}
        for intervention in interventions:
            key = (intervention["layer"], intervention["component"])
            label = intervention_label(*key)
            control_patched_margins = []
            recoveries = []
            signed_shifts = []
            for control_index, control in enumerate(controls):
                emitted_index = example_index[(row_index, control_index, "emitted")]
                reference_index = example_index[(row_index, control_index, "reference")]
                patched_margin = (
                    patched_scores[key][emitted_index]
                    - patched_scores[key][reference_index]
                )
                control_patched_margins.append(patched_margin)
                recovery = normalized_recovery(
                    control["source_factual_margin"],
                    control["target_control_margin"],
                    patched_margin,
                    float(bundle["minimum_relation_effect"]),
                )
                recoveries.append(recovery)
                signed_shifts.append(
                    expected_sign(record["catalog_label"])
                    * (patched_margin - control["target_control_margin"])
                )
            valid = [float(value) for value in recoveries if value is not None]
            intervention_results[label] = {
                "patched_control_margins": control_patched_margins,
                "recoveries": recoveries,
                "mean_recovery": sum(valid) / len(valid) if valid else None,
                "mean_signed_shift": sum(signed_shifts) / len(signed_shifts),
            }
        row = {
            **record,
            "controls": controls,
            "mean_observed_relation_effect": observed_delta,
            "mean_signed_observed_relation_effect": expected_sign(
                record["catalog_label"]
            )
            * observed_delta,
            "contrast_reproduction_error": contrast_reproduction_errors[-1],
            "interventions": intervention_results,
        }
        rows.append(row)

    def curve_for(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
        curve = []
        for intervention in interventions:
            label = intervention_label(
                intervention["layer"], intervention["component"]
            )
            recoveries = [
                recovery
                for row in selected
                for recovery in row["interventions"][label]["recoveries"]
                if recovery is not None
            ]
            signed_shifts = [
                float(row["interventions"][label]["mean_signed_shift"])
                for row in selected
            ]
            curve.append(
                {
                    **intervention,
                    "nonweak_count": len(recoveries),
                    "mean_recovery": (
                        sum(recoveries) / len(recoveries) if recoveries else None
                    ),
                    "median_recovery": (
                        statistics.median(recoveries) if recoveries else None
                    ),
                    "toward_source_rate": (
                        sum(value > 0 for value in recoveries) / len(recoveries)
                        if recoveries
                        else None
                    ),
                    "mean_signed_shift": sum(signed_shifts) / len(signed_shifts),
                }
            )
        return curve

    analysis_groups = {
        "all": rows,
        "verified_exact": [
            row for row in rows if row["catalog_label"] == "verified_exact"
        ],
        "catalog_conflict": [
            row for row in rows if row["catalog_label"] == "catalog_conflict"
        ],
    }
    for role in sorted({row["focus_role"] for row in rows}):
        analysis_groups[f"focus:{role}"] = [
            row for row in rows if row["focus_role"] == role
        ]
    group_curves = {}
    for group, selected in analysis_groups.items():
        curve = curve_for(selected)
        residual_curve = [
            point for point in curve if point["component"] == "full_residual"
        ]
        group_curves[group] = {
            "row_count": len(selected),
            "curve": curve,
            "earliest_sustained_residual_layer": earliest_sustained_layer(
                residual_curve
            ),
        }

    endpoint_key = (num_layers, "full_residual")
    endpoint_errors = [
        abs(patched_scores[endpoint_key][index] - source_scores[index])
        for index in range(len(example_specs))
    ]

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
    intervention_gate = (
        len(source_vectors) == len(interventions)
        and len(patched_scores) == len(interventions)
        and all(len(row["interventions"]) == len(interventions) for row in rows)
    )
    source_duplicate_gate = max(source_duplicate_errors) <= 1e-6
    endpoint_gate = max(endpoint_errors) <= float(bundle["endpoint_tolerance"])
    contrast_reproduction_gate = max(contrast_reproduction_errors) <= float(
        bundle["contrast_reproduction_tolerance"]
    )
    technical_gate = (
        architecture_gate
        and token_alignment_gate
        and len(rows) == 17
        and len(example_specs) == 68
        and intervention_gate
        and source_duplicate_gate
        and endpoint_gate
        and contrast_reproduction_gate
        and all_values_finite
    )
    summary = {
        "run_id": run_id,
        "model_id": bundle["model_id"],
        "model_revision": model_revision,
        "row_count": len(rows),
        "example_count": len(example_specs),
        "focus_role_counts": dict(Counter(row["focus_role"] for row in rows)),
        "group_curves": group_curves,
        "maximum_source_duplicate_error": max(source_duplicate_errors),
        "maximum_contrast_reproduction_error": max(contrast_reproduction_errors),
        "maximum_endpoint_error": max(endpoint_errors),
        "architecture_gate": architecture_gate,
        "token_alignment_gate": token_alignment_gate,
        "intervention_gate": intervention_gate,
        "source_duplicate_gate": source_duplicate_gate,
        "contrast_reproduction_gate": contrast_reproduction_gate,
        "endpoint_gate": endpoint_gate,
        "all_values_finite": all_values_finite,
        "technical_gate": technical_gate,
        "interpretation_scope": "posthoc_full_sequence_causal_title_trace",
        "source_title_contrast_job": bundle["source_title_contrast_job"],
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    chunks = encode_artifact_chunks({"rows": rows, "summary": summary})
    for index, data in enumerate(chunks):
        print(
            "SEQ_CAUSAL_ARTIFACT_CHUNK_JSON="
            + json.dumps(
                {"index": index, "total": len(chunks), "data": data},
                separators=(",", ":"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
