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
"""Run layerwise logit-lens and residual-patching song relation probes."""

from __future__ import annotations

import argparse
import base64
import json
import math
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


def relation_margin(
    title_correct: float,
    neutral_correct: float,
    title_wrong: float,
    neutral_wrong: float,
) -> float:
    return (title_correct - neutral_correct) - (title_wrong - neutral_wrong)


def find_earliest_sustained(
    accuracies: list[float],
    threshold: float,
    count: int,
) -> int | None:
    for start in range(0, len(accuracies) - count + 1):
        if all(value >= threshold for value in accuracies[start : start + count]):
            return start
    return None


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
    depth_labels = ["embedding"] + [f"layer_{index + 1}" for index in range(layer_count)]

    sequence_specs: list[dict[str, Any]] = []
    for control in controls:
        for title_condition, title in (
            ("title", control["title"]),
            ("neutral", control["neutral_title"]),
        ):
            prefix = bundle["likelihood_template"].format(title=title)
            for artist_condition, artist in (
                ("correct", control["correct_artist"]),
                ("wrong", control["wrong_artist"]),
            ):
                sequence_specs.append(
                    {
                        "relation_id": control["relation_id"],
                        "key": f"{title_condition}_{artist_condition}",
                        "prefix": prefix,
                        "artist": artist,
                        "text": prefix + artist,
                    }
                )

    max_final_target_logit_error = 0.0
    for batch_start in range(0, len(sequence_specs), 8):
        batch = sequence_specs[batch_start : batch_start + 8]
        encoded = tokenizer(
            [spec["text"] for spec in batch],
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=256,
        )
        offsets_batch = encoded.pop("offset_mapping").tolist()
        inputs = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )
        for batch_index, spec in enumerate(batch):
            span = [len(spec["prefix"]), len(spec["text"])]
            offsets = [tuple(item) for item in offsets_batch[batch_index]]
            target_positions = overlapping_positions(offsets, span)
            if target_positions[0] == 0:
                raise ValueError("artist continuation begins at token zero")
            prediction_positions = [position - 1 for position in target_positions]
            target_ids = inputs["input_ids"][batch_index, target_positions]

            actual_token_logits = outputs.logits[
                batch_index,
                prediction_positions,
                target_ids,
            ].to(dtype=torch.float32)
            actual_step_logits = outputs.logits[
                batch_index,
                prediction_positions,
            ].to(dtype=torch.float32)
            actual_logps = actual_token_logits - torch.logsumexp(
                actual_step_logits,
                dim=-1,
            )
            spec["actual_mean_logp"] = float(actual_logps.mean().item())
            spec["actual_mean_target_logit"] = float(actual_token_logits.mean().item())
            spec["token_count"] = len(target_positions)

            layer_target_logits: list[float] = []
            for depth, hidden in enumerate(outputs.hidden_states):
                prediction_hidden = hidden[batch_index, prediction_positions]
                normalized = (
                    prediction_hidden
                    if depth == len(outputs.hidden_states) - 1
                    else model.model.norm(prediction_hidden)
                )
                target_weights = model.lm_head.weight[target_ids]
                target_logits = (normalized * target_weights).sum(dim=-1)
                if model.lm_head.bias is not None:
                    target_logits = target_logits + model.lm_head.bias[target_ids]
                layer_target_logits.append(
                    float(target_logits.to(dtype=torch.float32).mean().item())
                )
            spec["layer_target_logits"] = layer_target_logits
            max_final_target_logit_error = max(
                max_final_target_logit_error,
                abs(layer_target_logits[-1] - spec["actual_mean_target_logit"]),
            )
        del outputs

    specs_by_relation: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in sequence_specs:
        specs_by_relation.setdefault(spec["relation_id"], {})[spec["key"]] = spec

    relation_rows: list[dict[str, Any]] = []
    for control in controls:
        specs = specs_by_relation[control["relation_id"]]
        final_direct_margin = relation_margin(
            specs["title_correct"]["actual_mean_logp"],
            specs["neutral_correct"]["actual_mean_logp"],
            specs["title_wrong"]["actual_mean_logp"],
            specs["neutral_wrong"]["actual_mean_logp"],
        )
        layer_margins = [
            relation_margin(
                specs["title_correct"]["layer_target_logits"][depth],
                specs["neutral_correct"]["layer_target_logits"][depth],
                specs["title_wrong"]["layer_target_logits"][depth],
                specs["neutral_wrong"]["layer_target_logits"][depth],
            )
            for depth in range(len(depth_labels))
        ]
        relation_rows.append(
            {
                **control,
                "final_direct_pmi_margin": final_direct_margin,
                "layer_target_logit_pmi_margins": layer_margins,
                "continuation_token_counts": {
                    key: spec["token_count"] for key, spec in specs.items()
                },
            }
        )

    patch_prefixes_real = [
        bundle["patch_prefix_template"].format(title=row["title"])
        for row in relation_rows
    ]
    patch_prefixes_neutral = [
        bundle["patch_prefix_template"].format(title=row["neutral_title"])
        for row in relation_rows
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

    correct_first_ids = [
        first_continuation_token_id(prefix, row["correct_artist"])
        for prefix, row in zip(patch_prefixes_real, relation_rows)
    ]
    wrong_first_ids = [
        first_continuation_token_id(prefix, row["wrong_artist"])
        for prefix, row in zip(patch_prefixes_real, relation_rows)
    ]
    if any(correct == wrong for correct, wrong in zip(correct_first_ids, wrong_first_ids)):
        raise ValueError("a correct/wrong artist pair shares its first continuation token")

    real_encoded = tokenizer(
        patch_prefixes_real,
        return_tensors="pt",
        padding=True,
    ).to(device)
    neutral_encoded = tokenizer(
        patch_prefixes_neutral,
        return_tensors="pt",
        padding=True,
    ).to(device)

    def final_nonpad_positions(attention_mask: Any) -> Any:
        positions = []
        for row_mask in attention_mask:
            positions.append(
                int(torch.nonzero(row_mask, as_tuple=False)[-1].item())
            )
        return torch.tensor(positions, device=device, dtype=torch.long)

    real_positions = final_nonpad_positions(real_encoded["attention_mask"])
    neutral_positions = final_nonpad_positions(neutral_encoded["attention_mask"])
    batch_indices = torch.arange(len(relation_rows), device=device)
    correct_id_tensor = torch.tensor(correct_first_ids, device=device, dtype=torch.long)
    wrong_id_tensor = torch.tensor(wrong_first_ids, device=device, dtype=torch.long)

    neutral_layer_vectors: list[Any] = [None] * layer_count
    capture_handles = []
    for layer_index, layer in enumerate(model.model.layers):
        def capture_hook(
            _module: Any,
            _inputs: Any,
            output: Any,
            index: int = layer_index,
        ) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            neutral_layer_vectors[index] = hidden[
                batch_indices,
                neutral_positions,
            ].detach().clone()

        capture_handles.append(layer.register_forward_hook(capture_hook))
    try:
        with torch.no_grad():
            neutral_outputs = model(**neutral_encoded, use_cache=False)
    finally:
        for handle in capture_handles:
            handle.remove()
    if any(vector is None for vector in neutral_layer_vectors):
        raise ValueError("failed to capture every neutral layer residual")
    with torch.no_grad():
        real_outputs = model(**real_encoded, use_cache=False)

    def first_token_margins(logits: Any, positions: Any) -> Any:
        final_logits = logits[batch_indices, positions]
        return (
            final_logits[batch_indices, correct_id_tensor]
            - final_logits[batch_indices, wrong_id_tensor]
        ).to(dtype=torch.float32)

    neutral_first_margins = first_token_margins(
        neutral_outputs.logits,
        neutral_positions,
    )
    real_first_margins = first_token_margins(real_outputs.logits, real_positions)
    del neutral_outputs
    del real_outputs
    patched_by_layer: list[list[float]] = []
    for layer_index in range(layer_count):
        neutral_vectors = neutral_layer_vectors[layer_index]

        def patch_hook(
            _module: Any,
            _inputs: Any,
            output: Any,
            replacement: Any = neutral_vectors,
        ) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            patched = hidden.clone()
            patched[batch_indices, real_positions] = replacement
            if isinstance(output, tuple):
                return (patched, *output[1:])
            return patched

        handle = model.model.layers[layer_index].register_forward_hook(patch_hook)
        try:
            with torch.no_grad():
                patched_outputs = model(**real_encoded, use_cache=False)
        finally:
            handle.remove()
        patched_margins = first_token_margins(
            patched_outputs.logits,
            real_positions,
        )
        patched_by_layer.append([float(value) for value in patched_margins.tolist()])
        del patched_outputs

    for row_index, row in enumerate(relation_rows):
        real_margin = float(real_first_margins[row_index].item())
        neutral_margin = float(neutral_first_margins[row_index].item())
        patched_margins = [values[row_index] for values in patched_by_layer]
        denominator = real_margin - neutral_margin
        row["correct_first_token_id"] = correct_first_ids[row_index]
        row["wrong_first_token_id"] = wrong_first_ids[row_index]
        row["real_first_token_margin"] = real_margin
        row["neutral_first_token_margin"] = neutral_margin
        row["patched_first_token_margins"] = patched_margins
        row["patch_aligned_effects"] = [
            (real_margin - patched) * (1.0 if denominator >= 0 else -1.0)
            for patched in patched_margins
        ]
        row["patch_remaining_fractions"] = [
            abs(patched - neutral_margin) / abs(denominator)
            if abs(denominator) >= 0.05
            else None
            for patched in patched_margins
        ]

    layer_curve: list[dict[str, Any]] = []
    layer_accuracies: list[float] = []
    for depth, label in enumerate(depth_labels):
        margins = [row["layer_target_logit_pmi_margins"][depth] for row in relation_rows]
        accuracy = sum(margin > 0 for margin in margins) / len(margins)
        layer_accuracies.append(accuracy)
        layer_curve.append(
            {
                "depth": depth,
                "label": label,
                "accuracy": accuracy,
                "mean_margin": sum(margins) / len(margins),
                "english_accuracy": sum(
                    row["layer_target_logit_pmi_margins"][depth] > 0
                    for row in relation_rows
                    if row["language"] == "en"
                )
                / sum(row["language"] == "en" for row in relation_rows),
                "chinese_accuracy": sum(
                    row["layer_target_logit_pmi_margins"][depth] > 0
                    for row in relation_rows
                    if row["language"] == "zh"
                )
                / sum(row["language"] == "zh" for row in relation_rows),
            }
        )

    patch_curve: list[dict[str, Any]] = []
    for layer_index in range(layer_count):
        aligned_effects = [row["patch_aligned_effects"][layer_index] for row in relation_rows]
        remaining = [
            row["patch_remaining_fractions"][layer_index]
            for row in relation_rows
            if row["patch_remaining_fractions"][layer_index] is not None
        ]
        toward_neutral = [
            abs(row["patched_first_token_margins"][layer_index] - row["neutral_first_token_margin"])
            < abs(row["real_first_token_margin"] - row["neutral_first_token_margin"])
            for row in relation_rows
        ]
        patch_curve.append(
            {
                "layer": layer_index + 1,
                "mean_aligned_effect": sum(aligned_effects) / len(aligned_effects),
                "toward_neutral_accuracy": sum(toward_neutral) / len(toward_neutral),
                "mean_remaining_fraction": (
                    sum(remaining) / len(remaining) if remaining else None
                ),
            }
        )

    final_direct_accuracy = sum(
        row["final_direct_pmi_margin"] > 0 for row in relation_rows
    ) / len(relation_rows)
    earliest_depth = find_earliest_sustained(
        layer_accuracies,
        float(bundle["sustained_accuracy_threshold"]),
        int(bundle["sustained_layer_count"]),
    )
    max_patch_endpoint_error = max(
        abs(row["patched_first_token_margins"][-1] - row["neutral_first_token_margin"])
        for row in relation_rows
    )
    tolerance = float(bundle["consistency_tolerance"])
    technical_gate = (
        len(relation_rows) == len(controls)
        and all(
            len(row["layer_target_logit_pmi_margins"]) == len(depth_labels)
            and len(row["patched_first_token_margins"]) == layer_count
            for row in relation_rows
        )
        and max_final_target_logit_error <= tolerance
        and max_patch_endpoint_error <= tolerance
    )
    behavioral_gate = final_direct_accuracy >= float(bundle["behavior_threshold"])
    interpretation_gate = technical_gate and behavioral_gate
    summary = {
        "run_id": run_id,
        "mode": args.mode,
        "model_id": bundle["model_id"],
        "relation_count": len(relation_rows),
        "layer_count": layer_count,
        "depth_labels": depth_labels,
        "final_direct_pmi_accuracy": final_direct_accuracy,
        "layer_target_logit_curve": layer_curve,
        "earliest_sustained_depth": earliest_depth,
        "earliest_sustained_label": (
            depth_labels[earliest_depth] if earliest_depth is not None else None
        ),
        "patch_curve": patch_curve,
        "max_final_target_logit_error": max_final_target_logit_error,
        "max_patch_endpoint_error": max_patch_endpoint_error,
        "technical_gate": technical_gate,
        "behavioral_gate": behavioral_gate,
        "interpretation_gate": interpretation_gate,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if not all(
        math.isfinite(float(value))
        for row in relation_rows
        for value in (
            row["final_direct_pmi_margin"],
            row["real_first_token_margin"],
            row["neutral_first_token_margin"],
        )
    ):
        raise ValueError("non-finite relation metric")

    for row in relation_rows:
        print(
            "LAYER_ATTR_ROW_JSON="
            + json.dumps(row, ensure_ascii=True, separators=(",", ":"))
        )
    print(
        "LAYER_ATTR_SUMMARY_JSON="
        + json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
