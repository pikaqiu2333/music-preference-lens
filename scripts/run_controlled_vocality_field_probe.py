# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.8.0",
#   "torch==2.7.1",
#   "transformers==4.53.2",
# ]
# ///
"""Diagnose title-versus-artist vocality effects with frozen layer-18 heads."""

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


def render_pair_text(reason: str, title: str, artist: str) -> tuple[str, dict[str, list[int]]]:
    text = f"Music recommendation plan.\nReason: {reason}\nTitle: "
    title_span = [len(text), len(text) + len(title)]
    text += title
    text += "\nArtist: "
    artist_span = [len(text), len(text) + len(artist)]
    text += artist
    return text, {"title": title_span, "artist": artist_span}


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


def head_bounds(head: int, head_dim: int, num_heads: int, width: int) -> tuple[int, int]:
    if head < 0 or head >= num_heads:
        raise ValueError(f"head index {head} is outside 0..{num_heads - 1}")
    if width != head_dim * num_heads:
        raise ValueError("concatenated attention width does not match architecture")
    return head * head_dim, (head + 1) * head_dim


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


def classify_failure_locus(
    title_effect: float,
    artist_effect: float,
    minimum_effect: float,
) -> str:
    title_negative = title_effect < 0
    artist_negative = artist_effect < 0
    if title_negative and artist_negative:
        label = "both_fields"
    elif title_negative:
        label = "title_only"
    elif artist_negative:
        label = "artist_only"
    else:
        label = "neither_field"
    if abs(title_effect) < minimum_effect or abs(artist_effect) < minimum_effect:
        label += "_weak_boundary"
    return label


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
    selected_heads = [int(head) for head in bundle["selected_heads"]]
    scopes = list(bundle["intervention_scopes"])
    minimum_field_effect = float(bundle["minimum_field_effect"])
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

    target_layer = int(bundle["target_layer"])
    layer = model.model.layers[target_layer - 1]
    attention = layer.self_attn
    o_proj = attention.o_proj
    num_heads = int(model.config.num_attention_heads)
    head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // num_heads))
    architecture_matches = (
        num_heads == int(bundle["expected_num_attention_heads"])
        and head_dim == int(bundle["expected_head_dim"])
        and o_proj.in_features == num_heads * head_dim
        and all(0 <= head < num_heads for head in selected_heads)
    )
    if not architecture_matches:
        raise ValueError("model attention architecture does not match the frozen bundle")

    def output_hidden(output: Any) -> Any:
        return output[0] if isinstance(output, tuple) else output

    def encode_side(side: str) -> tuple[Any, list[dict[str, list[int]]], list[dict[str, list[int]]]]:
        texts = []
        spans = []
        for row in focus:
            matched = row["vocality"]
            flipped = "instrumental" if matched == "vocal" else "vocal"
            condition = matched if side == "source" else flipped
            text, field_spans = render_pair_text(
                bundle["reasons"][condition], row["title"], row["artist"]
            )
            texts.append(text)
            spans.append(field_spans)
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
        )
        offsets_batch = encoded.pop("offset_mapping").tolist()
        predictions = []
        target_ids = []
        for row_index, raw_offsets in enumerate(offsets_batch):
            offsets = [tuple(item) for item in raw_offsets]
            row_predictions = {}
            row_ids = {}
            for field in ("title", "artist"):
                token_positions = overlapping_positions(offsets, spans[row_index][field])
                row_predictions[field] = [position - 1 for position in token_positions]
                row_ids[field] = [
                    int(encoded["input_ids"][row_index, position])
                    for position in token_positions
                ]
            predictions.append(row_predictions)
            target_ids.append(row_ids)
        return encoded.to(device), predictions, target_ids

    source_inputs, source_positions, source_ids = encode_side("source")
    target_inputs, target_positions, target_ids = encode_side("target")
    if source_ids != target_ids:
        raise ValueError("title or artist target tokenization differs between conditions")

    source_head_vectors: dict[str, list[Any]] = {"title": [], "artist": []}
    source_attention_vectors: dict[str, list[Any]] = {"title": [], "artist": []}

    def capture_heads(_module: Any, inputs: Any) -> None:
        hidden = inputs[0]
        for field in ("title", "artist"):
            source_head_vectors[field].clear()
            source_head_vectors[field].extend(
                hidden[row_index, row_positions[field]].detach().clone()
                for row_index, row_positions in enumerate(source_positions)
            )

    def capture_attention(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output_hidden(output)
        for field in ("title", "artist"):
            source_attention_vectors[field].clear()
            source_attention_vectors[field].extend(
                hidden[row_index, row_positions[field]].detach().clone()
                for row_index, row_positions in enumerate(source_positions)
            )

    handles = [
        o_proj.register_forward_pre_hook(capture_heads),
        attention.register_forward_hook(capture_attention),
    ]
    try:
        with torch.no_grad():
            source_outputs = model(**source_inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    with torch.no_grad():
        target_outputs = model(**target_inputs, use_cache=False)

    def score_fields(logits: Any, positions: list[dict[str, list[int]]]) -> list[dict[str, Any]]:
        rows = []
        for row_index, row_positions in enumerate(positions):
            field_scores = {}
            field_counts = {}
            all_logps = []
            for field in ("title", "artist"):
                prediction_positions = row_positions[field]
                step_logits = logits[row_index, prediction_positions].to(dtype=torch.float32)
                ids = torch.tensor(source_ids[row_index][field], device=device)
                token_logits = step_logits[
                    torch.arange(len(prediction_positions), device=device), ids
                ]
                token_logps = token_logits - torch.logsumexp(step_logits, dim=-1)
                field_scores[field] = float(token_logps.mean().item())
                field_counts[field] = len(prediction_positions)
                all_logps.append(token_logps)
            pair_logps = torch.cat(all_logps)
            rows.append(
                {
                    "title_mean_logp": field_scores["title"],
                    "artist_mean_logp": field_scores["artist"],
                    "pair_mean_logp": float(pair_logps.mean().item()),
                    "title_token_count": field_counts["title"],
                    "artist_token_count": field_counts["artist"],
                }
            )
        return rows

    source_scores = score_fields(source_outputs.logits, source_positions)
    target_scores = score_fields(target_outputs.logits, target_positions)
    del source_outputs
    del target_outputs

    def fields_for_scope(scope: str) -> tuple[str, ...]:
        if scope == "title":
            return ("title",)
        if scope == "artist":
            return ("artist",)
        if scope == "both":
            return ("title", "artist")
        raise ValueError(f"unknown intervention scope: {scope}")

    def run_head_patch(heads: list[int], scope: str) -> list[dict[str, Any]]:
        patch_fields = fields_for_scope(scope)

        def patch_hook(_module: Any, inputs: Any) -> Any:
            hidden = inputs[0]
            patched = hidden.clone()
            for row_index, row_positions in enumerate(target_positions):
                for field in patch_fields:
                    positions = row_positions[field]
                    for head in heads:
                        start, end = head_bounds(head, head_dim, num_heads, hidden.shape[-1])
                        patched[row_index, positions, start:end] = source_head_vectors[field][
                            row_index
                        ][:, start:end]
            return (patched, *inputs[1:])

        handle = o_proj.register_forward_pre_hook(patch_hook)
        try:
            with torch.no_grad():
                outputs = model(**target_inputs, use_cache=False)
        finally:
            handle.remove()
        scores = score_fields(outputs.logits, target_positions)
        del outputs
        return scores

    single_head_scores = {
        scope: {head: run_head_patch([head], scope) for head in selected_heads}
        for scope in scopes
    }
    selected_group_scores = {
        scope: run_head_patch(selected_heads, scope) for scope in scopes
    }
    all_head_scores = run_head_patch(list(range(num_heads)), "both")

    def patch_direct_attention(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output_hidden(output)
        patched = hidden.clone()
        for row_index, row_positions in enumerate(target_positions):
            for field in ("title", "artist"):
                patched[row_index, row_positions[field]] = source_attention_vectors[field][
                    row_index
                ]
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    handle = attention.register_forward_hook(patch_direct_attention)
    try:
        with torch.no_grad():
            outputs = model(**target_inputs, use_cache=False)
    finally:
        handle.remove()
    direct_attention_scores = score_fields(outputs.logits, target_positions)
    del outputs

    metric_names = ("title_mean_logp", "artist_mean_logp", "pair_mean_logp")

    def intervention_metrics(
        source: dict[str, Any],
        target: dict[str, Any],
        patched: dict[str, Any],
    ) -> dict[str, Any]:
        raw_deltas = {}
        recoveries = {}
        for metric in metric_names:
            raw_deltas[metric] = patched[metric] - target[metric]
            threshold = minimum_field_effect if metric != "pair_mean_logp" else 1e-12
            recoveries[metric] = normalized_recovery(
                source[metric], target[metric], patched[metric], threshold
            )
        return {
            "patched_scores": {metric: patched[metric] for metric in metric_names},
            "raw_deltas": raw_deltas,
            "recoveries": recoveries,
        }

    rows = []
    pair_reconstruction_errors = []
    all_head_direct_errors = []
    for row_index, record in enumerate(focus):
        source = source_scores[row_index]
        target = target_scores[row_index]
        effects = {
            metric: source[metric] - target[metric] for metric in metric_names
        }
        token_total = source["title_token_count"] + source["artist_token_count"]
        reconstructed_pair_effect = (
            effects["title_mean_logp"] * source["title_token_count"]
            + effects["artist_mean_logp"] * source["artist_token_count"]
        ) / token_total
        pair_reconstruction_error = abs(
            reconstructed_pair_effect - effects["pair_mean_logp"]
        )
        pair_reconstruction_errors.append(pair_reconstruction_error)

        single = {}
        for scope in scopes:
            single[scope] = {
                str(head): intervention_metrics(
                    source, target, single_head_scores[scope][head][row_index]
                )
                for head in selected_heads
            }
        selected_group = {
            scope: intervention_metrics(
                source, target, selected_group_scores[scope][row_index]
            )
            for scope in scopes
        }
        all_heads = intervention_metrics(source, target, all_head_scores[row_index])
        direct_attention = intervention_metrics(
            source, target, direct_attention_scores[row_index]
        )
        control_errors = {
            metric: abs(
                all_heads["patched_scores"][metric]
                - direct_attention["patched_scores"][metric]
            )
            for metric in metric_names
        }
        all_head_direct_errors.extend(control_errors.values())
        row = {
            "track_key": record["track_key"],
            "title": record["title"],
            "artist": record["artist"],
            "vocality": record["vocality"],
            "sentinel_role": record["sentinel_role"],
            "source_scores": source,
            "target_scores": target,
            "source_target_effects": effects,
            "failure_locus": classify_failure_locus(
                effects["title_mean_logp"],
                effects["artist_mean_logp"],
                minimum_field_effect,
            ),
            "reconstructed_pair_effect": reconstructed_pair_effect,
            "pair_reconstruction_error": pair_reconstruction_error,
            "single_head_interventions": single,
            "selected_head_group_interventions": selected_group,
            "all_head_control": all_heads,
            "direct_attention_control": direct_attention,
            "all_head_direct_errors": control_errors,
        }
        rows.append(row)
        print(
            "VOCALITY_FIELD_ROW_JSON="
            + json.dumps(row, ensure_ascii=True, separators=(",", ":"))
        )

    def finite_numbers(value: Any) -> list[float]:
        if value is None:
            return []
        if isinstance(value, bool):
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
    interventions_complete = all(
        set(row["single_head_interventions"]) == set(scopes)
        and all(
            set(row["single_head_interventions"][scope])
            == {str(head) for head in selected_heads}
            for scope in scopes
        )
        and set(row["selected_head_group_interventions"]) == set(scopes)
        for row in rows
    )
    pair_reconstruction_gate = max(pair_reconstruction_errors) <= float(
        bundle["pair_reconstruction_tolerance"]
    )
    all_head_reproduction_gate = max(all_head_direct_errors) <= float(
        bundle["all_head_reproduction_tolerance"]
    )

    head_field_curve = []
    for head in selected_heads:
        key = str(head)
        for field, scope in (("title_mean_logp", "title"), ("artist_mean_logp", "artist")):
            values = [
                row["single_head_interventions"][scope][key]["recoveries"][field]
                for row in rows
            ]
            non_null = [float(value) for value in values if value is not None]
            head_field_curve.append(
                {
                    "head": head,
                    "field": field.removesuffix("_mean_logp"),
                    "nonweak_count": len(non_null),
                    "mean_recovery": (
                        sum(non_null) / len(non_null) if non_null else None
                    ),
                    "median_recovery": (
                        statistics.median(non_null) if non_null else None
                    ),
                    "toward_source_rate": (
                        sum(value > 0 for value in non_null) / len(non_null)
                        if non_null
                        else None
                    ),
                }
            )

    technical_gate = (
        architecture_matches
        and len(rows) == 4
        and interventions_complete
        and pair_reconstruction_gate
        and all_head_reproduction_gate
        and all_values_finite
    )
    summary = {
        "run_id": run_id,
        "model_id": bundle["model_id"],
        "target_layer": target_layer,
        "selected_heads": selected_heads,
        "row_count": len(rows),
        "failure_loci": {row["track_key"]: row["failure_locus"] for row in rows},
        "field_effects": {
            row["track_key"]: row["source_target_effects"] for row in rows
        },
        "head_field_curve": head_field_curve,
        "maximum_pair_reconstruction_error": max(pair_reconstruction_errors),
        "maximum_all_head_direct_error": max(all_head_direct_errors),
        "architecture_matches": architecture_matches,
        "interventions_complete": interventions_complete,
        "pair_reconstruction_gate": pair_reconstruction_gate,
        "all_head_reproduction_gate": all_head_reproduction_gate,
        "all_values_finite": all_values_finite,
        "technical_gate": technical_gate,
        "interpretation_scope": "exploratory_title_artist_failure_locus",
        "behavior_reference": bundle["behavior_reference"],
        "head_reference_run": bundle["head_reference_run"],
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    print(
        "VOCALITY_FIELD_SUMMARY_JSON="
        + json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
