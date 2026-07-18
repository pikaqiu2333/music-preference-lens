# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.8.0",
#   "torch==2.7.1",
#   "transformers==4.53.2",
# ]
# ///
"""Localize the layer-18 vocality effect to Qwen3 attention heads."""

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


def head_bounds(head: int, head_dim: int, num_heads: int, width: int) -> tuple[int, int]:
    if head < 0 or head >= num_heads:
        raise ValueError(f"head index {head} is outside 0..{num_heads - 1}")
    if width != head_dim * num_heads:
        raise ValueError("concatenated attention width does not match architecture")
    return head * head_dim, (head + 1) * head_dim


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
    records_by_key = {row["track_key"]: row for row in bundle["all_records"]}
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
    )
    if not architecture_matches:
        raise ValueError(
            f"unexpected attention architecture: heads={num_heads}, "
            f"head_dim={head_dim}, o_proj_width={o_proj.in_features}"
        )

    def output_hidden(output: Any) -> Any:
        return output[0] if isinstance(output, tuple) else output

    def encode_pair_side(side: str) -> tuple[Any, list[list[int]], list[list[int]]]:
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
        predictions = []
        target_ids = []
        for row_index, raw_offsets in enumerate(offsets_batch):
            offsets = [tuple(item) for item in raw_offsets]
            title_span, artist_span = spans[row_index]
            positions = overlapping_positions(offsets, title_span)
            positions += overlapping_positions(offsets, artist_span)
            prediction_positions = [position - 1 for position in positions]
            predictions.append(prediction_positions)
            target_ids.append(
                [int(encoded["input_ids"][row_index, position]) for position in positions]
            )
        return encoded.to(device), predictions, target_ids

    source_pair, source_pair_positions, source_pair_ids = encode_pair_side("source")
    target_pair, target_pair_positions, target_pair_ids = encode_pair_side("target")
    if source_pair_ids != target_pair_ids:
        raise ValueError("pair target tokenization differs between reason conditions")

    pair_source_heads: list[Any] = []
    pair_source_attention: list[Any] = []

    def capture_pair_heads(_module: Any, inputs: Any) -> None:
        hidden = inputs[0]
        pair_source_heads.clear()
        pair_source_heads.extend(
            hidden[row_index, positions].detach().clone()
            for row_index, positions in enumerate(source_pair_positions)
        )

    def capture_pair_attention(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output_hidden(output)
        pair_source_attention.clear()
        pair_source_attention.extend(
            hidden[row_index, positions].detach().clone()
            for row_index, positions in enumerate(source_pair_positions)
        )

    handles = [
        o_proj.register_forward_pre_hook(capture_pair_heads),
        attention.register_forward_hook(capture_pair_attention),
    ]
    try:
        with torch.no_grad():
            source_pair_outputs = model(**source_pair, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    with torch.no_grad():
        target_pair_outputs = model(**target_pair, use_cache=False)

    def pair_mean_logps(logits: Any, positions: list[list[int]]) -> list[float]:
        scores = []
        for row_index, prediction_positions in enumerate(positions):
            step_logits = logits[row_index, prediction_positions].to(dtype=torch.float32)
            ids = torch.tensor(source_pair_ids[row_index], device=device)
            token_logits = step_logits[
                torch.arange(len(prediction_positions), device=device), ids
            ]
            token_logps = token_logits - torch.logsumexp(step_logits, dim=-1)
            scores.append(float(token_logps.mean().item()))
        return scores

    pair_source_scores = pair_mean_logps(source_pair_outputs.logits, source_pair_positions)
    pair_target_scores = pair_mean_logps(target_pair_outputs.logits, target_pair_positions)
    del source_pair_outputs
    del target_pair_outputs

    def run_pair_head_patch(head: int | None) -> list[float]:
        def patch_heads(_module: Any, inputs: Any) -> Any:
            hidden = inputs[0]
            patched = hidden.clone()
            for row_index, positions in enumerate(target_pair_positions):
                if head is None:
                    patched[row_index, positions] = pair_source_heads[row_index]
                else:
                    start, end = head_bounds(head, head_dim, num_heads, hidden.shape[-1])
                    patched[row_index, positions, start:end] = pair_source_heads[row_index][
                        :, start:end
                    ]
            return (patched, *inputs[1:])

        handle = o_proj.register_forward_pre_hook(patch_heads)
        try:
            with torch.no_grad():
                outputs = model(**target_pair, use_cache=False)
        finally:
            handle.remove()
        scores = pair_mean_logps(outputs.logits, target_pair_positions)
        del outputs
        return scores

    pair_head_scores = {head: run_pair_head_patch(head) for head in range(num_heads)}
    pair_all_head_scores = run_pair_head_patch(None)

    def patch_pair_attention(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output_hidden(output)
        patched = hidden.clone()
        for row_index, positions in enumerate(target_pair_positions):
            patched[row_index, positions] = pair_source_attention[row_index]
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    handle = attention.register_forward_hook(patch_pair_attention)
    try:
        with torch.no_grad():
            outputs = model(**target_pair, use_cache=False)
    finally:
        handle.remove()
    pair_direct_attention_scores = pair_mean_logps(outputs.logits, target_pair_positions)
    del outputs

    pair_rows = []
    for row_index, record in enumerate(focus):
        source = pair_source_scores[row_index]
        target = pair_target_scores[row_index]
        patched_scores = {
            str(head): pair_head_scores[head][row_index] for head in range(num_heads)
        }
        recoveries = {
            key: recovery(source, target, value) for key, value in patched_scores.items()
        }
        all_heads = pair_all_head_scores[row_index]
        direct_attention = pair_direct_attention_scores[row_index]
        row = {
            "track_key": record["track_key"],
            "title": record["title"],
            "artist": record["artist"],
            "vocality": record["vocality"],
            "sentinel_role": record["sentinel_role"],
            "source_pair_mean_logp": source,
            "target_pair_mean_logp": target,
            "source_target_effect": source - target,
            "head_patched_pair_mean_logps": patched_scores,
            "head_recoveries": recoveries,
            "all_heads_patched_pair_mean_logp": all_heads,
            "all_heads_recovery": recovery(source, target, all_heads),
            "direct_attention_patched_pair_mean_logp": direct_attention,
            "direct_attention_recovery": recovery(source, target, direct_attention),
            "all_heads_direct_error": abs(all_heads - direct_attention),
        }
        pair_rows.append(row)
        print(
            "VOCALITY_HEAD_PAIR_ROW_JSON="
            + json.dumps(row, ensure_ascii=True, separators=(",", ":"))
        )

    choice_specs = []
    for class_label in ("vocal", "instrumental"):
        target_label = "instrumental" if class_label == "vocal" else "vocal"
        for order_index, order in enumerate(bundle["candidate_orders"]):
            source_prompt, mapping = render_choice_prompt(
                bundle["reasons"][class_label], order, records_by_key
            )
            target_prompt, target_mapping = render_choice_prompt(
                bundle["reasons"][target_label], order, records_by_key
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
    choice_source_heads: Any = None
    choice_source_attention: Any = None

    def capture_choice_heads(_module: Any, inputs: Any) -> None:
        nonlocal choice_source_heads
        choice_source_heads = inputs[0][
            batch_indices, source_choice_positions
        ].detach().clone()

    def capture_choice_attention(_module: Any, _inputs: Any, output: Any) -> None:
        nonlocal choice_source_attention
        choice_source_attention = output_hidden(output)[
            batch_indices, source_choice_positions
        ].detach().clone()

    handles = [
        o_proj.register_forward_pre_hook(capture_choice_heads),
        attention.register_forward_hook(capture_choice_attention),
    ]
    try:
        with torch.no_grad():
            source_choice_outputs = model(**source_choice, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    with torch.no_grad():
        target_choice_outputs = model(**target_choice, use_cache=False)

    token_id_cache: dict[tuple[str, str], int] = {}

    def continuation_token_id(prompt: str, letter: str) -> int:
        cache_key = (prompt, letter)
        if cache_key in token_id_cache:
            return token_id_cache[cache_key]
        text = prompt + " " + letter
        encoded = tokenizer(text, return_offsets_mapping=True)
        offsets = [tuple(item) for item in encoded["offset_mapping"]]
        positions = overlapping_positions(offsets, [len(prompt), len(text)])
        if len(positions) != 1:
            raise ValueError(f"choice {letter} is not one continuation token")
        token_id = int(encoded["input_ids"][positions[0]])
        token_id_cache[cache_key] = token_id
        return token_id

    def class_margins(logits: Any, positions: Any, prompt_key: str) -> list[float]:
        margins = []
        for row_index, spec in enumerate(choice_specs):
            final_logits = logits[row_index, positions[row_index]].to(dtype=torch.float32)
            class_values = []
            other_values = []
            prompt = spec[prompt_key]
            for letter, track_key in spec["mapping"].items():
                value = final_logits[continuation_token_id(prompt, letter)]
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
        source_choice_outputs.logits, source_choice_positions, "source_prompt"
    )
    choice_target_scores = class_margins(
        target_choice_outputs.logits, target_choice_positions, "target_prompt"
    )
    del source_choice_outputs
    del target_choice_outputs

    def run_choice_head_patch(head: int | None) -> list[float]:
        def patch_heads(_module: Any, inputs: Any) -> Any:
            hidden = inputs[0]
            patched = hidden.clone()
            if head is None:
                patched[batch_indices, target_choice_positions] = choice_source_heads
            else:
                start, end = head_bounds(head, head_dim, num_heads, hidden.shape[-1])
                patched[batch_indices, target_choice_positions, start:end] = (
                    choice_source_heads[:, start:end]
                )
            return (patched, *inputs[1:])

        handle = o_proj.register_forward_pre_hook(patch_heads)
        try:
            with torch.no_grad():
                outputs = model(**target_choice, use_cache=False)
        finally:
            handle.remove()
        scores = class_margins(outputs.logits, target_choice_positions, "target_prompt")
        del outputs
        return scores

    choice_head_scores = {head: run_choice_head_patch(head) for head in range(num_heads)}
    choice_all_head_scores = run_choice_head_patch(None)

    def patch_choice_attention(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output_hidden(output)
        patched = hidden.clone()
        patched[batch_indices, target_choice_positions] = choice_source_attention
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    handle = attention.register_forward_hook(patch_choice_attention)
    try:
        with torch.no_grad():
            outputs = model(**target_choice, use_cache=False)
    finally:
        handle.remove()
    choice_direct_attention_scores = class_margins(
        outputs.logits, target_choice_positions, "target_prompt"
    )
    del outputs

    choice_rows = []
    for row_index, spec in enumerate(choice_specs):
        source = choice_source_scores[row_index]
        target = choice_target_scores[row_index]
        patched_scores = {
            str(head): choice_head_scores[head][row_index] for head in range(num_heads)
        }
        recoveries = {
            key: recovery(source, target, value) for key, value in patched_scores.items()
        }
        all_heads = choice_all_head_scores[row_index]
        direct_attention = choice_direct_attention_scores[row_index]
        row = {
            "class_label": spec["class_label"],
            "source_reason": spec["source_reason"],
            "target_reason": spec["target_reason"],
            "order_index": spec["order_index"],
            "mapping": spec["mapping"],
            "source_class_margin": source,
            "target_class_margin": target,
            "source_target_effect": source - target,
            "head_patched_class_margins": patched_scores,
            "head_recoveries": recoveries,
            "all_heads_patched_class_margin": all_heads,
            "all_heads_recovery": recovery(source, target, all_heads),
            "direct_attention_patched_class_margin": direct_attention,
            "direct_attention_recovery": recovery(source, target, direct_attention),
            "all_heads_direct_error": abs(all_heads - direct_attention),
        }
        choice_rows.append(row)
        print(
            "VOCALITY_HEAD_CHOICE_ROW_JSON="
            + json.dumps(row, ensure_ascii=True, separators=(",", ":"))
        )

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    choice_head_curve = []
    pair_head_curve = []
    success_rows = [row for row in pair_rows if "success" in row["sentinel_role"]]
    failure_rows = [row for row in pair_rows if "failure" in row["sentinel_role"]]
    consistent_heads = []
    for head in range(num_heads):
        key = str(head)
        choice_values = [float(row["head_recoveries"][key]) for row in choice_rows]
        choice_mean = mean(choice_values)
        choice_rate = sum(value > 0 for value in choice_values) / len(choice_values)
        choice_head_curve.append(
            {
                "head": head,
                "mean_recovery": choice_mean,
                "median_recovery": statistics.median(choice_values),
                "toward_source_rate": choice_rate,
                "minimum_recovery": min(choice_values),
            }
        )
        if (
            choice_rate == 1.0
            and choice_mean >= float(bundle["choice_consistent_min_mean_recovery"])
        ):
            consistent_heads.append(head)

        success_values = [float(row["head_recoveries"][key]) for row in success_rows]
        failure_values = [float(row["head_recoveries"][key]) for row in failure_rows]
        pair_head_curve.append(
            {
                "head": head,
                "success_mean_recovery": mean(success_values),
                "failure_mean_recovery": mean(failure_values),
                "success_toward_source_rate": sum(value > 0 for value in success_values)
                / len(success_values),
                "failure_toward_source_rate": sum(value > 0 for value in failure_values)
                / len(failure_values),
                "success_minus_failure_mean": mean(success_values)
                - mean(failure_values),
            }
        )

    finite_values = []
    for row in pair_rows:
        finite_values.extend(
            [
                row["source_pair_mean_logp"],
                row["target_pair_mean_logp"],
                row["source_target_effect"],
                row["all_heads_patched_pair_mean_logp"],
                row["all_heads_recovery"],
                row["direct_attention_patched_pair_mean_logp"],
                row["direct_attention_recovery"],
            ]
        )
        finite_values.extend(row["head_patched_pair_mean_logps"].values())
        finite_values.extend(row["head_recoveries"].values())
    for row in choice_rows:
        finite_values.extend(
            [
                row["source_class_margin"],
                row["target_class_margin"],
                row["source_target_effect"],
                row["all_heads_patched_class_margin"],
                row["all_heads_recovery"],
                row["direct_attention_patched_class_margin"],
                row["direct_attention_recovery"],
            ]
        )
        finite_values.extend(row["head_patched_class_margins"].values())
        finite_values.extend(row["head_recoveries"].values())
    all_values_finite = all(math.isfinite(float(value)) for value in finite_values)
    max_pair_reproduction_error = max(row["all_heads_direct_error"] for row in pair_rows)
    max_choice_reproduction_error = max(
        row["all_heads_direct_error"] for row in choice_rows
    )
    all_interventions_present = (
        all(len(row["head_recoveries"]) == num_heads for row in pair_rows)
        and all(len(row["head_recoveries"]) == num_heads for row in choice_rows)
        and len(pair_rows) == 4
        and len(choice_rows) == 4
    )
    technical_gate = (
        architecture_matches
        and all_interventions_present
        and all(
            abs(row["source_target_effect"])
            >= float(bundle["minimum_pair_effect"])
            for row in pair_rows + choice_rows
        )
        and max_pair_reproduction_error
        <= float(bundle["all_head_reproduction_tolerance"])
        and max_choice_reproduction_error
        <= float(bundle["all_head_reproduction_tolerance"])
        and all_values_finite
    )
    summary = {
        "run_id": run_id,
        "model_id": bundle["model_id"],
        "target_layer": target_layer,
        "num_attention_heads": num_heads,
        "head_dim": head_dim,
        "pair_count": len(pair_rows),
        "choice_relation_count": len(choice_rows),
        "choice_head_curve": choice_head_curve,
        "pair_head_curve": pair_head_curve,
        "choice_consistent_heads": consistent_heads,
        "choice_consistent_rule": {
            "toward_source_rate": 1.0,
            "minimum_mean_recovery": bundle["choice_consistent_min_mean_recovery"],
        },
        "max_pair_all_heads_direct_error": max_pair_reproduction_error,
        "max_choice_all_heads_direct_error": max_choice_reproduction_error,
        "architecture_matches": architecture_matches,
        "all_interventions_present": all_interventions_present,
        "all_values_finite": all_values_finite,
        "technical_gate": technical_gate,
        "interpretation_scope": "exploratory_layer18_head_localization",
        "behavior_reference": bundle["behavior_reference"],
        "component_reference_run": bundle["component_reference_run"],
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    print(
        "VOCALITY_HEAD_SUMMARY_JSON="
        + json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
