# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "accelerate==1.8.1",
#   "torch==2.7.1",
#   "transformers==4.53.2",
# ]
# ///
"""Diagnose holdout failures with factual-versus-control title contrasts."""

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


def render_choice_prompt(
    title: str,
    emitted_artist: str,
    reference_artist: str,
    order: list[str],
) -> tuple[str, dict[str, str]]:
    artists = {"emitted": emitted_artist, "reference": reference_artist}
    lines = [f'Which artist recorded the track titled "{title}"?', "Options:"]
    mapping = {}
    for index, role in enumerate(order):
        letter = chr(ord("A") + index)
        lines.append(f"{letter}. {artists[role]}")
        mapping[letter] = role
    lines.extend(["Answer with one letter only.", "Answer:"])
    return "\n".join(lines), mapping


def expected_sign(label: str) -> int:
    if label == "verified_exact":
        return 1
    if label == "catalog_conflict":
        return -1
    raise ValueError(f"unknown catalog label: {label}")


def diagnostic_class(
    label: str, factual_margin: float, relation_delta: float
) -> str:
    sign = expected_sign(label)
    absolute_correct = sign * float(factual_margin) > 0
    delta_correct = sign * float(relation_delta) > 0
    if absolute_correct and delta_correct:
        return "correct_and_relation_specific"
    if absolute_correct:
        return "correct_without_relation_contrast"
    if delta_correct:
        return "latent_relation_masked_by_prior"
    return "relation_not_recovered"


def summarize_path(rows: list[dict[str, Any]], path: str) -> dict[str, Any]:
    result = {}
    for label in ("all", "verified_exact", "catalog_conflict"):
        selected = (
            rows if label == "all" else [row for row in rows if row["catalog_label"] == label]
        )
        if not selected:
            result[label] = {
                "count": 0,
                "absolute_correct": 0,
                "delta_direction_correct": 0,
                "diagnostic_classes": {},
                "mean_signed_relation_delta": None,
                "median_signed_relation_delta": None,
                "positive_signed_delta_rate": None,
            }
            continue
        diagnostics = [row["paths"][path] for row in selected]
        signed_deltas = [float(item["signed_relation_delta"]) for item in diagnostics]
        result[label] = {
            "count": len(selected),
            "absolute_correct": sum(item["absolute_correct"] for item in diagnostics),
            "delta_direction_correct": sum(
                item["delta_direction_correct"] for item in diagnostics
            ),
            "diagnostic_classes": dict(
                Counter(item["diagnostic_class"] for item in diagnostics)
            ),
            "mean_signed_relation_delta": sum(signed_deltas) / len(signed_deltas),
            "median_signed_relation_delta": statistics.median(signed_deltas),
            "positive_signed_delta_rate": (
                sum(value > 0 for value in signed_deltas) / len(signed_deltas)
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--model-revision")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + "_title_contrast"
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

    def first_token_id(prefix: str, continuation: str) -> int:
        text, span = append_continuation(prefix, continuation)
        encoded = tokenizer(text, return_offsets_mapping=True)
        offsets = [tuple(item) for item in encoded["offset_mapping"]]
        positions = overlapping_positions(offsets, span)
        return int(encoded["input_ids"][positions[0]])

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

    conditions = []
    for row_index, row in enumerate(records):
        row_conditions = [("factual", row["title"])] + [
            (f"control_{index + 1}", control["title"])
            for index, control in enumerate(row["control_titles"])
        ]
        for condition_id, title in row_conditions:
            conditions.append(
                {
                    "row_index": row_index,
                    "condition_id": condition_id,
                    "title": title,
                }
            )

    sequence_margins: dict[tuple[int, str], float] = {}
    for condition in conditions:
        row = records[condition["row_index"]]
        prefix = catalog_prefix(condition["title"])
        margin = sequence_mean_logp(prefix, row["emitted_artist"]) - sequence_mean_logp(
            prefix, row["reference_artist"]
        )
        sequence_margins[(condition["row_index"], condition["condition_id"])] = margin

    choice_specs = []
    for condition in conditions:
        row = records[condition["row_index"]]
        for order_index, order in enumerate(
            (["emitted", "reference"], ["reference", "emitted"])
        ):
            prompt, mapping = render_choice_prompt(
                condition["title"],
                row["emitted_artist"],
                row["reference_artist"],
                list(order),
            )
            choice_specs.append(
                {
                    **condition,
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
    choice_positions = torch.tensor(
        [
            int(torch.nonzero(row, as_tuple=False)[-1].item())
            for row in choice_inputs["attention_mask"]
        ],
        device=device,
        dtype=torch.long,
    )
    with torch.no_grad():
        choice_outputs = model(**choice_inputs, use_cache=False)
    choice_order_margins: dict[tuple[int, str], list[float]] = defaultdict(list)
    for spec_index, spec in enumerate(choice_specs):
        logits = choice_outputs.logits[
            spec_index, choice_positions[spec_index]
        ].to(dtype=torch.float32)
        role_to_letter = {role: letter for letter, role in spec["mapping"].items()}
        emitted_id = first_token_id(spec["prompt"], role_to_letter["emitted"])
        reference_id = first_token_id(spec["prompt"], role_to_letter["reference"])
        choice_order_margins[(spec["row_index"], spec["condition_id"])].append(
            float((logits[emitted_id] - logits[reference_id]).item())
        )
    del choice_outputs
    choice_margins = {
        key: sum(values) / len(values) for key, values in choice_order_margins.items()
    }

    rows = []
    reproduction_errors = []
    for row_index, record in enumerate(records):
        paths = {}
        for path, values, reference_key in (
            (
                "choice",
                choice_margins,
                "reference_choice_margin",
            ),
            (
                "catalog_sequence",
                sequence_margins,
                "reference_catalog_sequence_margin",
            ),
        ):
            factual = float(values[(row_index, "factual")])
            controls = [
                float(values[(row_index, f"control_{index + 1}")])
                for index in range(len(record["control_titles"]))
            ]
            control_mean = sum(controls) / len(controls)
            delta = factual - control_mean
            sign = expected_sign(record["catalog_label"])
            classification = diagnostic_class(record["catalog_label"], factual, delta)
            reproduction_error = abs(factual - float(record[reference_key]))
            reproduction_errors.append(reproduction_error)
            paths[path] = {
                "factual_emitted_margin": factual,
                "control_emitted_margins": controls,
                "control_mean_emitted_margin": control_mean,
                "relation_delta": delta,
                "signed_relation_delta": sign * delta,
                "absolute_correct": sign * factual > 0,
                "delta_direction_correct": sign * delta > 0,
                "diagnostic_class": classification,
                "reference_reproduction_error": reproduction_error,
            }
        row = {
            **record,
            "choice_order_emitted_margins": {
                condition_id: choice_order_margins[(row_index, condition_id)]
                for condition_id in ["factual", "control_1", "control_2"]
            },
            "paths": paths,
        }
        rows.append(row)
        print(
            "TITLE_CONTRAST_ROW_JSON="
            + json.dumps(row, ensure_ascii=True, separators=(",", ":"))
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

    control_gate = all(
        len(row["control_titles"]) == int(bundle["control_count_per_event"])
        and all(
            control["known_true_artist"].casefold()
            not in {
                row["emitted_artist"].casefold(),
                row["reference_artist"].casefold(),
            }
            for control in row["control_titles"]
        )
        for row in rows
    )
    both_orders_gate = all(
        all(len(values) == 2 for values in row["choice_order_emitted_margins"].values())
        for row in rows
    )
    all_values_finite = all(
        math.isfinite(number) for row in rows for number in finite_numbers(row)
    )
    # Different prompt batch sizes can shift fp16 logits by one 1/32 step.
    reproduction_tolerance = 0.04
    reproduction_gate = max(reproduction_errors) <= reproduction_tolerance
    technical_gate = (
        len(rows) == 17
        and control_gate
        and both_orders_gate
        and all_values_finite
        and reproduction_gate
    )
    role_summary = {}
    for role in sorted({row["focus_role"] for row in rows}):
        role_rows = [row for row in rows if row["focus_role"] == role]
        role_summary[role] = {
            path: summarize_path(role_rows, path)["all"]
            for path in ("choice", "catalog_sequence")
        }
    summary = {
        "run_id": run_id,
        "model_id": bundle["model_id"],
        "model_revision": model_revision,
        "row_count": len(rows),
        "focus_role_counts": dict(Counter(row["focus_role"] for row in rows)),
        "choice_summary": summarize_path(rows, "choice"),
        "catalog_sequence_summary": summarize_path(rows, "catalog_sequence"),
        "focus_role_summary": role_summary,
        "maximum_reference_reproduction_error": max(reproduction_errors),
        "reference_reproduction_tolerance": reproduction_tolerance,
        "control_association_gate": control_gate,
        "both_choice_orders_gate": both_orders_gate,
        "all_values_finite": all_values_finite,
        "reference_reproduction_gate": reproduction_gate,
        "technical_gate": technical_gate,
        "interpretation_scope": "posthoc_causal_title_contrast_diagnostic",
        "source_confirmation_job": bundle["source_confirmation_job"],
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    print(
        "TITLE_CONTRAST_SUMMARY_JSON="
        + json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
