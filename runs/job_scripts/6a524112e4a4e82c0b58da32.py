# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.8.0",
#   "torch==2.7.1",
#   "transformers==4.53.2",
# ]
# ///
"""Run the preregistered independent relation-conflict verifier holdout."""

from __future__ import annotations

import argparse
import base64
import json
import math
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EMBEDDED_BUNDLE_B64 = "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"


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


def binary_metrics(
    rows: list[dict[str, Any]], prediction_key: str
) -> dict[str, Any]:
    exact = [row for row in rows if row["catalog_label"] == "verified_exact"]
    conflict = [row for row in rows if row["catalog_label"] == "catalog_conflict"]
    exact_correct = sum(not bool(row[prediction_key]) for row in exact)
    conflict_correct = sum(bool(row[prediction_key]) for row in conflict)
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


def margin_prediction(margin: float) -> bool:
    return float(margin) < 0


def combined_prediction(choice_margin: float, catalog_sequence_margin: float) -> bool:
    return margin_prediction(choice_margin) or margin_prediction(
        catalog_sequence_margin
    )


def confirmation_passes(
    metrics: dict[str, Any], rule: dict[str, Any], technical_gate: bool = True
) -> bool:
    return (
        technical_gate
        and int(metrics["exact_count"]) >= int(rule["minimum_events_per_label"])
        and int(metrics["conflict_count"]) >= int(rule["minimum_events_per_label"])
        and float(metrics["balanced_accuracy"])
        >= float(rule["minimum_balanced_accuracy"])
        and float(metrics["exact_specificity"])
        >= float(rule["minimum_exact_specificity"])
        and float(metrics["conflict_sensitivity"])
        >= float(rule["minimum_conflict_sensitivity"])
    )


def path_overlap(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for label in ("verified_exact", "catalog_conflict"):
        counts = Counter()
        for row in rows:
            if row["catalog_label"] != label:
                continue
            choice = bool(row["choice_predicts_conflict"])
            sequence = bool(row["catalog_sequence_predicts_conflict"])
            if choice and sequence:
                counts["both"] += 1
            elif choice:
                counts["choice_only"] += 1
            elif sequence:
                counts["sequence_only"] += 1
            else:
                counts["neither"] += 1
        result[label] = dict(counts)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + "_confirm"
    bundle = load_bundle(args.bundle)
    records = bundle["records"]
    rule = bundle["frozen_rule"]
    if not torch.cuda.is_available():
        raise RuntimeError("this probe requires a CUDA GPU")
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(
        bundle["model_id"], trust_remote_code=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        bundle["model_id"],
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    def final_positions(attention_mask: Any) -> Any:
        return torch.tensor(
            [
                int(torch.nonzero(row, as_tuple=False)[-1].item())
                for row in attention_mask
            ],
            device=device,
            dtype=torch.long,
        )

    def first_token_id(prefix: str, artist: str) -> int:
        text, span = append_continuation(prefix, artist)
        encoded = tokenizer(text, return_offsets_mapping=True)
        offsets = [tuple(item) for item in encoded["offset_mapping"]]
        positions = overlapping_positions(offsets, span)
        return int(encoded["input_ids"][positions[0]])

    catalog_token_ids = []
    generation_token_ids = []
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
        if catalog_ids["emitted"] == catalog_ids["reference"]:
            raise ValueError(
                f"artists share catalog first continuation token: {row['record_id']}"
            )
        if generation_ids["emitted"] == generation_ids["reference"]:
            raise ValueError(
                f"artists share generation first continuation token: {row['record_id']}"
            )
        catalog_token_ids.append(catalog_ids)
        generation_token_ids.append(generation_ids)

    def batched_first_token_margins(
        prefixes: list[str], token_ids: list[dict[str, int]]
    ) -> list[float]:
        encoded = tokenizer(
            [prefix.rstrip() for prefix in prefixes],
            return_tensors="pt",
            padding=True,
        ).to(device)
        positions = final_positions(encoded["attention_mask"])
        with torch.no_grad():
            outputs = model(**encoded, use_cache=False)
        margins = []
        for index, ids in enumerate(token_ids):
            logits = outputs.logits[index, positions[index]].to(dtype=torch.float32)
            margins.append(
                float((logits[ids["emitted"]] - logits[ids["reference"]]).item())
            )
        del outputs
        return margins

    catalog_first_margins = batched_first_token_margins(
        [row["catalog_prefix"] for row in records], catalog_token_ids
    )
    generation_first_margins = batched_first_token_margins(
        [row["generation_prefix"] for row in records], generation_token_ids
    )

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
    with torch.no_grad():
        choice_outputs = model(**choice_inputs, use_cache=False)
    order_margins: list[list[float]] = [[] for _ in records]
    for spec_index, spec in enumerate(choice_specs):
        logits = choice_outputs.logits[
            spec_index, choice_positions[spec_index]
        ].to(dtype=torch.float32)
        role_to_letter = {role: letter for letter, role in spec["mapping"].items()}
        emitted_id = first_token_id(spec["prompt"], role_to_letter["emitted"])
        reference_id = first_token_id(spec["prompt"], role_to_letter["reference"])
        order_margins[spec["row_index"]].append(
            float((logits[emitted_id] - logits[reference_id]).item())
        )
    del choice_outputs
    choice_margins = [sum(values) / len(values) for values in order_margins]

    rows = []
    for index, record in enumerate(records):
        choice_warning = margin_prediction(choice_margins[index])
        catalog_sequence_warning = margin_prediction(
            catalog_sequence_margins[index]
        )
        combined_warning = combined_prediction(
            choice_margins[index], catalog_sequence_margins[index]
        )
        row = {
            **{key: value for key, value in record.items() if not key.endswith("_prefix")},
            "catalog_first_token_emitted_margin": catalog_first_margins[index],
            "generation_first_token_emitted_margin": generation_first_margins[index],
            "catalog_sequence_emitted_margin": catalog_sequence_margins[index],
            "generation_sequence_emitted_margin": generation_sequence_margins[index],
            "choice_order_emitted_margins": order_margins[index],
            "choice_emitted_margin": choice_margins[index],
            "choice_predicts_conflict": choice_warning,
            "catalog_sequence_predicts_conflict": catalog_sequence_warning,
            "combined_predicts_conflict": combined_warning,
            "choice_order_sign_consistent": (
                margin_prediction(order_margins[index][0])
                == margin_prediction(order_margins[index][1])
            ),
        }
        rows.append(row)
        print(
            "HOLDOUT_VERIFY_ROW_JSON="
            + json.dumps(row, ensure_ascii=True, separators=(",", ":"))
        )

    for row in rows:
        row["catalog_first_token_predicts_conflict"] = margin_prediction(
            row["catalog_first_token_emitted_margin"]
        )
        row["generation_first_token_predicts_conflict"] = margin_prediction(
            row["generation_first_token_emitted_margin"]
        )
        row["generation_sequence_predicts_conflict"] = margin_prediction(
            row["generation_sequence_emitted_margin"]
        )

    primary_metrics = binary_metrics(rows, "combined_predicts_conflict")
    choice_metrics = binary_metrics(rows, "choice_predicts_conflict")
    catalog_sequence_metrics = binary_metrics(
        rows, "catalog_sequence_predicts_conflict"
    )
    catalog_first_metrics = binary_metrics(
        rows, "catalog_first_token_predicts_conflict"
    )
    generation_first_metrics = binary_metrics(
        rows, "generation_first_token_predicts_conflict"
    )
    generation_sequence_metrics = binary_metrics(
        rows, "generation_sequence_predicts_conflict"
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

    labels = Counter(row["catalog_label"] for row in rows)
    expected = int(bundle["selection"]["selected_per_label"])
    all_values_finite = all(
        math.isfinite(number) for row in rows for number in finite_numbers(row)
    )
    both_orders_complete = all(
        len(row["choice_order_emitted_margins"]) == 2 for row in rows
    )
    relation_clusters = defaultdict(set)
    for row in rows:
        relation_clusters[row["catalog_label"]].add(row["relation_cluster_id"])
    technical_gate = (
        labels == Counter({"verified_exact": expected, "catalog_conflict": expected})
        and expected >= int(rule["minimum_events_per_label"])
        and both_orders_complete
        and all_values_finite
    )
    confirmed = confirmation_passes(primary_metrics, rule, technical_gate)
    summary = {
        "run_id": run_id,
        "model_id": bundle["model_id"],
        "row_count": len(rows),
        "label_counts": dict(labels),
        "unique_relation_counts": {
            label: len(values) for label, values in relation_clusters.items()
        },
        "primary_rule": rule["name"],
        "primary_metrics": primary_metrics,
        "choice_metrics": choice_metrics,
        "catalog_sequence_metrics": catalog_sequence_metrics,
        "catalog_first_token_metrics": catalog_first_metrics,
        "generation_first_token_metrics": generation_first_metrics,
        "generation_sequence_metrics": generation_sequence_metrics,
        "path_overlap": path_overlap(rows),
        "choice_order_sign_consistency_rate": (
            sum(row["choice_order_sign_consistent"] for row in rows) / len(rows)
        ),
        "confirmation_status": "confirmed" if confirmed else "not_confirmed",
        "selection": bundle["selection"],
        "both_choice_orders_complete": both_orders_complete,
        "all_values_finite": all_values_finite,
        "technical_gate": technical_gate,
        "interpretation_scope": "preregistered_independent_holdout_confirmation",
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    print(
        "HOLDOUT_VERIFY_SUMMARY_JSON="
        + json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

