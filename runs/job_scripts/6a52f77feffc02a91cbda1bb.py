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

