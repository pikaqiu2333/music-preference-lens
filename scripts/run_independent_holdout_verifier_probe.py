# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "accelerate==1.8.1",
#   "torch==2.7.1",
#   "transformers==4.53.2",
# ]
# ///
"""Run the pre-specified independent relation-conflict verifier holdout."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
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
    parser.add_argument("--model-revision")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + "_confirm"
    bundle = load_bundle(args.bundle)
    model_revision = (
        args.model_revision
        or os.environ.get("MODEL_REVISION")
        or bundle.get("model_revision")
    )
    model_source = {"revision": model_revision} if model_revision else {}
    records = bundle["records"]
    rule = bundle["frozen_rule"]
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
        "model_revision": model_revision,
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
        "interpretation_scope": "prespecified_independent_holdout_confirmation",
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
