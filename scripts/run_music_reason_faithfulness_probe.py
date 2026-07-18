# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.8.0",
#   "torch==2.7.1",
#   "transformers==4.53.2",
# ]
# ///
"""Score complete title-artist pairs under need counterfactuals."""

from __future__ import annotations

import argparse
import base64
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


EMBEDDED_BUNDLE_B64 = "__EXPERIMENT_BUNDLE_B64__"


def load_bundle(path: Path | None = None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    if EMBEDDED_BUNDLE_B64 == "__EXPERIMENT_BUNDLE_B64__":
        raise ValueError("no bundle supplied and embedded bundle was not injected")
    payload = base64.b64decode(EMBEDDED_BUNDLE_B64).decode("utf-8")
    return json.loads(payload)


def render_pair_text(prompt: str, title: str, artist: str) -> tuple[str, list[int], list[int]]:
    text = prompt.rstrip() + " "
    title_span = [len(text), len(text) + len(title)]
    text += title
    text += "\n   Artist: "
    artist_span = [len(text), len(text) + len(artist)]
    text += artist
    return text, title_span, artist_span


def overlapping_positions(offsets: list[tuple[int, int]], span: list[int]) -> list[int]:
    start, end = span
    return [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_start < end and token_end > start
    ]


def summarize_record(condition_scores: dict[str, float]) -> dict[str, Any]:
    original = condition_scores["original"]
    paraphrase = condition_scores["paraphrase"]
    opposite = condition_scores["opposite"]
    neutral = condition_scores["neutral"]
    lower_equivalent = min(original, paraphrase)
    return {
        "paraphrase_shift": abs(original - paraphrase),
        "opposite_margin": lower_equivalent - opposite,
        "neutral_margin": lower_equivalent - neutral,
        "opposite_below_both": opposite < lower_equivalent,
        "neutral_below_both": neutral < lower_equivalent,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "opposite_below_both_rate": 0.0,
            "neutral_below_both_rate": 0.0,
            "mean_opposite_margin": 0.0,
            "median_opposite_margin": 0.0,
            "mean_neutral_margin": 0.0,
            "mean_paraphrase_shift": 0.0,
        }
    return {
        "count": len(rows),
        "opposite_below_both_rate": mean(
            float(row["opposite_below_both"]) for row in rows
        ),
        "neutral_below_both_rate": mean(
            float(row["neutral_below_both"]) for row in rows
        ),
        "mean_opposite_margin": mean(row["opposite_margin"] for row in rows),
        "median_opposite_margin": median(row["opposite_margin"] for row in rows),
        "mean_neutral_margin": mean(row["neutral_margin"] for row in rows),
        "mean_paraphrase_shift": mean(row["paraphrase_shift"] for row in rows),
    }


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
    records = bundle["records"]
    if args.mode == "smoke":
        records = [row for row in records if row["smoke"]]

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

    contexts = {row["context_id"]: row for row in bundle["contexts"]}
    specs: list[dict[str, Any]] = []
    for record in records:
        context = contexts[record["context_id"]]
        for condition in bundle["conditions"]:
            text, title_span, artist_span = render_pair_text(
                context["condition_prompts"][condition],
                record["title"],
                record["artist"],
            )
            specs.append(
                {
                    "record_id": record["record_id"],
                    "condition": condition,
                    "text": text,
                    "title_span": title_span,
                    "artist_span": artist_span,
                }
            )

    max_tokens = 0
    for batch_start in range(0, len(specs), 8):
        batch = specs[batch_start : batch_start + 8]
        encoded = tokenizer(
            [row["text"] for row in batch],
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
        )
        offsets_batch = encoded.pop("offset_mapping").tolist()
        inputs = {key: value.to(device) for key, value in encoded.items()}
        max_tokens = max(max_tokens, int(inputs["attention_mask"].sum(dim=1).max().item()))
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)

        for batch_index, spec in enumerate(batch):
            offsets = [tuple(item) for item in offsets_batch[batch_index]]
            title_positions = overlapping_positions(offsets, spec["title_span"])
            artist_positions = overlapping_positions(offsets, spec["artist_span"])
            if not title_positions or not artist_positions:
                raise ValueError(f"empty target span for {spec['record_id']}")
            if title_positions[0] == 0 or artist_positions[0] == 0:
                raise ValueError("target continuation begins at token zero")

            def score_positions(positions: list[int]) -> dict[str, Any]:
                prediction_positions = [position - 1 for position in positions]
                target_ids = inputs["input_ids"][batch_index, positions]
                step_logits = outputs.logits[
                    batch_index, prediction_positions
                ].to(dtype=torch.float32)
                token_logits = step_logits[
                    torch.arange(len(positions), device=device), target_ids
                ]
                token_logps = token_logits - torch.logsumexp(step_logits, dim=-1)
                return {
                    "token_count": len(positions),
                    "sum_logp": float(token_logps.sum().item()),
                    "mean_logp": float(token_logps.mean().item()),
                }

            title_score = score_positions(title_positions)
            artist_score = score_positions(artist_positions)
            pair_count = title_score["token_count"] + artist_score["token_count"]
            pair_sum = title_score["sum_logp"] + artist_score["sum_logp"]
            spec["title_score"] = title_score
            spec["artist_score"] = artist_score
            spec["pair_score"] = {
                "token_count": pair_count,
                "sum_logp": pair_sum,
                "mean_logp": pair_sum / pair_count,
            }
        del outputs

    specs_by_record: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for spec in specs:
        specs_by_record[spec["record_id"]][spec["condition"]] = spec

    result_rows: list[dict[str, Any]] = []
    technical_complete = True
    for record in records:
        condition_specs = specs_by_record[record["record_id"]]
        condition_scores = {
            condition: condition_specs[condition]["pair_score"]["mean_logp"]
            for condition in bundle["conditions"]
        }
        metrics = summarize_record(condition_scores)
        row = {
            **record,
            "condition_pair_mean_logps": condition_scores,
            "condition_title_mean_logps": {
                condition: condition_specs[condition]["title_score"]["mean_logp"]
                for condition in bundle["conditions"]
            },
            "condition_artist_mean_logps": {
                condition: condition_specs[condition]["artist_score"]["mean_logp"]
                for condition in bundle["conditions"]
            },
            "pair_token_count": condition_specs["original"]["pair_score"]["token_count"],
            **metrics,
        }
        all_values = [
            *row["condition_pair_mean_logps"].values(),
            *row["condition_title_mean_logps"].values(),
            *row["condition_artist_mean_logps"].values(),
        ]
        technical_complete = technical_complete and all(math.isfinite(value) for value in all_values)
        result_rows.append(row)
        print("REASON_FAITH_ROW_JSON=" + json.dumps(row, ensure_ascii=False, separators=(",", ":")))

    overall = aggregate_rows(result_rows)
    by_catalog_label = {
        label: aggregate_rows([row for row in result_rows if row["catalog_label"] == label])
        for label in sorted({row["catalog_label"] for row in result_rows})
    }
    gates = bundle["registered_gates"]
    label_counts = Counter(row["catalog_label"] for row in result_rows)
    expected_count = gates["smoke_record_count"] if args.mode == "smoke" else len(records)
    technical_gate = (
        technical_complete
        and len(result_rows) == expected_count
        and all(len(specs_by_record[row["record_id"]]) == len(bundle["conditions"]) for row in records)
    )
    if args.mode == "smoke":
        technical_gate = technical_gate and dict(label_counts) == gates["smoke_label_counts"]
    followup_gate = (
        technical_gate
        and overall["opposite_below_both_rate"]
        >= gates["minimum_opposite_below_both_rate"]
        and overall["median_opposite_margin"] > 0
    )
    finished = datetime.now(timezone.utc)
    summary = {
        "run_id": run_id,
        "mode": args.mode,
        "model_id": bundle["model_id"],
        "record_count": len(result_rows),
        "catalog_label_counts": dict(label_counts),
        "conditions": bundle["conditions"],
        "overall": overall,
        "by_catalog_label": by_catalog_label,
        "max_input_tokens": max_tokens,
        "technical_gate": technical_gate,
        "mechanistic_followup_gate": followup_gate,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
    }
    print("REASON_FAITH_SUMMARY_JSON=" + json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
