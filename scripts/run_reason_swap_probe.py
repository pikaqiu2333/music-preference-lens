# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.8.0",
#   "torch==2.7.1",
#   "transformers==4.53.2",
# ]
# ///
"""Score fixed title-artist pairs after visible-reason counterfactuals."""

from __future__ import annotations

import argparse
import base64
import json
import math
import statistics
from collections import Counter, defaultdict
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


def render_pair_text(
    prefix: str,
    title: str,
    artist: str,
) -> tuple[str, list[int], list[int]]:
    text = prefix.rstrip() + " "
    title_span = [len(text), len(text) + len(title)]
    text += title
    text += "\n   Artist: "
    artist_span = [len(text), len(text) + len(artist)]
    text += artist
    return text, title_span, artist_span


def render_reason_prefix(context: dict[str, Any], reason: str) -> str:
    return (
        "Music playlist request.\n\n"
        f"User profile: {context['profile']}\n\n"
        f"Current need: {context['current_need']}\n\n"
        "Recommend real, existing tracks. For every item, write one short "
        "reason first, followed by its title and artist.\n\n"
        "Playlist:\n"
        f"1. Reason: {reason}\n"
        "   Title:"
    )


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


def summarize_scores(scores: dict[str, float]) -> dict[str, Any]:
    own = float(scores["own"])
    output: dict[str, Any] = {}
    for condition in ("same_context_swap", "opposite_context", "neutral"):
        margin = own - float(scores[condition])
        output[f"own_vs_{condition}_margin"] = margin
        output[f"own_beats_{condition}"] = margin > 0
    return output


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    result: dict[str, Any] = {"count": len(rows)}
    for condition in ("same_context_swap", "opposite_context", "neutral"):
        margin_key = f"own_vs_{condition}_margin"
        win_key = f"own_beats_{condition}"
        margins = [float(row[margin_key]) for row in rows]
        result[f"mean_{margin_key}"] = sum(margins) / len(margins)
        result[f"median_{margin_key}"] = statistics.median(margins)
        result[f"{win_key}_rate"] = sum(bool(row[win_key]) for row in rows) / len(rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + "_pilot"
    bundle = load_bundle(args.bundle)
    conditions = list(bundle["conditions"])
    records = bundle["records"]
    reason_by_record_id = {record["record_id"]: record["reason"] for record in records}

    def condition_reason(record: dict[str, Any], condition: str) -> str:
        if condition == "neutral":
            return str(bundle["neutral_reason"])
        source_id = record["condition_reason_record_ids"][condition]
        return reason_by_record_id[source_id]
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

    specs = []
    for record in records:
        for condition in conditions:
            reason = condition_reason(record, condition)
            text, title_span, artist_span = render_pair_text(
                render_reason_prefix(
                    bundle["contexts"][record["context_id"]],
                    reason,
                ),
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

    scores: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    max_input_tokens = 0
    technical_complete = True
    for batch_start in range(0, len(specs), 8):
        batch = specs[batch_start : batch_start + 8]
        inputs = tokenizer(
            [row["text"] for row in batch],
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
        )
        offsets_batch = inputs.pop("offset_mapping").tolist()
        max_input_tokens = max(max_input_tokens, int(inputs["input_ids"].shape[1]))
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
        for batch_index, spec in enumerate(batch):
            offsets = [tuple(item) for item in offsets_batch[batch_index]]
            title_positions = overlapping_positions(offsets, spec["title_span"])
            artist_positions = overlapping_positions(offsets, spec["artist_span"])
            pair_positions = title_positions + artist_positions
            if min(pair_positions) < 1:
                raise ValueError("target token has no causal prediction position")

            def score_positions(positions: list[int]) -> dict[str, Any]:
                prediction_positions = [position - 1 for position in positions]
                target_ids = inputs["input_ids"][batch_index, positions]
                step_logits = outputs.logits[
                    batch_index,
                    prediction_positions,
                ].to(dtype=torch.float32)
                token_logits = step_logits[
                    torch.arange(len(positions), device=device),
                    target_ids,
                ]
                token_logps = token_logits - torch.logsumexp(step_logits, dim=-1)
                return {
                    "token_count": len(positions),
                    "mean_logp": float(token_logps.mean().item()),
                    "sum_logp": float(token_logps.sum().item()),
                }

            title_score = score_positions(title_positions)
            artist_score = score_positions(artist_positions)
            pair_score = score_positions(pair_positions)
            values = (
                title_score["mean_logp"],
                artist_score["mean_logp"],
                pair_score["mean_logp"],
            )
            technical_complete = technical_complete and all(
                math.isfinite(float(value)) for value in values
            )
            scores[spec["record_id"]][spec["condition"]] = {
                "title_mean_logp": title_score["mean_logp"],
                "artist_mean_logp": artist_score["mean_logp"],
                "pair_mean_logp": pair_score["mean_logp"],
                "title_token_count": title_score["token_count"],
                "artist_token_count": artist_score["token_count"],
                "pair_token_count": pair_score["token_count"],
            }
        del outputs

    rows = []
    for record in records:
        condition_scores = scores[record["record_id"]]
        pair_scores = {
            condition: float(condition_scores[condition]["pair_mean_logp"])
            for condition in conditions
        }
        row = {
            **record,
            "condition_reasons": {
                condition: condition_reason(record, condition)
                for condition in conditions
            },
            "condition_scores": condition_scores,
            "condition_pair_mean_logps": pair_scores,
            **summarize_scores(pair_scores),
        }
        rows.append(row)
        print(
            "REASON_SWAP_ROW_JSON="
            + json.dumps(row, ensure_ascii=True, separators=(",", ":"))
        )

    by_group = {
        group: aggregate_rows([row for row in rows if row["entity_group"] == group])
        for group in sorted({row["entity_group"] for row in rows})
    }
    by_context = {
        context: aggregate_rows([row for row in rows if row["context_id"] == context])
        for context in sorted({row["context_id"] for row in rows})
    }
    gates = bundle["registered_gates"]
    actual_group_counts = Counter(row["entity_group"] for row in rows)
    condition_count = sum(len(scores[row["record_id"]]) for row in rows)
    technical_gate = (
        technical_complete
        and len(rows) == int(gates["record_count"])
        and condition_count == int(gates["condition_count"])
        and all(set(scores[row["record_id"]]) == set(conditions) for row in rows)
        and all(
            row["condition_reasons"]["same_context_swap"]
            != row["condition_reasons"]["own"]
            for row in rows
        )
        and dict(actual_group_counts) == gates["entity_group_counts"]
    )
    summary = {
        "run_id": run_id,
        "model_id": bundle["model_id"],
        "record_count": len(rows),
        "condition_count": condition_count,
        "entity_group_counts": dict(actual_group_counts),
        "overall": aggregate_rows(rows),
        "by_entity_group": by_group,
        "by_context": by_context,
        "max_input_tokens": max_input_tokens,
        "technical_gate": technical_gate,
        "interpretation_scope": "exploratory_visible_reason_causality",
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    print(
        "REASON_SWAP_SUMMARY_JSON="
        + json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
