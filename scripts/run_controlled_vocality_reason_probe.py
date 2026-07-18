# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.8.0",
#   "torch==2.7.1",
#   "transformers==4.53.2",
# ]
# ///
"""Run controlled vocal-versus-instrumental reason interventions."""

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
CONDITIONS = ("vocal", "instrumental", "neutral")


def load_bundle(path: Path | None = None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    if EMBEDDED_BUNDLE_B64 == "__EXPERIMENT_BUNDLE_B64__":
        raise ValueError("pass --bundle or embed the experiment bundle")
    return json.loads(base64.b64decode(EMBEDDED_BUNDLE_B64).decode("utf-8"))


def render_pair_text(
    reason: str,
    title: str,
    artist: str,
) -> tuple[str, list[int], list[int]]:
    text = f"Music recommendation plan.\nReason: {reason}\nTitle: "
    title_span = [len(text), len(text) + len(title)]
    text += title
    text += "\nArtist: "
    artist_span = [len(text), len(text) + len(artist)]
    text += artist
    return text, title_span, artist_span


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
    letter_to_key = {}
    for index, track_key in enumerate(order):
        letter = chr(ord("A") + index)
        row = records_by_key[track_key]
        lines.append(f"{letter}. {row['title']} - {row['artist']}")
        letter_to_key[letter] = track_key
    lines.extend(["Answer with one letter only.", "Answer:"])
    return "\n".join(lines), letter_to_key


def aggregate_pair_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [float(row["matched_margin"]) for row in rows]
    return {
        "count": len(rows),
        "mean_matched_margin": sum(margins) / len(margins),
        "median_matched_margin": statistics.median(margins),
        "matched_direction_accuracy": sum(margin > 0 for margin in margins)
        / len(margins),
    }


def summarize_choice_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_reason: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_reason[row["reason_condition"]].append(float(row["vocal_minus_instrumental_margin"]))
    means = {
        condition: sum(by_reason[condition]) / len(by_reason[condition])
        for condition in CONDITIONS
    }
    return {
        "mean_vocal_minus_instrumental_by_reason": means,
        "vocal_reason_shift_from_neutral": means["vocal"] - means["neutral"],
        "instrumental_reason_shift_from_neutral": (
            means["neutral"] - means["instrumental"]
        ),
        "vocal_shift_positive_in_all_orders": all(
            row["vocal_minus_instrumental_margin"]
            > next(
                neutral["vocal_minus_instrumental_margin"]
                for neutral in rows
                if neutral["order_index"] == row["order_index"]
                and neutral["reason_condition"] == "neutral"
            )
            for row in rows
            if row["reason_condition"] == "vocal"
        ),
        "instrumental_shift_positive_in_all_orders": all(
            row["vocal_minus_instrumental_margin"]
            < next(
                neutral["vocal_minus_instrumental_margin"]
                for neutral in rows
                if neutral["order_index"] == row["order_index"]
                and neutral["reason_condition"] == "neutral"
            )
            for row in rows
            if row["reason_condition"] == "instrumental"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + "_pilot"
    bundle = load_bundle(args.bundle)
    records = bundle["records"]
    records_by_key = {row["track_key"]: row for row in records}
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
        for condition in CONDITIONS:
            text, title_span, artist_span = render_pair_text(
                bundle["reasons"][condition],
                record["title"],
                record["artist"],
            )
            specs.append(
                {
                    "track_key": record["track_key"],
                    "condition": condition,
                    "text": text,
                    "title_span": title_span,
                    "artist_span": artist_span,
                }
            )

    pair_scores: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    max_input_tokens = 0
    all_finite = True
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
            positions = title_positions + artist_positions
            prediction_positions = [position - 1 for position in positions]
            if min(prediction_positions) < 0:
                raise ValueError("target token has no causal prediction position")
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
            mean_logp = float(token_logps.mean().item())
            all_finite = all_finite and math.isfinite(mean_logp)
            pair_scores[spec["track_key"]][spec["condition"]] = {
                "pair_mean_logp": mean_logp,
                "pair_token_count": len(positions),
                "title_token_count": len(title_positions),
                "artist_token_count": len(artist_positions),
            }
        del outputs

    pair_rows = []
    for record in records:
        scores = pair_scores[record["track_key"]]
        matched_condition = record["vocality"]
        flipped_condition = (
            "instrumental" if matched_condition == "vocal" else "vocal"
        )
        matched = float(scores[matched_condition]["pair_mean_logp"])
        flipped = float(scores[flipped_condition]["pair_mean_logp"])
        neutral = float(scores["neutral"]["pair_mean_logp"])
        row = {
            **record,
            "condition_scores": scores,
            "matched_condition": matched_condition,
            "flipped_condition": flipped_condition,
            "matched_margin": matched - flipped,
            "matched_vs_neutral_margin": matched - neutral,
        }
        pair_rows.append(row)
        print(
            "VOCALITY_PAIR_ROW_JSON="
            + json.dumps(row, ensure_ascii=True, separators=(",", ":"))
        )

    choice_specs = []
    for order_index, order in enumerate(bundle["candidate_orders"]):
        for condition in CONDITIONS:
            prompt, mapping = render_choice_prompt(
                bundle["reasons"][condition],
                order,
                records_by_key,
            )
            choice_specs.append(
                {
                    "order_index": order_index,
                    "reason_condition": condition,
                    "prompt": prompt,
                    "mapping": mapping,
                }
            )

    choice_inputs = tokenizer(
        [row["prompt"] for row in choice_specs],
        return_tensors="pt",
        padding=True,
    ).to(device)
    max_input_tokens = max(max_input_tokens, int(choice_inputs["input_ids"].shape[1]))
    with torch.no_grad():
        choice_outputs = model(**choice_inputs, use_cache=False)
    lengths = choice_inputs["attention_mask"].sum(dim=1)

    def continuation_token_id(prompt: str, letter: str) -> int:
        text = prompt + " " + letter
        encoded = tokenizer(text, return_offsets_mapping=True)
        offsets = [tuple(item) for item in encoded["offset_mapping"]]
        positions = overlapping_positions(offsets, [len(prompt), len(text)])
        if len(positions) != 1:
            raise ValueError(f"choice {letter} is not one continuation token")
        return int(encoded["input_ids"][positions[0]])

    choice_rows = []
    for index, spec in enumerate(choice_specs):
        final_logits = choice_outputs.logits[
            index,
            int(lengths[index].item()) - 1,
        ].to(dtype=torch.float32)
        letter_logits = {
            letter: float(final_logits[continuation_token_id(spec["prompt"], letter)].item())
            for letter in spec["mapping"]
        }
        vocal_logits = torch.tensor(
            [
                letter_logits[letter]
                for letter, key in spec["mapping"].items()
                if records_by_key[key]["vocality"] == "vocal"
            ],
            device=device,
        )
        instrumental_logits = torch.tensor(
            [
                letter_logits[letter]
                for letter, key in spec["mapping"].items()
                if records_by_key[key]["vocality"] == "instrumental"
            ],
            device=device,
        )
        margin = float(
            (torch.logsumexp(vocal_logits, dim=0) - torch.logsumexp(instrumental_logits, dim=0)).item()
        )
        all_finite = all_finite and math.isfinite(margin)
        top_letter = max(letter_logits, key=letter_logits.get)
        row = {
            **spec,
            "letter_logits": letter_logits,
            "top_letter": top_letter,
            "top_track_key": spec["mapping"][top_letter],
            "top_vocality": records_by_key[spec["mapping"][top_letter]]["vocality"],
            "vocal_minus_instrumental_margin": margin,
        }
        choice_rows.append(row)
        print(
            "VOCALITY_CHOICE_ROW_JSON="
            + json.dumps(row, ensure_ascii=True, separators=(",", ":"))
        )

    overall_pair = aggregate_pair_rows(pair_rows)
    pair_by_vocality = {
        label: aggregate_pair_rows(
            [row for row in pair_rows if row["vocality"] == label]
        )
        for label in ("instrumental", "vocal")
    }
    choice_summary = summarize_choice_rows(choice_rows)
    gates = bundle["registered_gates"]
    condition_count = sum(len(pair_scores[row["track_key"]]) for row in records)
    technical_gate = (
        all_finite
        and len(records) == int(gates["record_count"])
        and condition_count == int(gates["pair_condition_count"])
        and Counter(row["vocality"] for row in records)
        == Counter(gates["vocality_counts"])
        and all(set(pair_scores[row["track_key"]]) == set(CONDITIONS) for row in records)
        and len(choice_rows) == len(bundle["candidate_orders"]) * len(CONDITIONS)
    )
    behavioral_gate = (
        technical_gate
        and overall_pair["matched_direction_accuracy"]
        >= float(gates["minimum_pair_direction_accuracy"])
        and all(
            pair_by_vocality[label]["median_matched_margin"] > 0
            for label in ("instrumental", "vocal")
        )
        and choice_summary["vocal_reason_shift_from_neutral"] > 0
        and choice_summary["instrumental_reason_shift_from_neutral"] > 0
    )
    summary = {
        "run_id": run_id,
        "model_id": bundle["model_id"],
        "record_count": len(records),
        "pair_condition_count": condition_count,
        "pair_overall": overall_pair,
        "pair_by_vocality": pair_by_vocality,
        "choice_summary": choice_summary,
        "max_input_tokens": max_input_tokens,
        "technical_gate": technical_gate,
        "behavioral_followup_gate": behavioral_gate,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    print(
        "VOCALITY_SUMMARY_JSON="
        + json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
