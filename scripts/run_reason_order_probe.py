# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.8.0",
#   "torch==2.7.1",
#   "transformers==4.53.2",
# ]
# ///
"""Generate matched playlists with reason-first and pair-first field order."""

from __future__ import annotations

import argparse
import base64
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EMBEDDED_BUNDLE_B64 = "__EXPERIMENT_BUNDLE_B64__"
FIELD_PATTERN = re.compile(
    r"^\s*(?:(?P<rank>\d+)[.)]\s*)?(?:\*\*)?"
    r"(?P<field>reason|title|artist)(?:\*\*)?\s*:\s*(?P<value>.*?)\s*$",
    re.IGNORECASE,
)
LABEL_PATTERN = re.compile(
    r"(?:(?P<rank>\d+)[.)]\s*)?(?:\*\*)?"
    r"(?P<field>reason|title|artist)(?:\*\*)?\s*:\s*",
    re.IGNORECASE,
)
TITLE_BY_PATTERN = re.compile(r"^(?P<title>.+?)\s+by\s+(?P<artist>.+)$", re.IGNORECASE)
EXPECTED_FIELDS = {
    "pair_first": ["title", "artist", "reason"],
    "reason_first": ["reason", "title", "artist"],
}


def load_bundle(path: Path | None = None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    if EMBEDDED_BUNDLE_B64 == "__EXPERIMENT_BUNDLE_B64__":
        raise ValueError("pass --bundle or embed the experiment bundle")
    return json.loads(base64.b64decode(EMBEDDED_BUNDLE_B64).decode("utf-8"))


def clean_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value


def parse_playlist(full_text: str, order: str) -> list[dict[str, Any]]:
    marker = full_text.rfind("Playlist:")
    if marker < 0:
        return []
    text = full_text[marker + len("Playlist:") :]
    blocks: dict[int, dict[str, Any]] = {}
    current_rank: int | None = None
    pending: tuple[int, str] | None = None
    offset = 0
    def assign(
        rank: int,
        field: str,
        value: str,
        position: int,
        *,
        explicit: bool,
    ) -> None:
        block = blocks.setdefault(rank, {"rank": rank, "explicit_fields": []})
        block[field] = clean_value(value)
        block[f"{field}_position"] = position
        if explicit and field not in block["explicit_fields"]:
            block["explicit_fields"].append(field)

    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        matches = list(LABEL_PATTERN.finditer(line))
        if pending:
            prefix_end = matches[0].start() if matches else len(line)
            prefix = line[:prefix_end]
            if prefix.strip():
                rank, field = pending
                assign(
                    rank,
                    field,
                    prefix,
                    offset + len(prefix) - len(prefix.lstrip()),
                    explicit=True,
                )
                pending = None
        if not matches:
            offset += len(raw_line)
            continue
        for index, match in enumerate(matches):
            if match.group("rank"):
                current_rank = int(match.group("rank"))
            if current_rank is None or current_rank > 5:
                continue
            field = match.group("field").casefold()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
            raw_value = line[match.end() : end]
            value = clean_value(raw_value)
            pending = None
            if not value:
                pending = (current_rank, field)
                continue
            value_position = offset + match.end() + len(raw_value) - len(raw_value.lstrip())
            if field == "title":
                title_by = TITLE_BY_PATTERN.match(value)
                if title_by and not any(
                    later.group("field").casefold() == "artist"
                    for later in matches[index + 1 :]
                ):
                    assign(
                        current_rank,
                        "title",
                        title_by.group("title"),
                        value_position,
                        explicit=True,
                    )
                    artist_start = value.casefold().rfind(" by ") + 4
                    assign(
                        current_rank,
                        "artist",
                        title_by.group("artist"),
                        value_position + artist_start,
                        explicit=False,
                    )
                    continue
            assign(
                current_rank,
                field,
                value,
                value_position,
                explicit=True,
            )
        offset += len(raw_line)

    expected = EXPECTED_FIELDS[order]
    rows = []
    for rank in sorted(blocks):
        block = blocks[rank]
        if not all(block.get(field) for field in ("reason", "title", "artist")):
            continue
        artist_suffix = " - " + block["artist"]
        if block["title"].casefold().endswith(artist_suffix.casefold()):
            block["title"] = clean_value(block["title"][: -len(artist_suffix)])
        actual = sorted(
            (field for field in expected),
            key=lambda field: block[f"{field}_position"],
        )
        rows.append(
            {
                "rank": rank,
                "reason": block["reason"],
                "title": block["title"],
                "artist": block["artist"],
                "actual_field_order": actual,
                "order_compliant": actual == expected,
                "explicit_artist_field": "artist" in block["explicit_fields"],
                "placeholder_artist": bool(
                    re.fullmatch(r"\[?artist name\]?", block["artist"], re.IGNORECASE)
                ),
            }
        )
    return rows[:5]


def normalize_pair(title: str, artist: str) -> str:
    value = unicodedata.normalize("NFKC", f"{title}\n{artist}").casefold()
    return "".join(character for character in value if character.isalnum())


def matched_overlap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> dict[str, Any]:
    left_pairs = {normalize_pair(row["title"], row["artist"]) for row in left}
    right_pairs = {normalize_pair(row["title"], row["artist"]) for row in right}
    union = left_pairs | right_pairs
    intersection = left_pairs & right_pairs
    return {
        "pair_first_count": len(left_pairs),
        "reason_first_count": len(right_pairs),
        "exact_pair_overlap_count": len(intersection),
        "exact_pair_jaccard": len(intersection) / len(union) if union else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + "_smoke"
    bundle = load_bundle(args.bundle)
    if not torch.cuda.is_available():
        raise RuntimeError("this probe requires a CUDA GPU")
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(bundle["model_id"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        bundle["model_id"],
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    generations = []
    rows = []
    for context in bundle["contexts"]:
        for seed in bundle["seeds"]:
            for order in bundle["orders"]:
                prompt = context["prompts"][order].rstrip()
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                torch.manual_seed(int(seed))
                torch.cuda.manual_seed_all(int(seed))
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=float(bundle["generation"]["temperature"]),
                        top_p=float(bundle["generation"]["top_p"]),
                        max_new_tokens=int(bundle["generation"]["max_new_tokens"]),
                        pad_token_id=tokenizer.pad_token_id,
                    )
                completion_ids = output_ids[0, inputs["input_ids"].shape[1] :]
                completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
                parsed = parse_playlist(prompt + completion, order)
                generation_id = f"{context['context_id']}__seed{seed}__{order}"
                generation_row = {
                    "generation_id": generation_id,
                    "context_id": context["context_id"],
                    "seed": int(seed),
                    "order": order,
                    "raw_generation": completion,
                    "parsed_count": len(parsed),
                    "valid_generation": len(parsed) >= 3,
                    "rows": parsed,
                }
                generations.append(generation_row)
                for row in parsed:
                    rows.append(
                        {
                            "record_type": "generated_pair",
                            "record_id": f"{generation_id}__rank{row['rank']}",
                            "generation_id": generation_id,
                            "context_id": context["context_id"],
                            "seed": int(seed),
                            "order": order,
                            **row,
                        }
                    )
                print(
                    "REASON_ORDER_GENERATION_JSON="
                    + json.dumps(generation_row, ensure_ascii=True, separators=(",", ":"))
                )

    overlaps = []
    for context in bundle["contexts"]:
        for seed in bundle["seeds"]:
            keyed = {
                row["order"]: row["rows"]
                for row in generations
                if row["context_id"] == context["context_id"]
                and row["seed"] == seed
            }
            overlaps.append(
                {
                    "context_id": context["context_id"],
                    "seed": int(seed),
                    **matched_overlap(keyed["pair_first"], keyed["reason_first"]),
                }
            )

    gates = bundle["registered_gates"]
    valid_generations = sum(row["valid_generation"] for row in generations)
    compliance = (
        sum(row["order_compliant"] for row in rows) / len(rows) if rows else 0.0
    )
    counts_by_order = {
        order: sum(row["order"] == order for row in rows)
        for order in bundle["orders"]
    }
    placeholder_counts = {
        order: sum(
            row["order"] == order and row["placeholder_artist"] for row in rows
        )
        for order in bundle["orders"]
    }
    explicit_artist_rates = {
        order: (
            sum(
                row["order"] == order and row["explicit_artist_field"]
                for row in rows
            )
            / counts_by_order[order]
            if counts_by_order[order]
            else 0.0
        )
        for order in bundle["orders"]
    }
    technical_gate = (
        len(generations) == int(gates["expected_generations"])
        and valid_generations >= int(gates["minimum_valid_generations"])
        and len(rows) >= int(gates["minimum_complete_rows"])
        and compliance >= float(gates["minimum_order_compliance"])
        and all(row["reason"] and row["title"] and row["artist"] for row in rows)
    )
    summary = {
        "run_id": run_id,
        "model_id": bundle["model_id"],
        "generation_count": len(generations),
        "valid_generation_count": valid_generations,
        "complete_row_count": len(rows),
        "row_counts_by_order": counts_by_order,
        "placeholder_artist_counts_by_order": placeholder_counts,
        "explicit_artist_field_rates_by_order": explicit_artist_rates,
        "order_compliance": compliance,
        "matched_overlaps": overlaps,
        "mean_exact_pair_jaccard": sum(row["exact_pair_jaccard"] for row in overlaps)
        / len(overlaps),
        "technical_gate": technical_gate,
        "catalog_verification_required": True,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    print(
        "REASON_ORDER_SUMMARY_JSON="
        + json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
