"""Reparse archived reason-order generations and compute corrected metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_reason_order_probe import matched_overlap, parse_playlist


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def analyze(
    raw_generations: list[dict[str, Any]],
    bundle: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    context_by_id = {row["context_id"]: row for row in bundle["contexts"]}
    generations = []
    rows = []
    for archived in raw_generations:
        context = context_by_id[archived["context_id"]]
        prompt = context["prompts"][archived["order"]].rstrip()
        parsed = parse_playlist(
            prompt + archived["raw_generation"],
            archived["order"],
        )
        generation = {
            **archived,
            "original_parser_count": archived["parsed_count"],
            "parsed_count": len(parsed),
            "valid_generation": len(parsed) >= 3,
            "rows": parsed,
        }
        generations.append(generation)
        for row in parsed:
            rows.append(
                {
                    "record_type": "generated_pair",
                    "record_id": f"{archived['generation_id']}__rank{row['rank']}",
                    "generation_id": archived["generation_id"],
                    "context_id": archived["context_id"],
                    "seed": archived["seed"],
                    "order": archived["order"],
                    **row,
                }
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
                    "seed": seed,
                    **matched_overlap(keyed["pair_first"], keyed["reason_first"]),
                }
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
    valid_generations = sum(row["valid_generation"] for row in generations)
    order_compliance = (
        sum(row["order_compliant"] for row in rows) / len(rows) if rows else 0.0
    )
    gates = bundle["registered_gates"]
    technical_gate = (
        len(generations) == int(gates["expected_generations"])
        and valid_generations >= int(gates["minimum_valid_generations"])
        and len(rows) >= int(gates["minimum_complete_rows"])
        and order_compliance >= float(gates["minimum_order_compliance"])
        and all(row["reason"] and row["title"] and row["artist"] for row in rows)
    )
    summary = {
        "source_job_id": "6a51ae56effc02a91cbd9235",
        "parser_revision": "inline_fields_and_title_by_artist_v2",
        "model_id": bundle["model_id"],
        "generation_count": len(generations),
        "valid_generation_count": valid_generations,
        "complete_row_count": len(rows),
        "row_counts_by_order": counts_by_order,
        "placeholder_artist_counts_by_order": placeholder_counts,
        "explicit_artist_field_rates_by_order": explicit_artist_rates,
        "order_compliance": order_compliance,
        "matched_overlaps": overlaps,
        "mean_exact_pair_jaccard": sum(
            row["exact_pair_jaccard"] for row in overlaps
        )
        / len(overlaps),
        "technical_gate": technical_gate,
        "catalog_verification_required": True,
    }
    return generations, rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "qwen_scope_music_reason_order_raw_generations.jsonl",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT / "runs" / "qwen_scope_music_reason_order_bundle.json",
    )
    parser.add_argument(
        "--generations-output",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "qwen_scope_music_reason_order_generations.jsonl",
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "qwen_scope_music_reason_order_rows.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "qwen_scope_music_reason_order_summary.json",
    )
    args = parser.parse_args()

    generations, rows, summary = analyze(load_jsonl(args.raw), load_json(args.bundle))
    write_jsonl(args.generations_output, generations)
    write_jsonl(args.rows_output, rows)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Reparsed {len(generations)} generations into {len(rows)} rows")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
