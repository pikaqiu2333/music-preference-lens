"""Parse archived holdout completions without changing the frozen parser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_song_entity_generation_time_probe import parse_playlist


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def analyze(
    bundle: dict[str, Any],
    raw_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contexts = {row["context_id"]: row for row in bundle["contexts"]}
    expected_ids = {
        f"{context_id}__seed{seed}"
        for context_id in contexts
        for seed in bundle["seeds"]
    }
    actual_ids = {row["generation_id"] for row in raw_rows}
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(f"holdout generation IDs differ: missing={missing}, extra={extra}")

    parsed_rows = []
    parsed_generation_count = 0
    generation_counts = {}
    for raw in sorted(raw_rows, key=lambda row: row["generation_id"]):
        context = contexts[raw["context_id"]]
        full_text = context["generation_prompt"].rstrip() + raw["completion"]
        parsed = parse_playlist(full_text)
        generation_counts[raw["generation_id"]] = len(parsed)
        if parsed:
            parsed_generation_count += 1
        for rank, item in enumerate(parsed, 1):
            parsed_rows.append(
                {
                    "record_type": "generated_pair",
                    "record_id": f"{raw['generation_id']}__rank{rank}",
                    "generation_id": raw["generation_id"],
                    "context_id": raw["context_id"],
                    "seed": int(raw["seed"]),
                    "rank": rank,
                    "title": item["title"],
                    "artist": item["artist"],
                    "reason": item["reason"],
                }
            )
    summary = {
        "raw_generation_count": len(raw_rows),
        "parsed_generation_count": parsed_generation_count,
        "parsed_pair_count": len(parsed_rows),
        "generation_pair_counts": generation_counts,
        "minimum_parsed_generation_count": bundle["minimum_parsed_generation_count"],
        "minimum_parsed_pair_count": bundle["minimum_parsed_pair_count"],
        "technical_gate": (
            parsed_generation_count >= int(bundle["minimum_parsed_generation_count"])
            and len(parsed_rows) >= int(bundle["minimum_parsed_pair_count"])
        ),
    }
    return parsed_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT / "runs" / "independent_holdout_generation_bundle.json",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=PROJECT_ROOT / "runs" / "independent_holdout_raw_generations.jsonl",
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=PROJECT_ROOT / "runs" / "independent_holdout_generated_pairs.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "runs" / "independent_holdout_parse_summary.json",
    )
    args = parser.parse_args()

    rows, summary = analyze(load_json(args.bundle), load_jsonl(args.raw))
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    args.rows.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Parsed {summary['parsed_generation_count']}/{summary['raw_generation_count']} "
        f"generations and {summary['parsed_pair_count']} pairs"
    )
    print(f"Technical gate: {summary['technical_gate']}")
    print(f"- {args.rows}")
    print(f"- {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
