"""Export a balanced real-track vocality-reason causal bundle."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REASONS = {
    "vocal": (
        "The track features prominent vocals and clearly audible sung lyrics throughout."
    ),
    "instrumental": (
        "The track features no vocals and only instrumental musical passages throughout."
    ),
    "neutral": (
        "The track has musical qualities that may suit this listening request."
    ),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def word_count(value: str) -> int:
    return len(value.split())


def build_candidate_orders(records: list[dict[str, Any]]) -> list[list[str]]:
    by_label = {
        label: sorted(
            (row for row in records if row["vocality"] == label),
            key=lambda row: row["track_key"],
        )
        for label in ("instrumental", "vocal")
    }
    first = []
    for instrumental, vocal in zip(by_label["instrumental"], by_label["vocal"]):
        first.extend([instrumental["track_key"], vocal["track_key"]])
    return [first, list(reversed(first))]


def build_bundle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 8:
        raise ValueError(f"expected eight controlled tracks, found {len(rows)}")
    counts = Counter(row["vocality"] for row in rows)
    if counts != Counter({"instrumental": 4, "vocal": 4}):
        raise ValueError(f"unexpected vocality counts: {dict(counts)}")
    if any(row.get("catalog_label") != "verified_exact" for row in rows):
        raise ValueError("all controlled tracks must be catalog verified exact")
    if len({row["artist"].casefold() for row in rows}) != len(rows):
        raise ValueError("controlled tracks must use distinct artists")
    if {word_count(reason) for reason in REASONS.values()} != {11}:
        raise ValueError("all controlled reasons must contain exactly 11 words")
    for row in rows:
        entity_text = f"{row['title']} {row['artist']}".casefold()
        if any(row["title"].casefold() in reason.casefold() or row["artist"].casefold() in reason.casefold() for reason in REASONS.values()):
            raise ValueError(f"reason leaks entity text for {entity_text}")

    records = [
        {
            "track_key": row["track_key"],
            "title": row["title"],
            "artist": row["artist"],
            "vocality": row["vocality"],
            "catalog_label": row["catalog_label"],
            "attribute_evidence_url": row["attribute_evidence_url"],
            "attribute_evidence_note": row["attribute_evidence_note"],
        }
        for row in sorted(rows, key=lambda item: item["track_key"])
    ]
    return {
        "bundle_version": "controlled_vocality_reason_v1",
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "reasons": REASONS,
        "records": records,
        "candidate_orders": build_candidate_orders(records),
        "registered_gates": {
            "record_count": 8,
            "pair_condition_count": 24,
            "vocality_counts": dict(counts),
            "minimum_pair_direction_accuracy": 0.75,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verified-tracks",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "controlled_vocality_tracks_catalog_verified.jsonl",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "controlled_vocality_reason_probe_bundle.json",
    )
    args = parser.parse_args()

    bundle = build_bundle(load_jsonl(args.verified_tracks))
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    args.bundle.write_text(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote controlled vocality bundle with {len(bundle['records'])} tracks")
    print(f"- {args.bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
