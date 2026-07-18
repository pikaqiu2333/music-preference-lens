"""Export visible-reason counterfactuals for fixed title-artist pairs."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("own", "same_context_swap", "opposite_context", "neutral")
NEUTRAL_REASON = "This track is a reasonable option for the current request."


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in value if character.isalnum())


def entity_group(row: dict[str, Any]) -> str:
    if row.get("placeholder_artist"):
        return "invalid_placeholder"
    return str(row["catalog_label"])


def choose_different_reason(
    row: dict[str, Any],
    context_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    start = next(index for index, item in enumerate(context_rows) if item["record_id"] == row["record_id"])
    for offset in range(1, len(context_rows)):
        candidate = context_rows[(start + offset) % len(context_rows)]["reason"]
        if normalize(candidate) != normalize(row["reason"]):
            return context_rows[(start + offset) % len(context_rows)]
    raise ValueError(f"no different same-context reason for {row['record_id']}")


def build_bundle(
    catalog_rows: list[dict[str, Any]],
    order_bundle: dict[str, Any],
) -> dict[str, Any]:
    rows = [row for row in catalog_rows if row.get("order") == "reason_first"]
    rows.sort(key=lambda row: (row["context_id"], int(row["seed"]), int(row["rank"])))
    if len(rows) != 20:
        raise ValueError(f"expected 20 reason-first rows, found {len(rows)}")
    contexts = {row["context_id"]: row for row in order_bundle["contexts"]}
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_slot = {}
    for row in rows:
        by_context[row["context_id"]].append(row)
        slot = (int(row["seed"]), int(row["rank"]))
        by_slot[(row["context_id"], *slot)] = row

    context_ids = sorted(by_context)
    if len(context_ids) != 2:
        raise ValueError(f"expected two contexts, found {context_ids}")
    records = []
    for row in rows:
        other_context = next(
            context_id for context_id in context_ids if context_id != row["context_id"]
        )
        opposite = by_slot[(other_context, int(row["seed"]), int(row["rank"]))]
        same_context = choose_different_reason(
            row,
            by_context[row["context_id"]],
        )
        if normalize(row["title"]) in normalize(row["reason"]):
            raise ValueError(f"own reason leaks title for {row['record_id']}")
        records.append(
            {
                "record_id": row["record_id"],
                "context_id": row["context_id"],
                "seed": int(row["seed"]),
                "rank": int(row["rank"]),
                "title": row["title"],
                "artist": row["artist"],
                "reason": row["reason"],
                "catalog_label": row["catalog_label"],
                "entity_group": entity_group(row),
                "condition_reason_record_ids": {
                    "own": row["record_id"],
                    "same_context_swap": same_context["record_id"],
                    "opposite_context": opposite["record_id"],
                },
            }
        )

    return {
        "bundle_version": "music_reason_swap_v1",
        "model_id": order_bundle["model_id"],
        "conditions": list(CONDITIONS),
        "neutral_reason": NEUTRAL_REASON,
        "contexts": {
            context_id: {
                "profile": context["profile"],
                "current_need": context["current_need"],
            }
            for context_id, context in contexts.items()
        },
        "records": records,
        "registered_gates": {
            "record_count": 20,
            "condition_count": 80,
            "entity_group_counts": dict(Counter(row["entity_group"] for row in records)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog-rows",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "qwen_scope_music_reason_order_catalog_verified.jsonl",
    )
    parser.add_argument(
        "--order-bundle",
        type=Path,
        default=PROJECT_ROOT / "runs" / "qwen_scope_music_reason_order_bundle.json",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT / "runs" / "qwen_scope_music_reason_swap_bundle.json",
    )
    args = parser.parse_args()

    bundle = build_bundle(load_jsonl(args.catalog_rows), load_json(args.order_bundle))
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    args.bundle.write_text(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote reason-swap bundle with {len(bundle['records'])} records")
    print(f"- {args.bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
