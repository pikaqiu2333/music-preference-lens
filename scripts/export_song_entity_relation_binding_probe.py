"""Validate relation-binding controls and export a standalone HF Job bundle."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LANGUAGE_COUNTS = {"en": 10, "zh": 10}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_controls(rows: list[dict[str, Any]]) -> None:
    counts = Counter(row.get("language") for row in rows)
    if dict(counts) != EXPECTED_LANGUAGE_COUNTS:
        raise ValueError(
            f"unexpected language counts: {dict(counts)}; "
            f"expected {EXPECTED_LANGUAGE_COUNTS}"
        )
    required = {
        "relation_id",
        "block_id",
        "title",
        "correct_artist",
        "wrong_artist",
        "neutral_title",
        "language",
    }
    relation_ids: set[str] = set()
    titles: set[tuple[str, str]] = set()
    blocks: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"control missing fields: {sorted(missing)}")
        if row["relation_id"] in relation_ids:
            raise ValueError(f"duplicate relation_id: {row['relation_id']}")
        title_key = (row["language"], row["title"].casefold())
        if title_key in titles:
            raise ValueError(f"duplicate title: {title_key}")
        if row["correct_artist"].casefold() == row["wrong_artist"].casefold():
            raise ValueError(f"unmatched control: {row['relation_id']}")
        relation_ids.add(row["relation_id"])
        titles.add(title_key)
        blocks.setdefault(row["block_id"], []).append(row)
    for language in EXPECTED_LANGUAGE_COUNTS:
        subset = [row for row in rows if row["language"] == language]
        exact = Counter(row["correct_artist"].casefold() for row in subset)
        mismatch = Counter(row["wrong_artist"].casefold() for row in subset)
        if exact != mismatch:
            raise ValueError(f"artist marginals differ for language {language}")
    for block_id, block_rows in blocks.items():
        if len(block_rows) != 2:
            raise ValueError(f"swap block {block_id} must contain exactly two relations")
        if len({row["language"] for row in block_rows}) != 1:
            raise ValueError(f"swap block {block_id} crosses languages")
        exact = Counter(row["correct_artist"].casefold() for row in block_rows)
        mismatch = Counter(row["wrong_artist"].casefold() for row in block_rows)
        if exact != mismatch:
            raise ValueError(f"swap block {block_id} is not a closed two-way swap")


def select_smoke_controls(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    smoke_blocks = {
        "en_swap_1",
        "en_swap_2",
        "en_swap_3",
        "zh_swap_1",
        "zh_swap_2",
        "zh_swap_3",
    }
    return [row for row in rows if row["block_id"] in smoke_blocks]


def build_bundle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "bundle_version": "song_entity_relation_binding_v1",
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "sae_repo": "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50",
        "sae_layer": 24,
        "controls": rows,
        "sae_template": "Catalog fact check.\nTitle: {title}\nArtist: {artist}\nRelation:",
        "likelihood_template": "Complete the factual catalog entry.\nTitle: {title}\nArtist: ",
        "choice_template": (
            "Which artist is correctly associated with the song title "
            "\"{title}\"?\nA. {artist_a}\nB. {artist_b}\nAnswer:"
        ),
        "feature_count": 32,
        "pair_folds": 5,
        "interpretation_threshold": 0.80,
    }


def build_catalog_precheck_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for row in rows:
        checks.append(
            {
                "pair_id": f"{row['relation_id']}__exact",
                "relation_id": row["relation_id"],
                "block_id": row["block_id"],
                "group": "known_exact",
                "title": row["title"],
                "artist": row["correct_artist"],
                "accepted_artists": row.get("accepted_artists", []),
                "language": row["language"],
            }
        )
        checks.append(
            {
                "pair_id": f"{row['relation_id']}__mismatch",
                "relation_id": row["relation_id"],
                "block_id": row["block_id"],
                "group": "artist_mismatch",
                "title": row["title"],
                "artist": row["wrong_artist"],
                "language": row["language"],
            }
        )
    return checks


def write_plan(path: Path, rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["language"] for row in rows)
    lines = [
        "# Qwen-Scope Song Entity Relation-Binding Run Plan",
        "",
        f"- Relations: {len(rows)}",
        f"- English: {counts['en']}",
        f"- Chinese: {counts['zh']}",
        "- Exact and mismatch artist marginals: identical within language",
        "- Primary behavioral metric: prior-corrected direct likelihood accuracy",
        "- Primary mechanism metric: pair-end SAE paired CV accuracy",
        "- Registered threshold: 0.80 for both primary metrics",
        "- Hardware target: Hugging Face `t4-small`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controls",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "qwen_scope_song_entity_relation_binding_controls.jsonl",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "qwen_scope_song_entity_relation_binding_bundle.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "qwen_scope_song_entity_relation_binding_probe_plan.md",
    )
    parser.add_argument(
        "--catalog-precheck-input",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "qwen_scope_song_entity_relation_binding_catalog_precheck_input.jsonl",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.controls)
    validate_controls(rows)
    bundle = build_bundle(rows)
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    args.bundle.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.catalog_precheck_input, build_catalog_precheck_rows(rows))
    write_plan(args.markdown, rows)
    print(f"Wrote relation-binding bundle with {len(rows)} controls")
    print(f"- {args.bundle}")
    print(f"- {args.catalog_precheck_input}")
    print(f"- {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
