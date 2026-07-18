"""Export option-letter song-title grounding prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TITLE_PROBE_TEMPLATE = (
    "Music catalog lookup.\n"
    "If the title is not a known real music release, answer Unknown.\n"
    'Song title: "{{title}}"\n'
    "Artist:\n"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_template(template: str, *, title: str, options: str = "") -> str:
    return template.replace("{{title}}", title).replace("{{options}}", options)


def validate_spec(spec: dict[str, Any]) -> None:
    candidates = spec.get("candidates", [])
    if not candidates:
        raise ValueError(f"{spec.get('item_id', '<missing>')}: no candidates")
    expected_type = spec["expected_best_type"]
    if expected_type not in {candidate["type"] for candidate in candidates}:
        raise ValueError(
            f"{spec['item_id']}: expected type {expected_type!r} missing from candidates"
        )
    labels = [candidate["label"] for candidate in candidates]
    if len(labels) != len(set(labels)):
        raise ValueError(f"{spec['item_id']}: duplicate candidate labels")
    if len(candidates) > len(LETTERS):
        raise ValueError(f"{spec['item_id']}: too many candidates")


def shuffled_candidates(candidates: list[dict[str, Any]], item_id: str, seed: int) -> list[dict[str, Any]]:
    seed_bytes = hashlib.sha256(f"{item_id}:{seed}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))
    copied = [dict(candidate) for candidate in candidates]
    rng.shuffle(copied)
    return copied


def option_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        option = {**candidate, "letter": LETTERS[index]}
        rows.append(option)
    return rows


def build_rows(specs: list[dict[str, Any]], template: str, permutations: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        validate_spec(spec)
        seen_orders: set[tuple[str, ...]] = set()
        permutation_index = 0
        attempt_index = 0
        while permutation_index < permutations:
            shuffled = shuffled_candidates(spec["candidates"], spec["item_id"], attempt_index)
            attempt_index += 1
            order_key = tuple(candidate["label"] for candidate in shuffled)
            if order_key in seen_orders:
                if attempt_index > permutations * 20:
                    raise ValueError(f"{spec['item_id']}: could not generate unique orders")
                continue
            seen_orders.add(order_key)
            options = option_rows(shuffled)
            option_text = "\n".join(
                f"{option['letter']}. {option['text']}" for option in options
            )
            rows.append(
                {
                    "prompt_id": f"{spec['item_id']}__perm{permutation_index}",
                    "item_id": spec["item_id"],
                    "permutation_index": permutation_index,
                    "group": spec["group"],
                    "title": spec["title"],
                    "expected_best_type": spec["expected_best_type"],
                    "expected_letters": [
                        option["letter"]
                        for option in options
                        if option["type"] == spec["expected_best_type"]
                    ],
                    "accepted_artists": spec.get("accepted_artists", []),
                    "source": spec.get("source"),
                    "notes": spec.get("notes", ""),
                    "options": options,
                    "title_probe_prompt": render_template(
                        TITLE_PROBE_TEMPLATE,
                        title=spec["title"],
                    ),
                    "prompt": render_template(
                        template,
                        title=spec["title"],
                        options=option_text,
                    ),
                }
            )
            permutation_index += 1
    return rows


def write_markdown(path: Path, specs: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    group_counts = Counter(row["group"] for row in rows)
    item_counts = Counter(spec["group"] for spec in specs)
    lines = [
        "# Qwen-Scope Song Entity Option-Letter Grounding Probe Plan",
        "",
        "## Purpose",
        "",
        "This probe scores only answer letters, avoiding the full-string length",
        "and title-copy biases observed in the previous forced-choice run.",
        "",
        "## Summary",
        "",
        f"- Total titles: {len(specs)}",
        f"- Total prompt variants: {len(rows)}",
        f"- Permutations per title: {len(rows) // max(1, len(specs))}",
        f"- Known real titles: {item_counts.get('known_real', 0)}",
        f"- Invented controls: {item_counts.get('invented_control', 0)}",
        f"- Free-generated titles: {item_counts.get('free_generated', 0)}",
        "",
        "## Variant Counts",
        "",
        "| Group | Variants |",
        "|---|---:|",
    ]
    for group, count in sorted(group_counts.items()):
        lines.append(f"| `{group}` | {count} |")
    lines.extend(
        [
            "",
            "## Titles",
            "",
            "| Group | Item | Title | Expected type | Candidate types |",
            "|---|---|---|---|---|",
        ]
    )
    for spec in specs:
        candidate_types = ", ".join(
            f"{candidate['text']} ({candidate['type']})"
            for candidate in spec["candidates"]
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{spec['group']}`",
                    f"`{spec['item_id']}`",
                    spec["title"],
                    f"`{spec['expected_best_type']}`",
                    candidate_types,
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--specs",
        type=Path,
        default=PROJECT_ROOT / "data" / "qwen_scope_song_entity_forced_choice_specs.jsonl",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=PROJECT_ROOT / "prompts" / "song_entity_option_letter_prompt.md",
    )
    parser.add_argument("--permutations", type=int, default=3)
    parser.add_argument(
        "--prompt-pack",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "qwen_scope_song_entity_option_letter_prompt_pack.jsonl",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "qwen_scope_song_entity_option_letter_probe_plan.md",
    )
    args = parser.parse_args()

    if args.permutations <= 0:
        raise ValueError("--permutations must be positive")

    specs = load_jsonl(args.specs)
    template = args.template.read_text(encoding="utf-8")
    rows = build_rows(specs, template, args.permutations)
    write_jsonl(args.prompt_pack, rows)
    write_markdown(args.markdown, specs, rows)

    print(f"Wrote {len(rows)} option-letter song-entity grounding prompts")
    print(f"- {args.prompt_pack}")
    print(f"- {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
