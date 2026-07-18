"""Export forced-choice song-title grounding prompts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def render_prompt(template: str, title: str) -> str:
    return template.replace("{{title}}", title)


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


def build_rows(specs: list[dict[str, Any]], template: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        validate_spec(spec)
        rows.append(
            {
                "prompt_id": spec["item_id"],
                "item_id": spec["item_id"],
                "group": spec["group"],
                "title": spec["title"],
                "expected_best_type": spec["expected_best_type"],
                "accepted_artists": spec.get("accepted_artists", []),
                "source": spec.get("source"),
                "notes": spec.get("notes", ""),
                "candidates": spec["candidates"],
                "prompt": render_prompt(template, spec["title"]),
            }
        )
    return rows


def write_markdown(path: Path, specs: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(row["group"] for row in rows)
    lines = [
        "# Qwen-Scope Song Entity Forced-Choice Grounding Probe Plan",
        "",
        "## Purpose",
        "",
        "This probe scores candidate artist continuations instead of using open",
        "generation. It tests whether each song title is treated as a known entity",
        "or as an unverified / invented placeholder.",
        "",
        "## Summary",
        "",
        f"- Total prompts: {len(rows)}",
        f"- Known real controls: {counts.get('known_real', 0)}",
        f"- Invented controls: {counts.get('invented_control', 0)}",
        f"- Free-generated titles: {counts.get('free_generated', 0)}",
        "",
        "| Group | Item | Title | Expected best type | Candidates |",
        "|---|---|---|---|---|",
    ]
    for spec in specs:
        candidates = ", ".join(
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
                    candidates,
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
        default=PROJECT_ROOT / "prompts" / "song_entity_forced_choice_prompt.md",
    )
    parser.add_argument(
        "--prompt-pack",
        type=Path,
        default=PROJECT_ROOT / "runs" / "qwen_scope_song_entity_forced_choice_prompt_pack.jsonl",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "qwen_scope_song_entity_forced_choice_probe_plan.md",
    )
    args = parser.parse_args()

    specs = load_jsonl(args.specs)
    template = args.template.read_text(encoding="utf-8")
    rows = build_rows(specs, template)
    write_jsonl(args.prompt_pack, rows)
    write_markdown(args.markdown, specs, rows)

    print(f"Wrote {len(rows)} forced-choice song-entity grounding prompts")
    print(f"- {args.prompt_pack}")
    print(f"- {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
