"""Export song-title entity-grounding prompts."""

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


def build_rows(specs: list[dict[str, Any]], template: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        rows.append(
            {
                "prompt_id": spec["item_id"],
                "item_id": spec["item_id"],
                "group": spec["group"],
                "title": spec["title"],
                "accepted_artists": spec.get("accepted_artists", []),
                "source": spec.get("source"),
                "notes": spec.get("notes", ""),
                "prompt": render_prompt(template, spec["title"]),
            }
        )
    return rows


def write_markdown(path: Path, specs: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(row["group"] for row in rows)
    lines = [
        "# Qwen-Scope Song Entity Grounding Probe Plan",
        "",
        "## Purpose",
        "",
        "This probe tests whether generated song titles behave more like known",
        "song entities or deliberately invented title controls.",
        "",
        "The prompt asks for a catalog-style artist continuation:",
        "",
        "```text",
        'Music catalog lookup.',
        'Song title: "{{title}}"',
        "Artist:",
        "```",
        "",
        "## Summary",
        "",
        f"- Total prompts: {len(rows)}",
        f"- Known real controls: {counts.get('known_real', 0)}",
        f"- Invented controls: {counts.get('invented_control', 0)}",
        f"- Free-generated titles: {counts.get('free_generated', 0)}",
        "",
        "| Group | Item | Title | Accepted Artists | Source |",
        "|---|---|---|---|---|",
    ]
    for spec in specs:
        artists = ", ".join(spec.get("accepted_artists", [])) or "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{spec['group']}`",
                    f"`{spec['item_id']}`",
                    spec["title"],
                    artists,
                    spec.get("source", ""),
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
        default=PROJECT_ROOT / "data" / "qwen_scope_song_entity_grounding_specs.jsonl",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=PROJECT_ROOT / "prompts" / "song_entity_grounding_prompt.md",
    )
    parser.add_argument(
        "--prompt-pack",
        type=Path,
        default=PROJECT_ROOT / "runs" / "qwen_scope_song_entity_grounding_prompt_pack.jsonl",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=PROJECT_ROOT / "reports" / "qwen_scope_song_entity_grounding_probe_plan.md",
    )
    args = parser.parse_args()

    specs = load_jsonl(args.specs)
    template = args.template.read_text(encoding="utf-8")
    rows = build_rows(specs, template)
    write_jsonl(args.prompt_pack, rows)
    write_markdown(args.markdown, specs, rows)

    print(f"Wrote {len(rows)} song-entity grounding prompts")
    print(f"- {args.prompt_pack}")
    print(f"- {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
