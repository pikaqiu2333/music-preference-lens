"""Export free playlist generation prompts for answer-token interpretability."""

from __future__ import annotations

import argparse
import json
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


def render_prompt(template: str, spec: dict[str, Any], *, current_need: str) -> str:
    return template.replace("{{profile}}", spec["profile"]).replace(
        "{{current_need}}",
        current_need,
    )


def build_rows(specs: list[dict[str, Any]], template: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        variants = [
            (
                "original",
                spec["original_current_need"],
                spec["expected_original_playlist_factors"],
            ),
            (
                "counterfactual",
                spec["counterfactual_current_need"],
                spec["expected_counterfactual_playlist_factors"],
            ),
        ]
        for variant, current_need, expected_playlist_factors in variants:
            rows.append(
                {
                    "prompt_id": f"{spec['probe_id']}__{variant}",
                    "probe_id": spec["probe_id"],
                    "dimension": spec["dimension"],
                    "variant": variant,
                    "current_need": current_need,
                    "expected_playlist_factors": expected_playlist_factors,
                    "feature_hypotheses": spec["feature_hypotheses"],
                    "interpretability_question": spec["interpretability_question"],
                    "prompt": render_prompt(template, spec, current_need=current_need),
                }
            )
    return rows


def write_markdown(path: Path, specs: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Qwen-Scope Free Playlist Generation Probe Plan",
        "",
        "## Purpose",
        "",
        "This prompt pack removes candidate tracks and asks the base model to",
        "generate a short playlist directly from the user profile and current",
        "listening need.",
        "",
        "The goal is not to judge whether the generated song titles are factual.",
        "The goal is to test whether the generated playlist language and",
        "answer-token SAE features move together under controlled preference",
        "counterfactuals.",
        "",
        "## Summary",
        "",
        f"- Probe pairs: {len(specs)}",
        f"- Prompt rows: {len(rows)}",
        "- Variants per pair: original and counterfactual",
        "- Candidate tracks: none",
        "- Generated object: 5-track playlist with title, sound, and reason",
        "",
        "| Probe | Dimension | Original Need | Counterfactual Need | Key Features |",
        "|---|---|---|---|---|",
    ]
    for spec in specs:
        features = ", ".join(item["feature"] for item in spec["feature_hypotheses"])
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{spec['probe_id']}`",
                    f"`{spec['dimension']}`",
                    spec["original_current_need"],
                    spec["counterfactual_current_need"],
                    features,
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
        default=PROJECT_ROOT / "data" / "qwen_scope_music_free_playlist_specs.jsonl",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=PROJECT_ROOT / "prompts" / "free_playlist_generation_probe_prompt.md",
    )
    parser.add_argument(
        "--prompt-pack",
        type=Path,
        default=PROJECT_ROOT / "runs" / "qwen_scope_music_free_playlist_prompt_pack.jsonl",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=PROJECT_ROOT / "reports" / "qwen_scope_music_free_playlist_probe_plan.md",
    )
    args = parser.parse_args()

    specs = load_jsonl(args.specs)
    template = args.template.read_text(encoding="utf-8")
    rows = build_rows(specs, template)
    write_jsonl(args.prompt_pack, rows)
    write_markdown(args.markdown, specs, rows)

    print(f"Wrote {len(rows)} free-playlist prompts")
    print(f"- {args.prompt_pack}")
    print(f"- {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
