"""Export matched pair-first and reason-first music generation prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_IDS = ("emotional_vocal", "strict_no_vocals")
SEEDS = (17, 29)
ORDERS = ("pair_first", "reason_first")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def render_prompt(context: dict[str, Any], order: str) -> str:
    if order == "pair_first":
        order_instruction = (
            "For every item, output the title and artist first. Only after "
            "choosing that pair, write one short reason."
        )
        first_field = "Title"
    elif order == "reason_first":
        order_instruction = (
            "For every item, write one short reason first. Only after writing "
            "that reason, output the title and artist it leads you to choose."
        )
        first_field = "Reason"
    else:
        raise ValueError(f"unknown order: {order}")
    return (
        "Music playlist request.\n\n"
        f"User profile: {context['profile']}\n\n"
        f"Current need: {context['current_need']}\n\n"
        "Recommend 5 real, existing tracks. Each item must contain exactly one "
        "short Reason, one Title, and one Artist.\n"
        f"{order_instruction}\n\n"
        "Playlist:\n"
        f"1. {first_field}:\n"
    )


def build_bundle(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in contexts if row["context_id"] in CONTEXT_IDS]
    if [row["context_id"] for row in selected] != list(CONTEXT_IDS):
        by_id = {row["context_id"]: row for row in selected}
        selected = [by_id[context_id] for context_id in CONTEXT_IDS]
    if len(selected) != len(CONTEXT_IDS):
        raise ValueError("reason-order contexts are missing")
    rendered = []
    for context in selected:
        rendered.append(
            {
                **context,
                "prompts": {
                    order: render_prompt(context, order) for order in ORDERS
                },
            }
        )
    return {
        "bundle_version": "music_reason_order_counterfactual_v1",
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "contexts": rendered,
        "orders": list(ORDERS),
        "seeds": list(SEEDS),
        "generation": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 512,
        },
        "registered_gates": {
            "expected_generations": 8,
            "minimum_valid_generations": 8,
            "minimum_complete_rows": 32,
            "minimum_order_compliance": 0.8,
        },
    }


def write_plan(path: Path, bundle: dict[str, Any]) -> None:
    lines = [
        "# Music Reason-Order HF Smoke Plan",
        "",
        f"- Contexts: {len(bundle['contexts'])}",
        f"- Seeds: {', '.join(str(seed) for seed in bundle['seeds'])}",
        "- Conditions: pair-first and reason-first",
        "- Expected generations: 8",
        "- Requested recommendations: 40 total",
        "- Minimum complete rows: 32",
        "- Minimum actual order compliance: 0.80",
        "- Hardware target: Hugging Face `t4-small`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contexts",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "qwen_scope_song_entity_generation_time_specs.jsonl",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT / "runs" / "qwen_scope_music_reason_order_bundle.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=PROJECT_ROOT / "reports" / "qwen_scope_music_reason_order_plan.md",
    )
    args = parser.parse_args()

    bundle = build_bundle(load_jsonl(args.contexts))
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    args.bundle.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_plan(args.markdown, bundle)
    print(f"Wrote reason-order bundle with {len(bundle['contexts'])} contexts")
    print(f"- {args.bundle}")
    print(f"- {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
