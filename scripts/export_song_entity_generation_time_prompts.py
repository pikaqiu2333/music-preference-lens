"""Export generation-time song-entity prompts and a standalone job bundle."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [17, 29, 43]
DEFAULT_GENERATION = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 384,
}
EXPECTED_CONTROL_COUNTS = {
    "known_exact": 20,
    "artist_mismatch": 10,
    "synthetic_pair": 10,
}


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


def render_generation_prompt(template: str, context: dict[str, Any]) -> str:
    return template.replace("{{profile}}", context["profile"]).replace(
        "{{current_need}}",
        context["current_need"],
    )


def validate_contexts(contexts: list[dict[str, Any]]) -> None:
    if len(contexts) != 4:
        raise ValueError(f"expected 4 contexts, found {len(contexts)}")
    required = {"context_id", "dimension", "profile", "current_need"}
    ids: set[str] = set()
    for context in contexts:
        missing = required - set(context)
        if missing:
            raise ValueError(f"context missing fields: {sorted(missing)}")
        if context["context_id"] in ids:
            raise ValueError(f"duplicate context_id: {context['context_id']}")
        ids.add(context["context_id"])


def validate_controls(controls: list[dict[str, Any]]) -> None:
    counts = Counter(control.get("group") for control in controls)
    if dict(counts) != EXPECTED_CONTROL_COUNTS:
        raise ValueError(
            f"unexpected control counts: {dict(counts)}; expected {EXPECTED_CONTROL_COUNTS}"
        )
    required = {"pair_id", "group", "title", "artist", "language"}
    pair_ids: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for control in controls:
        missing = required - set(control)
        if missing:
            raise ValueError(f"control missing fields: {sorted(missing)}")
        pair_id = control["pair_id"]
        pair = (control["title"].casefold(), control["artist"].casefold())
        if pair_id in pair_ids:
            raise ValueError(f"duplicate pair_id: {pair_id}")
        if pair in pairs:
            raise ValueError(f"duplicate title-artist pair: {pair}")
        pair_ids.add(pair_id)
        pairs.add(pair)


def build_prompt_rows(
    contexts: list[dict[str, Any]],
    generation_template: str,
    seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for context in contexts:
        prompt = render_generation_prompt(generation_template, context)
        for seed in seeds:
            rows.append(
                {
                    "generation_id": f"{context['context_id']}__seed{seed}",
                    "context_id": context["context_id"],
                    "dimension": context["dimension"],
                    "seed": seed,
                    "prompt": prompt,
                }
            )
    return rows


def build_bundle(
    contexts: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    generation_template: str,
    verification_template: str,
    seeds: list[int],
) -> dict[str, Any]:
    rendered_contexts = []
    for context in contexts:
        rendered_contexts.append(
            {
                **context,
                "generation_prompt": render_generation_prompt(
                    generation_template,
                    context,
                ),
            }
        )
    return {
        "bundle_version": "song_entity_generation_time_v1",
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "sae_repo": "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50",
        "sae_layer": 24,
        "seeds": seeds,
        "generation": DEFAULT_GENERATION,
        "contexts": rendered_contexts,
        "controls": controls,
        "verification_template": verification_template,
        "verification_options": {
            "known_exact": "Known exact title-artist release",
            "unknown": "Unknown or not a known exact title-artist release",
        },
        "hf_result_repo": "REDACTED/music-preference-lens-runs",
    }


def write_markdown(
    path: Path,
    contexts: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    prompt_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(control["group"] for control in controls)
    lines = [
        "# Qwen-Scope Song Entity Generation-Time Probe Plan",
        "",
        "## Summary",
        "",
        f"- Contexts: {len(contexts)}",
        f"- Seeds: {', '.join(str(seed) for seed in DEFAULT_SEEDS)}",
        f"- Playlist generations: {len(prompt_rows)}",
        "- Requested tracks per generation: 5",
        f"- Known exact controls: {counts['known_exact']}",
        f"- Artist-mismatch controls: {counts['artist_mismatch']}",
        f"- Synthetic controls: {counts['synthetic_pair']}",
        "- Primary internal signal: layer-24 pair-end SAE activation",
        "- Verification conditions: neutral and self-attributed",
        "",
        "## Contexts",
        "",
        "| Context | Dimension | Current need |",
        "|---|---|---|",
    ]
    for context in contexts:
        lines.append(
            f"| `{context['context_id']}` | `{context['dimension']}` | "
            f"{context['current_need']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Gate",
            "",
            "The full run is interpreted only if control balanced accuracy is at",
            "least 0.80, at least 10 of 12 generations are valid, and at least",
            "50 complete title-artist pairs are parsed.",
            "",
        ]
    )
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
        "--controls",
        type=Path,
        default=PROJECT_ROOT / "data" / "qwen_scope_song_entity_pair_controls.jsonl",
    )
    parser.add_argument(
        "--generation-template",
        type=Path,
        default=PROJECT_ROOT / "prompts" / "song_entity_generation_time_prompt.md",
    )
    parser.add_argument(
        "--verification-template",
        type=Path,
        default=PROJECT_ROOT / "prompts" / "song_entity_pair_verification_prompt.md",
    )
    parser.add_argument(
        "--prompt-pack",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "qwen_scope_song_entity_generation_time_prompt_pack.jsonl",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "qwen_scope_song_entity_generation_time_bundle.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "qwen_scope_song_entity_generation_time_probe_plan.md",
    )
    args = parser.parse_args()

    contexts = load_jsonl(args.contexts)
    controls = load_jsonl(args.controls)
    validate_contexts(contexts)
    validate_controls(controls)
    generation_template = args.generation_template.read_text(encoding="utf-8")
    verification_template = args.verification_template.read_text(encoding="utf-8")
    rows = build_prompt_rows(contexts, generation_template, DEFAULT_SEEDS)
    bundle = build_bundle(
        contexts,
        controls,
        generation_template,
        verification_template,
        DEFAULT_SEEDS,
    )

    write_jsonl(args.prompt_pack, rows)
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    args.bundle.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.markdown, contexts, controls, rows)
    print(f"Wrote {len(rows)} generation-time prompts")
    print(f"- {args.prompt_pack}")
    print(f"- {args.bundle}")
    print(f"- {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
