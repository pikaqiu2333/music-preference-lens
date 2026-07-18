"""Export generated-reason interpretability probes into prompt packs.

This script prepares the next mechanistic step: model outputs whose answer-token
features can be compared with the reason factors the model names.
"""

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


def render_prompt(template: str, spec: dict[str, Any], current_need: str) -> str:
    return (
        template.replace("{{profile}}", spec["profile"])
        .replace("{{current_need}}", current_need)
        .replace("{{candidate_cards}}", spec["candidate_cards"])
    )


def build_prompt_rows(specs: list[dict[str, Any]], template: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        variants = [
            (
                "original",
                spec["original_current_need"],
                spec["expected_original_reason_factors"],
            ),
            (
                "counterfactual",
                spec["counterfactual_current_need"],
                spec["expected_counterfactual_reason_factors"],
            ),
        ]
        for variant, current_need, expected_reason_factors in variants:
            rows.append(
                {
                    "prompt_id": f"{spec['probe_id']}__{variant}",
                    "probe_id": spec["probe_id"],
                    "dimension": spec["dimension"],
                    "variant": variant,
                    "candidate_track_ids": spec["candidate_track_ids"],
                    "expected_reason_factors": expected_reason_factors,
                    "feature_hypotheses": spec["feature_hypotheses"],
                    "interpretability_question": spec["interpretability_question"],
                    "prompt": render_prompt(template, spec, current_need),
                }
            )
    return rows


def write_markdown(path: Path, specs: list[dict[str, Any]], prompt_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Qwen-Scope Music Generated-Reason Probe Plan",
        "",
        "## Purpose",
        "",
        "This prompt pack is for generated-answer interpretability, not a music",
        "recommendation benchmark. The analysis target is whether answer reasons",
        "align with candidate SAE features discovered in phrase-level probes.",
        "",
        "## Summary",
        "",
        f"- Probe pairs: {len(specs)}",
        f"- Prompt rows: {len(prompt_rows)}",
        "- Variants per pair: original and counterfactual",
        "",
        "| Probe | Dimension | Expected Original Factors | Expected Counterfactual Factors | Feature Hypotheses |",
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
                    ", ".join(f"`{factor}`" for factor in spec["expected_original_reason_factors"]),
                    ", ".join(f"`{factor}`" for factor in spec["expected_counterfactual_reason_factors"]),
                    features,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Analysis Logic",
            "",
            "For each generated answer, identify the spans where the model names",
            "reason factors such as `high energy`, `emotional vocal`, or",
            "`no vocals`. Then compare candidate feature activations on those",
            "answer spans with the prompt-side feature direction.",
            "",
            "A useful failure case is an answer that confidently cites a reason while",
            "the expected feature movement is weak, missing, or contradicted.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--specs",
        type=Path,
        default=PROJECT_ROOT / "data" / "qwen_scope_music_generation_specs.jsonl",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=PROJECT_ROOT / "prompts" / "generated_reason_probe_prompt.md",
    )
    parser.add_argument(
        "--prompt-pack",
        type=Path,
        default=PROJECT_ROOT / "runs" / "qwen_scope_music_generation_prompt_pack.jsonl",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=PROJECT_ROOT / "reports" / "qwen_scope_music_generation_probe_plan.md",
    )
    args = parser.parse_args()

    specs = load_jsonl(args.specs)
    template = args.template.read_text(encoding="utf-8")
    prompt_rows = build_prompt_rows(specs, template)

    write_jsonl(args.prompt_pack, prompt_rows)
    write_markdown(args.markdown, specs, prompt_rows)

    print(f"Wrote {len(prompt_rows)} generated-reason prompts")
    print(f"- {args.prompt_pack}")
    print(f"- {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
