"""Render recommendation cases into LLM prompt packs.

The output JSONL has one prompt per original/counterfactual case variant. It is
model-provider agnostic: each row contains the prompt text plus metadata needed
to compare rankings later.
"""

from __future__ import annotations

import argparse
import copy
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


def format_candidate_tracks(candidates: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for candidate in candidates:
        lines = [
            f"- track_id: {candidate['track_id']}",
            f"  title: {candidate['title']}",
            f"  artist: {candidate['artist']}",
            f"  tags: {', '.join(candidate.get('tags', []))}",
            f"  mood: {', '.join(candidate.get('mood', []))}",
            f"  energy: {candidate.get('energy')}/5",
            f"  language: {candidate.get('language')}",
            f"  evidence: {candidate.get('evidence')}",
        ]
        notes = candidate.get("notes")
        if notes:
            lines.append(f"  notes: {notes}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def apply_counterfactual(case: dict[str, Any], counterfactual: dict[str, Any]) -> dict[str, Any]:
    variant = copy.deepcopy(case)
    variant["case_id"] = counterfactual["variant_id"]
    variant["source_case_id"] = case["case_id"]
    variant["variant_type"] = counterfactual["variant_type"]
    variant["expected_effect"] = counterfactual["expected_effect"]

    if "user_profile" in counterfactual:
        variant["user_profile"] = counterfactual["user_profile"]
    if "current_context" in counterfactual:
        variant["current_context"] = counterfactual["current_context"]

    overrides = counterfactual.get("candidate_overrides", [])
    if overrides:
        by_id = {candidate["track_id"]: candidate for candidate in variant["candidates"]}
        for override in overrides:
            track_id = override["track_id"]
            if track_id not in by_id:
                raise ValueError(
                    f"{case['case_id']}/{counterfactual['variant_id']}: "
                    f"unknown candidate override {track_id}"
                )
            by_id[track_id].update({k: v for k, v in override.items() if k != "track_id"})

    return variant


def render_prompt(template: str, case: dict[str, Any]) -> str:
    return (
        template.replace("{{user_profile}}", case["user_profile"])
        .replace("{{current_context}}", case["current_context"])
        .replace("{{candidate_tracks}}", format_candidate_tracks(case["candidates"]))
    )


def prompt_rows(cases: list[dict[str, Any]], template: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.append(
            {
                "prompt_id": f"{case['case_id']}__original",
                "case_id": case["case_id"],
                "source_case_id": case["case_id"],
                "variant_type": "original",
                "expected_effect": "baseline",
                "expected_sensitive_factors": case["expected_sensitive_factors"],
                "candidate_track_ids": [c["track_id"] for c in case["candidates"]],
                "prompt": render_prompt(template, case),
            }
        )

        for counterfactual in case.get("counterfactuals", []):
            variant = apply_counterfactual(case, counterfactual)
            rows.append(
                {
                    "prompt_id": variant["case_id"],
                    "case_id": variant["case_id"],
                    "source_case_id": variant["source_case_id"],
                    "variant_type": variant["variant_type"],
                    "expected_effect": variant["expected_effect"],
                    "expected_sensitive_factors": variant["expected_sensitive_factors"],
                    "candidate_track_ids": [c["track_id"] for c in variant["candidates"]],
                    "prompt": render_prompt(template, variant),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "data" / "seed_cases.jsonl",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=PROJECT_ROOT / "prompts" / "rerank_prompt.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "prompt_pack.jsonl",
    )
    args = parser.parse_args()

    cases = load_jsonl(args.cases)
    template = args.template.read_text(encoding="utf-8")
    rows = prompt_rows(cases, template)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} prompts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

