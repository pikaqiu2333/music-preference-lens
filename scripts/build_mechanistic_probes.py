"""Build contrastive mechanistic probe specs from recommendation cases."""

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


def load_dimensions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["dimensions"])


def tokenize_for_match(text: str) -> str:
    return text.lower()


def infer_dimensions(text: str, dimensions: list[dict[str, Any]]) -> list[str]:
    haystack = tokenize_for_match(text)
    matched: list[str] = []
    for dimension in dimensions:
        for keyword in dimension.get("keywords", []):
            if keyword.lower() in haystack:
                matched.append(dimension["dimension_id"])
                break
    return matched


def format_candidate(candidate: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"track_id: {candidate['track_id']}",
            f"title: {candidate['title']}",
            f"artist: {candidate['artist']}",
            f"tags: {', '.join(candidate.get('tags', []))}",
            f"mood: {', '.join(candidate.get('mood', []))}",
            f"energy: {candidate.get('energy')}/5",
            f"language: {candidate.get('language')}",
            f"evidence: {candidate.get('evidence')}",
        ]
    )


def format_probe_text(case: dict[str, Any]) -> str:
    candidates = "\n\n".join(format_candidate(c) for c in case["candidates"])
    return "\n\n".join(
        [
            "User profile:",
            case["user_profile"],
            "Current context:",
            case["current_context"],
            "Candidate tracks:",
            candidates,
        ]
    )


def condition_text(case: dict[str, Any]) -> str:
    return "\n".join(
        [
            case.get("user_profile", ""),
            case.get("current_context", ""),
            " ".join(case.get("expected_sensitive_factors", [])),
        ]
    )


def apply_counterfactual(case: dict[str, Any], counterfactual: dict[str, Any]) -> dict[str, Any]:
    variant = copy.deepcopy(case)
    if "user_profile" in counterfactual:
        variant["user_profile"] = counterfactual["user_profile"]
    if "current_context" in counterfactual:
        variant["current_context"] = counterfactual["current_context"]

    overrides = counterfactual.get("candidate_overrides", [])
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


def infer_candidate_focus(
    case: dict[str, Any],
    counterfactual: dict[str, Any],
    dimensions: list[dict[str, Any]],
    added_dimensions: set[str],
    removed_dimensions: set[str],
) -> list[dict[str, str]]:
    focus: list[dict[str, str]] = []
    for candidate in case["candidates"]:
        candidate_text = " ".join(
            [
                " ".join(candidate.get("tags", [])),
                " ".join(candidate.get("mood", [])),
                candidate.get("language", ""),
                candidate.get("evidence", ""),
            ]
        )
        candidate_dims = set(infer_dimensions(candidate_text, dimensions))
        added_overlap = sorted(candidate_dims & added_dimensions)
        removed_overlap = sorted(candidate_dims & removed_dimensions)
        if added_overlap:
            direction = "up"
            rationale = "overlaps added dimensions: " + ", ".join(added_overlap)
        elif removed_overlap and counterfactual["variant_type"] == "preference_ablation":
            direction = "same_or_reason_shift"
            rationale = "overlaps removed preference dimensions: " + ", ".join(removed_overlap)
        elif removed_overlap:
            direction = "down"
            rationale = "overlaps removed dimensions: " + ", ".join(removed_overlap)
        elif counterfactual["variant_type"] == "preference_ablation":
            direction = "same_or_reason_shift"
            rationale = "ablation may change rationale weight without changing winner"
        else:
            direction = "unknown"
            rationale = "no simple keyword overlap with changed dimensions"
        focus.append(
            {
                "track_id": candidate["track_id"],
                "expected_direction": direction,
                "rationale": rationale,
            }
        )
    return focus


def build_probe_specs(
    cases: list[dict[str, Any]],
    dimensions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for case in cases:
        original_text = format_probe_text(case)
        original_dimensions = set(infer_dimensions(condition_text(case), dimensions))
        for counterfactual in case.get("counterfactuals", []):
            variant_case = apply_counterfactual(case, counterfactual)
            counterfactual_text = format_probe_text(variant_case)
            counterfactual_dimensions = set(
                infer_dimensions(
                    "\n".join(
                        [
                            condition_text(variant_case),
                            counterfactual.get("expected_effect", ""),
                        ]
                    ),
                    dimensions,
                )
            )
            added_dimensions = counterfactual_dimensions - original_dimensions
            removed_dimensions = original_dimensions - counterfactual_dimensions
            changed_dimensions = added_dimensions | removed_dimensions
            if changed_dimensions:
                target_dimensions = sorted(changed_dimensions)
            else:
                target_dimensions = sorted(original_dimensions | counterfactual_dimensions)
            probes.append(
                {
                    "probe_id": f"{case['case_id']}__{counterfactual['variant_id']}",
                    "source_case_id": case["case_id"],
                    "variant_id": counterfactual["variant_id"],
                    "variant_type": counterfactual["variant_type"],
                    "original_text": original_text,
                    "counterfactual_text": counterfactual_text,
                    "target_dimensions": target_dimensions,
                    "original_dimensions": sorted(original_dimensions),
                    "counterfactual_dimensions": sorted(counterfactual_dimensions),
                    "added_dimensions": sorted(added_dimensions),
                    "removed_dimensions": sorted(removed_dimensions),
                    "expected_effect": counterfactual["expected_effect"],
                    "candidate_track_ids": [c["track_id"] for c in case["candidates"]],
                    "candidate_focus": infer_candidate_focus(
                        case,
                        counterfactual,
                        dimensions,
                        added_dimensions,
                        removed_dimensions,
                    ),
                }
            )
    return probes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "data" / "seed_cases.jsonl",
    )
    parser.add_argument(
        "--dimensions",
        type=Path,
        default=PROJECT_ROOT / "config" / "interpretability_dimensions.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "mechanistic_probe_pack.jsonl",
    )
    args = parser.parse_args()

    cases = load_jsonl(args.cases)
    dimensions = load_dimensions(args.dimensions)
    probes = build_probe_specs(cases, dimensions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for probe in probes:
            handle.write(json.dumps(probe, ensure_ascii=False) + "\n")

    print(f"Wrote {len(probes)} mechanistic probes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
