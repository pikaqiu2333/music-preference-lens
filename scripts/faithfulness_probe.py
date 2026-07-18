"""Validate case files and run a lightweight lexical faithfulness pre-check.

This script does not call an LLM. It gives us a cheap first pass over the
casebook:

- validate required fields,
- summarize counterfactual coverage,
- detect whether expected sensitive factors have lexical support in the case,
- produce a TODO-style report for manual case improvement.

Later, this can be extended to compare real LLM outputs across original and
counterfactual prompts.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_CASE_FIELDS = {
    "case_id",
    "user_profile",
    "current_context",
    "candidates",
    "expected_sensitive_factors",
    "counterfactuals",
}

REQUIRED_CANDIDATE_FIELDS = {
    "track_id",
    "title",
    "artist",
    "tags",
    "mood",
    "energy",
    "language",
    "evidence",
}

REQUIRED_COUNTERFACTUAL_FIELDS = {
    "variant_id",
    "variant_type",
    "changed_fields",
    "expected_effect",
}


def tokenize(text: str) -> set[str]:
    """Tokenize Chinese/English-ish text for rough lexical overlap checks."""
    lowered = text.lower()
    ascii_tokens = set(re.findall(r"[a-z0-9]+", lowered))
    chinese_chars = set(re.findall(r"[\u4e00-\u9fff]", text))
    return ascii_tokens | chinese_chars


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            cases.append(item)
    return cases


def validate_case(case: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    case_id = case.get("case_id", "<missing>")

    missing = REQUIRED_CASE_FIELDS - set(case)
    if missing:
        errors.append(f"{case_id}: missing case fields: {sorted(missing)}")

    candidates = case.get("candidates", [])
    if not isinstance(candidates, list) or len(candidates) < 2:
        errors.append(f"{case_id}: expected at least two candidates")
    for candidate in candidates:
        missing_candidate = REQUIRED_CANDIDATE_FIELDS - set(candidate)
        if missing_candidate:
            track_id = candidate.get("track_id", "<missing>")
            errors.append(
                f"{case_id}/{track_id}: missing candidate fields: "
                f"{sorted(missing_candidate)}"
            )

    counterfactuals = case.get("counterfactuals", [])
    if not isinstance(counterfactuals, list) or not counterfactuals:
        errors.append(f"{case_id}: expected at least one counterfactual")
    for cf in counterfactuals:
        missing_cf = REQUIRED_COUNTERFACTUAL_FIELDS - set(cf)
        if missing_cf:
            variant_id = cf.get("variant_id", "<missing>")
            errors.append(
                f"{case_id}/{variant_id}: missing counterfactual fields: "
                f"{sorted(missing_cf)}"
            )

    return errors


def lexical_factor_support(case: dict[str, Any]) -> list[dict[str, Any]]:
    text_parts = [
        case.get("user_profile", ""),
        case.get("current_context", ""),
    ]
    for candidate in case.get("candidates", []):
        text_parts.extend(
            [
                " ".join(candidate.get("tags", [])),
                " ".join(candidate.get("mood", [])),
                candidate.get("language", ""),
                candidate.get("evidence", ""),
                candidate.get("notes", ""),
            ]
        )
    case_tokens = tokenize("\n".join(text_parts))

    rows: list[dict[str, Any]] = []
    for factor in case.get("expected_sensitive_factors", []):
        factor_tokens = tokenize(factor)
        overlap = sorted(factor_tokens & case_tokens)
        rows.append(
            {
                "factor": factor,
                "token_count": len(factor_tokens),
                "overlap_count": len(overlap),
                "overlap": overlap[:10],
                "has_support": bool(overlap),
            }
        )
    return rows


def build_report(cases: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Faithfulness Probe Pre-Check")
    lines.append("")
    lines.append(f"Cases: {len(cases)}")
    lines.append("")

    variant_counts: Counter[str] = Counter()
    for case in cases:
        for cf in case.get("counterfactuals", []):
            variant_counts[cf.get("variant_type", "<missing>")] += 1

    lines.append("## Counterfactual Coverage")
    lines.append("")
    for variant_type, count in sorted(variant_counts.items()):
        lines.append(f"- {variant_type}: {count}")
    lines.append("")

    lines.append("## Case Factor Support")
    lines.append("")
    for case in cases:
        lines.append(f"### {case.get('case_id', '<missing>')}")
        lines.append("")
        for row in lexical_factor_support(case):
            status = "ok" if row["has_support"] else "check"
            overlap = ", ".join(row["overlap"]) if row["overlap"] else "none"
            lines.append(f"- {status}: {row['factor']} (overlap: {overlap})")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "This is only a lexical pre-check. A factor with weak lexical overlap can "
        "still be valid, and a factor with strong overlap can still be unfaithful "
        "in model output. The next step is to call an LLM and compare original "
        "versus counterfactual rankings."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "seed_cases.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "reports"
        / "precheck_report.md",
    )
    args = parser.parse_args()

    cases = load_jsonl(args.cases)
    errors: list[str] = []
    for case in cases:
        errors.extend(validate_case(case))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(build_report(cases), encoding="utf-8")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

