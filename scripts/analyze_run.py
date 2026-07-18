"""Analyze model outputs across original and counterfactual prompt variants."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def ranking_order(output: dict[str, Any] | None) -> list[str]:
    if not output:
        return []
    ranking = output.get("ranking", [])
    if not isinstance(ranking, list):
        return []
    return [str(item.get("track_id")) for item in ranking if isinstance(item, dict)]


def scores_by_track(output: dict[str, Any] | None) -> dict[str, float]:
    if not output:
        return {}
    ranking = output.get("ranking", [])
    if not isinstance(ranking, list):
        return {}
    scores: dict[str, float] = {}
    for item in ranking:
        if not isinstance(item, dict):
            continue
        track_id = item.get("track_id")
        score = item.get("score")
        if track_id is None or not isinstance(score, (int, float)):
            continue
        scores[str(track_id)] = float(score)
    return scores


def sensitive_factors_by_track(output: dict[str, Any] | None) -> dict[str, list[str]]:
    if not output:
        return {}
    ranking = output.get("ranking", [])
    if not isinstance(ranking, list):
        return {}
    factors: dict[str, list[str]] = {}
    for item in ranking:
        if not isinstance(item, dict):
            continue
        track_id = item.get("track_id")
        raw_factors = item.get("sensitive_factors", [])
        if track_id is None or not isinstance(raw_factors, list):
            continue
        factors[str(track_id)] = [str(factor) for factor in raw_factors]
    return factors


def top_choice(row: dict[str, Any]) -> str:
    output = row.get("model_output")
    if isinstance(output, dict) and output.get("top_choice"):
        return str(output["top_choice"])
    order = ranking_order(output)
    return order[0] if order else "<missing>"


def summarize_run(rows: list[dict[str, Any]]) -> str:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source_case_id"]].append(row)

    lines: list[str] = []
    lines.append("# Music Preference Lens Run Analysis")
    lines.append("")
    lines.append(f"Rows: {len(rows)}")
    models = sorted({row.get("model", "<missing>") for row in rows})
    lines.append(f"Models: {', '.join(models)}")
    error_count = sum(1 for row in rows if row.get("error"))
    parse_error_count = sum(1 for row in rows if row.get("parse_error"))
    lines.append(f"Errors: {error_count}")
    lines.append(f"Parse errors: {parse_error_count}")
    lines.append("")

    changed_top_count = 0
    variant_count = 0

    for source_case_id, case_rows in sorted(by_source.items()):
        original_rows = [row for row in case_rows if row["variant_type"] == "original"]
        original = original_rows[0] if original_rows else None
        original_top = top_choice(original) if original else "<missing>"
        original_output = original.get("model_output") if original else None
        original_order = ranking_order(original_output)
        original_scores = scores_by_track(original_output)
        original_factors = sensitive_factors_by_track(original_output)

        lines.append(f"## {source_case_id}")
        lines.append("")
        lines.append(f"- original_top: `{original_top}`")
        lines.append(f"- original_order: `{', '.join(original_order)}`")

        for row in sorted(case_rows, key=lambda r: r["prompt_id"]):
            if row["variant_type"] == "original":
                continue
            variant_count += 1
            variant_top = top_choice(row)
            variant_output = row.get("model_output")
            variant_order = ranking_order(variant_output)
            variant_scores = scores_by_track(variant_output)
            variant_factors = sensitive_factors_by_track(variant_output)
            top_changed = variant_top != original_top
            if top_changed:
                changed_top_count += 1
            status = "changed" if top_changed else "same"
            lines.append("")
            lines.append(f"### {row['prompt_id']}")
            lines.append("")
            lines.append(f"- variant_type: `{row['variant_type']}`")
            lines.append(f"- top_choice: `{variant_top}` ({status})")
            lines.append(f"- ranking_order: `{', '.join(variant_order)}`")
            score_deltas = []
            for track_id in sorted(set(original_scores) | set(variant_scores)):
                if track_id in original_scores and track_id in variant_scores:
                    delta = variant_scores[track_id] - original_scores[track_id]
                    score_deltas.append(
                        f"{track_id}: {original_scores[track_id]:.1f}->{variant_scores[track_id]:.1f} ({delta:+.1f})"
                    )
            if score_deltas:
                lines.append(f"- score_delta: {'; '.join(score_deltas)}")
            if variant_top in variant_factors:
                factors = ", ".join(variant_factors[variant_top])
                lines.append(f"- top_sensitive_factors: {factors}")
            if (
                variant_top == original_top
                and variant_top in original_factors
                and variant_top in variant_factors
            ):
                before = set(original_factors[variant_top])
                after = set(variant_factors[variant_top])
                removed = sorted(before - after)
                added = sorted(after - before)
                if removed or added:
                    lines.append(
                        "- top_factor_shift: "
                        f"removed [{', '.join(removed) or 'none'}]; "
                        f"added [{', '.join(added) or 'none'}]"
                    )
            lines.append(f"- expected_effect: {row['expected_effect']}")
            if row.get("error"):
                lines.append(f"- error: `{row['error']}`")
            if row.get("parse_error"):
                lines.append(f"- parse_error: `{row['parse_error']}`")
        lines.append("")

    lines.append("## Aggregate")
    lines.append("")
    if variant_count:
        rate = changed_top_count / variant_count
        lines.append(f"- top-choice change rate: {changed_top_count}/{variant_count} ({rate:.1%})")
    else:
        lines.append("- top-choice change rate: n/a")
    lines.append("")
    lines.append(
        "Interpretation note: top-choice changes are only a coarse signal. A faithful "
        "model can keep the same top choice while changing scores or reasons, and an "
        "unfaithful model can change rankings for the wrong reason."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "runs" / "openai_results.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "run_analysis.md",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.results)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(summarize_run(rows), encoding="utf-8")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
