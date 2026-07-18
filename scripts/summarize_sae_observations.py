"""Summarize SAE observation CSVs into a Markdown report."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def summarize(rows: list[dict[str, str]]) -> str:
    by_probe: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_probe[row.get("probe_id", "<missing>")].append(row)

    lines: list[str] = []
    lines.append("# SAE Observation Summary")
    lines.append("")
    lines.append(f"Rows: {len(rows)}")
    lines.append(f"Probes: {len(by_probe)}")
    lines.append("")

    comparable = [
        row
        for row in rows
        if row.get("expected_direction")
        and row.get("observed_direction")
        and row.get("expected_direction") != "inspect"
    ]
    if comparable:
        aligned = sum(
            1 for row in comparable if row["expected_direction"] == row["observed_direction"]
        )
        lines.append("## Direction Alignment")
        lines.append("")
        lines.append(f"- aligned: {aligned}/{len(comparable)} ({aligned / len(comparable):.1%})")
        lines.append("")

    for probe_id, probe_rows in sorted(by_probe.items()):
        lines.append(f"## `{probe_id}`")
        lines.append("")

        first = probe_rows[0]
        added = first.get("added_dimensions", "")
        removed = first.get("removed_dimensions", "")
        targets = first.get("target_dimensions", "")
        lines.append(f"- added_dimensions: {added or 'n/a'}")
        lines.append(f"- removed_dimensions: {removed or 'n/a'}")
        lines.append(f"- target_dimensions: {targets or 'n/a'}")
        lines.append("")

        numeric_rows: list[tuple[float, dict[str, Any]]] = []
        for row in probe_rows:
            delta = parse_float(row.get("delta", ""))
            if delta is not None:
                numeric_rows.append((abs(delta), row))

        if numeric_rows:
            lines.append("| Rank | Feature | Delta | Direction | Note |")
            lines.append("|---:|---:|---:|---|---|")
            for _, row in sorted(numeric_rows, reverse=True)[:10]:
                rank = row.get("feature_rank", "")
                feature_id = row.get("feature_id", "")
                delta = row.get("delta", "")
                direction = row.get("observed_direction", "")
                note = row.get("note", "").replace("|", "/")
                lines.append(f"| {rank} | {feature_id} | {delta} | {direction} | {note} |")
            lines.append("")
        else:
            dimensions = [
                row for row in probe_rows if row.get("candidate_dimension")
            ]
            if dimensions:
                lines.append("| Dimension | Expected | Observed | Feature | Note |")
                lines.append("|---|---|---|---|---|")
                for row in dimensions:
                    dimension = row.get("candidate_dimension", "")
                    expected = row.get("expected_direction", "")
                    observed = row.get("observed_direction", "")
                    feature = row.get("feature_id", "")
                    note = row.get("note", "").replace("|", "/")
                    lines.append(f"| {dimension} | {expected} | {observed} | {feature} | {note} |")
                lines.append("")
            else:
                lines.append("_No numeric feature deltas available yet._")
                lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observations",
        type=Path,
        default=PROJECT_ROOT / "runs" / "sae_probe_observations.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "sae_observation_summary.md",
    )
    args = parser.parse_args()

    rows = read_csv(args.observations)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(summarize(rows), encoding="utf-8")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

