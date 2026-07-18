"""Export mechanistic pilot specs into pairwise prompt files.

The output is intentionally simple:

- JSONL for programmatic notebooks.
- CSV for manual inspection or spreadsheet notes.
- Markdown table for reports.
"""

from __future__ import annotations

import argparse
import csv
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


def pair_rows(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        base = {
            "probe_id": spec["probe_id"],
            "source_case_id": spec["source_case_id"],
            "variant_id": spec["variant_id"],
            "variant_type": spec["variant_type"],
            "target_dimensions": spec["target_dimensions"],
            "added_dimensions": spec["added_dimensions"],
            "removed_dimensions": spec["removed_dimensions"],
            "expected_effect": spec["expected_effect"],
            "candidate_focus": spec["candidate_focus"],
        }
        rows.append(
            {
                **base,
                "side": "original",
                "text": spec["original_text"],
            }
        )
        rows.append(
            {
                **base,
                "side": "counterfactual",
                "text": spec["counterfactual_text"],
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "probe_id",
        "side",
        "variant_type",
        "target_dimensions",
        "added_dimensions",
        "removed_dimensions",
        "expected_effect",
        "text",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "probe_id": row["probe_id"],
                    "side": row["side"],
                    "variant_type": row["variant_type"],
                    "target_dimensions": ", ".join(row["target_dimensions"]),
                    "added_dimensions": ", ".join(row["added_dimensions"]),
                    "removed_dimensions": ", ".join(row["removed_dimensions"]),
                    "expected_effect": row["expected_effect"],
                    "text": row["text"],
                }
            )


def write_markdown(path: Path, specs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Mechanistic Probe Pair Table",
        "",
        "| Probe | Added | Removed | Expected Effect |",
        "|---|---|---|---|",
    ]
    for spec in specs:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{spec['probe_id']}`",
                    ", ".join(f"`{d}`" for d in spec["added_dimensions"]) or "none",
                    ", ".join(f"`{d}`" for d in spec["removed_dimensions"]) or "none",
                    spec["expected_effect"].replace("|", "/"),
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
        default=PROJECT_ROOT / "data" / "mechanistic_pilot_specs.jsonl",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=PROJECT_ROOT / "runs" / "mechanistic_probe_pairs.jsonl",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "runs" / "mechanistic_probe_pairs.csv",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=PROJECT_ROOT / "reports" / "mechanistic_probe_pair_table.md",
    )
    args = parser.parse_args()

    specs = load_jsonl(args.specs)
    rows = pair_rows(specs)
    write_jsonl(args.jsonl, rows)
    write_csv(args.csv, rows)
    write_markdown(args.markdown, specs)
    print(f"Wrote {len(rows)} pair rows")
    print(f"- {args.jsonl}")
    print(f"- {args.csv}")
    print(f"- {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

