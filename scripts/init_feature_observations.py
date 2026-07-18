"""Create a feature-observation template for mechanistic probe runs."""

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


def expected_direction(dimension: str, spec: dict[str, Any]) -> str:
    if dimension in spec["added_dimensions"]:
        return "increase"
    if dimension in spec["removed_dimensions"]:
        return "decrease"
    return "inspect"


def observation_rows(specs: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in specs:
        for dimension in spec["target_dimensions"]:
            rows.append(
                {
                    "probe_id": spec["probe_id"],
                    "dimension_id": dimension,
                    "expected_direction": expected_direction(dimension, spec),
                    "model_key": "",
                    "base_model": "",
                    "sae_repo": "",
                    "layer": "",
                    "hook_point": "",
                    "feature_id": "",
                    "feature_label": "",
                    "original_activation": "",
                    "counterfactual_activation": "",
                    "observed_delta": "",
                    "observed_direction": "",
                    "evidence_note": "",
                    "confidence": "",
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--specs",
        type=Path,
        default=PROJECT_ROOT / "data" / "mechanistic_pilot_specs.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "feature_observations_template.csv",
    )
    args = parser.parse_args()

    specs = load_jsonl(args.specs)
    rows = observation_rows(specs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} observation rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

