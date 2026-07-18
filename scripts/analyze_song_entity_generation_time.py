"""Analyze generation-time grounding against catalog and control evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "mean_knownness": sum(row["generation_knownness"] for row in rows) / len(rows),
        "knownness_negative": sum(row["generation_knownness"] < 0 for row in rows),
        "mean_neutral_unknown_logit": sum(
            row["neutral_unknown_logit"] for row in rows
        )
        / len(rows),
        "neutral_unknown_positive": sum(
            row["neutral_unknown_logit"] > 0 for row in rows
        ),
        "mean_self_unknown_logit": sum(
            row["self_attributed_unknown_logit"] for row in rows
        )
        / len(rows),
        "self_unknown_positive": sum(
            row["self_attributed_unknown_logit"] > 0 for row in rows
        ),
        "mean_self_minus_neutral": sum(
            row["self_attributed_unknown_logit"] - row["neutral_unknown_logit"]
            for row in rows
        )
        / len(rows),
    }


def point_biserial(values: list[float], labels: list[int]) -> float:
    mean_value = sum(values) / len(values)
    mean_label = sum(labels) / len(labels)
    covariance = sum(
        (value - mean_value) * (label - mean_label)
        for value, label in zip(values, labels)
    )
    value_variance = sum((value - mean_value) ** 2 for value in values)
    label_variance = sum((label - mean_label) ** 2 for label in labels)
    denominator = math.sqrt(value_variance * label_variance)
    return covariance / denominator if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "qwen_scope_song_entity_generation_time_full_catalog_verified.jsonl",
    )
    parser.add_argument(
        "--run-summary",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "qwen_scope_song_entity_generation_time_full_summary.json",
    )
    parser.add_argument(
        "--verification-controls",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "qwen_scope_song_entity_pair_verification_control_summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "qwen_scope_song_entity_generation_time_analysis.json",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.rows)
    run_summary = load_json(args.run_summary)
    verification_controls = load_json(args.verification_controls)
    labels = Counter(row["catalog_label"] for row in rows)
    if labels.get("verification_error", 0):
        raise ValueError("catalog verification errors remain; retry them before analysis")

    by_label = {
        label: group_stats([row for row in rows if row["catalog_label"] == label])
        for label in sorted(labels)
    }
    by_context = {
        context: {
            "catalog_labels": dict(
                Counter(
                    row["catalog_label"]
                    for row in rows
                    if row["context_id"] == context
                )
            ),
            **group_stats([row for row in rows if row["context_id"] == context]),
        }
        for context in sorted({row["context_id"] for row in rows})
    }
    exact_labels = [int(row["catalog_label"] == "verified_exact") for row in rows]
    knownness = [row["generation_knownness"] for row in rows]
    generated_self_delta = group_stats(rows)["mean_self_minus_neutral"]
    control_self_delta = verification_controls["mean_self_minus_neutral"]

    result = {
        "row_count": len(rows),
        "unique_pair_count": len(
            {(row["title"].casefold(), row["artist"].casefold()) for row in rows}
        ),
        "catalog_labels": dict(labels),
        "verified_exact_rate": labels["verified_exact"] / len(rows),
        "by_catalog_label": by_label,
        "by_context": by_context,
        "knownness_catalog_point_biserial": point_biserial(knownness, exact_labels),
        "generated_self_minus_neutral": generated_self_delta,
        "control_self_minus_neutral": control_self_delta,
        "self_attribution_net_of_control": generated_self_delta - control_self_delta,
        "generation_technical_gate": run_summary["technical_gate"],
        "generation_knownness_control_balanced_accuracy": run_summary[
            "control_balanced_accuracy"
        ],
        "generation_knownness_interpretation_allowed": run_summary[
            "control_balanced_accuracy"
        ]
        >= 0.80,
        "verification_neutral_balanced_accuracy": verification_controls[
            "neutral_balanced_accuracy"
        ],
        "verification_self_balanced_accuracy": verification_controls[
            "self_balanced_accuracy"
        ],
        "verification_interpretation_allowed": min(
            verification_controls["neutral_balanced_accuracy"],
            verification_controls["self_balanced_accuracy"],
        )
        >= 0.80,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
