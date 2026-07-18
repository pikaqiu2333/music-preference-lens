from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_reason_swap_probe import (  # noqa: E402
    CONDITIONS,
    build_bundle,
    load_json,
    load_jsonl,
    normalize,
)
from run_reason_swap_probe import (  # noqa: E402
    aggregate_rows,
    render_pair_text,
    summarize_scores,
)


class ReasonSwapExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog_rows = load_jsonl(
            PROJECT_ROOT
            / "runs"
            / "qwen_scope_music_reason_order_catalog_verified.jsonl"
        )
        self.order_bundle = load_json(
            PROJECT_ROOT / "runs" / "qwen_scope_music_reason_order_bundle.json"
        )

    def test_bundle_has_all_reason_first_rows_and_conditions(self) -> None:
        bundle = build_bundle(self.catalog_rows, self.order_bundle)
        self.assertEqual(len(bundle["records"]), 20)
        self.assertEqual(bundle["conditions"], list(CONDITIONS))
        self.assertEqual(
            Counter(row["entity_group"] for row in bundle["records"]),
            Counter(bundle["registered_gates"]["entity_group_counts"]),
        )
        for row in bundle["records"]:
            self.assertEqual(
                set(row["condition_reason_record_ids"]),
                set(CONDITIONS) - {"neutral"},
            )
            reason_by_id = {
                item["record_id"]: item["reason"] for item in bundle["records"]
            }
            self.assertNotEqual(
                normalize(reason_by_id[row["condition_reason_record_ids"]["own"]]),
                normalize(
                    reason_by_id[
                        row["condition_reason_record_ids"]["same_context_swap"]
                    ]
                ),
            )
            self.assertNotIn(normalize(row["title"]), normalize(row["reason"]))

    def test_placeholder_catalog_labels_are_overridden(self) -> None:
        bundle = build_bundle(self.catalog_rows, self.order_bundle)
        placeholders = [
            row for row in bundle["records"] if row["entity_group"] == "invalid_placeholder"
        ]
        self.assertEqual(len(placeholders), 5)
        self.assertTrue(all(row["artist"] == "[Artist Name]" for row in placeholders))


class ReasonSwapRunnerTests(unittest.TestCase):
    def test_renderer_tracks_complete_pair(self) -> None:
        text, title_span, artist_span = render_pair_text(
            "1. Reason: Emotional release.\n   Title:",
            "Someone Like You",
            "Adele",
        )
        self.assertEqual(text[slice(*title_span)], "Someone Like You")
        self.assertEqual(text[slice(*artist_span)], "Adele")

    def test_summary_uses_signed_own_margins(self) -> None:
        result = summarize_scores(
            {
                "own": -1.0,
                "same_context_swap": -1.2,
                "opposite_context": -0.8,
                "neutral": -1.5,
            }
        )
        self.assertAlmostEqual(result["own_vs_same_context_swap_margin"], 0.2)
        self.assertTrue(result["own_beats_same_context_swap"])
        self.assertFalse(result["own_beats_opposite_context"])

    def test_aggregate_keeps_win_rates_and_margins(self) -> None:
        rows = []
        for margin in (0.2, -0.1):
            row = {}
            for condition in ("same_context_swap", "opposite_context", "neutral"):
                row[f"own_vs_{condition}_margin"] = margin
                row[f"own_beats_{condition}"] = margin > 0
            rows.append(row)
        summary = aggregate_rows(rows)
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["own_beats_same_context_swap_rate"], 0.5)
        self.assertAlmostEqual(summary["mean_own_vs_neutral_margin"], 0.05)


if __name__ == "__main__":
    unittest.main()
