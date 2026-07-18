from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_controlled_vocality_reason_probe import (  # noqa: E402
    REASONS,
    build_bundle,
    load_jsonl,
    word_count,
)
from run_controlled_vocality_reason_probe import (  # noqa: E402
    aggregate_pair_rows,
    render_choice_prompt,
    render_pair_text,
    summarize_choice_rows,
)


class ControlledVocalityExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = load_jsonl(
            PROJECT_ROOT
            / "runs"
            / "controlled_vocality_tracks_catalog_verified.jsonl"
        )

    def test_bundle_is_balanced_exact_and_artist_disjoint(self) -> None:
        bundle = build_bundle(self.rows)
        self.assertEqual(len(bundle["records"]), 8)
        self.assertEqual(
            Counter(row["vocality"] for row in bundle["records"]),
            Counter({"instrumental": 4, "vocal": 4}),
        )
        self.assertTrue(
            all(row["catalog_label"] == "verified_exact" for row in bundle["records"])
        )
        self.assertEqual(len({row["artist"] for row in bundle["records"]}), 8)

    def test_reasons_are_equal_length_and_entity_free(self) -> None:
        self.assertEqual({word_count(reason) for reason in REASONS.values()}, {11})
        bundle = build_bundle(self.rows)
        for row in bundle["records"]:
            for reason in REASONS.values():
                self.assertNotIn(row["title"].casefold(), reason.casefold())
                self.assertNotIn(row["artist"].casefold(), reason.casefold())

    def test_candidate_orders_are_complete_reversals(self) -> None:
        bundle = build_bundle(self.rows)
        first, second = bundle["candidate_orders"]
        self.assertEqual(second, list(reversed(first)))
        self.assertEqual(set(first), {row["track_key"] for row in bundle["records"]})


class ControlledVocalityRunnerTests(unittest.TestCase):
    def test_pair_renderer_tracks_complete_entity(self) -> None:
        text, title_span, artist_span = render_pair_text(
            REASONS["vocal"],
            "Space Oddity",
            "David Bowie",
        )
        self.assertEqual(text[slice(*title_span)], "Space Oddity")
        self.assertEqual(text[slice(*artist_span)], "David Bowie")

    def test_choice_renderer_maps_all_letters(self) -> None:
        records = {
            f"k{index}": {"title": f"T{index}", "artist": f"A{index}"}
            for index in range(8)
        }
        prompt, mapping = render_choice_prompt(REASONS["neutral"], list(records), records)
        self.assertEqual(set(mapping), set("ABCDEFGH"))
        self.assertIn("Answer:", prompt)

    def test_pair_and_choice_summaries_keep_direction(self) -> None:
        pair = aggregate_pair_rows([{"matched_margin": 0.2}, {"matched_margin": -0.1}])
        self.assertEqual(pair["matched_direction_accuracy"], 0.5)
        rows = []
        values = {
            "vocal": [0.4, 0.2],
            "instrumental": [-0.3, -0.1],
            "neutral": [0.0, 0.0],
        }
        for condition, margins in values.items():
            for order_index, margin in enumerate(margins):
                rows.append(
                    {
                        "order_index": order_index,
                        "reason_condition": condition,
                        "vocal_minus_instrumental_margin": margin,
                    }
                )
        choice = summarize_choice_rows(rows)
        self.assertGreater(choice["vocal_reason_shift_from_neutral"], 0)
        self.assertGreater(choice["instrumental_reason_shift_from_neutral"], 0)
        self.assertTrue(choice["vocal_shift_positive_in_all_orders"])
        self.assertTrue(choice["instrumental_shift_positive_in_all_orders"])


if __name__ == "__main__":
    unittest.main()
