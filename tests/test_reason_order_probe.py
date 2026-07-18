from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_reason_order_probe import (  # noqa: E402
    ORDERS,
    build_bundle,
    load_jsonl,
)
from run_reason_order_probe import matched_overlap, parse_playlist  # noqa: E402
from analyze_reason_order_generations import analyze  # noqa: E402


class ReasonOrderExportTests(unittest.TestCase):
    def test_bundle_is_matched_except_for_order(self) -> None:
        contexts = load_jsonl(
            PROJECT_ROOT
            / "data"
            / "qwen_scope_song_entity_generation_time_specs.jsonl"
        )
        bundle = build_bundle(contexts)
        self.assertEqual(bundle["orders"], list(ORDERS))
        self.assertEqual(len(bundle["contexts"]), 2)
        self.assertEqual(bundle["seeds"], [17, 29])
        for context in bundle["contexts"]:
            self.assertIn(context["profile"], context["prompts"]["pair_first"])
            self.assertIn(context["profile"], context["prompts"]["reason_first"])
            self.assertIn(context["current_need"], context["prompts"]["pair_first"])
            self.assertIn(context["current_need"], context["prompts"]["reason_first"])


class ReasonOrderRunnerTests(unittest.TestCase):
    def test_parses_pair_first_and_checks_real_order(self) -> None:
        text = """Playlist:
1. Title:
All of Me
   Artist: John Legend
   Reason: Emotional vocal performance.
2. Title: Love Story
   Artist: Taylor Swift
   Reason: Clear lyrical story.
"""
        rows = parse_playlist(text, "pair_first")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["order_compliant"] for row in rows))

    def test_parses_reason_first_and_checks_real_order(self) -> None:
        text = """Playlist:
1. Reason:
Purely instrumental texture for writing.
   Title: An Ending (Ascent)
   Artist: Brian Eno
2. Reason: Quiet ambient focus.
   Title: Weightless
   Artist: Marconi Union
"""
        rows = parse_playlist(text, "reason_first")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["order_compliant"] for row in rows))

    def test_overlap_uses_complete_title_artist_pair(self) -> None:
        left = [{"title": "All of Me", "artist": "John Legend"}]
        right = [{"title": "All Of Me", "artist": "John Legend"}]
        overlap = matched_overlap(left, right)
        self.assertEqual(overlap["exact_pair_overlap_count"], 1)
        self.assertEqual(overlap["exact_pair_jaccard"], 1.0)

    def test_parses_inline_fields_and_title_by_artist_fallback(self) -> None:
        pair_first = """Playlist:
1. Title:
\"Invisible Light\" Artist: Billie Eilish Reason: Soft emotional vocals.
2. Title: \"Ocean Eyes\" Artist: Billie Eilish Reason: A dramatic chorus.
"""
        rows = parse_playlist(pair_first, "pair_first")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["title"], "Invisible Light")
        self.assertEqual(rows[0]["artist"], "Billie Eilish")

        reason_first = """Playlist:
1. Reason:
Purely instrumental focus. Title: \"Silent Melody\" by [Artist Name]
"""
        rows = parse_playlist(reason_first, "reason_first")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "Silent Melody")
        self.assertTrue(rows[0]["placeholder_artist"])
        self.assertFalse(rows[0]["explicit_artist_field"])

    def test_reanalysis_keeps_placeholder_failures_visible(self) -> None:
        contexts = load_jsonl(
            PROJECT_ROOT
            / "data"
            / "qwen_scope_song_entity_generation_time_specs.jsonl"
        )
        bundle = build_bundle(contexts)
        raw = []
        for context in bundle["contexts"]:
            for seed in bundle["seeds"]:
                for order in bundle["orders"]:
                    if order == "pair_first":
                        generation = (
                            ' "Weightless" Artist: Marconi Union '
                            "Reason: Quiet focus."
                        )
                    else:
                        generation = (
                            ' Quiet focus. Title: "Silent Melody" by [Artist Name]'
                        )
                    raw.append(
                        {
                            "generation_id": f"{context['context_id']}__seed{seed}__{order}",
                            "context_id": context["context_id"],
                            "seed": seed,
                            "order": order,
                            "raw_generation": generation,
                            "parsed_count": 0,
                            "valid_generation": False,
                            "rows": [],
                        }
                    )
        _, rows, summary = analyze(raw, bundle)
        self.assertEqual(len(rows), 8)
        self.assertEqual(summary["placeholder_artist_counts_by_order"]["reason_first"], 4)


if __name__ == "__main__":
    unittest.main()
