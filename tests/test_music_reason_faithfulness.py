from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_music_reason_faithfulness_probe import (  # noqa: E402
    CONDITIONS,
    build_bundle,
    load_jsonl,
    validate_inputs,
)
from run_music_reason_faithfulness_probe import (  # noqa: E402
    aggregate_rows,
    overlapping_positions,
    render_pair_text,
    summarize_record,
)


class ReasonFaithfulnessExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contexts = load_jsonl(
            PROJECT_ROOT / "data" / "qwen_scope_song_entity_generation_time_specs.jsonl"
        )
        self.counterfactuals = load_jsonl(
            PROJECT_ROOT / "data" / "qwen_scope_music_reason_counterfactuals.jsonl"
        )
        self.generated = load_jsonl(
            PROJECT_ROOT
            / "runs"
            / "qwen_scope_song_entity_generation_time_full_catalog_verified.jsonl"
        )
        self.template = (
            PROJECT_ROOT / "prompts" / "song_entity_generation_time_prompt.md"
        ).read_text(encoding="utf-8")

    def test_bundle_keeps_pair_labels_separate_from_conditions(self) -> None:
        validate_inputs(self.contexts, self.counterfactuals, self.generated)
        bundle = build_bundle(
            self.contexts,
            self.counterfactuals,
            self.generated,
            self.template,
        )
        self.assertEqual(bundle["conditions"], list(CONDITIONS))
        self.assertEqual(len(bundle["records"]), 53)
        self.assertNotIn(
            "recommendation_label",
            {key for row in bundle["records"] for key in row},
        )

    def test_smoke_has_six_exact_and_six_conflict_events(self) -> None:
        bundle = build_bundle(
            self.contexts,
            self.counterfactuals,
            self.generated,
            self.template,
        )
        smoke = [row for row in bundle["records"] if row["smoke"]]
        self.assertEqual(len(smoke), 12)
        self.assertEqual(
            Counter(row["catalog_label"] for row in smoke),
            Counter({"verified_exact": 6, "catalog_conflict": 6}),
        )
        self.assertEqual(len({row["record_id"] for row in smoke}), 12)
        self.assertTrue(all(row["smoke"] for row in smoke))

    def test_counterfactuals_change_need_but_preserve_profile(self) -> None:
        bundle = build_bundle(
            self.contexts,
            self.counterfactuals,
            self.generated,
            self.template,
        )
        for context in bundle["contexts"]:
            prompts = context["condition_prompts"]
            self.assertEqual(len(set(prompts.values())), 4)
            for prompt in prompts.values():
                self.assertIn(context["profile"], prompt)
            self.assertIn(context["current_need"], prompts["original"])
            self.assertNotIn(context["current_need"], prompts["opposite"])


class ReasonFaithfulnessRunnerTests(unittest.TestCase):
    def test_pair_renderer_tracks_complete_title_and_artist_spans(self) -> None:
        text, title_span, artist_span = render_pair_text(
            "Playlist:\n1. Title:",
            "Love on the Brain",
            "Coldplay",
        )
        self.assertEqual(text[title_span[0] : title_span[1]], "Love on the Brain")
        self.assertEqual(text[artist_span[0] : artist_span[1]], "Coldplay")
        self.assertLess(title_span[1], artist_span[0])

    def test_overlap_ignores_special_offsets(self) -> None:
        self.assertEqual(
            overlapping_positions([(0, 0), (0, 4), (4, 9)], [4, 9]),
            [2],
        )

    def test_summary_requires_opposite_below_both_equivalent_prompts(self) -> None:
        metrics = summarize_record(
            {
                "original": -1.0,
                "paraphrase": -1.1,
                "opposite": -1.5,
                "neutral": -1.3,
            }
        )
        self.assertTrue(metrics["opposite_below_both"])
        self.assertAlmostEqual(metrics["opposite_margin"], 0.4)
        self.assertAlmostEqual(metrics["paraphrase_shift"], 0.1)

    def test_aggregate_does_not_mix_catalog_labels(self) -> None:
        rows = [
            {
                "opposite_below_both": True,
                "neutral_below_both": False,
                "opposite_margin": 0.4,
                "neutral_margin": -0.1,
                "paraphrase_shift": 0.1,
            },
            {
                "opposite_below_both": False,
                "neutral_below_both": True,
                "opposite_margin": -0.2,
                "neutral_margin": 0.2,
                "paraphrase_shift": 0.2,
            },
        ]
        summary = aggregate_rows(rows)
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["opposite_below_both_rate"], 0.5)
        self.assertAlmostEqual(summary["median_opposite_margin"], 0.1)


if __name__ == "__main__":
    unittest.main()
