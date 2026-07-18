from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_free_generated_relation_conflict_probe import (  # noqa: E402
    build_bundle,
    load_json,
    load_jsonl,
    select_transfer_rows,
)
from run_free_generated_relation_conflict_probe import (  # noqa: E402
    append_continuation,
    binary_metrics,
    classify_conflict,
    normalized_recovery,
    render_choice_prompt,
    validation_status,
)


class FreeGeneratedRelationConflictExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog_rows = load_jsonl(
            PROJECT_ROOT
            / "runs"
            / "qwen_scope_song_entity_generation_time_full_catalog_verified.jsonl"
        )
        self.generation_bundle = load_json(
            PROJECT_ROOT / "runs" / "qwen_scope_song_entity_generation_time_bundle.json"
        )
        self.raw_generations = load_jsonl(
            PROJECT_ROOT / "runs" / "qwen_scope_song_entity_generation_format_raw.jsonl"
        )

    def test_selection_is_balanced_and_context_matched(self) -> None:
        selected = select_transfer_rows(self.catalog_rows)
        labels = Counter(row["catalog_label"] for row in selected)
        self.assertEqual(labels, Counter({"verified_exact": 6, "catalog_conflict": 6}))
        exact_contexts = Counter(
            row["context_id"]
            for row in selected
            if row["catalog_label"] == "verified_exact"
        )
        conflict_contexts = Counter(
            row["context_id"]
            for row in selected
            if row["catalog_label"] == "catalog_conflict"
        )
        self.assertEqual(exact_contexts, conflict_contexts)

    def test_deterministic_conflicts_follow_catalog_support_rule(self) -> None:
        selected = select_transfer_rows(self.catalog_rows)
        conflict_pairs = {
            (row["title"], row["artist"], row["reference_catalog"]["artist"])
            for row in selected
            if row["catalog_label"] == "catalog_conflict"
        }
        self.assertEqual(
            conflict_pairs,
            {
                ("Skinny Love", "Sam Smith", "Bon Iver"),
                ("Love on the Brain", "Coldplay", "Rihanna"),
                ("Everything In Its Right Place", "Kate Bush", "Radiohead"),
                ("River Flows in You", "Coldplay", "Yiruma"),
                ("Electric Youth", "A Tribe Called Quest", "Debbie Gibson"),
                ("Bittersweet", "Sam Smith", "Fuel"),
            },
        )

    def test_bundle_reconstructs_all_original_prefixes(self) -> None:
        bundle = build_bundle(
            self.catalog_rows, self.generation_bundle, self.raw_generations
        )
        self.assertEqual(len(bundle["records"]), 12)
        for row in bundle["records"]:
            self.assertTrue(row["generation_prefix"].strip())
            self.assertFalse(row["generation_prefix"].endswith(row["emitted_artist"]))
            self.assertNotEqual(
                row["emitted_artist"].casefold(), row["reference_artist"].casefold()
            )


class FreeGeneratedRelationConflictRunnerTests(unittest.TestCase):
    def test_append_continuation_tracks_artist_span(self) -> None:
        text, span = append_continuation("Title: X\nArtist:  ", "The Weeknd")
        self.assertEqual(text[slice(*span)], "The Weeknd")
        self.assertEqual(text, "Title: X\nArtist: The Weeknd")

    def test_choice_renderer_preserves_roles_under_reversal(self) -> None:
        first, mapping_first = render_choice_prompt(
            "Skinny Love", "Sam Smith", "Bon Iver", ["emitted", "reference"]
        )
        second, mapping_second = render_choice_prompt(
            "Skinny Love", "Sam Smith", "Bon Iver", ["reference", "emitted"]
        )
        self.assertEqual(mapping_first, {"A": "emitted", "B": "reference"})
        self.assertEqual(mapping_second, {"A": "reference", "B": "emitted"})
        self.assertNotEqual(first, second)

    def test_conflict_categories_cover_sampling_and_context_override(self) -> None:
        self.assertEqual(classify_conflict(0.1, -0.2), "relation_not_recovered")
        self.assertEqual(classify_conflict(-0.1, -0.2), "lower_probability_sample")
        self.assertEqual(classify_conflict(-0.1, 0.2), "context_override")

    def test_validation_status_uses_both_frozen_paths(self) -> None:
        rows = []
        for index in range(6):
            rows.append(
                {
                    "catalog_label": "verified_exact",
                    "catalog": 1.0 if index < 5 else -1.0,
                    "choice": 1.0 if index < 5 else -1.0,
                }
            )
            rows.append(
                {
                    "catalog_label": "catalog_conflict",
                    "catalog": -1.0 if index < 5 else 1.0,
                    "choice": -1.0 if index < 5 else 1.0,
                }
            )
        catalog = binary_metrics(rows, "catalog")
        choice = binary_metrics(rows, "choice")
        self.assertEqual(validation_status(catalog, choice, 0.75, 4), "validated_small_pilot")
        weak_choice = {**choice, "balanced_accuracy": 0.5, "conflict_correct": 3}
        self.assertEqual(
            validation_status(catalog, weak_choice, 0.75, 4),
            "promising_single_path",
        )

    def test_recovery_omits_small_source_target_difference(self) -> None:
        self.assertIsNone(normalized_recovery(0.05, 0.0, 0.02, 0.10))
        self.assertAlmostEqual(normalized_recovery(-1.0, 1.0, 0.0, 0.10), 0.5)


if __name__ == "__main__":
    unittest.main()
