from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_song_entity_layerwise_attribution_probe import (  # noqa: E402
    build_bundle,
    select_smoke_controls as select_export_smoke_controls,
)
from export_song_entity_relation_binding_probe import load_jsonl  # noqa: E402
from run_song_entity_layerwise_attribution_probe import (  # noqa: E402
    find_earliest_sustained,
    overlapping_positions,
    relation_margin,
    select_smoke_controls as select_run_smoke_controls,
)


class LayerwiseAttributionExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = load_jsonl(
            PROJECT_ROOT
            / "data"
            / "qwen_scope_song_entity_relation_binding_controls.jsonl"
        )

    def test_smoke_reuses_six_complete_swap_blocks(self) -> None:
        export_rows = select_export_smoke_controls(self.rows)
        run_rows = select_run_smoke_controls(self.rows)
        self.assertEqual(export_rows, run_rows)
        self.assertEqual(len(run_rows), 12)
        blocks = {row["block_id"] for row in run_rows}
        self.assertEqual(len(blocks), 6)
        for block_id in blocks:
            self.assertEqual(
                sum(row["block_id"] == block_id for row in run_rows),
                2,
            )

    def test_bundle_separates_sequence_and_patch_tokenization(self) -> None:
        bundle = build_bundle(self.rows)
        self.assertTrue(bundle["likelihood_template"].endswith("Artist: "))
        self.assertTrue(bundle["patch_prefix_template"].endswith("Artist:"))
        self.assertEqual(bundle["behavior_threshold"], 0.80)
        self.assertEqual(bundle["consistency_tolerance"], 0.02)


class LayerwiseAttributionRunnerTests(unittest.TestCase):
    def test_relation_margin_subtracts_artist_prior(self) -> None:
        self.assertAlmostEqual(
            relation_margin(5.0, 2.0, 4.0, 3.0),
            2.0,
        )

    def test_earliest_sustained_requires_consecutive_depths(self) -> None:
        accuracies = [0.5, 0.75, 0.5, 0.75, 0.83, 0.92, 0.7]
        self.assertEqual(find_earliest_sustained(accuracies, 0.75, 3), 3)
        self.assertIsNone(find_earliest_sustained(accuracies, 0.90, 3))

    def test_overlap_ignores_empty_special_token_offsets(self) -> None:
        offsets = [(0, 0), (0, 5), (5, 6), (6, 9)]
        self.assertEqual(overlapping_positions(offsets, [6, 9]), [3])


if __name__ == "__main__":
    unittest.main()
