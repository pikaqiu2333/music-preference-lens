from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_controlled_vocality_attention_head_probe import build_bundle  # noqa: E402
from run_controlled_vocality_attention_head_probe import (  # noqa: E402
    head_bounds,
    recovery,
    render_choice_prompt,
    render_pair_text,
)


class ControlledVocalityAttentionHeadExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path_bundle = json.loads(
            (
                PROJECT_ROOT
                / "runs"
                / "controlled_vocality_path_patching_bundle.json"
            ).read_text(encoding="utf-8")
        )

    def test_bundle_freezes_layer_architecture_and_four_roles(self) -> None:
        bundle = build_bundle(self.path_bundle)
        self.assertEqual(bundle["target_layer"], 18)
        self.assertEqual(bundle["expected_num_attention_heads"], 16)
        self.assertEqual(bundle["expected_head_dim"], 128)
        self.assertEqual(len(bundle["focus_records"]), 4)
        self.assertEqual(
            Counter("success" in row["sentinel_role"] for row in bundle["focus_records"]),
            Counter({True: 2, False: 2}),
        )
        self.assertTrue(
            all(
                abs(row["behavior_matched_margin"]) >= 0.10
                for row in bundle["focus_records"]
            )
        )

    def test_candidate_orders_remain_reversed(self) -> None:
        bundle = build_bundle(self.path_bundle)
        first, second = bundle["candidate_orders"]
        self.assertEqual(second, list(reversed(first)))


class ControlledVocalityAttentionHeadRunnerTests(unittest.TestCase):
    def test_head_bounds_partition_concatenated_width(self) -> None:
        spans = [head_bounds(head, 128, 16, 2048) for head in range(16)]
        self.assertEqual(spans[0], (0, 128))
        self.assertEqual(spans[-1], (1920, 2048))
        self.assertEqual(
            [end - start for start, end in spans],
            [128] * 16,
        )
        with self.assertRaises(ValueError):
            head_bounds(16, 128, 16, 2048)
        with self.assertRaises(ValueError):
            head_bounds(0, 128, 16, 1024)

    def test_recovery_handles_negative_behavior_effect(self) -> None:
        self.assertAlmostEqual(recovery(2.0, 1.0, 1.25), 0.25)
        self.assertAlmostEqual(recovery(1.0, 2.0, 1.75), 0.25)

    def test_renderers_preserve_complete_real_entities_and_mapping(self) -> None:
        pair, title_span, artist_span = render_pair_text(
            "A reason.", "Space Oddity", "David Bowie"
        )
        self.assertEqual(pair[slice(*title_span)], "Space Oddity")
        self.assertEqual(pair[slice(*artist_span)], "David Bowie")
        records = {
            "vocal": {"title": "Blinding Lights", "artist": "The Weeknd"},
            "instrumental": {"title": "Awake", "artist": "Tycho"},
        }
        prompt, mapping = render_choice_prompt(
            "A reason.", ["vocal", "instrumental"], records
        )
        self.assertEqual(mapping, {"A": "vocal", "B": "instrumental"})
        self.assertIn("A. Blinding Lights - The Weeknd", prompt)


if __name__ == "__main__":
    unittest.main()
