from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_controlled_vocality_field_probe import (  # noqa: E402
    EXPECTED_HEADS,
    build_bundle,
)
from run_controlled_vocality_field_probe import (  # noqa: E402
    classify_failure_locus,
    normalized_recovery,
    render_pair_text,
)


class ControlledVocalityFieldExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.head_bundle = json.loads(
            (
                PROJECT_ROOT
                / "runs"
                / "controlled_vocality_attention_head_bundle.json"
            ).read_text(encoding="utf-8")
        )
        self.head_summary = json.loads(
            (
                PROJECT_ROOT
                / "runs"
                / "controlled_vocality_attention_head_summary.json"
            ).read_text(encoding="utf-8")
        )

    def test_bundle_freezes_heads_layer_and_scopes(self) -> None:
        bundle = build_bundle(self.head_bundle, self.head_summary)
        self.assertEqual(bundle["selected_heads"], EXPECTED_HEADS)
        self.assertEqual(bundle["target_layer"], 18)
        self.assertEqual(bundle["intervention_scopes"], ["title", "artist", "both"])
        self.assertEqual(len(bundle["focus_records"]), 4)

    def test_changed_head_selection_is_rejected(self) -> None:
        changed = {**self.head_summary, "choice_consistent_heads": [0, 1, 8]}
        with self.assertRaises(ValueError):
            build_bundle(self.head_bundle, changed)


class ControlledVocalityFieldRunnerTests(unittest.TestCase):
    def test_failure_locus_rule_covers_all_sign_patterns(self) -> None:
        self.assertEqual(classify_failure_locus(-0.2, 0.1, 0.03), "title_only")
        self.assertEqual(classify_failure_locus(0.2, -0.1, 0.03), "artist_only")
        self.assertEqual(classify_failure_locus(-0.2, -0.1, 0.03), "both_fields")
        self.assertEqual(classify_failure_locus(0.2, 0.1, 0.03), "neither_field")
        self.assertEqual(
            classify_failure_locus(-0.01, 0.1, 0.03),
            "title_only_weak_boundary",
        )

    def test_recovery_omits_weak_field_but_preserves_negative_direction(self) -> None:
        self.assertIsNone(normalized_recovery(1.01, 1.0, 1.005, 0.03))
        self.assertAlmostEqual(normalized_recovery(2.0, 1.0, 1.5, 0.03), 0.5)
        self.assertAlmostEqual(normalized_recovery(1.0, 2.0, 1.5, 0.03), 0.5)

    def test_renderer_returns_disjoint_complete_field_spans(self) -> None:
        text, spans = render_pair_text("A reason.", "Space Oddity", "David Bowie")
        self.assertEqual(text[slice(*spans["title"])], "Space Oddity")
        self.assertEqual(text[slice(*spans["artist"])], "David Bowie")
        self.assertLessEqual(spans["title"][1], spans["artist"][0])


if __name__ == "__main__":
    unittest.main()
