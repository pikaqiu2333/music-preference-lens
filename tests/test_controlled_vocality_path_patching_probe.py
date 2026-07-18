from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_controlled_vocality_path_patching_probe import (  # noqa: E402
    FOCUS_ROLES,
    build_bundle,
    load_jsonl,
)
from run_controlled_vocality_path_patching_probe import (  # noqa: E402
    recovery,
    render_choice_prompt,
    render_pair_text,
)


class ControlledVocalityPathPatchingExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.behavior_bundle = json.loads(
            (
                PROJECT_ROOT
                / "runs"
                / "controlled_vocality_reason_probe_bundle.json"
            ).read_text(encoding="utf-8")
        )
        self.behavior_rows = load_jsonl(
            PROJECT_ROOT
            / "runs"
            / "controlled_vocality_reason_pair_rows.jsonl"
        )

    def test_focus_is_balanced_across_class_and_behavior(self) -> None:
        bundle = build_bundle(self.behavior_bundle, self.behavior_rows)
        focus = bundle["focus_records"]
        self.assertEqual({row["track_key"] for row in focus}, set(FOCUS_ROLES))
        self.assertEqual(
            Counter(row["vocality"] for row in focus),
            Counter({"vocal": 2, "instrumental": 2}),
        )
        self.assertEqual(
            Counter("success" in row["sentinel_role"] for row in focus),
            Counter({True: 2, False: 2}),
        )
        self.assertTrue(
            all(abs(row["behavior_matched_margin"]) >= 0.10 for row in focus)
        )

    def test_candidate_orders_remain_complete_reversals(self) -> None:
        bundle = build_bundle(self.behavior_bundle, self.behavior_rows)
        first, second = bundle["candidate_orders"]
        self.assertEqual(second, list(reversed(first)))
        self.assertEqual(
            set(first),
            {row["track_key"] for row in bundle["all_records"]},
        )


class ControlledVocalityPathPatchingRunnerTests(unittest.TestCase):
    def test_recovery_preserves_signed_effect_direction(self) -> None:
        self.assertAlmostEqual(recovery(2.0, 1.0, 1.5), 0.5)
        self.assertAlmostEqual(recovery(1.0, 2.0, 1.5), 0.5)
        self.assertAlmostEqual(recovery(2.0, 1.0, 0.5), -0.5)
        self.assertAlmostEqual(recovery(1.0, 2.0, 2.5), -0.5)

    def test_pair_renderer_tracks_complete_title_and_artist(self) -> None:
        text, title_span, artist_span = render_pair_text(
            "A reason.",
            "Space Oddity",
            "David Bowie",
        )
        self.assertEqual(text[slice(*title_span)], "Space Oddity")
        self.assertEqual(text[slice(*artist_span)], "David Bowie")

    def test_choice_renderer_keeps_candidate_mapping(self) -> None:
        records = {
            "vocal": {"title": "Blinding Lights", "artist": "The Weeknd"},
            "instrumental": {"title": "Awake", "artist": "Tycho"},
        }
        prompt, mapping = render_choice_prompt(
            "A reason.",
            ["instrumental", "vocal"],
            records,
        )
        self.assertEqual(mapping, {"A": "instrumental", "B": "vocal"})
        self.assertIn("A. Awake - Tycho", prompt)
        self.assertTrue(prompt.endswith("Answer:"))


if __name__ == "__main__":
    unittest.main()
