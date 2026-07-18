from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_music_reason_component_patching_probe import (  # noqa: E402
    CASE_ROLES,
    SELECTED_LAYERS,
    build_bundle,
    load_json,
    load_jsonl,
)
from run_music_reason_component_patching_probe import (  # noqa: E402
    component_recovery,
    render_pair_text,
)


class MusicReasonComponentExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.faithfulness_bundle = load_json(
            PROJECT_ROOT
            / "runs"
            / "qwen_scope_music_reason_faithfulness_smoke_bundle.json"
        )
        self.result_rows = load_jsonl(
            PROJECT_ROOT
            / "runs"
            / "qwen_scope_music_reason_faithfulness_smoke_rows.jsonl"
        )

    def test_bundle_has_three_distinct_case_roles(self) -> None:
        bundle = build_bundle(self.faithfulness_bundle, self.result_rows)
        self.assertEqual(len(bundle["controls"]), 3)
        self.assertEqual(
            {row["case_role"] for row in bundle["controls"]},
            set(CASE_ROLES.values()),
        )
        self.assertEqual(bundle["selected_layers"], SELECTED_LAYERS)

    def test_failure_case_keeps_negative_need_effect(self) -> None:
        bundle = build_bundle(self.faithfulness_bundle, self.result_rows)
        failures = [
            row
            for row in bundle["controls"]
            if row["case_role"] == "verified_constraint_failure"
        ]
        self.assertEqual(len(failures), 1)
        self.assertLess(failures[0]["archived_need_effect"], 0)


class MusicReasonComponentRunnerTests(unittest.TestCase):
    def test_signed_recovery_handles_positive_and_negative_effects(self) -> None:
        self.assertAlmostEqual(component_recovery(-1.0, -2.0, -1.5), 0.5)
        self.assertAlmostEqual(component_recovery(-2.0, -1.0, -1.5), 0.5)

    def test_renderer_tracks_complete_pair(self) -> None:
        text, title_span, artist_span = render_pair_text(
            "Playlist:\n1. Title:",
            "Space Oddity",
            "David Bowie",
        )
        self.assertEqual(text[slice(*title_span)], "Space Oddity")
        self.assertEqual(text[slice(*artist_span)], "David Bowie")


if __name__ == "__main__":
    unittest.main()
