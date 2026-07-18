from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_embedded_hf_job import extract_embedded_bundle  # noqa: E402
from export_phase3_natural_generation import build_bundle  # noqa: E402
from run_phase3_natural_generation import parse_playlist, summarize_rows  # noqa: E402


class Phase3NaturalGenerationExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol_path = (
            PROJECT_ROOT
            / "config"
            / "phase3_natural_relation_discovery_protocol.json"
        )
        cls.doc_path = (
            PROJECT_ROOT / "docs" / "phase3_natural_relation_discovery_protocol.md"
        )
        cls.protocol_bytes = cls.protocol_path.read_bytes()
        cls.doc_bytes = cls.doc_path.read_bytes()
        cls.protocol = json.loads(cls.protocol_bytes.decode("utf-8"))

    def build(self) -> dict:
        return build_bundle(
            self.protocol,
            self.protocol_bytes,
            self.doc_bytes,
        )

    def test_pilot_bundle_has_only_frozen_discovery_rows(self) -> None:
        bundle = self.build()
        self.assertEqual(bundle["mode"], "pilot")
        self.assertEqual(bundle["expected_generation_count"], 72)
        self.assertEqual(bundle["maximum_parsed_track_count"], 360)
        self.assertEqual(bundle["minimum_parsed_track_count"], 300)
        self.assertEqual(bundle["seeds"], self.protocol["generation"]["pilot_seeds"])
        self.assertEqual(len(bundle["contexts"]), 12)
        self.assertNotIn(
            self.protocol["generation"]["holdout_prompt_template"],
            {row["generation_prompt"] for row in bundle["contexts"]},
        )

    def test_generated_bundle_and_embedded_runner_match_sources(self) -> None:
        bundle_path = (
            PROJECT_ROOT / "runs" / "phase3_natural_pilot_generation_bundle.json"
        )
        stored = json.loads(bundle_path.read_text(encoding="utf-8"))
        self.assertEqual(stored, self.build())
        runner_path = PROJECT_ROOT / "scripts" / "run_phase3_natural_generation.py"
        embedded_path = (
            PROJECT_ROOT
            / "runs"
            / "jobs"
            / "run_phase3_natural_pilot_generation_embedded.py"
        )
        self.assertEqual(
            extract_embedded_bundle(
                runner_path.read_text(encoding="utf-8"),
                embedded_path.read_text(encoding="utf-8"),
                "__PHASE3_NATURAL_GENERATION_BUNDLE_B64_ZLIB__",
                "zlib",
            ),
            bundle_path.read_bytes().strip(),
        )


class Phase3NaturalGenerationRunnerTests(unittest.TestCase):
    def test_parser_recovers_seeded_first_row_and_five_lines(self) -> None:
        completion = (
            " First Song | Artist: First Artist | Reason: One.\n"
            "2. Title: Second Song | Artist: Second Artist | Reason: Two.\n"
            "3. Title: Third Song | Artist: Third Artist | Reason: Three.\n"
            "4. Title: Fourth Song | Artist: Fourth Artist | Reason: Four.\n"
            "5. Title: Fifth Song | Artist: Fifth Artist | Reason: Five."
        )
        parsed = parse_playlist("1. Title:" + completion)
        self.assertEqual(len(parsed), 5)
        self.assertEqual(parsed[0]["title"], "First Song")
        self.assertEqual(parsed[-1]["artist"], "Fifth Artist")

    def test_parser_does_not_need_the_instructional_prompt(self) -> None:
        prompt = (
            "Use N. Title: <title> | Artist: <artist> | Reason: <reason>.\n"
            "Playlist:\n1. Title:"
        )
        completion = " Real Song | Artist: Real Artist | Reason: Real reason."
        self.assertEqual(
            parse_playlist("1. Title:" + completion),
            [
                {
                    "title": "Real Song",
                    "artist": "Real Artist",
                    "reason": "Real reason.",
                }
            ],
        )
        self.assertEqual(prompt.rsplit("Playlist:\n", 1)[-1], "1. Title:")

    def test_summary_applies_only_the_frozen_parse_gate(self) -> None:
        bundle = {
            "expected_generation_count": 2,
            "minimum_parsed_track_count": 8,
            "maximum_parsed_track_count": 10,
        }
        rows = [
            {
                "generation_id": "a",
                "context_id": "one",
                "completion": "ok",
                "parsed_track_count": 4,
            },
            {
                "generation_id": "b",
                "context_id": "two",
                "completion": "ok",
                "parsed_track_count": 4,
            },
        ]
        self.assertTrue(summarize_rows(rows, bundle, True)["technical_gate"])
        rows[1]["parsed_track_count"] = 3
        self.assertFalse(summarize_rows(rows, bundle, True)["technical_gate"])


if __name__ == "__main__":
    unittest.main()
