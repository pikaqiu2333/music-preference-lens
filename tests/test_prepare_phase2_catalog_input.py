from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from prepare_phase2_catalog_input import prepare_catalog_rows  # noqa: E402


class PreparePhase2CatalogInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hashes = {"json_sha256": "a", "markdown_sha256": "b"}
        self.bundle = {
            "protocol_id": "phase2",
            "protocol_hashes": self.hashes,
            "model_id": "granite",
            "model_revision": "revision",
            "contexts": [
                {
                    "context_id": "context",
                    "generation_prompt": "Playlist:\n1. Title:",
                }
            ],
            "seeds": [139],
        }
        self.artifact = {
            "summary": {
                "mode": "primary",
                "protocol_id": "phase2",
                "protocol_hashes": self.hashes,
                "model_id": "granite",
                "model_revision": "revision",
                "technical_gate": True,
                "prompt_template_id": "primary",
                "run_id": "run",
            },
            "rows": [
                {
                    "generation_id": "context__seed139",
                    "context_id": "context",
                    "seed": 139,
                    "parsed_track_count": 2,
                    "completion": (
                        " Song A\nArtist: Artist A\nReason: Reason A.\n"
                        "2. Title: Song B | Artist: Artist B | Reason: Reason B."
                    ),
                }
            ],
        }

    def test_parses_and_provenances_every_track(self) -> None:
        rows, summary = prepare_catalog_rows(self.artifact, self.bundle)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["record_id"], "context__seed139__rank1")
        self.assertEqual(rows[0]["normalized_title"], "songa")
        self.assertEqual(rows[1]["artist"], "Artist B")
        self.assertEqual(summary["parsed_event_count"], 2)
        self.assertEqual(summary["unique_normalized_title_count"], 2)
        self.assertTrue(summary["record_sha256_unique"])

    def test_rejects_parser_count_drift(self) -> None:
        self.artifact["rows"][0]["parsed_track_count"] = 1
        with self.assertRaisesRegex(ValueError, "parser count drift"):
            prepare_catalog_rows(self.artifact, self.bundle)

    def test_rejects_unregistered_generation_id(self) -> None:
        self.artifact["rows"][0]["generation_id"] = "other__seed139"
        with self.assertRaisesRegex(ValueError, "generation IDs"):
            prepare_catalog_rows(self.artifact, self.bundle)

    def test_rejects_protocol_hash_mismatch(self) -> None:
        self.artifact["summary"]["protocol_hashes"] = {
            "json_sha256": "wrong",
            "markdown_sha256": "b",
        }
        with self.assertRaisesRegex(ValueError, "protocol hashes"):
            prepare_catalog_rows(self.artifact, self.bundle)

    def test_rejects_smoke_artifact(self) -> None:
        self.artifact["summary"]["mode"] = "smoke"
        with self.assertRaisesRegex(ValueError, "scientific generation"):
            prepare_catalog_rows(self.artifact, self.bundle)


if __name__ == "__main__":
    unittest.main()
