from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from prepare_phase3_catalog_input import prepare_catalog_rows  # noqa: E402


def fixture() -> tuple[dict, dict]:
    bundle = {
        "protocol_id": "phase3",
        "protocol_hashes": {"json_sha256": "a", "markdown_sha256": "b"},
        "mode": "pilot",
        "prompt_template_id": "discovery_v1",
        "model_id": "model",
        "model_revision": "revision",
        "contexts": [{"context_id": "context"}],
        "seeds": [11],
        "completion_prefix": "1. Title:",
        "generation": {"tracks_per_playlist": 5},
        "minimum_parsed_track_count": 1,
    }
    completion = " Song | Artist: Artist | Reason: Reason."
    summary = {
        **{
            key: bundle[key]
            for key in (
                "protocol_id",
                "protocol_hashes",
                "mode",
                "prompt_template_id",
                "model_id",
                "model_revision",
            )
        },
        "run_id": "run",
        "parsed_track_count": 1,
        "technical_gate": True,
    }
    artifact = {
        "rows": [
            {
                "generation_id": "context__seed11",
                "context_id": "context",
                "seed": 11,
                "completion": completion,
                "parsed_track_count": 1,
                "parsed_tracks": [
                    {"title": "Song", "artist": "Artist", "reason": "Reason."}
                ],
            }
        ],
        "summary": summary,
    }
    return artifact, bundle


class PreparePhase3CatalogInputTests(unittest.TestCase):
    def test_replays_every_track_and_binds_provenance(self) -> None:
        rows, summary = prepare_catalog_rows(*fixture())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["record_id"], "context__seed11__rank1")
        self.assertEqual(rows[0]["normalized_title"], "song")
        self.assertEqual(rows[0]["batch_mode"], "phase3_pilot")
        self.assertEqual(len(rows[0]["generation_row_sha256"]), 64)
        self.assertTrue(summary["parser_replay_gate"])
        self.assertTrue(summary["technical_gate"])

    def test_rejects_changed_generation_ids(self) -> None:
        artifact, bundle = fixture()
        artifact["rows"][0]["generation_id"] = "other__seed11"
        with self.assertRaisesRegex(ValueError, "generation IDs"):
            prepare_catalog_rows(artifact, bundle)

    def test_rejects_parser_replay_drift(self) -> None:
        artifact, bundle = fixture()
        artifact["rows"][0]["parsed_tracks"][0]["artist"] = "Changed"
        with self.assertRaisesRegex(ValueError, "parser replay drift"):
            prepare_catalog_rows(artifact, bundle)


if __name__ == "__main__":
    unittest.main()
