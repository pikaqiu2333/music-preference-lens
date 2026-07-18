from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from validate_publication import (  # noqa: E402
    find_broken_local_links,
    validate_publication,
)


class PublicationAssetTests(unittest.TestCase):
    def test_publication_bundle_is_internally_consistent(self) -> None:
        result = validate_publication(PROJECT_ROOT)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["local_link_failures"], 0)
        self.assertEqual(result["technical_gates_checked"], 11)
        self.assertEqual(result["publication_technical_gates_checked"], 5)
        self.assertEqual(result["exploratory_technical_gates_checked"], 6)
        self.assertEqual(result["hf_jobs_archived"], 5)
        self.assertEqual(result["phase2_hf_jobs_recorded"], 3)
        self.assertEqual(result["literature_sources_checked"], 24)
        self.assertEqual(result["literature_exclusions_checked"], 7)
        self.assertGreaterEqual(result["publication_files_hashed"], 70)

    def test_publication_runners_pin_runtime_and_accept_model_revision(self) -> None:
        runners = (
            "run_holdout_generation_probe.py",
            "run_independent_holdout_verifier_probe.py",
            "run_holdout_title_contrast_probe.py",
            "run_holdout_sequence_causal_trace.py",
        )
        for name in runners:
            source = (PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn('requires-python = ">=3.12,<3.13"', source)
            self.assertIn('"accelerate==1.8.1"', source)
            self.assertIn('parser.add_argument("--model-revision")', source)
            self.assertIn('os.environ.get("MODEL_REVISION")', source)

    def test_archived_job_metadata_uses_exact_model_revisions(self) -> None:
        metadata = json.loads(
            (PROJECT_ROOT / "runs" / "hf_job_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(metadata["jobs"]), 5)
        self.assertTrue(all(job["status"] == "COMPLETED" for job in metadata["jobs"]))
        self.assertEqual(
            metadata["model_revisions"]["Qwen/Qwen3-1.7B-Base"]["revision"],
            "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
        )
        self.assertEqual(
            metadata["model_revisions"]["Qwen/Qwen3-4B-Base"]["revision"],
            "906bfd4b4dc7f14ee4320094d8b41684abff8539",
        )

    def test_local_link_validator_ignores_web_and_detects_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "present.md").write_text("ok\n", encoding="utf-8")
            markdown = root / "index.md"
            markdown.write_text(
                "[present](present.md)\n"
                "[web](https://example.com)\n"
                "[missing](missing.md)\n",
                encoding="utf-8",
            )
            self.assertEqual(find_broken_local_links(markdown), ["missing.md"])

    def test_public_snapshot_omits_raw_third_party_response_archives(self) -> None:
        receipt = json.loads(
            (PROJECT_ROOT / "runs" / "private_evidence_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(receipt["distribution_status"], "not_publicly_redistributed")
        self.assertEqual(len(receipt["omitted_assets"]), 4)
        for row in receipt["omitted_assets"]:
            self.assertFalse((PROJECT_ROOT / row["path"]).exists())
            self.assertRegex(row["sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
