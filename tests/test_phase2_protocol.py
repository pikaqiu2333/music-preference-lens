from __future__ import annotations

import base64
import hashlib
import json
import re
import sys
import unittest
import zlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
SMOKE_BUNDLE = PROJECT_ROOT / "runs" / "phase2_granite_smoke_generation_bundle.json"
SMOKE_JOB = (
    PROJECT_ROOT / "runs" / "jobs" / "run_phase2_granite_smoke_generation_embedded.py"
)
PRIMARY_BUNDLE = (
    PROJECT_ROOT / "runs" / "phase2_granite_primary_generation_bundle.json"
)
PRIMARY_JOB = (
    PROJECT_ROOT / "runs" / "jobs" / "run_phase2_granite_primary_generation_embedded.py"
)
EXTENSION_BUNDLE = (
    PROJECT_ROOT / "runs" / "phase2_granite_extension_generation_bundle.json"
)
EXTENSION_JOB = (
    PROJECT_ROOT
    / "runs"
    / "jobs"
    / "run_phase2_granite_extension_generation_embedded.py"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from export_phase2_generation_probe import (  # noqa: E402
    DEFAULT_PROTOCOL,
    DEFAULT_PROTOCOL_DOC,
    build_bundle,
    generation_prompt,
)
from run_phase2_generation_probe import (  # noqa: E402
    encode_artifact_chunks,
    parse_playlist,
    validate_protocol_payloads,
)


class Phase2ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol_bytes = DEFAULT_PROTOCOL.read_bytes()
        cls.protocol_doc_bytes = DEFAULT_PROTOCOL_DOC.read_bytes()
        cls.protocol = json.loads(cls.protocol_bytes.decode("utf-8"))

    def test_protocol_freezes_cross_architecture_model_and_stop_rules(self) -> None:
        model = self.protocol["models"]["confirmatory"]
        self.assertEqual(model["model_id"], "ibm-granite/granite-4.1-3b-base")
        self.assertEqual(len(model["revision"]), 40)
        self.assertEqual(model["expected_num_layers"], 40)
        self.assertTrue(self.protocol["stop_conditions"]["forbid_third_model_rescue"])
        self.assertTrue(
            self.protocol["stop_conditions"][
                "forbid_post_result_threshold_prompt_layer_or_sample_changes"
            ]
        )

    def test_generation_counts_and_contexts_are_frozen(self) -> None:
        generation = self.protocol["generation"]
        primary_ids = [row["context_id"] for row in generation["primary_contexts"]]
        extension_ids = [
            row["context_id"] for row in generation["registered_extension_contexts"]
        ]
        self.assertEqual(len(primary_ids), 12)
        self.assertEqual(len(extension_ids), 4)
        self.assertEqual(len(set(primary_ids + extension_ids)), 16)
        self.assertEqual(len(generation["seeds"]), 5)
        self.assertEqual(generation["primary_playlist_count"], 60)
        self.assertEqual(generation["extension_playlist_count"], 20)
        self.assertEqual(generation["maximum_parsed_event_count"], 400)

    def test_diagnostic_and_intervention_controls_are_disjoint(self) -> None:
        diagnosis = self.protocol["mechanism_diagnosis"]
        self.assertEqual(diagnosis["diagnostic_neutral_title_count"], 3)
        self.assertEqual(diagnosis["heldout_manipulation_neutral_title_count"], 3)
        self.assertEqual(diagnosis["layers"], [20, 28, 36])
        self.assertTrue(diagnosis["label_requires_template_agreement"])
        self.assertEqual(
            self.protocol["hypothesis_gates"][
                "h2_minimum_normalized_title_clusters_per_class"
            ],
            15,
        )
        self.assertEqual(
            len(self.protocol["correction"]["naive_candidate_free_templates"]), 2
        )
        self.assertEqual(
            len(self.protocol["correction"]["anti_prior_candidate_free_templates"]),
            2,
        )

    def test_smoke_is_disjoint_and_never_uploaded(self) -> None:
        smoke = self.protocol["technical_smoke"]
        scientific_ids = {
            row["context_id"]
            for key in ("primary_contexts", "registered_extension_contexts")
            for row in self.protocol["generation"][key]
        }
        self.assertNotIn(smoke["context"]["context_id"], scientific_ids)
        self.assertNotIn(smoke["seed"], self.protocol["generation"]["seeds"])
        self.assertTrue(smoke["forbid_raw_artifact_upload"])
        self.assertEqual(
            smoke["prompt_attempt_order"], ["primary", "single_format_fallback"]
        )

    def test_generated_smoke_artifacts_match_current_protocol(self) -> None:
        bundle = json.loads(SMOKE_BUNDLE.read_text(encoding="utf-8"))
        self.assertEqual(bundle["contexts"][0]["context_id"], "format_only_smoke")
        self.assertEqual(bundle["seeds"], [997])
        self.assertEqual(bundle["prompt_template_id"], "primary")
        self.assertIsNone(bundle["output_repo"])
        self.assertEqual(
            bundle["protocol_hashes"]["json_sha256"],
            hashlib.sha256(self.protocol_bytes).hexdigest(),
        )
        self.assertEqual(
            bundle["protocol_hashes"]["markdown_sha256"],
            hashlib.sha256(self.protocol_doc_bytes).hexdigest(),
        )

        job_text = SMOKE_JOB.read_text(encoding="utf-8")
        match = re.search(r'^EMBEDDED_BUNDLE_B64 = "([A-Za-z0-9+/=]+)"$', job_text, re.MULTILINE)
        self.assertIsNotNone(match)
        embedded = json.loads(base64.b64decode(match.group(1)).decode("utf-8"))
        self.assertEqual(embedded, bundle)

    def test_generated_primary_artifacts_match_current_protocol(self) -> None:
        bundle = json.loads(PRIMARY_BUNDLE.read_text(encoding="utf-8"))
        self.assertEqual(bundle["mode"], "primary")
        self.assertEqual(bundle["expected_generation_count"], 60)
        self.assertEqual(bundle["seeds"], self.protocol["generation"]["seeds"])
        self.assertEqual(
            [row["context_id"] for row in bundle["contexts"]],
            [
                row["context_id"]
                for row in self.protocol["generation"]["primary_contexts"]
            ],
        )
        self.assertFalse(bundle["redact_completions"])
        self.assertIsNone(bundle["output_repo"])
        self.assertEqual(
            bundle["protocol_hashes"]["json_sha256"],
            hashlib.sha256(self.protocol_bytes).hexdigest(),
        )
        self.assertEqual(
            bundle["protocol_hashes"]["markdown_sha256"],
            hashlib.sha256(self.protocol_doc_bytes).hexdigest(),
        )

        job_text = PRIMARY_JOB.read_text(encoding="utf-8")
        match = re.search(
            r'^EMBEDDED_BUNDLE_B64 = "([A-Za-z0-9+/=]+)"$',
            job_text,
            re.MULTILINE,
        )
        self.assertIsNotNone(match)
        embedded = json.loads(base64.b64decode(match.group(1)).decode("utf-8"))
        self.assertEqual(embedded, bundle)

    def test_generated_extension_artifacts_match_current_protocol(self) -> None:
        bundle = json.loads(EXTENSION_BUNDLE.read_text(encoding="utf-8"))
        self.assertEqual(bundle["mode"], "extension")
        self.assertEqual(bundle["expected_generation_count"], 20)
        self.assertEqual(bundle["seeds"], self.protocol["generation"]["seeds"])
        self.assertEqual(
            [row["context_id"] for row in bundle["contexts"]],
            [
                row["context_id"]
                for row in self.protocol["generation"][
                    "registered_extension_contexts"
                ]
            ],
        )
        self.assertFalse(bundle["redact_completions"])
        self.assertIsNone(bundle["output_repo"])
        self.assertEqual(
            bundle["protocol_hashes"]["json_sha256"],
            hashlib.sha256(self.protocol_bytes).hexdigest(),
        )

        job_text = EXTENSION_JOB.read_text(encoding="utf-8")
        match = re.search(
            r'^EMBEDDED_BUNDLE_B64 = "([A-Za-z0-9+/=]+)"$',
            job_text,
            re.MULTILINE,
        )
        self.assertIsNotNone(match)
        embedded = json.loads(base64.b64decode(match.group(1)).decode("utf-8"))
        self.assertEqual(embedded, bundle)

    def test_h2_averages_two_independent_paraphrase_indicators(self) -> None:
        scoring = self.protocol["correction"]["candidate_free_response_scoring"]
        self.assertIn("arithmetic mean", scoring["per_title_condition_accuracy"])
        self.assertIn("0, 0.5, or 1", scoring["per_title_condition_accuracy"])

    def test_bundle_counts_and_embedded_protocol_hashes(self) -> None:
        primary = build_bundle(
            self.protocol,
            "primary",
            self.protocol_bytes,
            self.protocol_doc_bytes,
        )
        smoke = build_bundle(
            self.protocol,
            "smoke",
            self.protocol_bytes,
            self.protocol_doc_bytes,
        )
        self.assertEqual(primary["expected_generation_count"], 60)
        self.assertEqual(smoke["expected_generation_count"], 1)
        self.assertFalse(primary["redact_completions"])
        self.assertTrue(smoke["redact_completions"])
        self.assertEqual(validate_protocol_payloads(primary), primary["protocol_hashes"])

    def test_prompt_and_parser_round_trip(self) -> None:
        prompt = generation_prompt(
            self.protocol["generation"]["primary_contexts"][0],
            self.protocol["generation"]["prompt_templates"]["primary"],
        )
        completion = (
            " Example Song\nArtist: Example Artist\nReason: A concise reason.\n"
            "2. Title: Second Song\nArtist: Second Artist\nReason: Another reason.\n"
        )
        parsed = parse_playlist(prompt + completion)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["title"], "Example Song")
        self.assertEqual(parsed[1]["artist"], "Second Artist")

        inline = parse_playlist(
            "1. Title: Pipe Song | Artist: Pipe Artist | Reason: Short reason."
        )
        self.assertEqual(inline[0]["title"], "Pipe Song")
        self.assertEqual(inline[0]["artist"], "Pipe Artist")

    def test_artifact_chunks_round_trip(self) -> None:
        value = {"rows": [{"title": "Track", "artist": "Artist"}], "ok": True}
        chunks = encode_artifact_chunks(value, maximum_chars=12)
        decoded = json.loads(zlib.decompress(base64.b64decode("".join(chunks))))
        self.assertEqual(decoded, value)


if __name__ == "__main__":
    unittest.main()
