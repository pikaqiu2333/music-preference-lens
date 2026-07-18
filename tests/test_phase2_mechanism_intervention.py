from __future__ import annotations

import base64
import json
import hashlib
import re
import sys
import unittest
import zlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_phase2_mechanism_intervention_probe import (  # noqa: E402
    CATALOG_VERIFIER_VERSION,
    build_artifacts,
    generation_row_hash,
    render_embedded_runner,
    select_catalog_clusters,
    validate_final_catalog_rows,
)
from run_phase2_mechanism_intervention_probe import (  # noqa: E402
    append_continuation,
    contains_key,
    overlapping_positions,
    parse_candidate_free_response,
    patch_relation_shift,
)
from score_phase2_mechanism_intervention import score_phase2  # noqa: E402
from verify_phase2_catalog import apple_url, musicbrainz_url, verify_pair  # noqa: E402


class Phase2MechanismInterventionExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol_bytes = (
            PROJECT_ROOT / "config" / "phase2_mechanism_intervention_protocol.json"
        ).read_bytes()
        cls.doc_bytes = (
            PROJECT_ROOT / "docs" / "phase2_mechanism_intervention_protocol.md"
        ).read_bytes()
        cls.protocol = json.loads(cls.protocol_bytes)

    def make_exact(self, index: int, *, artist: str | None = None) -> dict:
        artist = artist or f"Exact Artist {index}"
        return {
            "record_type": "generated_pair",
            "record_id": f"exact-{index}",
            "generation_id": f"generation-{index}",
            "rank": 1,
            "title": f"Exact Title {index}",
            "artist": artist,
            "normalized_title": f"exacttitle{index}",
            "normalized_artist": artist.casefold().replace(" ", ""),
            "phase2_catalog_label": "strict_exact",
            "catalog_artist_keys": [artist],
        }

    def make_conflict(self, index: int) -> dict:
        return {
            "record_type": "generated_pair",
            "record_id": f"conflict-{index}",
            "generation_id": f"conflict-generation-{index}",
            "rank": 1,
            "title": f"Conflict Title {index}",
            "artist": f"Wrong Artist {index}",
            "normalized_title": f"conflicttitle{index}",
            "normalized_artist": f"wrongartist{index}",
            "phase2_catalog_label": "strict_conflict",
            "reference_artist": f"Reference Artist {index}",
            "catalog_artist_keys": [f"Reference Artist {index}"],
        }

    def test_one_row_is_selected_before_cluster_label_caps(self) -> None:
        exact = self.make_exact(0)
        conflict_same_title = {
            **self.make_conflict(0),
            "record_id": "other",
            "title": exact["title"],
            "normalized_title": exact["normalized_title"],
        }
        conflicts, exact_rows, summary = select_catalog_clusters(
            [exact, conflict_same_title], self.protocol
        )
        self.assertEqual(len(conflicts) + len(exact_rows), 1)
        self.assertEqual(summary["strict_cluster_count_before_caps"], 1)

    def test_bundles_physically_isolate_candidate_free_references(self) -> None:
        rows = [self.make_conflict(index) for index in range(2)] + [
            self.make_exact(index) for index in range(8)
        ]
        diagnosis, correction, scoring, summary = build_artifacts(
            rows,
            self.protocol,
            self.protocol_bytes,
            self.doc_bytes,
            enforce_final_stop=False,
        )
        self.assertEqual(len(diagnosis["records"]), 2)
        self.assertFalse(summary["correction_bundle_contains_reference_field"])
        correction_text = json.dumps(correction, ensure_ascii=False)
        for index in range(2):
            self.assertNotIn(f"Reference Artist {index}", correction_text)
        self.assertIn("Reference Artist 0", json.dumps(scoring))
        self.assertEqual(len(diagnosis["records"][0]["diagnostic_controls"]), 3)
        self.assertEqual(len(diagnosis["records"][0]["manipulation_controls"]), 3)
        self.assertTrue(
            set(
                row["normalized_title"]
                for row in diagnosis["records"][0]["diagnostic_controls"]
            ).isdisjoint(
                row["normalized_title"]
                for row in diagnosis["records"][0]["manipulation_controls"]
            )
        )

    def test_correction_prompts_are_prerendered_and_have_two_paraphrases(self) -> None:
        rows = [self.make_conflict(0)] + [self.make_exact(index) for index in range(7)]
        _, correction, _, _ = build_artifacts(
            rows, self.protocol, self.protocol_bytes, self.doc_bytes
        )
        conflict = next(
            row for row in correction["records"] if row["record_type"] == "strict_conflict"
        )
        self.assertEqual(len(conflict["prompts"]), 2)
        self.assertTrue(all(len(prompts) == 2 for prompts in conflict["prompts"].values()))
        self.assertTrue(
            all(
                "Wrong Artist 0" in prompt
                for prompts in conflict["prompts"].values()
                for prompt in prompts
            )
        )

    def test_embedded_runner_and_bundle_hash_are_exact(self) -> None:
        rows = [self.make_conflict(0)] + [self.make_exact(index) for index in range(7)]
        diagnosis, _, scoring, _ = build_artifacts(
            rows, self.protocol, self.protocol_bytes, self.doc_bytes
        )
        template = (
            PROJECT_ROOT / "scripts" / "run_phase2_mechanism_intervention_probe.py"
        ).read_bytes()
        rendered = render_embedded_runner(template, diagnosis).decode("ascii")
        match = re.search(
            r'^EMBEDDED_BUNDLE_B64 = "([A-Za-z0-9+/=]+)"$',
            rendered,
            re.MULTILINE,
        )
        self.assertIsNotNone(match)
        embedded = json.loads(zlib.decompress(base64.b64decode(match.group(1))))
        self.assertEqual(embedded, diagnosis)
        expected_hash = hashlib.sha256(
            json.dumps(
                diagnosis,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            scoring["execution_assets"]["diagnosis"]["canonical_bundle_sha256"],
            expected_hash,
        )

    def test_final_stop_rejects_underpowered_conflict_count(self) -> None:
        rows = [self.make_conflict(0)] + [self.make_exact(index) for index in range(7)]
        with self.assertRaisesRegex(ValueError, "fewer than 30"):
            build_artifacts(
                rows,
                self.protocol,
                self.protocol_bytes,
                self.doc_bytes,
                enforce_final_stop=True,
            )

    def test_same_title_duplicate_is_deterministic_under_input_reversal(self) -> None:
        first = self.make_conflict(0)
        second = {**first, "record_id": "conflict-alternate", "artist": "Another Wrong"}
        selected_a = select_catalog_clusters([first, second], self.protocol)[0]
        selected_b = select_catalog_clusters([second, first], self.protocol)[0]
        self.assertEqual(selected_a[0]["record_id"], selected_b[0]["record_id"])

    def test_legacy_non_strict_catalog_labels_cannot_enter_phase2(self) -> None:
        row = self.make_exact(0)
        row.pop("phase2_catalog_label")
        row["catalog_label"] = "verified_exact"
        conflicts, exact_rows, summary = select_catalog_clusters([row], self.protocol)
        self.assertEqual(conflicts, [])
        self.assertEqual(exact_rows, [])
        self.assertEqual(summary["strict_input_row_count"], 0)

    def test_final_catalog_gate_rejects_pending_or_wrong_provenance(self) -> None:
        row = self.make_exact(0)
        title = row["title"]
        artist = row["artist"]
        routes = {
            musicbrainz_url(title, artist): {
                "count": 1,
                "offset": 0,
                "recordings": [
                    {
                        "id": "mb-exact",
                        "title": title,
                        "artist-credit": [{"name": artist}],
                    }
                ],
            },
            apple_url(title, artist): {
                "resultCount": 1,
                "results": [
                    {
                        "wrapperType": "track",
                        "kind": "song",
                        "trackId": 123,
                        "trackName": title,
                        "artistName": artist,
                    }
                ],
            },
        }
        evidence: list[dict] = []
        catalog_result = verify_pair(
            title,
            artist,
            fetcher=lambda url: routes[url],
            archive=evidence,
            now=lambda: "2026-07-13T00:00:00Z",
            sleep_seconds=0.0,
            max_attempts=1,
            sleep_fn=lambda _: None,
        )
        row.update(
            {
                "protocol_id": self.protocol["protocol_id"],
                "protocol_hashes": {
                    "json_sha256": hashlib.sha256(self.protocol_bytes).hexdigest(),
                    "markdown_sha256": hashlib.sha256(self.doc_bytes).hexdigest(),
                },
                "model_id": self.protocol["models"]["confirmatory"]["model_id"],
                "model_revision": self.protocol["models"]["confirmatory"]["revision"],
                "reason": "Reason",
                **catalog_result,
            }
        )
        self.assertEqual(row["catalog_verifier_version"], CATALOG_VERIFIER_VERSION)
        row["generation_row_sha256"] = generation_row_hash(row)
        hashes = row["protocol_hashes"]
        summary = validate_final_catalog_rows(
            [row], self.protocol, hashes, evidence
        )
        self.assertEqual(summary["strict_catalog_row_count"], 1)

        pending = {**row, "phase2_catalog_label": "pending"}
        with self.assertRaisesRegex(ValueError, "pending or legacy"):
            validate_final_catalog_rows([pending], self.protocol, hashes, evidence)

        wrong_revision = {**row, "model_revision": "wrong"}
        with self.assertRaisesRegex(ValueError, "model revision"):
            validate_final_catalog_rows(
                [wrong_revision], self.protocol, hashes, evidence
            )

        with self.assertRaisesRegex(ValueError, "fully linked"):
            validate_final_catalog_rows([row], self.protocol, hashes, [])

        unrelated_evidence = [
            {
                **request,
                "request_url": f"https://example.test/unrelated/{index}",
                "query_parameters": {},
            }
            for index, request in enumerate(evidence)
        ]
        with self.assertRaisesRegex(ValueError, "unrelated request URL"):
            validate_final_catalog_rows(
                [row], self.protocol, hashes, unrelated_evidence
            )

        unrelated_bodies = [
            {**request, "raw_response_body": "{}", "source_ids": []}
            for request in evidence
        ]
        with self.assertRaisesRegex(ValueError, "evidence replay was incomplete"):
            validate_final_catalog_rows(
                [row], self.protocol, hashes, unrelated_bodies
            )


class Phase2MechanismInterventionRunnerTests(unittest.TestCase):
    def test_complete_continuation_positions_are_tracked(self) -> None:
        text, span = append_continuation("Title: X\nArtist: ", "The Weeknd")
        self.assertEqual(text[slice(*span)], "The Weeknd")
        offsets = [
            (0, 0),
            (0, span[0]),
            (span[0], span[0] + 3),
            (span[0] + 4, span[1]),
        ]
        self.assertEqual(overlapping_positions(offsets, span), [2, 3])

    def test_relation_shift_uses_reference_minus_emitted_direction(self) -> None:
        self.assertEqual(patch_relation_shift(-1.0, -3.0, -4.0, -3.0), 3.0)

    def test_recursive_reference_guard(self) -> None:
        self.assertFalse(contains_key({"rows": [{"title": "X"}]}, "reference_artist"))
        self.assertTrue(
            contains_key(
                {"rows": [{"scoring": {"reference_artist": "Answer"}}]},
                "reference_artist",
            )
        )

    def test_runner_parser_rejects_multiple_artist_fields(self) -> None:
        self.assertEqual(
            parse_candidate_free_response("Artist: A Artist: B")["status"],
            "invalid",
        )


class Phase2FinalScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(
            (
                PROJECT_ROOT
                / "config"
                / "phase2_mechanism_intervention_protocol.json"
            ).read_text(encoding="utf-8")
        )
        protocol_bytes = (
            PROJECT_ROOT / "config" / "phase2_mechanism_intervention_protocol.json"
        ).read_bytes()
        doc_bytes = (
            PROJECT_ROOT / "docs" / "phase2_mechanism_intervention_protocol.md"
        ).read_bytes()
        cls.hashes = {
            "json_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "markdown_sha256": hashlib.sha256(doc_bytes).hexdigest(),
        }
        cls.catalog_hashes = {
            "catalog_rows_sha256": "catalog",
            "catalog_evidence_sha256": "evidence",
            "catalog_verifier_script_sha256": "verifier",
        }

    def make_artifacts(self, count: int = 15) -> tuple[dict, dict, dict]:
        diagnosis_rows = []
        correction_rows = []
        key_rows = []
        for mechanism, sign in (("prior_masked", 0.2), ("wrong_binding", -0.2)):
            for index in range(count):
                record_id = f"{mechanism}-{index}"
                templates = [
                    {
                        "record_id": record_id,
                        "technical_failure": None,
                        "layer_median_shifts": {
                            "20": sign,
                            "28": sign,
                            "36": 0.0,
                        },
                    }
                    for _ in range(2)
                ]
                diagnosis_rows.append(
                    {
                        "record_id": record_id,
                        "normalized_title": record_id,
                        "template_results": templates,
                        "manipulation_check": {},
                    }
                )
                anti_response = "Artist: Correct" if mechanism == "prior_masked" else "ABSTAIN"
                correction_rows.append(
                    {
                        "record_id": record_id,
                        "record_type": "strict_conflict",
                        "conditions": {
                            "naive_candidate_free_self_check": [
                                {"response": "ABSTAIN"},
                                {"response": "ABSTAIN"},
                            ],
                            "anti_prior_candidate_free_recall": [
                                {"response": anti_response},
                                {"response": anti_response},
                            ],
                        },
                    }
                )
                key_rows.append(
                    {
                        "record_id": record_id,
                        "record_type": "strict_conflict",
                        "normalized_title": record_id,
                        "reference_artist": "Correct",
                        "catalog_safety_action": "return_reference",
                    }
                )
        summary = {
            "protocol_id": self.protocol["protocol_id"],
            "protocol_hashes": self.hashes,
            "technical_gate": True,
            "model_id": self.protocol["models"]["confirmatory"]["model_id"],
            "model_revision": self.protocol["models"]["confirmatory"]["revision"],
            "catalog_asset_hashes": self.catalog_hashes,
        }
        return (
            {
                "records": diagnosis_rows,
                "summary": {
                    **summary,
                    "canonical_bundle_sha256": "diagnosis-bundle",
                    "submitted_script_sha256": "diagnosis-runner",
                },
            },
            {
                "records": correction_rows,
                "summary": {
                    **summary,
                    "canonical_bundle_sha256": "correction-bundle",
                    "submitted_script_sha256": "correction-runner",
                },
            },
            {
                "protocol_id": self.protocol["protocol_id"],
                "protocol_hashes": self.hashes,
                "model_id": self.protocol["models"]["confirmatory"]["model_id"],
                "model_revision": self.protocol["models"]["confirmatory"]["revision"],
                "catalog_asset_hashes": self.catalog_hashes,
                "execution_assets": {
                    "diagnosis": {
                        "canonical_bundle_sha256": "diagnosis-bundle",
                        "submitted_script_sha256": "diagnosis-runner",
                    },
                    "correction": {
                        "canonical_bundle_sha256": "correction-bundle",
                        "submitted_script_sha256": "correction-runner",
                    },
                },
                "records": key_rows,
            },
        )

    def test_frozen_positive_case_passes_both_hypotheses(self) -> None:
        diagnosis, correction, key = self.make_artifacts()
        result = score_phase2(diagnosis, correction, key, self.protocol)
        self.assertTrue(result["H1"]["confirmatory_passes"])
        self.assertEqual(result["H2"]["interaction"], 1.0)
        self.assertTrue(result["H2"]["passes"])
        self.assertEqual(result["decision"], "CONFIRMATORY_PASS")

    def test_h2_is_not_tested_when_h1_fails(self) -> None:
        diagnosis, correction, key = self.make_artifacts()
        for row in diagnosis["records"]:
            row["template_results"][1]["layer_median_shifts"] = {
                "20": -0.2,
                "28": -0.2,
                "36": 0.0,
            }
        result = score_phase2(diagnosis, correction, key, self.protocol)
        self.assertFalse(result["H1"]["confirmatory_passes"])
        self.assertFalse(result["H2"]["tested"])

    def test_missing_correction_record_is_rejected(self) -> None:
        diagnosis, correction, key = self.make_artifacts()
        correction["records"].pop()
        with self.assertRaisesRegex(ValueError, "correction and scoring-key"):
            score_phase2(diagnosis, correction, key, self.protocol)

    def test_formal_scoring_recomputes_and_binds_every_asset(self) -> None:
        diagnosis, correction, key = self.make_artifacts()
        result = score_phase2(
            diagnosis,
            correction,
            key,
            self.protocol,
            observed_protocol_hashes=self.hashes,
            observed_catalog_asset_hashes=self.catalog_hashes,
            require_execution_assets=True,
        )
        self.assertEqual(result["decision"], "CONFIRMATORY_PASS")

        diagnosis["summary"]["submitted_script_sha256"] = "different-runner"
        with self.assertRaisesRegex(ValueError, "submitted runner hash"):
            score_phase2(
                diagnosis,
                correction,
                key,
                self.protocol,
                observed_protocol_hashes=self.hashes,
                observed_catalog_asset_hashes=self.catalog_hashes,
                require_execution_assets=True,
            )

        diagnosis, correction, key = self.make_artifacts()
        del diagnosis["summary"]["model_revision"]
        with self.assertRaisesRegex(ValueError, "model revision mismatch"):
            score_phase2(diagnosis, correction, key, self.protocol)

        diagnosis, correction, key = self.make_artifacts()
        del correction["summary"]["catalog_asset_hashes"]
        with self.assertRaisesRegex(ValueError, "catalog asset hashes mismatch"):
            score_phase2(diagnosis, correction, key, self.protocol)


if __name__ == "__main__":
    unittest.main()
