from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_relation_knowledge_recoverability_audit import (  # noqa: E402
    build_bundle,
    load_jsonl,
)
from build_embedded_hf_job import extract_embedded_bundle  # noqa: E402
from extract_relation_knowledge_audit import extract_audit  # noqa: E402
from run_relation_knowledge_recoverability_audit import (  # noqa: E402
    conflict_category,
    encode_artifact_chunks,
    matches_any,
    normalize_entity,
    parse_artist_generation,
    summarize_records,
)


class RelationKnowledgeAuditExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol_path = (
            PROJECT_ROOT / "config" / "relation_knowledge_recoverability_audit.json"
        )
        self.doc_path = (
            PROJECT_ROOT / "docs" / "relation_knowledge_recoverability_audit.md"
        )
        self.protocol_bytes = self.protocol_path.read_bytes()
        self.doc_bytes = self.doc_path.read_bytes()
        self.protocol = json.loads(self.protocol_bytes.decode("utf-8"))

    def build(self) -> dict:
        return build_bundle(
            self.protocol,
            self.protocol_bytes,
            self.doc_bytes,
            load_jsonl(PROJECT_ROOT / self.protocol["inputs"]["conflicts"]),
            load_jsonl(
                PROJECT_ROOT
                / self.protocol["inputs"]["generated_exact_controls"]
            ),
        )

    def test_bundle_freezes_all_three_groups(self) -> None:
        bundle = self.build()
        self.assertEqual(bundle["expected_record_count"], 40)
        self.assertEqual(bundle["expected_prompt_count"], 120)
        self.assertEqual(
            Counter(row["source_group"] for row in bundle["records"]),
            Counter(
                {
                    "phase2_conflict": 7,
                    "phase2_generated_exact": 25,
                    "canonical_positive": 8,
                }
            ),
        )
        self.assertEqual(len({row["record_id"] for row in bundle["records"]}), 40)

    def test_conflicts_keep_emitted_and_catalog_reference_separate(self) -> None:
        bundle = self.build()
        conflicts = [
            row for row in bundle["records"] if row["source_group"] == "phase2_conflict"
        ]
        self.assertTrue(
            all(
                normalize_entity(row["target_artist"])
                != normalize_entity(row["emitted_artist"])
                for row in conflicts
            )
        )
        self.assertTrue(all(row["accepted_target_artists"] for row in conflicts))

    def test_generated_bundle_and_embedded_runner_match_sources(self) -> None:
        expected_bundle = self.build()
        bundle_path = PROJECT_ROOT / "runs" / "relation_knowledge_audit_bundle.json"
        stored_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        self.assertEqual(stored_bundle, expected_bundle)

        runner_path = (
            PROJECT_ROOT / "scripts" / "run_relation_knowledge_recoverability_audit.py"
        )
        embedded_path = (
            PROJECT_ROOT / "runs" / "jobs" / "run_relation_knowledge_audit_embedded.py"
        )
        self.assertEqual(
            extract_embedded_bundle(
                runner_path.read_text(encoding="utf-8"),
                embedded_path.read_text(encoding="utf-8"),
                "__RELATION_KNOWLEDGE_AUDIT_BUNDLE_B64_ZLIB__",
                "zlib",
            ),
            bundle_path.read_bytes().strip(),
        )


class RelationKnowledgeAuditRunnerTests(unittest.TestCase):
    def test_normalizer_and_parser_are_frozen(self) -> None:
        self.assertEqual(normalize_entity("The Weeknd"), "theweeknd")
        self.assertEqual(normalize_entity("ＭØ"), "mø")
        self.assertEqual(parse_artist_generation("Artist: Queen.\nExtra"), "Queen")
        self.assertEqual(parse_artist_generation("Recorded by: Adele"), "Adele")
        self.assertTrue(matches_any("The Eagles", ["Eagles", "The Eagles"]))

    def test_conflict_categories_cover_registered_patterns(self) -> None:
        self.assertEqual(conflict_category(2, 0, 0.1), "reference_recoverable")
        self.assertEqual(conflict_category(2, 0, -0.1), "generation_only")
        self.assertEqual(conflict_category(1, 0, 0.1), "margin_only")
        self.assertEqual(
            conflict_category(0, 2, -0.1), "persistent_emitted_binding"
        )
        self.assertEqual(
            conflict_category(0, 0, -0.1), "unrecovered_or_indeterminate"
        )

    def test_summary_separates_technical_and_assay_gates(self) -> None:
        bundle = {
            "expected_record_count": 3,
            "expected_prompt_count": 9,
            "scoring": {"target_generation_recovery_minimum_prompts": 2},
            "validity_gates": {
                "minimum_nonempty_generation_fraction": 0.95,
                "minimum_canonical_control_recovery_fraction": 0.75,
            },
        }
        record_rows = [
            {
                "record_id": "c1",
                "source_group": "canonical_positive",
                "target_generation_count": 3,
                "audit_category": "target_recoverable",
            },
            {
                "record_id": "e1",
                "source_group": "phase2_generated_exact",
                "target_generation_count": 1,
                "audit_category": "target_unrecovered",
            },
            {
                "record_id": "x1",
                "source_group": "phase2_conflict",
                "target_generation_count": 2,
                "audit_category": "reference_recoverable",
            },
        ]
        prompt_rows = [{"raw_generation": "ok"} for _ in range(9)]
        summary = summarize_records(record_rows, prompt_rows, bundle)
        self.assertTrue(summary["technical_gate"])
        self.assertTrue(summary["assay_validity_gate"])
        self.assertEqual(summary["reference_recoverable_record_ids"], ["x1"])


class RelationKnowledgeAuditExtractionTests(unittest.TestCase):
    def test_extractor_cross_checks_rows_summary_and_chunks(self) -> None:
        rows = [{"record_id": "one"}, {"record_id": "two"}]
        summary = {"record_count": 2, "prompt_count": 3}
        artifact = {
            "record_rows": rows,
            "prompt_rows": [{"id": 1}, {"id": 2}, {"id": 3}],
            "summary": summary,
        }
        chunks = encode_artifact_chunks(artifact, maximum_chars=20)
        lines = [
            *("REL_KNOWLEDGE_ROW_JSON=" + json.dumps(row) for row in rows),
            *(
                "REL_KNOWLEDGE_ARTIFACT_CHUNK_JSON="
                + json.dumps({"index": index, "total": len(chunks), "data": data})
                for index, data in enumerate(chunks)
            ),
            "REL_KNOWLEDGE_SUMMARY_JSON=" + json.dumps(summary),
        ]
        self.assertEqual(extract_audit("\n".join(lines)), artifact)

    def test_extractor_rejects_marker_artifact_disagreement(self) -> None:
        artifact = {
            "record_rows": [{"record_id": "one"}],
            "prompt_rows": [{"id": 1}],
            "summary": {"record_count": 1, "prompt_count": 1},
        }
        chunk = encode_artifact_chunks(artifact)[0]
        text = "\n".join(
            [
                'REL_KNOWLEDGE_ROW_JSON={"record_id":"changed"}',
                "REL_KNOWLEDGE_ARTIFACT_CHUNK_JSON="
                + json.dumps({"index": 0, "total": 1, "data": chunk}),
                "REL_KNOWLEDGE_SUMMARY_JSON=" + json.dumps(artifact["summary"]),
            ]
        )
        with self.assertRaisesRegex(ValueError, "row markers"):
            extract_audit(text)


if __name__ == "__main__":
    unittest.main()
