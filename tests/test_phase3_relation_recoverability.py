from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_embedded_hf_job import extract_embedded_bundle  # noqa: E402
from export_phase3_relation_recoverability import build_bundle, load_jsonl  # noqa: E402
from run_phase3_relation_recoverability import (  # noqa: E402
    conflict_category,
    continuation_result,
)


def catalog_row(index: int, label: str, title: str | None = None) -> dict:
    title = title or f"Title {index}"
    normalized_title = "".join(character for character in title.lower() if character.isalnum())
    row = {
        "record_id": f"record-{index}",
        "generation_row_sha256": f"{index:064x}",
        "normalized_title": normalized_title,
        "title": title,
        "artist": f"Artist {index}",
        "context_id": "context",
        "model_id": "Qwen/Qwen3-4B-Base",
        "model_revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
        "catalog_verifier_version": "phase2_catalog_v2_complete_alias_audit",
        "phase2_catalog_label": label,
    }
    if label == "strict_conflict":
        row.update(
            {
                "reference_artist": f"Reference {index}",
                "reference_semantics": "catalog-supported reference, not unique real-world answer",
                "catalog_reference": {
                    "sources": {
                        "musicbrainz": {"artist_names": [f"Reference {index}"]},
                        "apple": {"artist_names": [f"Reference {index}"]},
                    }
                },
            }
        )
    return row


class Phase3RelationRecoverabilityExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol_path = (
            PROJECT_ROOT / "config" / "phase3_relation_recoverability_protocol.json"
        )
        cls.doc_path = (
            PROJECT_ROOT / "docs" / "phase3_relation_recoverability_protocol.md"
        )
        cls.protocol_bytes = cls.protocol_path.read_bytes()
        cls.doc_bytes = cls.doc_path.read_bytes()
        cls.protocol = json.loads(cls.protocol_bytes.decode("utf-8"))

    def test_selection_is_title_clustered_capped_and_conflict_disjoint(self) -> None:
        rows = [catalog_row(index, "excluded") for index in range(300)]
        rows[0] = catalog_row(1000, "strict_conflict", "Shared Title")
        rows[1] = catalog_row(1001, "strict_conflict", "Shared Title")
        rows[2] = catalog_row(1002, "strict_conflict", "Other Conflict")
        rows[3] = catalog_row(1003, "strict_exact", "Shared Title")
        for offset in range(30):
            rows[4 + offset] = catalog_row(2000 + offset, "strict_exact")
        bundle = build_bundle(
            self.protocol,
            self.protocol_bytes,
            self.doc_bytes,
            rows,
            b"verified catalog payload",
        )
        self.assertEqual(bundle["selected_conflict_cluster_count"], 2)
        self.assertEqual(bundle["selected_exact_cluster_count"], 24)
        conflicts = [
            row for row in bundle["records"] if row["source_group"] == "phase3_conflict"
        ]
        exact = [
            row
            for row in bundle["records"]
            if row["source_group"] == "phase3_generated_exact"
        ]
        self.assertEqual(len({row["normalized_title"] for row in conflicts}), 2)
        self.assertTrue(
            {row["normalized_title"] for row in conflicts}.isdisjoint(
                row["normalized_title"] for row in exact
            )
        )
        self.assertEqual(bundle["expected_record_count"], 2 + 24 + 8)
        self.assertEqual(bundle["expected_prompt_count"], (2 + 24 + 8) * 3)

    def test_protocol_is_frozen_before_catalog_results(self) -> None:
        self.assertEqual(
            self.protocol["status"], "frozen_before_phase3_pilot_catalog_results"
        )
        self.assertEqual(
            self.protocol["pilot_continuation_gate"][
                "minimum_unique_recoverable_conflict_title_clusters"
            ],
            8,
        )
        self.assertEqual(len(self.protocol["prompt_templates"]), 3)
        verifier_path = PROJECT_ROOT / "scripts" / "verify_phase2_catalog.py"
        import hashlib

        self.assertEqual(
            hashlib.sha256(verifier_path.read_bytes()).hexdigest(),
            self.protocol["input"]["catalog_verifier_sha256"],
        )

    def test_generated_bundle_and_embedded_runner_match_catalog(self) -> None:
        catalog_path = (
            PROJECT_ROOT / "runs" / "phase3_natural_pilot_catalog_verified.jsonl"
        )
        bundle_path = (
            PROJECT_ROOT / "runs" / "phase3_relation_recoverability_bundle.json"
        )
        stored = json.loads(bundle_path.read_text(encoding="utf-8"))
        expected = build_bundle(
            self.protocol,
            self.protocol_bytes,
            self.doc_bytes,
            load_jsonl(catalog_path),
            catalog_path.read_bytes(),
        )
        self.assertEqual(stored, expected)
        self.assertEqual(stored["selected_conflict_cluster_count"], 11)
        self.assertEqual(stored["selected_exact_cluster_count"], 20)
        runner_path = (
            PROJECT_ROOT / "scripts" / "run_phase3_relation_recoverability.py"
        )
        embedded_path = (
            PROJECT_ROOT
            / "runs"
            / "jobs"
            / "run_phase3_relation_recoverability_embedded.py"
        )
        self.assertEqual(
            extract_embedded_bundle(
                runner_path.read_text(encoding="utf-8"),
                embedded_path.read_text(encoding="utf-8"),
                "__PHASE3_RELATION_RECOVERABILITY_BUNDLE_B64_ZLIB__",
                "zlib",
            ),
            bundle_path.read_bytes().strip(),
        )

    def test_runner_only_reads_top_level_keys_present_in_bundle(self) -> None:
        bundle_path = (
            PROJECT_ROOT / "runs" / "phase3_relation_recoverability_bundle.json"
        )
        runner_path = (
            PROJECT_ROOT / "scripts" / "run_phase3_relation_recoverability.py"
        )
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        tree = ast.parse(runner_path.read_text(encoding="utf-8"))
        accessed = {
            node.slice.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "bundle"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        }
        self.assertEqual(accessed - bundle.keys(), set())


class Phase3RelationRecoverabilityRunnerTests(unittest.TestCase):
    def test_conflict_rule_requires_generation_and_positive_margin(self) -> None:
        self.assertEqual(
            conflict_category(2, 0, 0.1), "recoverable_relation_conflict"
        )
        self.assertEqual(conflict_category(2, 0, -0.1), "generation_only")
        self.assertEqual(conflict_category(1, 0, 0.1), "margin_only")

    def test_continuation_gate_is_not_relaxed_below_eight(self) -> None:
        bundle = {
            "source_catalog_row_count": 300,
            "pilot_continuation_gate": {
                "minimum_catalog_rows": 300,
                "minimum_unique_recoverable_conflict_title_clusters": 8,
                "pass_action": "continue",
                "failure_action": "stop",
            },
        }
        summary = {
            "assay_validity_gate": True,
            "recoverable_relation_conflict_record_ids": [f"r{i}" for i in range(7)],
        }
        self.assertFalse(continuation_result(summary, bundle)["pilot_continuation_gate"])
        summary["recoverable_relation_conflict_record_ids"].append("r7")
        self.assertTrue(continuation_result(summary, bundle)["pilot_continuation_gate"])
        bundle["source_catalog_row_count"] = 299
        self.assertFalse(continuation_result(summary, bundle)["pilot_continuation_gate"])


if __name__ == "__main__":
    unittest.main()
