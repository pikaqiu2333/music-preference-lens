from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from finalize_phase2_catalog import finalize_catalog  # noqa: E402


class FinalizePhase2CatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = {
            "protocol_id": "phase2",
            "generation": {"maximum_parsed_event_count": 400},
            "stop_conditions": {
                "minimum_unique_strict_conflict_title_clusters_after_cap": 30
            },
            "mechanism_diagnosis": {
                "minimum_global_unique_exact_pool_titles": 6
            },
        }
        self.hashes = {"json_sha256": "a", "markdown_sha256": "b"}
        self.primary = [
            {
                "record_id": "primary-1",
                "batch_mode": "primary",
                "phase2_catalog_label": "strict_conflict",
                "normalized_title": "primarytitle",
            }
        ]
        self.extension = [
            {
                "record_id": "extension-1",
                "batch_mode": "extension",
                "phase2_catalog_label": "excluded",
                "normalized_title": "extensiontitle",
            }
        ]

    def run_finalization(self, conflict_count: int, exact_count: int):
        conflicts = [{"record_id": f"conflict-{i}"} for i in range(conflict_count)]
        exact = [{"record_id": f"exact-{i}"} for i in range(exact_count)]
        selection = {
            "strict_conflict_cluster_count_before_cap": conflict_count,
            "strict_exact_cluster_count_before_cap": exact_count,
            "selected_conflict_cluster_count": conflict_count,
            "selected_exact_cluster_count": exact_count,
        }
        with (
            patch(
                "finalize_phase2_catalog.validate_final_catalog_rows",
                return_value={"validated": True},
            ),
            patch(
                "finalize_phase2_catalog.select_catalog_clusters",
                return_value=(conflicts, exact, selection),
            ),
        ):
            return finalize_catalog(
                self.primary,
                self.extension,
                [],
                self.protocol,
                self.hashes,
            )

    def test_proceeds_only_when_both_catalog_yield_gates_pass(self) -> None:
        combined, conflicts, exact, summary = self.run_finalization(30, 6)
        self.assertEqual(len(combined), 2)
        self.assertEqual(len(conflicts), 30)
        self.assertEqual(len(exact), 6)
        self.assertEqual(summary["decision"], "PROCEED_TO_MECHANISM_INTERVENTION")
        self.assertTrue(summary["formal_mechanism_run_allowed"])
        self.assertEqual(
            summary["combined_catalog_label_counts"],
            {"excluded": 1, "strict_conflict": 1},
        )
        self.assertEqual(summary["combined_unique_normalized_title_count"], 2)

    def test_stops_when_conflict_clusters_are_too_sparse(self) -> None:
        *_, summary = self.run_finalization(29, 20)
        self.assertEqual(
            summary["decision"], "STOP_INSUFFICIENT_STRICT_CONFLICT_CLUSTERS"
        )
        self.assertFalse(summary["formal_mechanism_run_allowed"])

    def test_stops_when_exact_control_pool_is_too_small(self) -> None:
        *_, summary = self.run_finalization(30, 5)
        self.assertEqual(summary["decision"], "STOP_INSUFFICIENT_STRICT_EXACT_POOL")

    def test_rejects_cross_batch_rows(self) -> None:
        self.extension[0]["batch_mode"] = "primary"
        with self.assertRaisesRegex(ValueError, "another batch"):
            finalize_catalog(
                self.primary, self.extension, [], self.protocol, self.hashes
            )

    def test_rejects_duplicate_record_ids_before_validation(self) -> None:
        self.extension[0]["record_id"] = "primary-1"
        with self.assertRaisesRegex(ValueError, "duplicate record IDs"):
            finalize_catalog(
                self.primary, self.extension, [], self.protocol, self.hashes
            )

    def test_rejects_rows_above_the_frozen_event_cap(self) -> None:
        self.protocol["generation"]["maximum_parsed_event_count"] = 1
        with self.assertRaisesRegex(ValueError, "exceeding cap"):
            finalize_catalog(
                self.primary, self.extension, [], self.protocol, self.hashes
            )


if __name__ == "__main__":
    unittest.main()
