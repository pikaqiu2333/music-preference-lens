from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from merge_phase2_shard_artifacts import (  # noqa: E402
    artifacts_from_log,
    merge_shard_artifacts,
)
from run_phase2_mechanism_intervention_probe import (  # noqa: E402
    batch_ranges,
    build_artifact_summary,
    build_receipt,
    canonical_json_sha256,
    emit_artifact_chunks,
    resolve_record_range,
)


class Phase2RunnerRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = {
            "bundle_version": "phase2_mechanism_diagnosis_v1",
            "protocol_id": "protocol-v1",
            "protocol_hashes": {
                "json_sha256": "protocol-json",
                "markdown_sha256": "protocol-markdown",
            },
            "catalog_asset_hashes": {
                "catalog_rows_sha256": "catalog-rows",
                "scoring_key_sha256": "scoring-key",
            },
            "model_id": "example/model",
            "model_revision": "model-revision",
            "expected_num_layers": 4,
            "endpoint_tolerance": 0.02,
            "records": [
                {"record_id": f"record-{index}", "input": index}
                for index in range(5)
            ],
        }
        self.technical = {
            "architecture_gate": True,
            "observed_num_layers": 4,
            "endpoint_gate": True,
            "endpoint_max_logit_error": 0.01,
        }

    def make_artifact(
        self,
        start: int,
        stop: int,
        *,
        script_sha256: str = "a" * 64,
        artifact_kind: str = "shard",
        requested_start: int | None = None,
        requested_stop: int | None = None,
    ) -> dict:
        requested_start = start if requested_start is None else requested_start
        requested_stop = stop if requested_stop is None else requested_stop
        receipt = build_receipt(
            self.bundle,
            mode="diagnosis",
            run_id=f"run-{start}-{stop}",
            script_sha256=script_sha256,
            bundle_sha256=canonical_json_sha256(self.bundle),
            record_start=requested_start,
            record_stop=requested_stop,
            batch_size=1,
            technical=self.technical,
        )
        records = [
            {"record_id": row["record_id"], "result": row["input"] * 10}
            for row in self.bundle["records"][start:stop]
        ]
        summary = build_artifact_summary(
            receipt=receipt,
            technical=self.technical,
            artifact_kind=artifact_kind,
            records=records,
            expected_records=self.bundle["records"][start:stop],
            record_start=start,
            record_stop=stop,
            maximum_gpu_memory_bytes=1234,
            started_at="2026-07-13T00:00:00+00:00",
            finished_at="2026-07-13T00:01:00+00:00",
            batch_index=0 if artifact_kind == "checkpoint" else None,
        )
        return {"records": records, "summary": summary}

    def test_half_open_ranges_and_batches_are_deterministic(self) -> None:
        self.assertEqual(resolve_record_range(5, 1, 5), (1, 5))
        self.assertEqual(resolve_record_range(5, 1, None), (1, 5))
        self.assertEqual(batch_ranges(1, 5, 2), [(1, 3), (3, 5)])
        with self.assertRaisesRegex(ValueError, "start < stop"):
            resolve_record_range(5, 2, 2)
        with self.assertRaisesRegex(ValueError, "exceeds full record count"):
            resolve_record_range(5, 0, 6)
        with self.assertRaisesRegex(ValueError, "positive"):
            batch_ranges(0, 5, 0)

    def test_receipt_hashes_full_bundle_and_preserves_catalog_assets(self) -> None:
        receipt = build_receipt(
            self.bundle,
            mode="diagnosis",
            run_id="run",
            script_sha256="a" * 64,
            bundle_sha256=canonical_json_sha256(self.bundle),
            record_start=2,
            record_stop=4,
            batch_size=1,
            technical=self.technical,
        )
        sliced_bundle = {**self.bundle, "records": self.bundle["records"][2:4]}
        self.assertEqual(
            receipt["canonical_bundle_sha256"], canonical_json_sha256(self.bundle)
        )
        self.assertNotEqual(
            receipt["canonical_bundle_sha256"], canonical_json_sha256(sliced_bundle)
        )
        self.assertEqual(receipt["catalog_asset_hashes"], self.bundle["catalog_asset_hashes"])
        self.assertEqual((receipt["record_start"], receipt["record_stop"]), (2, 4))

    def test_checkpoint_chunks_can_be_recovered_from_an_incomplete_log(self) -> None:
        lines = ["unrelated model output"]
        for start, stop in ((0, 1), (1, 2)):
            artifact = self.make_artifact(start, stop, artifact_kind="checkpoint")
            emit_artifact_chunks(
                "PHASE2_DIAG_CHECKPOINT_ARTIFACT_CHUNK_JSON=",
                artifact,
                metadata={
                    "checkpoint_id": f"{start:06d}-{stop:06d}",
                    "record_start": start,
                    "record_stop": stop,
                },
                maximum_chars=19,
                emit=lines.append,
            )
        recovered = artifacts_from_log("\n".join(lines))
        self.assertEqual(
            [row["record_id"] for artifact in recovered for row in artifact["records"]],
            ["record-0", "record-1"],
        )

    def test_merger_accepts_identical_overlap_and_orders_canonically(self) -> None:
        merged = merge_shard_artifacts(
            self.bundle,
            [self.make_artifact(0, 3), self.make_artifact(2, 5)],
        )
        self.assertEqual(
            [row["record_id"] for row in merged["records"]],
            [f"record-{index}" for index in range(5)],
        )
        self.assertTrue(merged["summary"]["technical_gate"])
        self.assertEqual(
            merged["summary"]["catalog_asset_hashes"],
            self.bundle["catalog_asset_hashes"],
        )

    def test_merger_rejects_catalog_asset_hash_drift(self) -> None:
        shard = self.make_artifact(0, 5)
        shard["summary"]["catalog_asset_hashes"] = {"catalog_rows_sha256": "other"}
        with self.assertRaisesRegex(ValueError, "catalog_asset_hashes mismatch"):
            merge_shard_artifacts(self.bundle, [shard])

    def test_merger_rejects_script_drift_between_shards(self) -> None:
        with self.assertRaisesRegex(ValueError, "submitted_script_sha256 mismatch"):
            merge_shard_artifacts(
                self.bundle,
                [
                    self.make_artifact(0, 2, script_sha256="a" * 64),
                    self.make_artifact(2, 5, script_sha256="b" * 64),
                ],
            )

    def test_merger_rejects_forged_range_membership(self) -> None:
        shard = self.make_artifact(0, 2)
        shard["records"][0]["record_id"] = "record-4"
        shard["summary"]["record_ids_sha256"] = canonical_json_sha256(
            [row["record_id"] for row in shard["records"]]
        )
        shard["summary"]["records_sha256"] = canonical_json_sha256(shard["records"])
        with self.assertRaisesRegex(ValueError, "canonical range"):
            merge_shard_artifacts(self.bundle, [shard])

    def test_merger_rejects_conflicting_duplicate_records(self) -> None:
        first = self.make_artifact(0, 3)
        second = self.make_artifact(2, 5)
        second["records"][0]["result"] = -1
        second["summary"]["records_sha256"] = canonical_json_sha256(second["records"])
        with self.assertRaisesRegex(ValueError, "conflicting duplicate.*record-2"):
            merge_shard_artifacts(self.bundle, [first, second])

    def test_merger_lists_every_missing_record_id(self) -> None:
        with self.assertRaisesRegex(
            ValueError, 'missing record IDs: \\["record-1", "record-2", "record-3"\\]'
        ):
            merge_shard_artifacts(
                self.bundle,
                [self.make_artifact(0, 1), self.make_artifact(4, 5)],
            )

    def test_merger_requires_catalog_asset_field_even_when_bundle_uses_empty_default(
        self,
    ) -> None:
        bundle = copy.deepcopy(self.bundle)
        del bundle["catalog_asset_hashes"]
        original = self.bundle
        self.bundle = bundle
        try:
            shard = self.make_artifact(0, 5)
            del shard["summary"]["catalog_asset_hashes"]
            with self.assertRaisesRegex(ValueError, "missing catalog_asset_hashes"):
                merge_shard_artifacts(bundle, [shard])
        finally:
            self.bundle = original


if __name__ == "__main__":
    unittest.main()
