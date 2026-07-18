from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from merge_phase2_catalog_evidence import (  # noqa: E402
    merge_catalog_evidence,
    run_cli,
)


class MergePhase2CatalogEvidenceTests(unittest.TestCase):
    def test_preserves_first_seen_bytes_and_accepts_identical_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            output = root / "merged.jsonl"
            first_b = b'{ "request_id": "b", "value": 2 }'
            first_a = b'{"request_id":"a","value":1}'
            second_c = b'{"request_id":"c","value":3}'
            first.write_bytes(first_b + b"\n" + first_a)
            second.write_bytes(first_a + b"\r\n" + second_c + b"\r\n")

            summary = merge_catalog_evidence([first, second], output)

            expected = first_b + b"\n" + first_a + b"\n" + second_c + b"\n"
            self.assertEqual(output.read_bytes(), expected)
            self.assertEqual(
                summary,
                {
                    "input_counts": [2, 2],
                    "unique_count": 3,
                    "duplicate_count": 1,
                    "output_sha256": hashlib.sha256(expected).hexdigest(),
                },
            )

    def test_rejects_semantically_equal_but_byte_different_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            output = root / "merged.jsonl"
            first.write_bytes(b'{"request_id":"same","value":1}\n')
            second.write_bytes(b'{"value":1,"request_id":"same"}\n')
            output.write_bytes(b"existing output\n")

            with self.assertRaisesRegex(
                ValueError, "conflicting duplicate request_id 'same'"
            ):
                merge_catalog_evidence([first, second], output)

            self.assertEqual(output.read_bytes(), b"existing output\n")

    def test_rejects_missing_or_empty_request_id(self) -> None:
        invalid_records = [b'{"value":1}\n', b'{"request_id":""}\n']
        for invalid_record in invalid_records:
            with self.subTest(record=invalid_record):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    first = root / "first.jsonl"
                    second = root / "second.jsonl"
                    output = root / "merged.jsonl"
                    first.write_bytes(b'{"request_id":"ok"}\n')
                    second.write_bytes(invalid_record)

                    with self.assertRaisesRegex(ValueError, "request_id"):
                        merge_catalog_evidence([first, second], output)

                    self.assertFalse(output.exists())

    def test_requires_two_archives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "only.jsonl"
            archive.write_bytes(b'{"request_id":"only"}\n')
            with self.assertRaisesRegex(ValueError, "at least two"):
                merge_catalog_evidence([archive], root / "merged.jsonl")

    def test_replace_failure_keeps_output_and_removes_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            output = root / "merged.jsonl"
            first.write_bytes(b'{"request_id":"first"}\n')
            second.write_bytes(b'{"request_id":"second"}\n')
            output.write_bytes(b"existing output\n")

            with patch(
                "merge_phase2_catalog_evidence.os.replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    merge_catalog_evidence([first, second], output)

            self.assertEqual(output.read_bytes(), b"existing output\n")
            self.assertEqual(list(root.glob(f".{output.name}.*.tmp")), [])

    def test_cli_emits_machine_readable_json_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            output = root / "merged.jsonl"
            first.write_bytes(b'{"request_id":"first"}\n')
            second.write_bytes(b'{"request_id":"second"}\n')
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                result = run_cli(
                    [
                        "--input",
                        str(first),
                        "--input",
                        str(second),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(result, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["input_counts"], [1, 1])
            self.assertEqual(summary["unique_count"], 2)
            self.assertEqual(summary["duplicate_count"], 0)
            self.assertEqual(
                summary["output_sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
