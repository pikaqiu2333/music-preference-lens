"""Merge Phase 2 catalog evidence archives without rewriting their records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _jsonl_record_bytes(line: bytes) -> bytes:
    if line.endswith(b"\n"):
        line = line[:-1]
        if line.endswith(b"\r"):
            line = line[:-1]
    return line


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def merge_catalog_evidence(
    input_paths: Sequence[Path | str], output_path: Path | str
) -> dict[str, Any]:
    """Merge JSONL archives in order and atomically replace ``output_path``."""

    inputs = [Path(path) for path in input_paths]
    if len(inputs) < 2:
        raise ValueError("at least two catalog evidence archives are required")

    seen: dict[str, tuple[bytes, Path, int]] = {}
    merged_records: list[bytes] = []
    input_counts: list[int] = []
    duplicate_count = 0

    for path in inputs:
        input_count = 0
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = _jsonl_record_bytes(line)
                if not record.strip():
                    continue
                input_count += 1
                try:
                    value = json.loads(record.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"{path}:{line_number}: invalid UTF-8 JSON: {error}"
                    ) from error
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{path}:{line_number}: expected a JSON object"
                    )
                request_id = value.get("request_id")
                if not isinstance(request_id, str) or not request_id.strip():
                    raise ValueError(
                        f"{path}:{line_number}: missing or empty request_id"
                    )

                existing = seen.get(request_id)
                if existing is None:
                    seen[request_id] = (record, path, line_number)
                    merged_records.append(record)
                    continue
                first_record, first_path, first_line = existing
                if record != first_record:
                    raise ValueError(
                        f"{path}:{line_number}: conflicting duplicate request_id "
                        f"{request_id!r}; first seen at {first_path}:{first_line}"
                    )
                duplicate_count += 1
        input_counts.append(input_count)

    payload = b"".join(record + b"\n" for record in merged_records)
    output = Path(output_path)
    _atomic_write(output, payload)
    return {
        "input_counts": input_counts,
        "unique_count": len(merged_records),
        "duplicate_count": duplicate_count,
        "output_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge Phase 2 catalog evidence JSONL archives by request_id."
    )
    parser.add_argument(
        "archives",
        nargs="*",
        type=Path,
        help="input archives in first-seen precedence order",
    )
    parser.add_argument(
        "--input",
        dest="input_options",
        action="append",
        type=Path,
        default=[],
        help="input archive; repeat at least twice",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    if args.archives and args.input_options:
        parser.error("use positional archives or repeated --input options, not both")
    inputs = args.input_options or args.archives
    if len(inputs) < 2:
        parser.error("at least two input archives are required")
    summary = merge_catalog_evidence(inputs, args.output)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


def main() -> int:
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
