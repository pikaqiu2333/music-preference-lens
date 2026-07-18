"""Extract row/summary JSON or chunked artifacts from captured job logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from decode_zlib_job_artifact import decode_artifact


def marker_records(text: str, marker: str) -> list[Any]:
    output = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(marker):
            output.append(json.loads(stripped[len(marker) :]))
    return output


def extract_plain(
    text: str, row_marker: str, summary_marker: str
) -> dict[str, Any]:
    rows = marker_records(text, row_marker)
    summaries = marker_records(text, summary_marker)
    if not rows:
        raise ValueError(f"no rows found for {row_marker}")
    if len(summaries) != 1:
        raise ValueError(f"expected one summary for {summary_marker}, found {len(summaries)}")
    return {"rows": rows, "summary": summaries[0]}


def extract_chunks(text: str, chunk_marker: str) -> dict[str, Any]:
    chunks = marker_records(text, chunk_marker)
    if not chunks:
        raise ValueError(f"no chunks found for {chunk_marker}")
    chunks.sort(key=lambda item: int(item["index"]))
    expected = int(chunks[0]["total"])
    if len(chunks) != expected or any(
        int(item["index"]) != index or int(item["total"]) != expected
        for index, item in enumerate(chunks)
    ):
        raise ValueError("artifact chunks are missing, duplicated, or inconsistent")
    return decode_artifact("".join(item["data"] for item in chunks))


def write_artifact(artifact: dict[str, Any], rows_path: Path, summary_path: Path) -> None:
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in artifact["rows"]
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(artifact["summary"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--chunk-marker")
    mode.add_argument("--row-marker")
    parser.add_argument("--summary-marker")
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8")
    if args.chunk_marker:
        artifact = extract_chunks(text, args.chunk_marker)
    else:
        if not args.summary_marker:
            parser.error("--summary-marker is required with --row-marker")
        artifact = extract_plain(text, args.row_marker, args.summary_marker)
    write_artifact(artifact, args.rows, args.summary)
    if args.artifact is not None:
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        args.artifact.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Extracted {len(artifact['rows'])} rows")
    print(f"- {args.rows}")
    print(f"- {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
