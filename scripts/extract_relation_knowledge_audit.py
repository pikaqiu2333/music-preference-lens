"""Extract and cross-check a relation-knowledge audit from HF Job logs."""

from __future__ import annotations

import argparse
import base64
import json
import zlib
from pathlib import Path
from typing import Any

from extract_job_log_artifacts import marker_records


ROW_MARKER = "REL_KNOWLEDGE_ROW_JSON="
CHUNK_MARKER = "REL_KNOWLEDGE_ARTIFACT_CHUNK_JSON="
SUMMARY_MARKER = "REL_KNOWLEDGE_SUMMARY_JSON="


def extract_audit(
    text: str,
    row_marker: str = ROW_MARKER,
    chunk_marker: str = CHUNK_MARKER,
    summary_marker: str = SUMMARY_MARKER,
) -> dict[str, Any]:
    chunks = marker_records(text, chunk_marker)
    if not chunks:
        raise ValueError("no relation-knowledge artifact chunks found")
    chunks.sort(key=lambda item: int(item["index"]))
    expected = int(chunks[0]["total"])
    if len(chunks) != expected or any(
        int(item["index"]) != index or int(item["total"]) != expected
        for index, item in enumerate(chunks)
    ):
        raise ValueError("artifact chunks are missing, duplicated, or inconsistent")
    artifact = json.loads(
        zlib.decompress(
            base64.b64decode("".join(item["data"] for item in chunks))
        ).decode("utf-8")
    )
    rows = marker_records(text, row_marker)
    summaries = marker_records(text, summary_marker)
    if len(summaries) != 1:
        raise ValueError(f"expected one audit summary, found {len(summaries)}")
    if rows != artifact.get("record_rows"):
        raise ValueError("row markers do not match the chunked artifact")
    if summaries[0] != artifact.get("summary"):
        raise ValueError("summary marker does not match the chunked artifact")
    summary = summaries[0]
    if len(rows) != int(summary["record_count"]):
        raise ValueError("record count does not match the summary")
    if len(artifact.get("prompt_rows", [])) != int(summary["prompt_count"]):
        raise ValueError("prompt count does not match the summary")
    return artifact


def write_outputs(
    artifact: dict[str, Any],
    artifact_path: Path,
    rows_path: Path,
    summary_path: Path,
) -> None:
    for path in (artifact_path, rows_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in artifact["record_rows"]
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
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--row-marker", default=ROW_MARKER)
    parser.add_argument("--chunk-marker", default=CHUNK_MARKER)
    parser.add_argument("--summary-marker", default=SUMMARY_MARKER)
    args = parser.parse_args()

    artifact = extract_audit(
        args.log.read_text(encoding="utf-8"),
        row_marker=args.row_marker,
        chunk_marker=args.chunk_marker,
        summary_marker=args.summary_marker,
    )
    write_outputs(artifact, args.artifact, args.rows, args.summary)
    print(
        f"Extracted {len(artifact['record_rows'])} records and "
        f"{len(artifact['prompt_rows'])} prompt rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
