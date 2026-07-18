"""Freeze the combined Phase 2 catalog and apply the preregistered stop gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from export_phase2_mechanism_intervention_probe import (
    DEFAULT_CATALOG_VERIFIER,
    DEFAULT_PROTOCOL,
    DEFAULT_PROTOCOL_DOC,
    read_jsonl,
    select_catalog_clusters,
    sha256_bytes,
    validate_final_catalog_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_batch_mode(rows: list[dict[str, Any]], expected: str) -> None:
    unexpected = [
        str(row.get("record_id", index))
        for index, row in enumerate(rows)
        if row.get("batch_mode") != expected
    ]
    if unexpected:
        raise ValueError(
            f"{expected} input contains rows from another batch: {unexpected[:3]}"
        )


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(Counter(str(row.get("phase2_catalog_label", "")) for row in rows).items())
    )


def finalize_catalog(
    primary_rows: list[dict[str, Any]],
    extension_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    protocol_hashes: dict[str, str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if not primary_rows:
        raise ValueError("primary catalog input is empty")
    validate_batch_mode(primary_rows, "primary")
    validate_batch_mode(extension_rows, "extension")
    combined = [*primary_rows, *extension_rows]
    maximum_events = int(protocol["generation"]["maximum_parsed_event_count"])
    if len(combined) > maximum_events:
        raise ValueError(
            f"combined catalog has {len(combined)} rows, exceeding cap {maximum_events}"
        )
    record_ids = [str(row.get("record_id", "")) for row in combined]
    if any(not record_id for record_id in record_ids):
        raise ValueError("combined catalog contains an empty record ID")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("combined catalog contains duplicate record IDs")

    validation = validate_final_catalog_rows(
        combined, protocol, protocol_hashes, evidence_rows
    )
    conflicts, exact_rows, selection = select_catalog_clusters(combined, protocol)
    minimum_conflicts = int(
        protocol["stop_conditions"][
            "minimum_unique_strict_conflict_title_clusters_after_cap"
        ]
    )
    minimum_exact = int(
        protocol["mechanism_diagnosis"]["minimum_global_unique_exact_pool_titles"]
    )
    if len(conflicts) < minimum_conflicts:
        decision = "STOP_INSUFFICIENT_STRICT_CONFLICT_CLUSTERS"
        reason = (
            f"{len(conflicts)} strict conflict title clusters remain; "
            f"the preregistered minimum is {minimum_conflicts}."
        )
    elif len(exact_rows) < minimum_exact:
        decision = "STOP_INSUFFICIENT_STRICT_EXACT_POOL"
        reason = (
            f"{len(exact_rows)} strict exact title clusters remain; "
            f"the preregistered minimum is {minimum_exact}."
        )
    else:
        decision = "PROCEED_TO_MECHANISM_INTERVENTION"
        reason = "All preregistered catalog-yield gates passed."

    summary = {
        "summary_version": "phase2_combined_catalog_final_v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_hashes": protocol_hashes,
        "primary_row_count": len(primary_rows),
        "extension_row_count": len(extension_rows),
        "combined_row_count": len(combined),
        "primary_catalog_label_counts": label_counts(primary_rows),
        "extension_catalog_label_counts": label_counts(extension_rows),
        "combined_catalog_label_counts": label_counts(combined),
        "combined_unique_normalized_title_count": len(
            {
                str(row.get("normalized_title", ""))
                for row in combined
                if row.get("normalized_title")
            }
        ),
        "maximum_parsed_event_count": maximum_events,
        "frozen_generation_batches_exhausted": True,
        **selection,
        "minimum_conflict_clusters_to_continue": minimum_conflicts,
        "minimum_exact_pool_clusters_to_continue": minimum_exact,
        "decision": decision,
        "decision_reason": reason,
        "formal_mechanism_run_allowed": decision
        == "PROCEED_TO_MECHANISM_INTERVENTION",
        "catalog_validation": validation,
    }
    return combined, conflicts, exact_rows, summary


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-doc", type=Path, default=DEFAULT_PROTOCOL_DOC)
    parser.add_argument(
        "--catalog-verifier", type=Path, default=DEFAULT_CATALOG_VERIFIER
    )
    parser.add_argument("--combined-output", type=Path, required=True)
    parser.add_argument("--conflicts-output", type=Path, required=True)
    parser.add_argument("--exact-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    protocol_bytes = args.protocol.read_bytes()
    protocol_doc_bytes = args.protocol_doc.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_hashes = {
        "json_sha256": sha256_bytes(protocol_bytes),
        "markdown_sha256": sha256_bytes(protocol_doc_bytes),
    }
    combined, conflicts, exact_rows, summary = finalize_catalog(
        read_jsonl(args.primary),
        read_jsonl(args.extension),
        read_jsonl(args.evidence),
        protocol,
        protocol_hashes,
    )
    atomic_write_jsonl(args.combined_output, combined)
    atomic_write_jsonl(args.conflicts_output, conflicts)
    atomic_write_jsonl(args.exact_output, exact_rows)
    summary["assets"] = {
        "primary_catalog_sha256": file_sha256(args.primary),
        "extension_catalog_sha256": file_sha256(args.extension),
        "combined_catalog_sha256": file_sha256(args.combined_output),
        "selected_conflicts_sha256": file_sha256(args.conflicts_output),
        "selected_exact_sha256": file_sha256(args.exact_output),
        "evidence_archive_sha256": file_sha256(args.evidence),
        "catalog_verifier_sha256": file_sha256(args.catalog_verifier),
    }
    atomic_write_json(args.summary, summary)
    print("PHASE2_FINAL_CATALOG_SUMMARY_JSON=" + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
