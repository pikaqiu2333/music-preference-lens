"""Validate and merge Phase 2 runner shards into a scorer artifact."""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import zlib
from pathlib import Path
from typing import Any

from run_phase2_mechanism_intervention_probe import (
    ARTIFACT_FORMAT_VERSION,
    bundle_mode,
    canonical_json_sha256,
    record_ids,
)


MERGED_ARTIFACT_FORMAT_VERSION = "phase2_mechanism_merged_artifact_v1"
SHA256 = re.compile(r"[0-9a-f]{64}")
CHUNK_MARKERS = (
    "PHASE2_DIAG_CHECKPOINT_ARTIFACT_CHUNK_JSON=",
    "PHASE2_CORRECTION_CHECKPOINT_ARTIFACT_CHUNK_JSON=",
    "PHASE2_DIAG_ARTIFACT_CHUNK_JSON=",
    "PHASE2_CORRECTION_ARTIFACT_CHUNK_JSON=",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require(summary: dict[str, Any], key: str, label: str) -> Any:
    if key not in summary:
        raise ValueError(f"{label} summary is missing {key}")
    return summary[key]


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def decode_chunk_group(
    envelopes: list[dict[str, Any]], label: str
) -> dict[str, Any]:
    if not envelopes:
        raise ValueError(f"{label} has no artifact chunks")
    expected_total = _require_int(envelopes[0].get("total"), f"{label} total")
    if expected_total <= 0:
        raise ValueError(f"{label} chunk total must be positive")
    metadata = {
        key: value
        for key, value in envelopes[0].items()
        if key not in {"index", "total", "data"}
    }
    by_index: dict[int, dict[str, Any]] = {}
    for envelope in envelopes:
        index = _require_int(envelope.get("index"), f"{label} chunk index")
        if index in by_index:
            raise ValueError(f"{label} contains duplicate chunk index {index}")
        if envelope.get("total") != expected_total:
            raise ValueError(f"{label} contains inconsistent chunk totals")
        observed_metadata = {
            key: value
            for key, value in envelope.items()
            if key not in {"index", "total", "data"}
        }
        if observed_metadata != metadata:
            raise ValueError(f"{label} contains inconsistent chunk metadata")
        if not isinstance(envelope.get("data"), str):
            raise ValueError(f"{label} chunk {index} has invalid data")
        by_index[index] = envelope
    missing_chunks = [
        index for index in range(expected_total) if index not in by_index
    ]
    extra_chunks = sorted(index for index in by_index if index >= expected_total)
    if missing_chunks or extra_chunks:
        raise ValueError(
            f"{label} chunk coverage mismatch; missing={missing_chunks}, "
            f"extra={extra_chunks}"
        )
    encoded = "".join(by_index[index]["data"] for index in range(expected_total))
    try:
        payload = zlib.decompress(base64.b64decode(encoded, validate=True))
        artifact = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, zlib.error, json.JSONDecodeError) as error:
        raise ValueError(f"{label} has an invalid compressed artifact") from error
    if not isinstance(artifact, dict):
        raise ValueError(f"{label} artifact must be a JSON object")
    return artifact


def artifacts_from_log(text: str, label: str = "job log") -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        marker = next(
            (candidate for candidate in CHUNK_MARKERS if stripped.startswith(candidate)),
            None,
        )
        if marker is None:
            continue
        try:
            envelope = json.loads(stripped[len(marker) :])
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} line {line_number} has invalid chunk JSON") from error
        if not isinstance(envelope, dict):
            raise ValueError(f"{label} line {line_number} chunk must be an object")
        if "CHECKPOINT" in marker:
            identity = envelope.get("checkpoint_id")
            if not isinstance(identity, str) or not identity:
                raise ValueError(
                    f"{label} line {line_number} checkpoint has no checkpoint_id"
                )
        else:
            identity = "final-shard"
        groups.setdefault((marker, identity), []).append(envelope)
    if not groups:
        raise ValueError(f"{label} contains no Phase 2 shard or checkpoint chunks")
    return [
        decode_chunk_group(envelopes, f"{label} {identity}")
        for (marker, identity), envelopes in sorted(groups.items())
    ]


def _validate_shard(
    artifact: dict[str, Any],
    *,
    label: str,
    bundle: dict[str, Any],
    mode: str,
    bundle_sha256: str,
    expected_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = artifact.get("records")
    summary = artifact.get("summary")
    if not isinstance(records, list) or not isinstance(summary, dict):
        raise ValueError(f"{label} must contain records and summary")
    if _require(summary, "artifact_format_version", label) != ARTIFACT_FORMAT_VERSION:
        raise ValueError(f"{label} artifact format version mismatch")
    if _require(summary, "artifact_kind", label) not in {"checkpoint", "shard"}:
        raise ValueError(f"{label} is not a checkpoint or shard artifact")
    expected_provenance = {
        "mode": mode,
        "protocol_id": bundle["protocol_id"],
        "protocol_hashes": bundle["protocol_hashes"],
        "catalog_asset_hashes": bundle.get("catalog_asset_hashes", {}),
        "canonical_bundle_sha256": bundle_sha256,
        "canonical_bundle_scope": "full_unsliced_bundle",
        "model_id": bundle["model_id"],
        "model_revision": bundle["model_revision"],
        "expected_num_layers": int(bundle["expected_num_layers"]),
        "full_record_count": len(expected_ids),
        "record_range_semantics": "zero_based_half_open",
    }
    for key, expected in expected_provenance.items():
        if _require(summary, key, label) != expected:
            raise ValueError(f"{label} {key} mismatch")
    script_sha256 = _require_sha256(
        _require(summary, "submitted_script_sha256", label),
        f"{label} submitted_script_sha256",
    )
    _require_sha256(
        _require(summary, "receipt_sha256", label), f"{label} receipt_sha256"
    )
    start = _require_int(_require(summary, "record_start", label), f"{label} start")
    stop = _require_int(_require(summary, "record_stop", label), f"{label} stop")
    if start < 0 or start >= stop or stop > len(expected_ids):
        raise ValueError(f"{label} has invalid record range [{start}, {stop})")
    requested_start = _require_int(
        _require(summary, "requested_record_start", label),
        f"{label} requested start",
    )
    requested_stop = _require_int(
        _require(summary, "requested_record_stop", label),
        f"{label} requested stop",
    )
    if (
        requested_start < 0
        or requested_start >= requested_stop
        or requested_stop > len(expected_ids)
        or start < requested_start
        or stop > requested_stop
    ):
        raise ValueError(f"{label} record range falls outside its requested range")
    if summary["artifact_kind"] == "shard" and (
        start != requested_start or stop != requested_stop
    ):
        raise ValueError(f"{label} final shard range differs from its requested range")
    shard_ids = record_ids(records, f"{label} records")
    expected_shard_ids = expected_ids[start:stop]
    if shard_ids != expected_shard_ids:
        raise ValueError(
            f"{label} record IDs do not match canonical range [{start}, {stop})"
        )
    expected_count = stop - start
    if _require(summary, "expected_record_count", label) != expected_count:
        raise ValueError(f"{label} expected record count mismatch")
    if _require(summary, "observed_record_count", label) != len(records):
        raise ValueError(f"{label} observed record count mismatch")
    if _require(summary, "record_ids_sha256", label) != canonical_json_sha256(
        shard_ids
    ):
        raise ValueError(f"{label} record ID hash mismatch")
    if _require(summary, "records_sha256", label) != canonical_json_sha256(records):
        raise ValueError(f"{label} records hash mismatch")
    if _require(summary, "architecture_gate", label) is not True:
        raise ValueError(f"{label} architecture gate failed")
    if _require(summary, "endpoint_gate", label) is not True:
        raise ValueError(f"{label} endpoint gate failed")
    if _require(summary, "technical_gate", label) is not True:
        raise ValueError(f"{label} technical gate failed")
    observed_layers = _require_int(
        _require(summary, "observed_num_layers", label), f"{label} observed layers"
    )
    if observed_layers != int(bundle["expected_num_layers"]):
        raise ValueError(f"{label} observed layer count mismatch")
    endpoint_error = _require(summary, "endpoint_max_logit_error", label)
    if isinstance(endpoint_error, bool) or not isinstance(endpoint_error, (int, float)):
        raise ValueError(f"{label} endpoint error must be numeric")
    endpoint_error = float(endpoint_error)
    if not math.isfinite(endpoint_error) or endpoint_error > float(
        bundle["endpoint_tolerance"]
    ):
        raise ValueError(f"{label} endpoint error exceeds the bundle tolerance")
    maximum_memory = _require_int(
        _require(summary, "maximum_gpu_memory_bytes", label),
        f"{label} maximum GPU memory",
    )
    if maximum_memory < 0:
        raise ValueError(f"{label} maximum GPU memory must be non-negative")
    for key in ("run_id", "started_at", "finished_at"):
        value = _require(summary, key, label)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} {key} must be a non-empty string")
    return records, {
        "label": label,
        "artifact_kind": summary["artifact_kind"],
        "run_id": summary["run_id"],
        "record_start": start,
        "record_stop": stop,
        "records_sha256": summary["records_sha256"],
        "receipt_sha256": summary["receipt_sha256"],
        "submitted_script_sha256": script_sha256,
        "observed_num_layers": observed_layers,
        "endpoint_max_logit_error": endpoint_error,
        "maximum_gpu_memory_bytes": maximum_memory,
        "started_at": summary["started_at"],
        "finished_at": summary["finished_at"],
    }


def merge_shard_artifacts(
    bundle: dict[str, Any], artifacts: list[dict[str, Any]]
) -> dict[str, Any]:
    mode, _ = bundle_mode(bundle)
    bundle_records = bundle.get("records")
    if not isinstance(bundle_records, list):
        raise ValueError("canonical bundle must contain records")
    expected_ids = record_ids(bundle_records, "canonical bundle records")
    if not expected_ids:
        raise ValueError("canonical bundle records must not be empty")
    bundle_sha256 = canonical_json_sha256(bundle)

    records_by_id: dict[str, dict[str, Any]] = {}
    source_details = []
    script_sha256: str | None = None
    conflicting_ids: set[str] = set()
    for index, artifact in enumerate(artifacts):
        label = f"shard {index}"
        records, details = _validate_shard(
            artifact,
            label=label,
            bundle=bundle,
            mode=mode,
            bundle_sha256=bundle_sha256,
            expected_ids=expected_ids,
        )
        if script_sha256 is None:
            script_sha256 = details["submitted_script_sha256"]
        elif details["submitted_script_sha256"] != script_sha256:
            raise ValueError(f"{label} submitted_script_sha256 mismatch")
        for row in records:
            record_id = str(row["record_id"])
            existing = records_by_id.get(record_id)
            if existing is None:
                records_by_id[record_id] = row
            elif canonical_json_sha256(existing) != canonical_json_sha256(row):
                conflicting_ids.add(record_id)
        source_details.append(details)

    if conflicting_ids:
        raise ValueError(
            "conflicting duplicate record IDs: "
            + json.dumps(sorted(conflicting_ids), ensure_ascii=False)
        )
    missing_ids = [record_id for record_id in expected_ids if record_id not in records_by_id]
    if missing_ids:
        raise ValueError(
            "missing record IDs: " + json.dumps(missing_ids, ensure_ascii=False)
        )
    merged_records = [records_by_id[record_id] for record_id in expected_ids]
    source_details.sort(
        key=lambda item: (
            item["record_start"],
            item["record_stop"],
            item["artifact_kind"],
            str(item["run_id"]),
        )
    )
    endpoint_error = max(item["endpoint_max_logit_error"] for item in source_details)
    summary = {
        "artifact_format_version": MERGED_ARTIFACT_FORMAT_VERSION,
        "artifact_kind": "merged",
        "run_id": f"phase2_{mode}_merged_{bundle_sha256[:16]}",
        "mode": mode,
        "protocol_id": bundle["protocol_id"],
        "protocol_hashes": bundle["protocol_hashes"],
        "catalog_asset_hashes": bundle.get("catalog_asset_hashes", {}),
        "submitted_script_sha256": script_sha256,
        "canonical_bundle_sha256": bundle_sha256,
        "canonical_bundle_scope": "full_unsliced_bundle",
        "model_id": bundle["model_id"],
        "model_revision": bundle["model_revision"],
        "expected_num_layers": int(bundle["expected_num_layers"]),
        "observed_num_layers": int(bundle["expected_num_layers"]),
        "full_record_count": len(expected_ids),
        "requested_record_start": 0,
        "requested_record_stop": len(expected_ids),
        "record_start": 0,
        "record_stop": len(expected_ids),
        "record_range_semantics": "zero_based_half_open",
        "expected_record_count": len(expected_ids),
        "observed_record_count": len(merged_records),
        "record_ids_sha256": canonical_json_sha256(expected_ids),
        "records_sha256": canonical_json_sha256(merged_records),
        "architecture_gate": True,
        "endpoint_gate": True,
        "endpoint_max_logit_error": endpoint_error,
        "technical_gate": True,
        "maximum_gpu_memory_bytes": max(
            item["maximum_gpu_memory_bytes"] for item in source_details
        ),
        "started_at": min(str(item["started_at"]) for item in source_details),
        "finished_at": max(str(item["finished_at"]) for item in source_details),
        "source_artifact_count": len(source_details),
        "source_run_ids": sorted({str(item["run_id"]) for item in source_details}),
        "source_receipt_sha256s": sorted(
            {str(item["receipt_sha256"]) for item in source_details}
        ),
        "source_shards": [
            {
                "artifact_kind": item["artifact_kind"],
                "run_id": item["run_id"],
                "record_start": item["record_start"],
                "record_stop": item["record_stop"],
                "records_sha256": item["records_sha256"],
            }
            for item in source_details
        ],
    }
    return {"records": merged_records, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--shard-artifact",
        "--shard",
        dest="shard_artifacts",
        action="append",
        type=Path,
        default=[],
        help="decoded runner checkpoint or final shard JSON; repeat as needed",
    )
    parser.add_argument(
        "--log",
        dest="logs",
        action="append",
        type=Path,
        default=[],
        help="runner log containing checkpoint/final compressed chunks",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.shard_artifacts and not args.logs:
        parser.error("at least one --shard-artifact or --log is required")

    artifacts = [load_json(path) for path in args.shard_artifacts]
    for path in args.logs:
        artifacts.extend(
            artifacts_from_log(path.read_text(encoding="utf-8"), label=str(path))
        )
    merged = merge_shard_artifacts(load_json(args.bundle), artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "PHASE2_MERGE_SUMMARY_JSON="
        + json.dumps(merged["summary"], ensure_ascii=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
