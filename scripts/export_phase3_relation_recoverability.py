"""Select Phase 3 catalog rows and export the frozen recoverability bundle."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "config" / "phase3_relation_recoverability_protocol.json"
DEFAULT_PROTOCOL_DOC = PROJECT_ROOT / "docs" / "phase3_relation_recoverability_protocol.md"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def selection_hash(row: dict[str, Any], salt: str) -> str:
    return sha256_bytes(f"{salt}|{row['generation_row_sha256']}".encode("utf-8"))


def one_per_title(
    rows: list[dict[str, Any]], label: str, label_field: str, salt: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get(label_field) == label:
            grouped[row["normalized_title"]].append(row)
    selected = []
    for title, group in grouped.items():
        representative = min(group, key=lambda row: selection_hash(row, salt))
        selected.append(
            {**representative, "phase3_recoverability_selection_hash": selection_hash(representative, salt)}
        )
    return sorted(selected, key=lambda row: row["phase3_recoverability_selection_hash"])


def accepted_reference_artists(row: dict[str, Any]) -> list[str]:
    names = {row["reference_artist"]}
    for source in row["catalog_reference"]["sources"].values():
        names.update(source.get("artist_names", []))
    return sorted(name for name in names if name)


def build_bundle(
    protocol: dict[str, Any],
    protocol_bytes: bytes,
    protocol_doc_bytes: bytes,
    verified_rows: list[dict[str, Any]],
    verified_payload: bytes,
) -> dict[str, Any]:
    input_spec = protocol["input"]
    verifier_path = PROJECT_ROOT / "scripts" / "verify_phase2_catalog.py"
    if sha256_bytes(verifier_path.read_bytes()) != input_spec[
        "catalog_verifier_sha256"
    ]:
        raise ValueError("catalog verifier source hash drift")
    if len(verified_rows) < int(input_spec["minimum_catalog_row_count"]):
        raise ValueError("catalog row count is below the frozen pilot minimum")
    if any(row.get("catalog_verifier_version") != input_spec["catalog_verifier_version"] for row in verified_rows):
        raise ValueError("catalog verifier version drift")
    model = protocol["model"]
    if any(row.get("model_id") != model["model_id"] for row in verified_rows):
        raise ValueError("catalog rows use the wrong model")
    if any(row.get("model_revision") != model["revision"] for row in verified_rows):
        raise ValueError("catalog rows use the wrong model revision")

    selection = protocol["selection"]
    label_field = input_spec["catalog_label_field"]
    salt = selection["selection_salt"]
    conflicts = one_per_title(
        verified_rows, input_spec["strict_conflict_label"], label_field, salt
    )
    conflict_titles = {row["normalized_title"] for row in conflicts}
    exact = one_per_title(
        verified_rows, input_spec["strict_exact_label"], label_field, salt
    )
    exact = [row for row in exact if row["normalized_title"] not in conflict_titles]
    exact = exact[: int(selection["maximum_strict_exact_controls"])]

    records = []
    for row in conflicts:
        records.append(
            {
                "record_id": row["record_id"],
                "source_group": "phase3_conflict",
                "title": row["title"],
                "target_artist": row["reference_artist"],
                "accepted_target_artists": accepted_reference_artists(row),
                "emitted_artist": row["artist"],
                "accepted_emitted_artists": [row["artist"]],
                "context_id": row["context_id"],
                "normalized_title": row["normalized_title"],
                "catalog_reference_semantics": row["reference_semantics"],
                "source_generation_sha256": row["generation_row_sha256"],
                "selection_hash": row["phase3_recoverability_selection_hash"],
            }
        )
    for row in exact:
        records.append(
            {
                "record_id": row["record_id"],
                "source_group": "phase3_generated_exact",
                "title": row["title"],
                "target_artist": row["artist"],
                "accepted_target_artists": [row["artist"]],
                "emitted_artist": None,
                "accepted_emitted_artists": [],
                "context_id": row["context_id"],
                "normalized_title": row["normalized_title"],
                "catalog_reference_semantics": "double-catalog strict exact",
                "source_generation_sha256": row["generation_row_sha256"],
                "selection_hash": row["phase3_recoverability_selection_hash"],
            }
        )
    for row in protocol["canonical_positive_controls"]:
        records.append(
            {
                "record_id": row["control_id"],
                "source_group": "canonical_positive",
                "title": row["title"],
                "target_artist": row["target_artist"],
                "accepted_target_artists": row["accepted_target_artists"],
                "emitted_artist": None,
                "accepted_emitted_artists": [],
                "context_id": None,
                "normalized_title": None,
                "catalog_reference_semantics": "frozen high-familiarity positive control",
                "source_generation_sha256": None,
                "selection_hash": None,
            }
        )
    records.sort(key=lambda row: (row["source_group"], row["record_id"]))
    if len({row["record_id"] for row in records}) != len(records):
        raise ValueError("recoverability records must have unique IDs")
    return {
        "bundle_version": "phase3_relation_recoverability_v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_status": protocol["status"],
        "protocol_hashes": {
            "json_sha256": sha256_bytes(protocol_bytes),
            "markdown_sha256": sha256_bytes(protocol_doc_bytes),
        },
        "protocol_payloads_b64": {
            "json": base64.b64encode(protocol_bytes).decode("ascii"),
            "markdown": base64.b64encode(protocol_doc_bytes).decode("ascii"),
        },
        "source_catalog_sha256": sha256_bytes(verified_payload),
        "source_catalog_row_count": len(verified_rows),
        "selected_conflict_cluster_count": len(conflicts),
        "selected_exact_cluster_count": len(exact),
        "model": model,
        "prompt_templates": protocol["prompt_templates"],
        "generation": protocol["generation"],
        "scoring": protocol["scoring"],
        "validity_gates": protocol["validity_gates"],
        "pilot_continuation_gate": protocol["pilot_continuation_gate"],
        "claim_boundaries": protocol["claim_boundaries"],
        "records": records,
        "expected_record_count": len(records),
        "expected_prompt_count": len(records) * len(protocol["prompt_templates"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-doc", type=Path, default=DEFAULT_PROTOCOL_DOC)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "phase3_relation_recoverability_bundle.json")
    args = parser.parse_args()

    protocol_bytes = args.protocol.read_bytes()
    protocol_doc_bytes = args.protocol_doc.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    catalog_path = args.catalog or PROJECT_ROOT / protocol["input"]["catalog_verified_rows"]
    catalog_payload = catalog_path.read_bytes()
    bundle = build_bundle(
        protocol,
        protocol_bytes,
        protocol_doc_bytes,
        load_jsonl(catalog_path),
        catalog_payload,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(f"conflicts={bundle['selected_conflict_cluster_count']}")
    print(f"exact_controls={bundle['selected_exact_cluster_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
