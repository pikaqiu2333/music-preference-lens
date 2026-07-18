"""Export the frozen Granite relation-knowledge recoverability audit bundle."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "config" / "relation_knowledge_recoverability_audit.json"
)
DEFAULT_PROTOCOL_DOC = (
    PROJECT_ROOT / "docs" / "relation_knowledge_recoverability_audit.md"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def reference_artist_names(row: dict[str, Any]) -> list[str]:
    names = {row["reference_artist"]}
    for source in row["catalog_reference"]["sources"].values():
        names.update(source.get("artist_names", []))
    return sorted(name for name in names if name)


def build_bundle(
    protocol: dict[str, Any],
    protocol_bytes: bytes,
    protocol_doc_bytes: bytes,
    conflicts: list[dict[str, Any]],
    exact_controls: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_conflicts = protocol["inputs"]["expected_conflict_count"]
    expected_exact = protocol["inputs"]["expected_generated_exact_count"]
    if len(conflicts) != expected_conflicts:
        raise ValueError(f"expected {expected_conflicts} conflicts, found {len(conflicts)}")
    if len(exact_controls) != expected_exact:
        raise ValueError(f"expected {expected_exact} exact rows, found {len(exact_controls)}")

    model = protocol["model"]
    source_rows = [*conflicts, *exact_controls]
    if any(row["model_id"] != model["model_id"] for row in source_rows):
        raise ValueError("source rows do not use the frozen model")
    if any(row["model_revision"] != model["revision"] for row in source_rows):
        raise ValueError("source rows do not use the frozen model revision")
    if any(row["catalog_label"] != "strict_conflict" for row in conflicts):
        raise ValueError("conflict input contains a non-conflict row")
    if any(row["catalog_label"] != "strict_exact" for row in exact_controls):
        raise ValueError("exact input contains a non-exact row")

    records = []
    for row in conflicts:
        records.append(
            {
                "record_id": row["record_id"],
                "source_group": "phase2_conflict",
                "title": row["title"],
                "target_artist": row["reference_artist"],
                "accepted_target_artists": reference_artist_names(row),
                "emitted_artist": row["artist"],
                "accepted_emitted_artists": [row["artist"]],
                "context_id": row["context_id"],
                "catalog_reference_semantics": row["reference_semantics"],
                "source_generation_sha256": row["generation_row_sha256"],
            }
        )
    for row in exact_controls:
        records.append(
            {
                "record_id": row["record_id"],
                "source_group": "phase2_generated_exact",
                "title": row["title"],
                "target_artist": row["artist"],
                "accepted_target_artists": [row["artist"]],
                "emitted_artist": None,
                "accepted_emitted_artists": [],
                "context_id": row["context_id"],
                "catalog_reference_semantics": "double-catalog strict exact",
                "source_generation_sha256": row["generation_row_sha256"],
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
                "catalog_reference_semantics": "frozen high-familiarity positive control",
                "source_generation_sha256": None,
            }
        )

    records.sort(key=lambda row: (row["source_group"], row["record_id"]))
    if len({row["record_id"] for row in records}) != len(records):
        raise ValueError("audit records must have unique ids")
    prompt_templates = protocol["prompt_templates"]
    expected_prompt_count = len(records) * len(prompt_templates)
    return {
        "bundle_version": "relation_knowledge_recoverability_audit_v1",
        "audit_id": protocol["audit_id"],
        "protocol_hashes": {
            "json_sha256": sha256_bytes(protocol_bytes),
            "markdown_sha256": sha256_bytes(protocol_doc_bytes),
        },
        "protocol_payloads_b64": {
            "json": base64.b64encode(protocol_bytes).decode("ascii"),
            "markdown": base64.b64encode(protocol_doc_bytes).decode("ascii"),
        },
        "source_hashes": {
            "conflicts_sha256": sha256_bytes(
                (PROJECT_ROOT / protocol["inputs"]["conflicts"]).read_bytes()
            ),
            "generated_exact_controls_sha256": sha256_bytes(
                (
                    PROJECT_ROOT
                    / protocol["inputs"]["generated_exact_controls"]
                ).read_bytes()
            ),
        },
        "model": model,
        "prompt_templates": prompt_templates,
        "generation": protocol["generation"],
        "scoring": protocol["scoring"],
        "validity_gates": protocol["validity_gates"],
        "claim_boundaries": protocol["claim_boundaries"],
        "downstream_rule": protocol["downstream_rule"],
        "records": records,
        "expected_record_count": len(records),
        "expected_prompt_count": expected_prompt_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-doc", type=Path, default=DEFAULT_PROTOCOL_DOC)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "relation_knowledge_audit_bundle.json",
    )
    args = parser.parse_args()

    protocol_bytes = args.protocol.read_bytes()
    protocol_doc_bytes = args.protocol_doc.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    bundle = build_bundle(
        protocol,
        protocol_bytes,
        protocol_doc_bytes,
        load_jsonl(PROJECT_ROOT / protocol["inputs"]["conflicts"]),
        load_jsonl(PROJECT_ROOT / protocol["inputs"]["generated_exact_controls"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(f"records={bundle['expected_record_count']}")
    print(f"prompts={bundle['expected_prompt_count']}")
    print(f"protocol_json_sha256={bundle['protocol_hashes']['json_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
