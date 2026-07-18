"""Freeze Phase 2 diagnosis, correction, and offline-scoring artifacts."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import urllib.parse
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from phase2_mechanism_analysis import normalize_entity, select_hash_ordered_controls
from verify_phase2_catalog import (
    CATALOG_VERIFIER_VERSION,
    apple_url,
    musicbrainz_url,
    request_cache_from_rows,
    verify_pair,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "config" / "phase2_mechanism_intervention_protocol.json"
DEFAULT_PROTOCOL_DOC = PROJECT_ROOT / "docs" / "phase2_mechanism_intervention_protocol.md"
DEFAULT_CATALOG_VERIFIER = PROJECT_ROOT / "scripts" / "verify_phase2_catalog.py"
DEFAULT_RUNNER_TEMPLATE = (
    PROJECT_ROOT / "scripts" / "run_phase2_mechanism_intervention_probe.py"
)
FINAL_CATALOG_LABELS = {
    "strict_exact",
    "strict_conflict",
    "ambiguous",
    "excluded",
    "error",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def generation_row_hash(row: dict[str, Any]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "generation_id": row["generation_id"],
                "rank": int(row["rank"]),
                "title": row["title"],
                "artist": row["artist"],
                "reason": row.get("reason", ""),
            }
        )
    )


def evidence_request_index(
    evidence_rows: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in evidence_rows:
        request_id = str(row.get("request_id", ""))
        if not request_id:
            raise ValueError("catalog evidence row has no request ID")
        if request_id in output:
            raise ValueError(f"catalog evidence request ID is not unique: {request_id}")
        for field in (
            "source",
            "request_url",
            "query_parameters",
            "queried_at_utc",
            "http_status",
            "request_status",
            "raw_response_body",
            "source_ids",
        ):
            if field not in row:
                raise ValueError(
                    f"catalog evidence {request_id} is missing required field {field}"
                )
        output[request_id] = row
    return output


def query_parameters(url: str) -> dict[str, list[str]]:
    return {
        key: values
        for key, values in urllib.parse.parse_qs(
            urllib.parse.urlsplit(url).query, keep_blank_values=True
        ).items()
    }


def expected_strict_query_identities(
    title: str, artist: str
) -> dict[str, tuple[str, str]]:
    expected = {
        musicbrainz_url(title, artist): ("musicbrainz", "title"),
        apple_url(title, artist): ("apple", "title"),
        musicbrainz_url(title, artist, targeted=True): (
            "musicbrainz",
            "title_artist",
        ),
        apple_url(title, artist, targeted=True): (
            "apple",
            "title_artist",
        ),
    }
    return expected


STRICT_RECOMPUTED_FIELDS = (
    "catalog_verifier_version",
    "catalog_label",
    "phase2_catalog_label",
    "classification_reason",
    "confirmatory_catalog_eligible",
    "catalog_evidence_complete_for_label",
    "alias_audit_complete",
    "reference_artist",
    "normalized_reference_artist",
    "reference_semantics",
    "catalog_reference",
    "shared_non_emitted_artists",
    "normalized_title",
    "normalized_artist",
)


def validate_and_recompute_strict_row(
    row: dict[str, Any],
    linked_evidence: list[dict[str, Any]],
) -> None:
    label = str(row["phase2_catalog_label"])
    expected = expected_strict_query_identities(
        str(row["title"]), str(row["artist"])
    )
    required_urls = {
        musicbrainz_url(str(row["title"]), str(row["artist"])),
        apple_url(str(row["title"]), str(row["artist"])),
    }
    if label == "strict_conflict":
        required_urls.update(
            {
                musicbrainz_url(
                    str(row["title"]), str(row["artist"]), targeted=True
                ),
                apple_url(str(row["title"]), str(row["artist"]), targeted=True),
            }
        )
    observed_urls: set[str] = set()
    for evidence in linked_evidence:
        url = str(evidence.get("request_url", ""))
        identity = expected.get(url)
        if identity is None:
            raise ValueError(
                f"strict row {row['record_id']} links an unrelated request URL"
            )
        if (evidence.get("source"), evidence.get("query_kind")) != identity:
            raise ValueError(
                f"strict row {row['record_id']} request source or query kind mismatch"
            )
        if evidence.get("query_parameters") != query_parameters(url):
            raise ValueError(
                f"strict row {row['record_id']} request parameters mismatch"
            )
        observed_urls.add(url)
    if not required_urls.issubset(observed_urls):
        raise ValueError(
            f"strict row {row['record_id']} does not link every expected query"
        )

    unexpected_fetches: list[str] = []

    def no_unlinked_network(url: str) -> Any:
        unexpected_fetches.append(url)
        raise RuntimeError("strict evidence replay attempted an unlinked request")

    replay_archive = [dict(evidence) for evidence in linked_evidence]
    recomputed = verify_pair(
        str(row["title"]),
        str(row["artist"]),
        accepted_artists=row.get("accepted_artists"),
        fetcher=no_unlinked_network,
        cache=request_cache_from_rows(linked_evidence),
        archive=replay_archive,
        sleep_seconds=0.0,
        max_attempts=1,
        sleep_fn=lambda _: None,
    )
    if unexpected_fetches:
        raise ValueError(
            f"strict row {row['record_id']} evidence replay was incomplete"
        )
    if set(recomputed["catalog_request_ids"]) != {
        str(evidence["request_id"]) for evidence in linked_evidence
    }:
        raise ValueError(f"strict row {row['record_id']} request IDs do not replay")
    for field in STRICT_RECOMPUTED_FIELDS:
        if row.get(field) != recomputed.get(field):
            raise ValueError(
                f"strict row {row['record_id']} replay mismatch for {field}"
            )


def validate_final_catalog_rows(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    protocol_hashes: dict[str, str],
    evidence_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("final catalog input is empty")
    expected_model = protocol["models"]["confirmatory"]
    evidence = evidence_request_index(evidence_rows)
    record_ids: set[str] = set()
    generation_hashes: set[str] = set()
    strict_count = 0
    for index, row in enumerate(rows):
        prefix = f"catalog row {index}"
        if row.get("record_type") != "generated_pair":
            raise ValueError(f"{prefix} is not a generated pair")
        record_id = str(row.get("record_id", ""))
        if not record_id or record_id in record_ids:
            raise ValueError(f"{prefix} has a missing or duplicate record ID")
        record_ids.add(record_id)
        label = row.get("phase2_catalog_label")
        if label not in FINAL_CATALOG_LABELS:
            raise ValueError(f"{prefix} has a pending or legacy catalog label")
        if row.get("catalog_verifier_version") != CATALOG_VERIFIER_VERSION:
            raise ValueError(f"{prefix} was not classified by the frozen v2 verifier")
        if row.get("protocol_id") != protocol["protocol_id"]:
            raise ValueError(f"{prefix} protocol ID mismatch")
        if row.get("protocol_hashes") != protocol_hashes:
            raise ValueError(f"{prefix} protocol hashes mismatch")
        if row.get("model_id") != expected_model["model_id"]:
            raise ValueError(f"{prefix} model ID mismatch")
        if row.get("model_revision") != expected_model["revision"]:
            raise ValueError(f"{prefix} model revision mismatch")
        row_hash = str(row.get("generation_row_sha256", ""))
        if row_hash != generation_row_hash(row) or row_hash in generation_hashes:
            raise ValueError(f"{prefix} generation row hash is invalid or duplicated")
        generation_hashes.add(row_hash)
        request_ids = [str(value) for value in row.get("catalog_request_ids", [])]
        if (
            not request_ids
            or len(set(request_ids)) != len(request_ids)
            or any(request_id not in evidence for request_id in request_ids)
        ):
            raise ValueError(f"{prefix} is not fully linked to the evidence archive")
        if label in {"strict_exact", "strict_conflict"}:
            strict_count += 1
            if not row.get("confirmatory_catalog_eligible"):
                raise ValueError(f"{prefix} strict label is not marked eligible")
            if not row.get("catalog_evidence_complete_for_label"):
                raise ValueError(f"{prefix} strict evidence is incomplete")
            if not row.get("alias_audit_complete"):
                raise ValueError(f"{prefix} strict alias audit is incomplete")
            if any(
                evidence[request_id].get("request_status") != "ok"
                for request_id in request_ids
            ):
                raise ValueError(f"{prefix} strict evidence includes a failed request")
            if any(
                not isinstance(evidence[request_id].get("http_status"), int)
                or not 200 <= int(evidence[request_id]["http_status"]) < 300
                or not str(evidence[request_id].get("raw_response_body", ""))
                for request_id in request_ids
            ):
                raise ValueError(f"{prefix} strict evidence has no successful raw body")
            linked_sources = {
                str(evidence[request_id].get("source")) for request_id in request_ids
            }
            if linked_sources != {"musicbrainz", "apple"}:
                raise ValueError(f"{prefix} strict evidence is not double-catalog")
            validate_and_recompute_strict_row(
                row, [evidence[request_id] for request_id in request_ids]
            )
    return {
        "catalog_row_count": len(rows),
        "strict_catalog_row_count": strict_count,
        "evidence_request_count": len(evidence),
        "record_ids_unique": True,
        "generation_row_hashes_valid_and_unique": True,
        "catalog_verifier_version": CATALOG_VERIFIER_VERSION,
    }


def render_embedded_runner(template_bytes: bytes, bundle: dict[str, Any]) -> bytes:
    placeholder = '__PHASE2_MECHANISM_BUNDLE_ZLIB_B64__'
    template = template_bytes.decode("ascii")
    if template.count(placeholder) != 2:
        raise ValueError("runner template bundle placeholder count changed")
    encoded = base64.b64encode(
        zlib.compress(canonical_json_bytes(bundle), level=9)
    ).decode("ascii")
    return template.replace(placeholder, encoded, 1).encode("ascii")


def strict_label(row: dict[str, Any]) -> str | None:
    value = row.get("phase2_catalog_label") or row.get("strict_catalog_label")
    return {
        "strict_exact": "strict_exact",
        "strict_conflict": "strict_conflict",
    }.get(value)


def normalized_title(row: dict[str, Any]) -> str:
    return row.get("normalized_title") or normalize_entity(row["title"])


def normalized_artist(row: dict[str, Any], field: str = "artist") -> str:
    key = "normalized_artist" if field == "artist" else f"normalized_{field}"
    return row.get(key) or normalize_entity(row[field])


def reference_artist(row: dict[str, Any]) -> str:
    if row.get("reference_artist"):
        return row["reference_artist"]
    for key in ("catalog_reference", "reference_catalog"):
        if isinstance(row.get(key), dict) and row[key].get("artist"):
            return row[key]["artist"]
    raise ValueError(f"strict conflict {row.get('record_id')} has no reference artist")


def catalog_artist_keys(row: dict[str, Any]) -> set[str]:
    direct = row.get("catalog_artist_keys")
    values: list[str] = []
    if isinstance(direct, dict):
        for artists in direct.values():
            values.extend(artists)
    elif isinstance(direct, list):
        values.extend(direct)
    for source in row.get("catalog_sources", []):
        evidence = source.get("evidence", source)
        candidates = evidence.get("title_matches", [])
        values.extend(
            candidate.get("artist", "")
            for candidate in candidates
            if isinstance(candidate, dict)
        )
    source_evidence = row.get("source_evidence", {})
    if isinstance(source_evidence, dict):
        for evidence in source_evidence.values():
            values.extend(evidence.get("title_artists", []))
            values.extend(
                candidate.get("artist", "")
                for candidate in evidence.get("title_matches", [])
                if isinstance(candidate, dict)
            )
    return {normalize_entity(value) for value in values if normalize_entity(value)}


def stable_row_hash(row: dict[str, Any], salt: str) -> str:
    identity = {
        "record_id": row.get("record_id"),
        "generation_id": row.get("generation_id"),
        "rank": row.get("rank"),
        "normalized_title": normalized_title(row),
        "normalized_artist": normalized_artist(row),
        "generation_row_sha256": row.get("generation_row_sha256"),
    }
    payload = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(f"{salt}:{payload}".encode("utf-8")).hexdigest()


def select_catalog_clusters(
    rows: Iterable[dict[str, Any]], protocol: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selection = protocol["generation"]["selection"]
    strict_rows = [row for row in rows if strict_label(row) is not None]
    by_title: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in strict_rows:
        title_key = normalized_title(row)
        if not title_key or not normalized_artist(row):
            continue
        by_title[title_key].append(row)

    cluster_rows = []
    for title_key, candidates in by_title.items():
        selected = min(
            candidates,
            key=lambda row: stable_row_hash(
                row, selection["within_cluster_row_hash_salt"]
            ),
        )
        cluster_rows.append({**selected, "normalized_title": title_key})

    def select_label(label: str, salt_key: str, maximum_key: str) -> list[dict[str, Any]]:
        candidates = [row for row in cluster_rows if strict_label(row) == label]
        candidates.sort(
            key=lambda row: hashlib.sha256(
                f"{selection[salt_key]}:{normalized_title(row)}".encode("utf-8")
            ).hexdigest()
        )
        return candidates[: int(selection[maximum_key])]

    conflicts = select_label(
        "strict_conflict", "conflict_hash_salt", "maximum_unique_conflict_title_clusters"
    )
    exact = select_label(
        "strict_exact", "exact_hash_salt", "maximum_unique_exact_title_clusters"
    )
    summary = {
        "strict_input_row_count": len(strict_rows),
        "strict_cluster_count_before_caps": len(cluster_rows),
        "strict_conflict_cluster_count_before_cap": sum(
            strict_label(row) == "strict_conflict" for row in cluster_rows
        ),
        "strict_exact_cluster_count_before_cap": sum(
            strict_label(row) == "strict_exact" for row in cluster_rows
        ),
        "selected_conflict_cluster_count": len(conflicts),
        "selected_exact_cluster_count": len(exact),
    }
    return conflicts, exact, summary


def eligible_neutral_rows(
    conflict: dict[str, Any], exact_rows: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    conflict_title = normalized_title(conflict)
    emitted_key = normalized_artist(conflict)
    reference_key = normalize_entity(reference_artist(conflict))
    output = []
    for row in exact_rows:
        title_key = normalized_title(row)
        exact_key = normalized_artist(row)
        listed = catalog_artist_keys(row)
        if title_key == conflict_title:
            continue
        if exact_key in {emitted_key, reference_key}:
            continue
        if emitted_key in listed or reference_key in listed:
            continue
        output.append(row)
    return output


def false_candidate_for_exact(
    row: dict[str, Any], exact_rows: Iterable[dict[str, Any]], salt: str
) -> str | None:
    title_key = normalized_title(row)
    actual_key = normalized_artist(row)
    listed = catalog_artist_keys(row)
    candidates = {
        candidate["artist"]
        for candidate in exact_rows
        if normalized_title(candidate) != title_key
        and normalized_artist(candidate) != actual_key
        and normalized_artist(candidate) not in listed
    }
    ordered = sorted(
        candidates,
        key=lambda artist: hashlib.sha256(
            f"{salt}:{title_key}:{normalize_entity(artist)}".encode("utf-8")
        ).hexdigest(),
    )
    return ordered[0] if ordered else None


def rendered_correction_record(
    record_id: str,
    record_type: str,
    title: str,
    emitted_artist: str,
    correction: dict[str, Any],
) -> dict[str, Any]:
    variables = {"title": title, "emitted_artist": emitted_artist}
    return {
        "record_id": record_id,
        "record_type": record_type,
        "title": title,
        "emitted_artist": emitted_artist,
        "prompts": {
            "naive_candidate_free_self_check": [
                template.format(**variables)
                for template in correction["naive_candidate_free_templates"]
            ],
            "anti_prior_candidate_free_recall": [
                template.format(**variables)
                for template in correction["anti_prior_candidate_free_templates"]
            ],
        },
    }


def build_artifacts(
    catalog_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    protocol_bytes: bytes,
    protocol_doc_bytes: bytes,
    *,
    enforce_final_stop: bool = False,
    catalog_evidence_rows: list[dict[str, Any]] | None = None,
    catalog_asset_hashes: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    conflicts, exact_rows, selection_summary = select_catalog_clusters(
        catalog_rows, protocol
    )
    minimum_exact = int(
        protocol["mechanism_diagnosis"]["minimum_global_unique_exact_pool_titles"]
    )
    if enforce_final_stop and len(exact_rows) < minimum_exact:
        raise ValueError("STOP_PHASE2: fewer than six unique strict exact pool titles")
    minimum_conflicts = int(
        protocol["stop_conditions"][
            "minimum_unique_strict_conflict_title_clusters_after_cap"
        ]
    )
    if enforce_final_stop and len(conflicts) < minimum_conflicts:
        raise ValueError("STOP_PHASE2: fewer than 30 strict conflict title clusters")

    protocol_hashes = {
        "json_sha256": sha256_bytes(protocol_bytes),
        "markdown_sha256": sha256_bytes(protocol_doc_bytes),
    }
    catalog_validation = None
    if enforce_final_stop:
        if catalog_evidence_rows is None or catalog_asset_hashes is None:
            raise ValueError("final export requires catalog evidence and asset hashes")
        catalog_validation = validate_final_catalog_rows(
            catalog_rows, protocol, protocol_hashes, catalog_evidence_rows
        )
    model = protocol["models"]["confirmatory"]
    diagnosis_spec = protocol["mechanism_diagnosis"]
    correction_spec = protocol["correction"]
    common = {
        "protocol_id": protocol["protocol_id"],
        "protocol_hashes": protocol_hashes,
        "model_id": model["model_id"],
        "model_revision": model["revision"],
        "expected_num_layers": model["expected_num_layers"],
        "endpoint_tolerance": 0.02,
        "catalog_asset_hashes": dict(catalog_asset_hashes or {}),
    }

    diagnosis_records = []
    correction_records = []
    scoring_records = []
    neutral_failure_count = 0
    for conflict in conflicts:
        eligible = eligible_neutral_rows(conflict, exact_rows)
        by_key = {normalized_title(row): row for row in eligible}
        control_keys = select_hash_ordered_controls(
            normalized_title(conflict), by_key, count=6
        )
        controls = [
            {"title": by_key[key]["title"], "normalized_title": key}
            for key in control_keys
        ]
        technical_failure = None
        if len(controls) < 6:
            technical_failure = "insufficient_eligible_neutral_titles"
            neutral_failure_count += 1
        record_id = str(conflict["record_id"])
        emitted = conflict["artist"]
        reference = reference_artist(conflict)
        diagnosis_records.append(
            {
                "record_id": record_id,
                "normalized_title": normalized_title(conflict),
                "title": conflict["title"],
                "emitted_artist": emitted,
                "reference_artist": reference,
                "diagnostic_controls": controls[:3],
                "manipulation_controls": controls[3:6],
                "technical_failure": technical_failure,
            }
        )
        correction_records.append(
            rendered_correction_record(
                record_id, "strict_conflict", conflict["title"], emitted, correction_spec
            )
        )
        scoring_records.append(
            {
                "record_id": record_id,
                "record_type": "strict_conflict",
                "normalized_title": normalized_title(conflict),
                "reference_artist": reference,
                "catalog_safety_action": "return_reference",
            }
        )

    unavailable_exact_controls = 0
    for row in exact_rows:
        false_artist = false_candidate_for_exact(
            row, exact_rows, correction_spec["exact_false_candidate_selection_salt"]
        )
        if false_artist is None:
            unavailable_exact_controls += 1
            continue
        record_id = f"exact_control::{row['record_id']}"
        correction_records.append(
            rendered_correction_record(
                record_id, "strict_exact_control", row["title"], false_artist, correction_spec
            )
        )
        scoring_records.append(
            {
                "record_id": record_id,
                "record_type": "strict_exact_control",
                "normalized_title": normalized_title(row),
                "reference_artist": row["artist"],
                "catalog_safety_action": "return_reference",
            }
        )

    diagnosis_bundle = {
        "bundle_version": "phase2_mechanism_diagnosis_v1",
        **common,
        "layers": diagnosis_spec["layers"],
        "prompt_templates": diagnosis_spec["diagnostic_prompt_templates"],
        "minimum_absolute_patch_relation_shift_nats": diagnosis_spec[
            "minimum_absolute_patch_relation_shift_nats"
        ],
        "minimum_layers_same_direction": diagnosis_spec[
            "minimum_layers_same_direction"
        ],
        "maximum_selected_cluster_technical_failure_rate": diagnosis_spec[
            "patch_specification"
        ]["maximum_selected_cluster_technical_failure_rate"],
        "closed_set_manipulation_prompt_template": correction_spec[
            "closed_set_manipulation_prompt_template"
        ],
        "records": diagnosis_records,
    }
    correction_bundle = {
        "bundle_version": "phase2_candidate_free_correction_v1",
        **common,
        "generation": correction_spec["generation"],
        "records": correction_records,
    }
    scoring_key = {
        "bundle_version": "phase2_candidate_free_scoring_key_v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_hashes": protocol_hashes,
        "model_id": model["model_id"],
        "model_revision": model["revision"],
        "catalog_asset_hashes": dict(catalog_asset_hashes or {}),
        "records": scoring_records,
    }
    scoring_key["execution_assets"] = {
        "diagnosis": {
            "canonical_bundle_sha256": sha256_bytes(
                canonical_json_bytes(diagnosis_bundle)
            ),
            "submitted_script_sha256": None,
        },
        "correction": {
            "canonical_bundle_sha256": sha256_bytes(
                canonical_json_bytes(correction_bundle)
            ),
            "submitted_script_sha256": None,
        },
    }
    parsed_event_count = sum(
        row.get("record_type") == "generated_pair" for row in catalog_rows
    )
    extension = protocol["generation"]["extension_trigger"]
    extension_required = (
        parsed_event_count < int(extension["minimum_primary_parsed_events"])
        or selection_summary["strict_conflict_cluster_count_before_cap"]
        < int(extension["minimum_primary_unique_strict_conflict_title_clusters"])
    )
    export_summary = {
        **selection_summary,
        "parsed_event_count": parsed_event_count,
        "extension_required_after_primary": extension_required,
        "minimum_conflicts_final_stop": minimum_conflicts,
        "minimum_exact_pool_titles": minimum_exact,
        "neutral_control_failure_count": neutral_failure_count,
        "neutral_control_failure_rate": (
            neutral_failure_count / len(conflicts) if conflicts else None
        ),
        "candidate_free_conflict_record_count": len(conflicts),
        "candidate_free_exact_control_count": sum(
            row["record_type"] == "strict_exact_control"
            for row in correction_records
        ),
        "unavailable_exact_control_count": unavailable_exact_controls,
        "correction_bundle_contains_reference_field": any(
            "reference_artist" in json.dumps(row, ensure_ascii=False)
            for row in correction_records
        ),
        "protocol_hashes": protocol_hashes,
        "catalog_validation": catalog_validation,
        "catalog_asset_hashes": dict(catalog_asset_hashes or {}),
        "execution_assets": scoring_key["execution_assets"],
    }
    return diagnosis_bundle, correction_bundle, scoring_key, export_summary


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-rows", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-doc", type=Path, default=DEFAULT_PROTOCOL_DOC)
    parser.add_argument("--catalog-evidence-archive", type=Path)
    parser.add_argument(
        "--catalog-verifier-script", type=Path, default=DEFAULT_CATALOG_VERIFIER
    )
    parser.add_argument("--runner-template", type=Path, default=DEFAULT_RUNNER_TEMPLATE)
    parser.add_argument("--diagnosis-bundle", type=Path, required=True)
    parser.add_argument("--correction-bundle", type=Path, required=True)
    parser.add_argument("--diagnosis-runner", type=Path)
    parser.add_argument("--correction-runner", type=Path)
    parser.add_argument("--scoring-key", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()

    if args.final and (
        args.catalog_evidence_archive is None
        or args.diagnosis_runner is None
        or args.correction_runner is None
    ):
        raise ValueError(
            "--final requires catalog evidence plus diagnosis/correction runner paths"
        )

    protocol_bytes = args.protocol.read_bytes()
    protocol_doc_bytes = args.protocol_doc.read_bytes()
    catalog_bytes = args.catalog_rows.read_bytes()
    catalog_rows = read_jsonl(args.catalog_rows)
    evidence_bytes = (
        args.catalog_evidence_archive.read_bytes()
        if args.catalog_evidence_archive is not None
        else b""
    )
    evidence_rows = (
        read_jsonl(args.catalog_evidence_archive)
        if args.catalog_evidence_archive is not None
        else None
    )
    catalog_asset_hashes = {
        "catalog_rows_sha256": sha256_bytes(catalog_bytes),
        "catalog_evidence_sha256": sha256_bytes(evidence_bytes),
        "catalog_verifier_script_sha256": sha256_bytes(
            args.catalog_verifier_script.read_bytes()
        ),
    }
    diagnosis, correction, scoring, summary = build_artifacts(
        catalog_rows,
        json.loads(protocol_bytes.decode("utf-8")),
        protocol_bytes,
        protocol_doc_bytes,
        enforce_final_stop=args.final,
        catalog_evidence_rows=evidence_rows,
        catalog_asset_hashes=catalog_asset_hashes,
    )
    if args.diagnosis_runner is not None or args.correction_runner is not None:
        if args.diagnosis_runner is None or args.correction_runner is None:
            raise ValueError("both embedded runner output paths are required")
        template_bytes = args.runner_template.read_bytes()
        diagnosis_runner = render_embedded_runner(template_bytes, diagnosis)
        correction_runner = render_embedded_runner(template_bytes, correction)
        args.diagnosis_runner.parent.mkdir(parents=True, exist_ok=True)
        args.diagnosis_runner.write_bytes(diagnosis_runner)
        args.correction_runner.parent.mkdir(parents=True, exist_ok=True)
        args.correction_runner.write_bytes(correction_runner)
        scoring["execution_assets"]["diagnosis"][
            "submitted_script_sha256"
        ] = sha256_bytes(diagnosis_runner)
        scoring["execution_assets"]["correction"][
            "submitted_script_sha256"
        ] = sha256_bytes(correction_runner)
        summary["execution_assets"] = scoring["execution_assets"]
    outputs = (diagnosis, correction, scoring, summary)
    for path, value in zip(
        (
            args.diagnosis_bundle,
            args.correction_bundle,
            args.scoring_key,
            args.summary,
        ),
        outputs,
    ):
        write_json(path, value)
    print("PHASE2_MECHANISM_EXPORT_SUMMARY_JSON=" + json.dumps(outputs[-1], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
