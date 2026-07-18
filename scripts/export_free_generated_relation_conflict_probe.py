"""Export a frozen exact-vs-catalog-conflict relation-verification pilot."""

from __future__ import annotations

import argparse
import base64
import json
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

from run_song_entity_generation_time_probe import parse_playlist
from verify_song_entity_catalog import normalize_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def exact_in_both_sources(row: dict[str, Any]) -> bool:
    emitted_key = normalize_name(row["artist"])
    sources = {source["source"]: source for source in row["catalog_sources"]}
    if set(sources) != {"musicbrainz", "apple"}:
        return False
    return all(
        source["status"] == "ok"
        and any(
            normalize_name(candidate.get("artist", "")) == emitted_key
            for candidate in source["evidence"]["exact"]
        )
        for source in sources.values()
    )


def shared_catalog_reference(row: dict[str, Any]) -> dict[str, Any] | None:
    emitted_key = normalize_name(row["artist"])
    sources = {
        source["source"]: source
        for source in row["catalog_sources"]
        if source["status"] == "ok"
    }
    if set(sources) != {"musicbrainz", "apple"}:
        return None
    counters = {}
    displays = {}
    for source_name, source in sources.items():
        artists = [
            candidate.get("artist", "")
            for candidate in source["evidence"]["title_matches"]
            if candidate.get("artist")
        ]
        counters[source_name] = Counter(normalize_name(artist) for artist in artists)
        displays[source_name] = {
            normalize_name(artist): artist for artist in artists
        }
    shared = set(counters["musicbrainz"]) & set(counters["apple"])
    shared.discard(emitted_key)
    if len(shared) != 1:
        return None
    reference_key = next(iter(shared))
    return {
        "artist": displays["apple"][reference_key],
        "normalized_artist": reference_key,
        "combined_support": counters["musicbrainz"][reference_key]
        + counters["apple"][reference_key],
    }


def select_transfer_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exact = sorted(
        [
            {**row, "reference_catalog": None}
            for row in rows
            if row.get("catalog_label") == "verified_exact"
            and exact_in_both_sources(row)
        ],
        key=lambda row: (row["generation_id"], int(row["rank"])),
    )
    if len(exact) != 6:
        raise ValueError(f"expected six double-source exact rows, found {len(exact)}")
    context_quota = Counter(row["context_id"] for row in exact)

    conflict_candidates = []
    for row in rows:
        if row.get("catalog_label") != "catalog_conflict":
            continue
        reference = shared_catalog_reference(row)
        if reference is not None:
            conflict_candidates.append({**row, "reference_catalog": reference})
    selected_conflicts = []
    for context_id, quota in sorted(context_quota.items()):
        candidates = sorted(
            [row for row in conflict_candidates if row["context_id"] == context_id],
            key=lambda row: (
                -int(row["reference_catalog"]["combined_support"]),
                row["generation_id"],
                int(row["rank"]),
            ),
        )
        if len(candidates) < quota:
            raise ValueError(f"not enough deterministic conflicts for {context_id}")
        selected_conflicts.extend(candidates[:quota])
    if len(selected_conflicts) != 6:
        raise ValueError("expected six context-matched conflict rows")
    return exact + sorted(
        selected_conflicts,
        key=lambda row: (row["generation_id"], int(row["rank"])),
    )


def attach_references(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exact = [row for row in rows if row["catalog_label"] == "verified_exact"]
    output = []
    for row in rows:
        if row["catalog_label"] == "catalog_conflict":
            reference_artist = row["reference_catalog"]["artist"]
            reference_role = "double_source_catalog_artist"
        else:
            row_index = exact.index(row)
            reference_artist = None
            for offset in range(1, len(exact) + 1):
                candidate = exact[(row_index + offset) % len(exact)]["artist"]
                if normalize_name(candidate) != normalize_name(row["artist"]):
                    reference_artist = candidate
                    break
            if reference_artist is None:
                raise ValueError("could not construct an exact-row wrong control")
            reference_role = "deranged_exact_artist_control"
        output.append(
            {
                **row,
                "reference_artist": reference_artist,
                "reference_role": reference_role,
                "emitted_is_correct": row["catalog_label"] == "verified_exact",
            }
        )
    return output


def reconstruct_prefixes(
    rows: list[dict[str, Any]],
    generation_bundle: dict[str, Any],
    raw_generations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    contexts = {row["context_id"]: row for row in generation_bundle["contexts"]}
    raw_by_id = {row["generation_id"]: row["completion"] for row in raw_generations}
    output = []
    for row in rows:
        context = contexts[row["context_id"]]
        full_text = context["generation_prompt"].rstrip() + raw_by_id[row["generation_id"]]
        parsed = parse_playlist(full_text)
        rank = int(row["rank"])
        if rank < 1 or rank > len(parsed):
            raise ValueError(f"rank missing from raw generation: {row['generation_id']}#{rank}")
        parsed_row = parsed[rank - 1]
        if (
            normalize_name(parsed_row["title"]) != normalize_name(row["title"])
            or normalize_name(parsed_row["artist"]) != normalize_name(row["artist"])
        ):
            raise ValueError(f"raw prefix mismatch: {row['generation_id']}#{rank}")
        generation_prefix = full_text[: parsed_row["artist_span"][0]].rstrip()
        output.append(
            {
                "record_id": f"{row['generation_id']}__rank{rank}",
                "generation_id": row["generation_id"],
                "context_id": row["context_id"],
                "seed": int(row["seed"]),
                "rank": rank,
                "title": row["title"],
                "emitted_artist": row["artist"],
                "reference_artist": row["reference_artist"],
                "reference_role": row["reference_role"],
                "catalog_label": row["catalog_label"],
                "emitted_is_correct": bool(row["emitted_is_correct"]),
                "reason": row["reason"],
                "generation_prefix": generation_prefix,
                "catalog_prefix": (
                    f"Complete the factual catalog entry.\nTitle: {row['title']}\nArtist:"
                ),
                "catalog_reference_support": (
                    row["reference_catalog"]["combined_support"]
                    if row["reference_catalog"] is not None
                    else None
                ),
            }
        )
    return output


def build_bundle(
    catalog_rows: list[dict[str, Any]],
    generation_bundle: dict[str, Any],
    raw_generations: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = attach_references(select_transfer_rows(catalog_rows))
    records = reconstruct_prefixes(selected, generation_bundle, raw_generations)
    return {
        "bundle_version": "free_generated_relation_conflict_v1",
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "records": records,
        "selected_layers": [14, 18, 21, 24, 27, 28],
        "interventions": [
            {"layer": 21, "component": "attention"},
            {"layer": 24, "component": "mlp"},
            {"layer": 27, "component": "full_residual"},
            {"layer": 28, "component": "full_residual"},
        ],
        "minimum_patch_effect": 0.10,
        "endpoint_tolerance": 0.02,
        "minimum_balanced_accuracy": 0.75,
        "minimum_group_correct": 4,
        "field_reference_run": "20260711T121410Z_pilot",
        "raw_generation_job": "6a506af0a9bcc59cfbc4b812",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog-rows",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "qwen_scope_song_entity_generation_time_full_catalog_verified.jsonl",
    )
    parser.add_argument(
        "--generation-bundle",
        type=Path,
        default=PROJECT_ROOT / "runs" / "qwen_scope_song_entity_generation_time_bundle.json",
    )
    parser.add_argument(
        "--raw-generations",
        type=Path,
        default=PROJECT_ROOT / "runs" / "qwen_scope_song_entity_generation_format_raw.jsonl",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT / "runs" / "free_generated_relation_conflict_bundle.json",
    )
    parser.add_argument(
        "--encoded",
        type=Path,
        default=PROJECT_ROOT / "runs" / "free_generated_relation_conflict_bundle.zlib.b64",
    )
    args = parser.parse_args()

    bundle = build_bundle(
        load_jsonl(args.catalog_rows),
        load_json(args.generation_bundle),
        load_jsonl(args.raw_generations),
    )
    payload = json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    args.bundle.write_bytes(payload + b"\n")
    args.encoded.write_text(
        base64.b64encode(zlib.compress(payload, level=9)).decode("ascii") + "\n",
        encoding="ascii",
    )
    labels = Counter(row["catalog_label"] for row in bundle["records"])
    print(f"Wrote transfer bundle: {dict(labels)}")
    print(f"- {args.bundle}")
    print(f"- {args.encoded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
