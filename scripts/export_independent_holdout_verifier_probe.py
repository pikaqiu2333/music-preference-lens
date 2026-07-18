"""Export the pre-specified independent relation-conflict verifier holdout."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

from export_free_generated_relation_conflict_probe import (
    exact_in_both_sources,
    load_json,
    load_jsonl,
    shared_catalog_reference,
)
from run_song_entity_generation_time_probe import parse_playlist
from verify_song_entity_catalog import normalize_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HASH_SALT = "holdout_v1"


def selection_hash(row: dict[str, Any]) -> str:
    key = f"{HASH_SALT}:{row['generation_id']}:{int(row['rank'])}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def qualifying_pools(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exact = [
        {**row, "reference_catalog": None, "selection_hash": selection_hash(row)}
        for row in rows
        if row.get("catalog_label") == "verified_exact"
        and exact_in_both_sources(row)
    ]
    conflict = []
    for row in rows:
        if row.get("catalog_label") != "catalog_conflict":
            continue
        reference = shared_catalog_reference(row)
        if reference is not None:
            conflict.append(
                {
                    **row,
                    "reference_catalog": reference,
                    "selection_hash": selection_hash(row),
                }
            )
    return (
        sorted(exact, key=lambda row: row["selection_hash"]),
        sorted(conflict, key=lambda row: row["selection_hash"]),
    )


def select_holdout_rows(
    rows: list[dict[str, Any]],
    maximum_per_label: int = 12,
    minimum_per_label: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exact, conflict = qualifying_pools(rows)
    count = min(len(exact), len(conflict), maximum_per_label)
    if count < minimum_per_label:
        raise ValueError(
            "independent holdout is too small: "
            f"exact={len(exact)}, conflict={len(conflict)}, required={minimum_per_label}"
        )
    selected = exact[:count] + conflict[:count]
    return selected, {
        "hash_salt": HASH_SALT,
        "double_source_exact_pool_count": len(exact),
        "unique_shared_conflict_pool_count": len(conflict),
        "maximum_per_label": maximum_per_label,
        "minimum_per_label": minimum_per_label,
        "selected_per_label": count,
    }


def attach_references(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exact = sorted(
        [row for row in rows if row["catalog_label"] == "verified_exact"],
        key=lambda row: row["selection_hash"],
    )
    output = []
    for row in rows:
        if row["catalog_label"] == "catalog_conflict":
            reference_artist = row["reference_catalog"]["artist"]
            reference_role = "double_source_catalog_artist"
        else:
            index = exact.index(row)
            reference_artist = None
            for offset in range(1, len(exact) + 1):
                candidate = exact[(index + offset) % len(exact)]["artist"]
                if normalize_name(candidate) != normalize_name(row["artist"]):
                    reference_artist = candidate
                    break
            if reference_artist is None:
                raise ValueError("could not construct an exact-row wrong control")
            reference_role = "hash_order_deranged_exact_artist_control"
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
            raise ValueError(f"rank missing from raw generation: {row['record_id']}")
        parsed_row = parsed[rank - 1]
        if (
            normalize_name(parsed_row["title"]) != normalize_name(row["title"])
            or normalize_name(parsed_row["artist"]) != normalize_name(row["artist"])
        ):
            raise ValueError(f"raw prefix mismatch: {row['record_id']}")
        generation_prefix = full_text[: parsed_row["artist_span"][0]].rstrip()
        output.append(
            {
                "record_id": row["record_id"],
                "generation_id": row["generation_id"],
                "context_id": row["context_id"],
                "seed": int(row["seed"]),
                "rank": rank,
                "selection_hash": row["selection_hash"],
                "title": row["title"],
                "emitted_artist": row["artist"],
                "reference_artist": row["reference_artist"],
                "reference_role": row["reference_role"],
                "catalog_label": row["catalog_label"],
                "emitted_is_correct": bool(row["emitted_is_correct"]),
                "reason": row["reason"],
                "generation_prefix": generation_prefix,
                "catalog_prefix": (
                    f"Complete the factual catalog entry.\n"
                    f"Title: {row['title']}\nArtist:"
                ),
                "relation_cluster_id": (
                    f"{normalize_name(row['title'])}::"
                    f"{normalize_name(row['artist'])}"
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
    model_id: str = "Qwen/Qwen3-1.7B-Base",
) -> dict[str, Any]:
    selected, selection = select_holdout_rows(catalog_rows)
    selected = attach_references(selected)
    records = reconstruct_prefixes(selected, generation_bundle, raw_generations)
    required_correct = math.ceil(0.75 * selection["selected_per_label"])
    return {
        "bundle_version": "independent_holdout_verifier_v1",
        "model_id": model_id,
        "records": records,
        "selection": selection,
        "frozen_rule": {
            "name": "choice_or_factual_complete_artist_negative",
            "predict_conflict_when": (
                "choice_emitted_margin < 0 or "
                "catalog_sequence_emitted_margin < 0"
            ),
            "minimum_balanced_accuracy": 0.75,
            "minimum_exact_specificity": 0.75,
            "minimum_conflict_sensitivity": 0.75,
            "minimum_events_per_label": 8,
            "required_correct_per_label": required_correct,
        },
        "diagnostic_paths": [
            "catalog_first_token_emitted_margin",
            "generation_first_token_emitted_margin",
            "generation_sequence_emitted_margin",
        ],
        "generation_run_id": "20260711T124528Z_holdout",
        "generation_job_id": "6a523ac0effc02a91cbd98aa",
        "discovery_result_reference": "free_generated_relation_conflict_summary.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog-rows",
        type=Path,
        default=PROJECT_ROOT / "runs" / "independent_holdout_catalog_verified.jsonl",
    )
    parser.add_argument(
        "--generation-bundle",
        type=Path,
        default=PROJECT_ROOT / "runs" / "independent_holdout_generation_bundle.json",
    )
    parser.add_argument(
        "--raw-generations",
        type=Path,
        default=PROJECT_ROOT / "runs" / "independent_holdout_raw_generations.jsonl",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT / "runs" / "independent_holdout_verifier_bundle.json",
    )
    parser.add_argument(
        "--encoded",
        type=Path,
        default=PROJECT_ROOT / "runs" / "independent_holdout_verifier_bundle.zlib.b64",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    args = parser.parse_args()

    bundle = build_bundle(
        load_jsonl(args.catalog_rows),
        load_json(args.generation_bundle),
        load_jsonl(args.raw_generations),
        model_id=args.model_id,
    )
    payload = json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    args.bundle.write_bytes(payload + b"\n")
    args.encoded.write_text(
        base64.b64encode(zlib.compress(payload, level=9)).decode("ascii") + "\n",
        encoding="ascii",
    )
    labels = Counter(row["catalog_label"] for row in bundle["records"])
    print(f"Wrote independent verifier bundle: {dict(labels)}")
    print(f"- {args.bundle}")
    print(f"- {args.encoded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
