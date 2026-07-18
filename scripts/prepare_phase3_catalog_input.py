"""Replay Phase 3 pilot parsing and prepare catalog-verification records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from phase2_mechanism_analysis import normalize_entity
from run_phase3_natural_generation import parse_playlist


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def generation_row_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "generation_id": row["generation_id"],
            "rank": int(row["rank"]),
            "title": row["title"],
            "artist": row["artist"],
            "reason": row["reason"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_catalog_rows(
    artifact: dict[str, Any], bundle: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = artifact["summary"]
    for field in (
        "protocol_id",
        "protocol_hashes",
        "mode",
        "prompt_template_id",
        "model_id",
        "model_revision",
    ):
        if summary[field] != bundle[field]:
            raise ValueError(f"artifact and bundle differ on {field}")
    if summary["mode"] != "pilot" or not summary.get("technical_gate"):
        raise ValueError("Phase 3 pilot generation technical gate did not pass")

    expected_ids = {
        f"{context['context_id']}__seed{seed}"
        for context in bundle["contexts"]
        for seed in bundle["seeds"]
    }
    actual_ids = {row["generation_id"] for row in artifact["rows"]}
    if actual_ids != expected_ids or len(artifact["rows"]) != len(expected_ids):
        raise ValueError("generation IDs do not match the frozen pilot bundle")

    output: list[dict[str, Any]] = []
    for generation in artifact["rows"]:
        replayed = parse_playlist(
            bundle["completion_prefix"] + generation["completion"],
            maximum_tracks=int(bundle["generation"]["tracks_per_playlist"]),
        )
        if replayed != generation["parsed_tracks"]:
            raise ValueError(
                f"parser replay drift for {generation['generation_id']}"
            )
        if len(replayed) != int(generation["parsed_track_count"]):
            raise ValueError(
                f"parser count drift for {generation['generation_id']}"
            )
        completion_sha256 = hashlib.sha256(
            generation["completion"].encode("utf-8")
        ).hexdigest()
        for rank, track in enumerate(replayed, 1):
            row = {
                "record_type": "generated_pair",
                "record_id": f"{generation['generation_id']}__rank{rank}",
                "generation_id": generation["generation_id"],
                "context_id": generation["context_id"],
                "seed": int(generation["seed"]),
                "rank": rank,
                "title": track["title"],
                "artist": track["artist"],
                "reason": track["reason"],
                "normalized_title": normalize_entity(track["title"]),
                "normalized_artist": normalize_entity(track["artist"]),
                "batch_mode": "phase3_pilot",
                "prompt_template_id": summary["prompt_template_id"],
                "model_id": summary["model_id"],
                "model_revision": summary["model_revision"],
                "protocol_id": summary["protocol_id"],
                "protocol_hashes": summary["protocol_hashes"],
                "source_completion_sha256": completion_sha256,
            }
            if not row["normalized_title"] or not row["normalized_artist"]:
                raise ValueError(f"empty normalized entity in {row['record_id']}")
            row["generation_row_sha256"] = generation_row_hash(row)
            output.append(row)

    if len(output) != int(summary["parsed_track_count"]):
        raise ValueError("artifact parsed-track total does not replay")
    title_counts = Counter(row["normalized_title"] for row in output)
    result_summary = {
        "protocol_id": summary["protocol_id"],
        "protocol_hashes": summary["protocol_hashes"],
        "source_run_id": summary["run_id"],
        "source_mode": summary["mode"],
        "generation_count": len(artifact["rows"]),
        "parsed_event_count": len(output),
        "minimum_parsed_event_count": bundle["minimum_parsed_track_count"],
        "unique_normalized_title_count": len(title_counts),
        "duplicate_title_cluster_count": sum(
            count > 1 for count in title_counts.values()
        ),
        "record_sha256_unique": (
            len({row["generation_row_sha256"] for row in output}) == len(output)
        ),
        "parser_replay_gate": True,
        "technical_gate": len(output) >= int(bundle["minimum_parsed_track_count"]),
    }
    return output, result_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--generation-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    rows, summary = prepare_catalog_rows(
        load_json(args.artifact), load_json(args.generation_bundle)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE3_CATALOG_INPUT_SUMMARY_JSON=" + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
