"""Parse archived Phase 2 generations into catalog-verification records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from phase2_mechanism_analysis import normalize_entity
from run_phase2_generation_probe import parse_playlist


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def generation_row_hash(row: dict[str, Any]) -> str:
    fields = {
        "generation_id": row["generation_id"],
        "rank": int(row["rank"]),
        "title": row["title"],
        "artist": row["artist"],
        "reason": row.get("reason", ""),
    }
    payload = json.dumps(
        fields, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_catalog_rows(
    artifact: dict[str, Any], generation_bundle: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = artifact["summary"]
    if summary["mode"] not in {"primary", "extension"}:
        raise ValueError("catalog input must come from a scientific generation batch")
    if summary["protocol_id"] != generation_bundle["protocol_id"]:
        raise ValueError("artifact and generation bundle protocol IDs differ")
    if summary["protocol_hashes"] != generation_bundle["protocol_hashes"]:
        raise ValueError("artifact and generation bundle protocol hashes differ")
    if summary["model_id"] != generation_bundle["model_id"]:
        raise ValueError("artifact and generation bundle model IDs differ")
    if summary["model_revision"] != generation_bundle["model_revision"]:
        raise ValueError("artifact and generation bundle model revisions differ")
    if not summary.get("technical_gate"):
        raise ValueError("generation technical gate did not pass")

    contexts = {
        row["context_id"]: row["generation_prompt"]
        for row in generation_bundle["contexts"]
    }
    expected_ids = {
        f"{context_id}__seed{seed}"
        for context_id in contexts
        for seed in generation_bundle["seeds"]
    }
    actual_ids = {row["generation_id"] for row in artifact["rows"]}
    if actual_ids != expected_ids:
        raise ValueError("generation IDs do not match the frozen bundle")

    output: list[dict[str, Any]] = []
    for generation in artifact["rows"]:
        prompt = contexts[generation["context_id"]]
        parsed = parse_playlist(prompt.rstrip() + generation["completion"])
        if len(parsed) != int(generation["parsed_track_count"]):
            raise ValueError(
                f"parser count drift for {generation['generation_id']}: "
                f"{len(parsed)} != {generation['parsed_track_count']}"
            )
        for rank, track in enumerate(parsed, 1):
            row = {
                "record_type": "generated_pair",
                "record_id": f"{generation['generation_id']}__rank{rank}",
                "generation_id": generation["generation_id"],
                "context_id": generation["context_id"],
                "seed": int(generation["seed"]),
                "rank": rank,
                "title": track["title"],
                "artist": track["artist"],
                "reason": track.get("reason", ""),
                "normalized_title": normalize_entity(track["title"]),
                "normalized_artist": normalize_entity(track["artist"]),
                "batch_mode": summary["mode"],
                "prompt_template_id": summary["prompt_template_id"],
                "model_id": summary["model_id"],
                "model_revision": summary["model_revision"],
                "protocol_id": summary["protocol_id"],
                "protocol_hashes": summary["protocol_hashes"],
            }
            if not row["normalized_title"] or not row["normalized_artist"]:
                raise ValueError(f"empty normalized entity in {row['record_id']}")
            row["generation_row_sha256"] = generation_row_hash(row)
            output.append(row)

    title_counts = Counter(row["normalized_title"] for row in output)
    parse_summary = {
        "protocol_id": summary["protocol_id"],
        "protocol_hashes": summary["protocol_hashes"],
        "source_run_id": summary["run_id"],
        "source_mode": summary["mode"],
        "generation_count": len(artifact["rows"]),
        "parsed_event_count": len(output),
        "unique_normalized_title_count": len(title_counts),
        "duplicate_title_cluster_count": sum(count > 1 for count in title_counts.values()),
        "record_sha256_unique": len({row["generation_row_sha256"] for row in output})
        == len(output),
    }
    return output, parse_summary


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
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE2_CATALOG_INPUT_SUMMARY_JSON=" + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
