"""Export post-hoc title counterfactuals for holdout verifier failures."""

from __future__ import annotations

import argparse
import base64
import json
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

from export_free_generated_relation_conflict_probe import load_json, load_jsonl
from verify_song_entity_catalog import normalize_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def event_key(row: dict[str, Any]) -> str:
    return "::".join(
        normalize_name(row[key])
        for key in ("title", "emitted_artist", "reference_artist")
    )


def unique_candidate_events(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for row in sorted(records, key=lambda item: item["selection_hash"]):
        key = event_key(row)
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def known_true_artist(row: dict[str, Any]) -> str:
    if row["catalog_label"] == "verified_exact":
        return row["emitted_artist"]
    return row["reference_artist"]


def select_control_titles(
    target: dict[str, Any],
    candidates: list[dict[str, Any]],
    count: int = 2,
) -> list[dict[str, str]]:
    target_artists = {
        normalize_name(target["emitted_artist"]),
        normalize_name(target["reference_artist"]),
    }
    target_title = normalize_name(target["title"])
    start = candidates.index(target)
    controls = []
    for offset in range(1, len(candidates) + 1):
        candidate = candidates[(start + offset) % len(candidates)]
        if normalize_name(candidate["title"]) == target_title:
            continue
        true_artist = known_true_artist(candidate)
        if normalize_name(true_artist) in target_artists:
            continue
        controls.append(
            {
                "source_record_id": candidate["record_id"],
                "title": candidate["title"],
                "known_true_artist": true_artist,
            }
        )
        if len(controls) == count:
            return controls
    raise ValueError(f"not enough nonassociated title controls for {target['record_id']}")


def focus_role(result: dict[str, Any]) -> str:
    label = result["catalog_label"]
    if label == "catalog_conflict":
        return (
            "choice_conflict_hit"
            if float(result["choice_emitted_margin"]) < 0
            else "choice_conflict_miss"
        )
    return (
        "sequence_exact_false_positive"
        if float(result["catalog_sequence_emitted_margin"]) < 0
        else "exact_clean_control"
    )


def build_bundle(
    verifier_bundle: dict[str, Any],
    verifier_rows: list[dict[str, Any]],
    model_id: str = "Qwen/Qwen3-1.7B-Base",
) -> dict[str, Any]:
    records = unique_candidate_events(verifier_bundle["records"])
    result_by_id = {row["record_id"]: row for row in verifier_rows}
    output = []
    for row in records:
        result = result_by_id[row["record_id"]]
        output.append(
            {
                "record_id": row["record_id"],
                "context_id": row["context_id"],
                "seed": row["seed"],
                "rank": row["rank"],
                "selection_hash": row["selection_hash"],
                "title": row["title"],
                "emitted_artist": row["emitted_artist"],
                "reference_artist": row["reference_artist"],
                "catalog_label": row["catalog_label"],
                "relation_cluster_id": row["relation_cluster_id"],
                "focus_role": focus_role(result),
                "reference_choice_margin": result["choice_emitted_margin"],
                "reference_catalog_sequence_margin": result[
                    "catalog_sequence_emitted_margin"
                ],
                "control_titles": select_control_titles(row, records),
            }
        )
    return {
        "bundle_version": "holdout_title_contrast_v1",
        "model_id": model_id,
        "records": output,
        "control_count_per_event": 2,
        "diagnostic_rule": {
            "name": "factual_minus_nonassociated_title_control",
            "expected_delta": {
                "verified_exact": "positive",
                "catalog_conflict": "negative",
            },
            "interpretation": (
                "correct delta with wrong absolute margin indicates latent relation "
                "support masked by candidate prior; wrong delta indicates relation "
                "retrieval failure under this probe"
            ),
        },
        "source_confirmation_run": "20260711T131420Z_confirm",
        "source_confirmation_job": "6a524112e4a4e82c0b58da32",
        "focus_role_counts": dict(Counter(row["focus_role"] for row in output)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verifier-bundle",
        type=Path,
        default=PROJECT_ROOT / "runs" / "independent_holdout_verifier_bundle.json",
    )
    parser.add_argument(
        "--verifier-rows",
        type=Path,
        default=PROJECT_ROOT / "runs" / "independent_holdout_verifier_rows.jsonl",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT / "runs" / "holdout_title_contrast_bundle.json",
    )
    parser.add_argument(
        "--encoded",
        type=Path,
        default=PROJECT_ROOT / "runs" / "holdout_title_contrast_bundle.zlib.b64",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    args = parser.parse_args()

    bundle = build_bundle(
        load_json(args.verifier_bundle),
        load_jsonl(args.verifier_rows),
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
    print(f"Wrote title-contrast bundle: {bundle['focus_role_counts']}")
    print(f"- {args.bundle}")
    print(f"- {args.encoded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
