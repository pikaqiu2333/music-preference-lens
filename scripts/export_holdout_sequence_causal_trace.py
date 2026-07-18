"""Export full-sequence causal tracing for holdout title contrasts."""

from __future__ import annotations

import argparse
import base64
import json
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

from export_free_generated_relation_conflict_probe import load_json, load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_bundle(
    title_bundle: dict[str, Any],
    title_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result_by_id = {row["record_id"]: row for row in title_rows}
    records = []
    for record in title_bundle["records"]:
        result = result_by_id[record["record_id"]]
        records.append(
            {
                **record,
                "choice_diagnostic_class": result["paths"]["choice"][
                    "diagnostic_class"
                ],
                "sequence_diagnostic_class": result["paths"]["catalog_sequence"][
                    "diagnostic_class"
                ],
                "observed_sequence_relation_delta": result["paths"][
                    "catalog_sequence"
                ]["relation_delta"],
            }
        )
    return {
        "bundle_version": "holdout_sequence_causal_trace_v1",
        "model_id": title_bundle["model_id"],
        "records": records,
        "full_residual_layers": list(range(1, 29)),
        "component_interventions": [
            {"layer": layer, "component": component}
            for layer in (18, 21, 24, 27)
            for component in ("attention", "mlp")
        ],
        "expected_num_layers": 28,
        "minimum_relation_effect": 0.10,
        "endpoint_tolerance": 0.02,
        "contrast_reproduction_tolerance": 0.05,
        "source_title_contrast_job": "6a52f913e4a4e82c0b58ea8f",
        "focus_role_counts": dict(Counter(row["focus_role"] for row in records)),
        "interpretation": (
            "patch factual-title states into nonassociated-title sequences at all "
            "artist prediction positions; recovery measures causal sufficiency for "
            "the observed factual-minus-control complete-artist effect"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--title-bundle",
        type=Path,
        default=PROJECT_ROOT / "runs" / "holdout_title_contrast_bundle.json",
    )
    parser.add_argument(
        "--title-rows",
        type=Path,
        default=PROJECT_ROOT / "runs" / "holdout_title_contrast_rows.jsonl",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT / "runs" / "holdout_sequence_causal_trace_bundle.json",
    )
    parser.add_argument(
        "--encoded",
        type=Path,
        default=PROJECT_ROOT / "runs" / "holdout_sequence_causal_trace_bundle.zlib.b64",
    )
    args = parser.parse_args()

    bundle = build_bundle(load_json(args.title_bundle), load_jsonl(args.title_rows))
    payload = json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    args.bundle.write_bytes(payload + b"\n")
    args.encoded.write_text(
        base64.b64encode(zlib.compress(payload, level=9)).decode("ascii") + "\n",
        encoding="ascii",
    )
    print(f"Wrote sequence causal-trace bundle: {bundle['focus_role_counts']}")
    print(f"- {args.bundle}")
    print(f"- {args.encoded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
