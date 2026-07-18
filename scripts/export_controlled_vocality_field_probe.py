"""Export the frozen title-versus-artist field-diagnosis bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HEADS = [0, 1, 8, 9]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_bundle(
    head_bundle: dict[str, Any],
    head_summary: dict[str, Any],
) -> dict[str, Any]:
    selected_heads = [int(head) for head in head_summary["choice_consistent_heads"]]
    if selected_heads != EXPECTED_HEADS:
        raise ValueError(
            f"frozen choice-consistent heads changed: {selected_heads} != {EXPECTED_HEADS}"
        )
    if int(head_bundle["target_layer"]) != 18:
        raise ValueError("field diagnosis is registered only for layer 18")
    focus = head_bundle["focus_records"]
    if len(focus) != 4:
        raise ValueError("field diagnosis requires four frozen sentinels")
    return {
        "bundle_version": "controlled_vocality_field_diagnosis_v1",
        "model_id": head_bundle["model_id"],
        "reasons": head_bundle["reasons"],
        "focus_records": focus,
        "target_layer": 18,
        "selected_heads": selected_heads,
        "expected_num_attention_heads": head_bundle["expected_num_attention_heads"],
        "expected_head_dim": head_bundle["expected_head_dim"],
        "intervention_scopes": ["title", "artist", "both"],
        "minimum_field_effect": 0.03,
        "pair_reconstruction_tolerance": 0.002,
        "all_head_reproduction_tolerance": 0.002,
        "behavior_reference": head_bundle["behavior_reference"],
        "head_reference_run": head_summary["run_id"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--head-bundle",
        type=Path,
        default=PROJECT_ROOT / "runs" / "controlled_vocality_attention_head_bundle.json",
    )
    parser.add_argument(
        "--head-summary",
        type=Path,
        default=PROJECT_ROOT / "runs" / "controlled_vocality_attention_head_summary.json",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT / "runs" / "controlled_vocality_field_probe_bundle.json",
    )
    args = parser.parse_args()

    bundle = build_bundle(load_json(args.head_bundle), load_json(args.head_summary))
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    args.bundle.write_text(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote field bundle with {len(bundle['focus_records'])} sentinels and "
        f"heads {bundle['selected_heads']}"
    )
    print(f"- {args.bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
