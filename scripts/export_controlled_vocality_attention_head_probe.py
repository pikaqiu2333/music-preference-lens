"""Export the frozen layer-18 vocality attention-head probe bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_bundle(path_bundle: dict[str, Any]) -> dict[str, Any]:
    focus = path_bundle["focus_records"]
    if len(focus) != 4:
        raise ValueError("attention-head probe requires four frozen sentinels")
    roles = {row["sentinel_role"] for row in focus}
    expected_roles = {
        "vocal_pair_success",
        "vocal_pair_failure",
        "instrumental_pair_success",
        "instrumental_pair_failure",
    }
    if roles != expected_roles:
        raise ValueError("sentinel roles changed after component localization")
    if any(abs(float(row["behavior_matched_margin"])) < 0.10 for row in focus):
        raise ValueError("a frozen sentinel no longer has a nontrivial effect")
    return {
        "bundle_version": "controlled_vocality_attention_head_v1",
        "model_id": path_bundle["model_id"],
        "reasons": path_bundle["reasons"],
        "all_records": path_bundle["all_records"],
        "focus_records": focus,
        "candidate_orders": path_bundle["candidate_orders"],
        "target_layer": 18,
        "expected_num_attention_heads": 16,
        "expected_head_dim": 128,
        "minimum_pair_effect": 0.10,
        "all_head_reproduction_tolerance": 0.002,
        "choice_consistent_min_mean_recovery": 0.05,
        "behavior_reference": path_bundle["behavior_reference"],
        "component_reference_run": "20260711T075330Z_pilot",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-bundle",
        type=Path,
        default=PROJECT_ROOT / "runs" / "controlled_vocality_path_patching_bundle.json",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT / "runs" / "controlled_vocality_attention_head_bundle.json",
    )
    args = parser.parse_args()

    bundle = build_bundle(load_json(args.path_bundle))
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    args.bundle.write_text(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        "Wrote layer-18 head bundle with "
        f"{len(bundle['focus_records'])} sentinels and "
        f"{bundle['expected_num_attention_heads']} heads"
    )
    print(f"- {args.bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
