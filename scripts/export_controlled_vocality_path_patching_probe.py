"""Export the four-sentinel pair-versus-choice component-patching bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTED_LAYERS = [14, 16, 18, 21, 24, 27, 28]
FOCUS_ROLES = {
    "vocal_blinding_lights": "vocal_pair_success",
    "vocal_space_oddity": "vocal_pair_failure",
    "instrumental_river_flows_in_you": "instrumental_pair_success",
    "instrumental_awake": "instrumental_pair_failure",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_bundle(
    behavior_bundle: dict[str, Any],
    behavior_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    records = {row["track_key"]: row for row in behavior_bundle["records"]}
    results = {row["track_key"]: row for row in behavior_rows}
    if not set(FOCUS_ROLES) <= set(records) or not set(FOCUS_ROLES) <= set(results):
        raise ValueError("controlled vocality sentinels are missing")
    focus = []
    for track_key, role in FOCUS_ROLES.items():
        record = records[track_key]
        result = results[track_key]
        effect = float(result["matched_margin"])
        if abs(effect) < 0.10:
            raise ValueError(f"sentinel effect is too small: {track_key}={effect}")
        focus.append(
            {
                **record,
                "sentinel_role": role,
                "behavior_matched_margin": effect,
                "behavior_matched_pair_mean_logp": float(
                    result["condition_scores"][record["vocality"]]["pair_mean_logp"]
                ),
                "behavior_flipped_pair_mean_logp": float(
                    result["condition_scores"][
                        "instrumental" if record["vocality"] == "vocal" else "vocal"
                    ]["pair_mean_logp"]
                ),
            }
        )
    return {
        "bundle_version": "controlled_vocality_path_patching_v1",
        "model_id": behavior_bundle["model_id"],
        "reasons": behavior_bundle["reasons"],
        "all_records": behavior_bundle["records"],
        "focus_records": focus,
        "candidate_orders": behavior_bundle["candidate_orders"],
        "selected_layers": SELECTED_LAYERS,
        "analysis_layers": SELECTED_LAYERS[:-1],
        "components": ["attention", "mlp", "full_residual"],
        "minimum_pair_effect": 0.10,
        "endpoint_tolerance": 0.02,
        "behavior_reference": "20260711T073819Z_pilot",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--behavior-bundle",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "controlled_vocality_reason_probe_bundle.json",
    )
    parser.add_argument(
        "--behavior-rows",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "controlled_vocality_reason_pair_rows.jsonl",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "controlled_vocality_path_patching_bundle.json",
    )
    args = parser.parse_args()

    bundle = build_bundle(load_json(args.behavior_bundle), load_jsonl(args.behavior_rows))
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    args.bundle.write_text(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote path-patching bundle with {len(bundle['focus_records'])} sentinels")
    print(f"- {args.bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
