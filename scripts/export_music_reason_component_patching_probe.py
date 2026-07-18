"""Export the three-case recommendation-reason component-patching bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTED_LAYERS = [14, 16, 18, 21, 24, 27, 28]
CASE_ROLES = {
    "emotional_vocal__seed17__rank5": "verified_need_sensitive",
    "peak_time_rave__seed17__rank2": "conflict_need_sensitive",
    "strict_no_vocals__seed29__rank5": "verified_constraint_failure",
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
    faithfulness_bundle: dict[str, Any],
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    contexts = {
        row["context_id"]: row for row in faithfulness_bundle["contexts"]
    }
    records = {row["record_id"]: row for row in faithfulness_bundle["records"]}
    results = {row["record_id"]: row for row in result_rows}
    if not set(CASE_ROLES) <= set(records) or not set(CASE_ROLES) <= set(results):
        raise ValueError("registered component-patching cases are missing")

    controls = []
    for record_id, case_role in CASE_ROLES.items():
        record = records[record_id]
        result = results[record_id]
        context = contexts[record["context_id"]]
        original = float(result["condition_pair_mean_logps"]["original"])
        opposite = float(result["condition_pair_mean_logps"]["opposite"])
        controls.append(
            {
                **record,
                "case_role": case_role,
                "original_prompt": context["condition_prompts"]["original"],
                "opposite_prompt": context["condition_prompts"]["opposite"],
                "archived_original_pair_mean_logp": original,
                "archived_opposite_pair_mean_logp": opposite,
                "archived_need_effect": original - opposite,
            }
        )

    return {
        "bundle_version": "music_reason_component_patching_v1",
        "model_id": faithfulness_bundle["model_id"],
        "controls": controls,
        "selected_layers": SELECTED_LAYERS,
        "analysis_layers": SELECTED_LAYERS[:-1],
        "components": ["attention", "mlp", "full_residual"],
        "minimum_need_effect": 0.05,
        "minimum_valid_cases": 3,
        "baseline_tolerance": 0.02,
        "endpoint_tolerance": 0.02,
        "behavior_reference": "20260711T015944Z_smoke",
    }


def write_plan(path: Path, bundle: dict[str, Any]) -> None:
    lines = [
        "# Music Reason Component-Patching HF Plan",
        "",
        f"- Cases: {len(bundle['controls'])}",
        "- Layers: 14, 16, 18, 21, 24, 27; endpoint layer 28",
        "- Components: attention, MLP, full residual",
        "- Target: complete title-artist token sequence",
        "- Source condition: opposite need",
        "- Target condition: original need",
        "- Minimum absolute need effect: 0.05",
        "- Baseline and endpoint tolerance: 0.02",
        "- Hardware target: Hugging Face `t4-small`",
        "- Interpretation: exploratory three-case mechanism pilot",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--faithfulness-bundle",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "qwen_scope_music_reason_faithfulness_smoke_bundle.json",
    )
    parser.add_argument(
        "--result-rows",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "qwen_scope_music_reason_faithfulness_smoke_rows.jsonl",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "qwen_scope_music_reason_component_patching_bundle.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "qwen_scope_music_reason_component_patching_probe_plan.md",
    )
    args = parser.parse_args()

    bundle = build_bundle(
        load_json(args.faithfulness_bundle),
        load_jsonl(args.result_rows),
    )
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    args.bundle.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_plan(args.markdown, bundle)
    print(f"Wrote component-patching bundle with {len(bundle['controls'])} cases")
    print(f"- {args.bundle}")
    print(f"- {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
