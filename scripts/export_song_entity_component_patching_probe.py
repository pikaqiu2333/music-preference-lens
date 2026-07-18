"""Export the attention-vs-MLP component-patching HF Job bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from export_song_entity_relation_binding_probe import load_jsonl, validate_controls


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_BLOCKS = {
    "en_swap_1",
    "en_swap_2",
    "en_swap_3",
    "zh_swap_1",
    "zh_swap_2",
    "zh_swap_3",
}
SELECTED_LAYERS = [14, 16, 18, 21, 24, 27, 28]


def select_smoke_controls(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["block_id"] in SMOKE_BLOCKS]


def build_bundle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "bundle_version": "song_entity_component_patching_v1",
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "controls": rows,
        "patch_prefix_template": (
            "Complete the factual catalog entry.\nTitle: {title}\nArtist:"
        ),
        "selected_layers": SELECTED_LAYERS,
        "analysis_layers": SELECTED_LAYERS[:-1],
        "components": ["attention", "mlp", "full_residual"],
        "minimum_valid_relations": 8,
        "minimum_title_effect": 0.05,
        "endpoint_tolerance": 0.02,
        "behavior_reference": {
            "run_id": "20260710T090506Z_smoke",
            "final_sequence_pmi_accuracy": 0.8333333333333334,
        },
    }


def write_plan(path: Path, rows: list[dict[str, Any]]) -> None:
    smoke = select_smoke_controls(rows)
    lines = [
        "# Qwen Song Entity Component Patching Run Plan",
        "",
        f"- Smoke relations: {len(smoke)}",
        "- Analysis layers: 14, 16, 18, 21, 24, 27",
        "- Endpoint check layer: 28",
        "- Components: self-attention, MLP, full residual",
        "- Minimum valid relations: 8",
        "- Minimum real-vs-neutral first-token effect: 0.05",
        "- Layer-28 endpoint tolerance: 0.02",
        "- Trained probes: none",
        "- Hardware target: Hugging Face `t4-small`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controls",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "qwen_scope_song_entity_relation_binding_controls.jsonl",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "qwen_scope_song_entity_component_patching_bundle.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "qwen_scope_song_entity_component_patching_probe_plan.md",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.controls)
    validate_controls(rows)
    bundle = build_bundle(rows)
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    args.bundle.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_plan(args.markdown, rows)
    print(f"Wrote component-patching bundle with {len(rows)} controls")
    print(f"- {args.bundle}")
    print(f"- {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
