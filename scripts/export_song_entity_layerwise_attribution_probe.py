"""Export the layerwise title-artist attribution Hugging Face Job bundle."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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


def select_smoke_controls(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["block_id"] in SMOKE_BLOCKS]


def build_bundle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "bundle_version": "song_entity_layerwise_attribution_v1",
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "controls": rows,
        "likelihood_template": (
            "Complete the factual catalog entry.\nTitle: {title}\nArtist: "
        ),
        "patch_prefix_template": (
            "Complete the factual catalog entry.\nTitle: {title}\nArtist:"
        ),
        "behavior_threshold": 0.80,
        "sustained_accuracy_threshold": 0.75,
        "sustained_layer_count": 3,
        "consistency_tolerance": 0.02,
    }


def write_plan(path: Path, rows: list[dict[str, Any]]) -> None:
    smoke = select_smoke_controls(rows)
    language_counts = Counter(row["language"] for row in smoke)
    lines = [
        "# Qwen Song Entity Layerwise Attribution Run Plan",
        "",
        f"- Full controls available: {len(rows)}",
        f"- Smoke relations: {len(smoke)}",
        f"- Smoke English: {language_counts['en']}",
        f"- Smoke Chinese: {language_counts['zh']}",
        "- Readout: final RMS norm plus observed-token unembedding row",
        "- Causal test: neutral-title residual patch at the Artist prediction position",
        "- Trained probes: none",
        "- Behavioral gate: 0.80 final sequence-PMI accuracy",
        "- Consistency tolerance: 0.02 logit",
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
        / "qwen_scope_song_entity_layerwise_attribution_bundle.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "qwen_scope_song_entity_layerwise_attribution_probe_plan.md",
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
    print(f"Wrote layerwise attribution bundle with {len(rows)} controls")
    print(f"- {args.bundle}")
    print(f"- {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
