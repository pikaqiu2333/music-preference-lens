"""Export the counterfactual recommendation-reason faithfulness probe."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("original", "paraphrase", "opposite", "neutral")
SMOKE_KEYS = {
    ("emotional_vocal__seed17", 3),
    ("emotional_vocal__seed17", 5),
    ("emotional_vocal__seed29", 4),
    ("emotional_vocal__seed29", 5),
    ("emotional_vocal__seed43", 1),
    ("emotional_vocal__seed43", 2),
    ("emotional_vocal__seed43", 4),
    ("emotional_vocal__seed43", 5),
    ("peak_time_rave__seed17", 1),
    ("peak_time_rave__seed17", 2),
    ("strict_no_vocals__seed29", 1),
    ("strict_no_vocals__seed29", 5),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def render_prompt(template: str, profile: str, current_need: str) -> str:
    return template.replace("{{profile}}", profile).replace(
        "{{current_need}}", current_need
    )


def validate_inputs(
    contexts: list[dict[str, Any]],
    counterfactuals: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
) -> None:
    if len(contexts) != 4 or len(counterfactuals) != 4:
        raise ValueError("expected four contexts and four counterfactual rows")
    context_ids = {row["context_id"] for row in contexts}
    counterfactual_ids = {row["context_id"] for row in counterfactuals}
    if context_ids != counterfactual_ids:
        raise ValueError("context and counterfactual IDs differ")
    required_cf = {"context_id", "paraphrase_need", "opposite_need", "neutral_need"}
    for row in counterfactuals:
        missing = required_cf - set(row)
        if missing:
            raise ValueError(f"counterfactual row missing fields: {sorted(missing)}")
        values = [row[key] for key in sorted(required_cf - {"context_id"})]
        if len(set(values)) != len(values):
            raise ValueError(f"counterfactual needs are not distinct: {row['context_id']}")
    keys: set[tuple[str, int]] = set()
    for row in generated_rows:
        key = (row["generation_id"], int(row["rank"]))
        if key in keys:
            raise ValueError(f"duplicate generated record: {key}")
        keys.add(key)
        if row["catalog_label"] not in {"verified_exact", "catalog_conflict", "unverified"}:
            raise ValueError(f"unexpected catalog label: {row['catalog_label']}")
    missing_smoke = SMOKE_KEYS - keys
    if missing_smoke:
        raise ValueError(f"missing smoke records: {sorted(missing_smoke)}")


def build_bundle(
    contexts: list[dict[str, Any]],
    counterfactuals: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
    prompt_template: str,
) -> dict[str, Any]:
    context_by_id = {row["context_id"]: row for row in contexts}
    cf_by_id = {row["context_id"]: row for row in counterfactuals}
    rendered_contexts: list[dict[str, Any]] = []
    for context_id in sorted(context_by_id):
        context = context_by_id[context_id]
        cf = cf_by_id[context_id]
        needs = {
            "original": context["current_need"],
            "paraphrase": cf["paraphrase_need"],
            "opposite": cf["opposite_need"],
            "neutral": cf["neutral_need"],
        }
        rendered_contexts.append(
            {
                **context,
                "condition_needs": needs,
                "condition_prompts": {
                    key: render_prompt(prompt_template, context["profile"], need)
                    for key, need in needs.items()
                },
            }
        )

    records: list[dict[str, Any]] = []
    for row in generated_rows:
        if row["catalog_label"] == "unverified":
            continue
        key = (row["generation_id"], int(row["rank"]))
        records.append(
            {
                "record_id": f"{row['generation_id']}__rank{row['rank']}",
                "generation_id": row["generation_id"],
                "context_id": row["context_id"],
                "seed": int(row["seed"]),
                "rank": int(row["rank"]),
                "title": row["title"],
                "artist": row["artist"],
                "reason": row["reason"],
                "catalog_label": row["catalog_label"],
                "smoke": key in SMOKE_KEYS,
            }
        )

    return {
        "bundle_version": "music_reason_counterfactual_faithfulness_v1",
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "conditions": list(CONDITIONS),
        "contexts": rendered_contexts,
        "records": records,
        "registered_gates": {
            "smoke_record_count": 12,
            "smoke_label_counts": {"verified_exact": 6, "catalog_conflict": 6},
            "minimum_opposite_below_both_rate": 2 / 3,
            "require_positive_median_opposite_margin": True,
        },
        "hf_result_repo": "REDACTED/music-preference-lens-runs",
    }


def write_plan(path: Path, bundle: dict[str, Any]) -> None:
    smoke = [row for row in bundle["records"] if row["smoke"]]
    counts = Counter(row["catalog_label"] for row in smoke)
    lines = [
        "# Music Recommendation Reason Faithfulness Probe Plan",
        "",
        f"- Model: `{bundle['model_id']}`",
        f"- Full catalog-resolved records: {len(bundle['records'])}",
        f"- Smoke records: {len(smoke)}",
        f"- Verified exact in smoke: {counts['verified_exact']}",
        f"- Catalog conflict in smoke: {counts['catalog_conflict']}",
        "- Decision unit: complete title-artist token sequence",
        "- Conditions: original, semantic paraphrase, opposite need, neutral need",
        "- Trained probe or LLM judge: none",
        "",
        "## Mechanistic Follow-up Gate",
        "",
        "Proceed only if at least 8/12 opposite needs score below both",
        "semantically equivalent needs and the median opposite margin is positive.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contexts",
        type=Path,
        default=PROJECT_ROOT / "data" / "qwen_scope_song_entity_generation_time_specs.jsonl",
    )
    parser.add_argument(
        "--counterfactuals",
        type=Path,
        default=PROJECT_ROOT / "data" / "qwen_scope_music_reason_counterfactuals.jsonl",
    )
    parser.add_argument(
        "--generated",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "qwen_scope_song_entity_generation_time_full_catalog_verified.jsonl",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=PROJECT_ROOT / "prompts" / "song_entity_generation_time_prompt.md",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT / "runs" / "qwen_scope_music_reason_faithfulness_bundle.json",
    )
    parser.add_argument(
        "--smoke-bundle",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "qwen_scope_music_reason_faithfulness_smoke_bundle.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=PROJECT_ROOT / "reports" / "qwen_scope_music_reason_faithfulness_probe_plan.md",
    )
    args = parser.parse_args()

    contexts = load_jsonl(args.contexts)
    counterfactuals = load_jsonl(args.counterfactuals)
    generated_rows = load_jsonl(args.generated)
    validate_inputs(contexts, counterfactuals, generated_rows)
    bundle = build_bundle(
        contexts,
        counterfactuals,
        generated_rows,
        args.prompt_template.read_text(encoding="utf-8"),
    )
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    args.bundle.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    smoke_bundle = {
        **bundle,
        "records": [row for row in bundle["records"] if row["smoke"]],
    }
    args.smoke_bundle.write_text(
        json.dumps(smoke_bundle, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    write_plan(args.markdown, bundle)
    print(f"Wrote {len(bundle['records'])} catalog-resolved records")
    print(f"- {args.bundle}")
    print(f"- {args.smoke_bundle}")
    print(f"- {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
