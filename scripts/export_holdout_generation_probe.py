"""Export the frozen new-seed free-generation holdout bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_SEEDS = [59, 71, 83, 101, 127]
DISCOVERY_SEEDS = {17, 29, 43}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_bundle(source: dict[str, Any]) -> dict[str, Any]:
    if set(source["seeds"]) != DISCOVERY_SEEDS:
        raise ValueError("source discovery seeds changed")
    if DISCOVERY_SEEDS & set(HOLDOUT_SEEDS):
        raise ValueError("holdout seeds overlap discovery seeds")
    contexts = source["contexts"]
    if len(contexts) != 4 or len({row["context_id"] for row in contexts}) != 4:
        raise ValueError("expected four distinct frozen contexts")
    return {
        "bundle_version": "independent_holdout_generation_v1",
        "model_id": source["model_id"],
        "contexts": contexts,
        "seeds": HOLDOUT_SEEDS,
        "generation": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 384,
        },
        "expected_generation_count": len(contexts) * len(HOLDOUT_SEEDS),
        "minimum_parsed_generation_count": 16,
        "minimum_parsed_pair_count": 75,
        "discovery_generation_run": "20260710T035224Z_full",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-bundle",
        type=Path,
        default=PROJECT_ROOT / "runs" / "qwen_scope_song_entity_generation_time_bundle.json",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT / "runs" / "independent_holdout_generation_bundle.json",
    )
    args = parser.parse_args()
    bundle = build_bundle(load_json(args.source_bundle))
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    args.bundle.write_text(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote holdout bundle with {len(bundle['contexts'])} contexts, "
        f"{len(bundle['seeds'])} seeds, and "
        f"{bundle['expected_generation_count']} generations"
    )
    print(f"- {args.bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
