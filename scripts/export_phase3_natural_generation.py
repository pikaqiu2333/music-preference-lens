"""Export the frozen Phase 3 natural-generation pilot bundle."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "config" / "phase3_natural_relation_discovery_protocol.json"
)
DEFAULT_PROTOCOL_DOC = (
    PROJECT_ROOT / "docs" / "phase3_natural_relation_discovery_protocol.md"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "phase3_natural_pilot_generation_bundle.json"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def render_prompt(context: dict[str, str], template: str) -> str:
    return template.format(
        profile=context["profile"],
        current_need=context["current_need"],
    )


def build_bundle(
    protocol: dict[str, Any],
    protocol_bytes: bytes,
    protocol_doc_bytes: bytes,
) -> dict[str, Any]:
    generation = protocol["generation"]
    contexts = [
        {
            **context,
            "generation_prompt": render_prompt(
                context, generation["discovery_prompt_template"]
            ),
        }
        for context in generation["discovery_contexts"]
    ]
    seeds = list(generation["pilot_seeds"])
    expected = len(contexts) * len(seeds)
    if expected != int(generation["pilot_playlist_count"]):
        raise ValueError("pilot playlist count does not match the frozen protocol")
    if set(seeds) & set(generation["expansion_seeds"]):
        raise ValueError("pilot and expansion seeds overlap")
    if set(seeds) & set(generation["holdout_seeds"]):
        raise ValueError("pilot and holdout seeds overlap")
    model = protocol["model"]
    return {
        "bundle_version": "phase3_natural_generation_v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_status": protocol["status"],
        "protocol_hashes": {
            "json_sha256": sha256_bytes(protocol_bytes),
            "markdown_sha256": sha256_bytes(protocol_doc_bytes),
        },
        "protocol_payloads_b64": {
            "json": base64.b64encode(protocol_bytes).decode("ascii"),
            "markdown": base64.b64encode(protocol_doc_bytes).decode("ascii"),
        },
        "mode": "pilot",
        "prompt_template_id": "discovery_v1",
        "model_id": model["model_id"],
        "model_revision": model["revision"],
        "expected_num_layers": model["expected_num_layers"],
        "contexts": contexts,
        "seeds": seeds,
        "generation": {
            "tracks_per_playlist": generation["tracks_per_playlist"],
            "temperature": generation["temperature"],
            "top_p": generation["top_p"],
            "max_new_tokens": generation["max_new_tokens"],
        },
        "completion_prefix": "1. Title:",
        "expected_generation_count": expected,
        "maximum_parsed_track_count": (
            expected * int(generation["tracks_per_playlist"])
        ),
        "minimum_parsed_track_count": protocol["pilot_continuation_gate"][
            "minimum_parsed_track_events"
        ],
        "continuation_gate": protocol["pilot_continuation_gate"],
        "claim_boundaries": protocol["claim_boundaries"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-doc", type=Path, default=DEFAULT_PROTOCOL_DOC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    protocol_bytes = args.protocol.read_bytes()
    protocol_doc_bytes = args.protocol_doc.read_bytes()
    bundle = build_bundle(
        json.loads(protocol_bytes.decode("utf-8")),
        protocol_bytes,
        protocol_doc_bytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(f"expected_generations={bundle['expected_generation_count']}")
    print(f"maximum_tracks={bundle['maximum_parsed_track_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
