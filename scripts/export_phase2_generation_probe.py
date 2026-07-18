"""Export frozen Phase 2 free-generation bundles."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "config" / "phase2_mechanism_intervention_protocol.json"
DEFAULT_PROTOCOL_DOC = PROJECT_ROOT / "docs" / "phase2_mechanism_intervention_protocol.md"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def generation_prompt(context: dict[str, str], template: str) -> str:
    return template.format(
        profile=context["profile"],
        current_need=context["current_need"],
    )


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_bundle(
    protocol: dict[str, Any],
    mode: str,
    protocol_bytes: bytes,
    protocol_doc_bytes: bytes,
    output_repo: str | None = None,
    output_repo_private: bool = False,
    prompt_template_id: str = "primary",
    model_role: str = "confirmatory",
) -> dict[str, Any]:
    generation = protocol["generation"]
    if prompt_template_id not in generation["prompt_templates"]:
        raise ValueError(f"unknown prompt template: {prompt_template_id}")
    if model_role not in ("confirmatory", "technical_backup_only"):
        raise ValueError(f"unknown model role: {model_role}")
    if mode == "smoke":
        contexts = [protocol["technical_smoke"]["context"]]
        seeds = [protocol["technical_smoke"]["seed"]]
        minimum_parsed_tracks = protocol["technical_smoke"]["minimum_parsed_tracks"]
        if output_repo is not None:
            raise ValueError("technical smoke cannot upload raw artifacts")
    elif mode == "primary":
        contexts = generation["primary_contexts"]
        seeds = generation["seeds"]
        minimum_parsed_tracks = generation["extension_trigger"][
            "minimum_primary_parsed_events"
        ]
    elif mode == "extension":
        contexts = generation["registered_extension_contexts"]
        seeds = generation["seeds"]
        minimum_parsed_tracks = 0
    else:
        raise ValueError(f"unknown mode: {mode}")

    prompt_template = generation["prompt_templates"][prompt_template_id]
    rendered_contexts = [
        {
            **context,
            "generation_prompt": generation_prompt(context, prompt_template),
        }
        for context in contexts
    ]
    expected_generation_count = len(rendered_contexts) * len(seeds)
    if mode == "primary" and expected_generation_count != generation["primary_playlist_count"]:
        raise ValueError("primary playlist count does not match frozen protocol")
    if mode == "extension" and expected_generation_count != generation["extension_playlist_count"]:
        raise ValueError("extension playlist count does not match frozen protocol")

    model = protocol["models"][model_role]
    return {
        "bundle_version": "phase2_generation_v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_hashes": {
            "json_sha256": sha256_bytes(protocol_bytes),
            "markdown_sha256": sha256_bytes(protocol_doc_bytes),
        },
        "protocol_payloads_b64": {
            "json": base64.b64encode(protocol_bytes).decode("ascii"),
            "markdown": base64.b64encode(protocol_doc_bytes).decode("ascii"),
        },
        "mode": mode,
        "prompt_template_id": prompt_template_id,
        "model_role": model_role,
        "model_id": model["model_id"],
        "model_revision": model["revision"],
        "expected_num_layers": model["expected_num_layers"],
        "contexts": rendered_contexts,
        "seeds": seeds,
        "generation": {
            "temperature": generation["temperature"],
            "top_p": generation["top_p"],
            "max_new_tokens": generation["max_new_tokens"],
        },
        "expected_generation_count": expected_generation_count,
        "minimum_parsed_tracks": minimum_parsed_tracks,
        "endpoint_tolerance": 0.02,
        "redact_completions": mode == "smoke",
        "output_repo": output_repo,
        "output_repo_private": output_repo_private,
    }


def default_output(mode: str) -> Path:
    return PROJECT_ROOT / "runs" / f"phase2_granite_{mode}_generation_bundle.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-doc", type=Path, default=DEFAULT_PROTOCOL_DOC)
    parser.add_argument("--mode", choices=("smoke", "primary", "extension"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-repo")
    parser.add_argument("--output-repo-private", action="store_true")
    parser.add_argument(
        "--prompt-template-id",
        choices=("primary", "single_format_fallback"),
        default="primary",
    )
    parser.add_argument(
        "--model-role",
        choices=("confirmatory", "technical_backup_only"),
        default="confirmatory",
    )
    args = parser.parse_args()

    protocol_bytes = args.protocol.read_bytes()
    protocol_doc_bytes = args.protocol_doc.read_bytes()
    bundle = build_bundle(
        json.loads(protocol_bytes.decode("utf-8")),
        args.mode,
        protocol_bytes,
        protocol_doc_bytes,
        output_repo=args.output_repo,
        output_repo_private=args.output_repo_private,
        prompt_template_id=args.prompt_template_id,
        model_role=args.model_role,
    )
    output = args.output or default_output(args.mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(f"protocol_json_sha256={bundle['protocol_hashes']['json_sha256']}")
    print(f"protocol_markdown_sha256={bundle['protocol_hashes']['markdown_sha256']}")
    print(f"expected_generations={bundle['expected_generation_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
