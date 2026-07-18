# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "accelerate==1.8.1",
#   "torch==2.7.1",
#   "transformers==4.53.3",
# ]
# ///
"""Run the frozen Phase 3 natural music-recommendation pilot on HF Jobs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EMBEDDED_BUNDLE_B64_ZLIB = "__PHASE3_NATURAL_GENERATION_BUNDLE_B64_ZLIB__"

INLINE_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?(?:\*\*)?title"
    r"(?:\*\*)?\s*:\s*(?P<title>.*?)\s*\|\s*(?:\*\*)?artist"
    r"(?:\*\*)?\s*:\s*(?P<artist>.*?)\s*\|\s*(?:\*\*)?reason"
    r"(?:\*\*)?\s*:\s*(?P<reason>.*?)"
    r"(?=\n\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?(?:\*\*)?title"
    r"(?:\*\*)?\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
FIELD_PATTERN = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?(?:\*\*)?"
    r"(?P<field>title|artist|reason)(?:\*\*)?\s*:\s*(?P<value>.*?)\s*$",
    re.IGNORECASE,
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_bundle(path: Path | None = None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    if (
        EMBEDDED_BUNDLE_B64_ZLIB
        == "__PHASE3_NATURAL_GENERATION_BUNDLE_B64_ZLIB__"
    ):
        raise ValueError("pass --bundle or embed the Phase 3 generation bundle")
    payload = zlib.decompress(base64.b64decode(EMBEDDED_BUNDLE_B64_ZLIB))
    return json.loads(payload.decode("utf-8"))


def validate_protocol_payloads(bundle: dict[str, Any]) -> None:
    for key in ("json", "markdown"):
        payload = base64.b64decode(bundle["protocol_payloads_b64"][key])
        if sha256_bytes(payload) != bundle["protocol_hashes"][f"{key}_sha256"]:
            raise ValueError(f"embedded protocol {key} hash mismatch")


def clean_field_value(value: str) -> str:
    return value.strip().strip("| ").strip('"\'').strip()


def parse_playlist(text: str, maximum_tracks: int = 5) -> list[dict[str, str]]:
    inline = [
        {
            field: clean_field_value(match.group(field))
            for field in ("title", "artist", "reason")
        }
        for match in INLINE_PATTERN.finditer(text)
    ]
    inline = [row for row in inline if all(row.values())]
    if inline:
        return inline[:maximum_tracks]

    rows: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in text.splitlines():
        match = FIELD_PATTERN.match(line)
        if not match:
            continue
        field = match.group("field").lower()
        value = clean_field_value(match.group("value"))
        if field == "title" and current:
            if all(current.get(name) for name in ("title", "artist", "reason")):
                rows.append(current)
            current = {}
        current[field] = value
    if all(current.get(name) for name in ("title", "artist", "reason")):
        rows.append(current)
    return rows[:maximum_tracks]


def encode_artifact_chunks(value: Any, maximum_chars: int = 7500) -> list[str]:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    encoded = base64.b64encode(zlib.compress(payload, level=9)).decode("ascii")
    return [
        encoded[start : start + maximum_chars]
        for start in range(0, len(encoded), maximum_chars)
    ]


def summarize_rows(
    rows: list[dict[str, Any]],
    bundle: dict[str, Any],
    architecture_gate: bool,
) -> dict[str, Any]:
    expected = int(bundle["expected_generation_count"])
    parsed_count = sum(int(row["parsed_track_count"]) for row in rows)
    nonempty_count = sum(bool(row["completion"].strip()) for row in rows)
    generation_ids = {row["generation_id"] for row in rows}
    context_counts = Counter()
    for row in rows:
        context_counts[row["context_id"]] += int(row["parsed_track_count"])
    return {
        "generation_count": len(rows),
        "expected_generation_count": expected,
        "unique_generation_count": len(generation_ids),
        "nonempty_completion_count": nonempty_count,
        "parsed_track_count": parsed_count,
        "minimum_parsed_track_count": int(bundle["minimum_parsed_track_count"]),
        "maximum_parsed_track_count": int(bundle["maximum_parsed_track_count"]),
        "parsed_tracks_by_context": dict(sorted(context_counts.items())),
        "architecture_gate": architecture_gate,
        "technical_gate": (
            architecture_gate
            and len(rows) == expected
            and len(generation_ids) == expected
            and nonempty_count == expected
            and parsed_count >= int(bundle["minimum_parsed_track_count"])
            and parsed_count <= int(bundle["maximum_parsed_track_count"])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = datetime.now(timezone.utc)
    bundle = load_bundle(args.bundle)
    validate_protocol_payloads(bundle)
    if bundle["mode"] != "pilot":
        raise ValueError("this frozen runner accepts only the Phase 3 pilot")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 3 natural generation requires a CUDA GPU")
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(
        bundle["model_id"],
        revision=bundle["model_revision"],
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        bundle["model_id"],
        revision=bundle["model_revision"],
        torch_dtype=torch.float16,
        trust_remote_code=False,
    ).to(device)
    model.eval()
    architecture_gate = (
        hasattr(model, "model")
        and hasattr(model.model, "layers")
        and len(model.model.layers) == int(bundle["expected_num_layers"])
    )
    if not architecture_gate:
        raise ValueError("loaded model does not match the frozen architecture")

    rows: list[dict[str, Any]] = []
    for context in bundle["contexts"]:
        prompt = context["generation_prompt"].rstrip()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        for seed in bundle["seeds"]:
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=float(bundle["generation"]["temperature"]),
                    top_p=float(bundle["generation"]["top_p"]),
                    max_new_tokens=int(bundle["generation"]["max_new_tokens"]),
                    pad_token_id=tokenizer.pad_token_id,
                )
            completion_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            completion = tokenizer.decode(
                completion_ids, skip_special_tokens=True
            )
            parsed_tracks = parse_playlist(
                bundle["completion_prefix"] + completion,
                maximum_tracks=int(bundle["generation"]["tracks_per_playlist"]),
            )
            rows.append(
                {
                    "generation_id": f"{context['context_id']}__seed{seed}",
                    "context_id": context["context_id"],
                    "seed": int(seed),
                    "completion": completion,
                    "parsed_track_count": len(parsed_tracks),
                    "parsed_tracks": parsed_tracks,
                }
            )

    summary = summarize_rows(rows, bundle, architecture_gate)
    summary.update(
        {
            "run_id": started.strftime("%Y%m%dT%H%M%SZ")
            + "_phase3_natural_pilot",
            "protocol_id": bundle["protocol_id"],
            "protocol_hashes": bundle["protocol_hashes"],
            "mode": bundle["mode"],
            "prompt_template_id": bundle["prompt_template_id"],
            "model_id": bundle["model_id"],
            "model_revision": bundle["model_revision"],
            "expected_num_layers": bundle["expected_num_layers"],
            "observed_num_layers": len(model.model.layers),
            "bundle_canonical_sha256": sha256_bytes(
                json.dumps(
                    bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ),
            "submitted_script_sha256": None,
            "script_execution_mode": "inline_python_c_no_file",
            "maximum_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    artifact = {
        "rows": rows,
        "summary": summary,
        "claim_boundaries": bundle["claim_boundaries"],
        "continuation_gate": bundle["continuation_gate"],
    }
    for row in rows:
        print(
            "PHASE3_GENERATION_ROW_JSON="
            + json.dumps(row, ensure_ascii=True, separators=(",", ":"))
        )
    chunks = encode_artifact_chunks(artifact)
    for index, data in enumerate(chunks):
        print(
            "PHASE3_GENERATION_ARTIFACT_CHUNK_JSON="
            + json.dumps(
                {"index": index, "total": len(chunks), "data": data},
                separators=(",", ":"),
            )
        )
    print(
        "PHASE3_GENERATION_SUMMARY_JSON="
        + json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    )
    return 0 if summary["technical_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
