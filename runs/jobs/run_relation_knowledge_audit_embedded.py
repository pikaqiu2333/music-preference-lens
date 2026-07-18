# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "accelerate==1.8.1",
#   "torch==2.7.1",
#   "transformers==4.53.3",
# ]
# ///
"""Run the frozen relation-knowledge recoverability audit on HF Jobs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import statistics
import unicodedata
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EMBEDDED_BUNDLE_B64_ZLIB = "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"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_bundle(path: Path | None = None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    if (
        EMBEDDED_BUNDLE_B64_ZLIB
        == "__RELATION_KNOWLEDGE_AUDIT_BUNDLE_B64_ZLIB__"
    ):
        raise ValueError("pass --bundle or embed the audit bundle")
    payload = zlib.decompress(base64.b64decode(EMBEDDED_BUNDLE_B64_ZLIB))
    return json.loads(payload.decode("utf-8"))


def validate_protocol_payloads(bundle: dict[str, Any]) -> None:
    for key in ("json", "markdown"):
        payload = base64.b64decode(bundle["protocol_payloads_b64"][key])
        if sha256_bytes(payload) != bundle["protocol_hashes"][f"{key}_sha256"]:
            raise ValueError(f"embedded protocol {key} hash mismatch")


def normalize_entity(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def parse_artist_generation(value: str) -> str:
    first_line = value.strip().splitlines()[0].strip() if value.strip() else ""
    first_line = re.sub(
        r"^(?:artist|recorded\s+by)\s*:\s*", "", first_line, flags=re.IGNORECASE
    )
    return first_line.strip().strip("\"'`*_ |.,;:()[]{}")


def matches_any(value: str, accepted: list[str]) -> bool:
    normalized = normalize_entity(value)
    return bool(normalized) and normalized in {
        normalize_entity(candidate) for candidate in accepted
    }


def conflict_category(
    target_generation_count: int,
    emitted_generation_count: int,
    primary_margin: float,
    minimum_recoveries: int = 2,
) -> str:
    target_recovered = target_generation_count >= minimum_recoveries
    positive_margin = primary_margin > 0.0
    if target_recovered and positive_margin:
        return "reference_recoverable"
    if target_recovered:
        return "generation_only"
    if positive_margin:
        return "margin_only"
    if emitted_generation_count >= minimum_recoveries:
        return "persistent_emitted_binding"
    return "unrecovered_or_indeterminate"


def encode_artifact_chunks(value: Any, maximum_chars: int = 7500) -> list[str]:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    encoded = base64.b64encode(zlib.compress(payload, level=9)).decode("ascii")
    return [
        encoded[start : start + maximum_chars]
        for start in range(0, len(encoded), maximum_chars)
    ]


def continuation_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    continuation: str,
    device: str,
) -> dict[str, float | int]:
    import torch

    full_text = prompt + continuation
    encoded = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    encoded = encoded.to(device)
    with torch.no_grad():
        logits = model(**encoded, use_cache=False).logits[0]
    input_ids = encoded["input_ids"][0]
    token_logprobs = torch.log_softmax(logits[:-1].float(), dim=-1)
    values = []
    boundary = len(prompt)
    for token_index in range(1, len(input_ids)):
        start, end = offsets[token_index]
        if end <= boundary or end <= start:
            continue
        values.append(
            float(token_logprobs[token_index - 1, input_ids[token_index]].item())
        )
    if not values:
        raise ValueError("candidate continuation produced no scorable tokens")
    return {
        "token_count": len(values),
        "sum_logprob": sum(values),
        "mean_logprob": sum(values) / len(values),
    }


def summarize_records(
    record_rows: list[dict[str, Any]],
    prompt_rows: list[dict[str, Any]],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    group_counts = Counter(row["source_group"] for row in record_rows)
    category_counts = Counter(
        row["audit_category"]
        for row in record_rows
        if row["source_group"] == "phase2_conflict"
    )
    canonical = [
        row for row in record_rows if row["source_group"] == "canonical_positive"
    ]
    exact = [
        row
        for row in record_rows
        if row["source_group"] == "phase2_generated_exact"
    ]
    minimum = int(bundle["scoring"]["target_generation_recovery_minimum_prompts"])
    canonical_recovered = sum(
        row["target_generation_count"] >= minimum for row in canonical
    )
    exact_recovered = sum(row["target_generation_count"] >= minimum for row in exact)
    nonempty = sum(bool(row["raw_generation"].strip()) for row in prompt_rows)
    expected_prompts = int(bundle["expected_prompt_count"])
    nonempty_fraction = nonempty / expected_prompts if expected_prompts else 0.0
    canonical_fraction = canonical_recovered / len(canonical) if canonical else 0.0
    technical_gate = (
        len(record_rows) == int(bundle["expected_record_count"])
        and len(prompt_rows) == expected_prompts
        and nonempty_fraction
        >= float(bundle["validity_gates"]["minimum_nonempty_generation_fraction"])
    )
    assay_validity_gate = (
        technical_gate
        and canonical_fraction
        >= float(
            bundle["validity_gates"][
                "minimum_canonical_control_recovery_fraction"
            ]
        )
    )
    return {
        "record_count": len(record_rows),
        "prompt_count": len(prompt_rows),
        "group_counts": dict(sorted(group_counts.items())),
        "nonempty_generation_count": nonempty,
        "nonempty_generation_fraction": nonempty_fraction,
        "canonical_recovered_count": canonical_recovered,
        "canonical_control_count": len(canonical),
        "canonical_recovery_fraction": canonical_fraction,
        "generated_exact_recovered_count": exact_recovered,
        "generated_exact_control_count": len(exact),
        "generated_exact_recovery_fraction": (
            exact_recovered / len(exact) if exact else 0.0
        ),
        "conflict_category_counts": dict(sorted(category_counts.items())),
        "reference_recoverable_record_ids": [
            row["record_id"]
            for row in record_rows
            if row["audit_category"] == "reference_recoverable"
        ],
        "technical_gate": technical_gate,
        "assay_validity_gate": assay_validity_gate,
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
    if not torch.cuda.is_available():
        raise RuntimeError("relation-knowledge audit requires a CUDA GPU")
    device = "cuda"
    model_spec = bundle["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["model_id"],
        revision=model_spec["revision"],
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["model_id"],
        revision=model_spec["revision"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    ).to(device)
    model.eval()
    observed_layers = len(model.model.layers)
    if observed_layers != int(model_spec["expected_num_layers"]):
        raise ValueError("loaded model does not match the frozen architecture")

    prompt_rows = []
    record_rows = []
    minimum = int(bundle["scoring"]["target_generation_recovery_minimum_prompts"])
    for record in bundle["records"]:
        per_prompt = []
        for template in bundle["prompt_templates"]:
            prompt = template["template"].format(title=record["title"])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    do_sample=bool(bundle["generation"]["do_sample"]),
                    max_new_tokens=int(bundle["generation"]["max_new_tokens"]),
                    pad_token_id=tokenizer.pad_token_id,
                )
            completion_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            raw_generation = tokenizer.decode(
                completion_ids, skip_special_tokens=True
            )
            parsed_artist = parse_artist_generation(raw_generation)
            target_score = continuation_logprob(
                model,
                tokenizer,
                prompt,
                record["target_artist"],
                device,
            )
            emitted_score = None
            mean_margin = None
            sum_margin = None
            if record["emitted_artist"]:
                emitted_score = continuation_logprob(
                    model,
                    tokenizer,
                    prompt,
                    record["emitted_artist"],
                    device,
                )
                mean_margin = (
                    float(target_score["mean_logprob"])
                    - float(emitted_score["mean_logprob"])
                )
                sum_margin = (
                    float(target_score["sum_logprob"])
                    - float(emitted_score["sum_logprob"])
                )
            row = {
                "record_id": record["record_id"],
                "source_group": record["source_group"],
                "title": record["title"],
                "target_artist": record["target_artist"],
                "emitted_artist": record["emitted_artist"],
                "template_id": template["template_id"],
                "raw_generation": raw_generation,
                "parsed_artist": parsed_artist,
                "normalized_parsed_artist": normalize_entity(parsed_artist),
                "target_generated": matches_any(
                    parsed_artist, record["accepted_target_artists"]
                ),
                "emitted_generated": matches_any(
                    parsed_artist, record["accepted_emitted_artists"]
                ),
                "target_score": target_score,
                "emitted_score": emitted_score,
                "target_minus_emitted_mean_logprob": mean_margin,
                "target_minus_emitted_sum_logprob": sum_margin,
            }
            prompt_rows.append(row)
            per_prompt.append(row)

        target_count = sum(row["target_generated"] for row in per_prompt)
        emitted_count = sum(row["emitted_generated"] for row in per_prompt)
        margins = [
            float(row["target_minus_emitted_mean_logprob"])
            for row in per_prompt
            if row["target_minus_emitted_mean_logprob"] is not None
        ]
        primary_margin = statistics.median(margins) if margins else None
        if record["source_group"] == "phase2_conflict":
            audit_category = conflict_category(
                target_count,
                emitted_count,
                float(primary_margin),
                minimum,
            )
        else:
            audit_category = (
                "target_recoverable" if target_count >= minimum else "target_unrecovered"
            )
        record_rows.append(
            {
                **record,
                "target_generation_count": target_count,
                "emitted_generation_count": emitted_count,
                "primary_mean_logprob_margin": primary_margin,
                "template_mean_logprob_margins": margins,
                "audit_category": audit_category,
            }
        )

    summary = summarize_records(record_rows, prompt_rows, bundle)
    summary.update(
        {
            "audit_id": bundle["audit_id"],
            "model_id": model_spec["model_id"],
            "model_revision": model_spec["revision"],
            "observed_num_layers": observed_layers,
            "bundle_sha256": sha256_bytes(
                json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            ),
            "submitted_script_sha256": None,
            "script_execution_mode": "inline_python_c_no_file",
            "maximum_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    artifact = {
        "summary": summary,
        "record_rows": record_rows,
        "prompt_rows": prompt_rows,
        "claim_boundaries": bundle["claim_boundaries"],
        "downstream_rule": bundle["downstream_rule"],
    }
    for row in record_rows:
        print(
            "REL_KNOWLEDGE_ROW_JSON="
            + json.dumps(row, ensure_ascii=True, separators=(",", ":"))
        )
    chunks = encode_artifact_chunks(artifact)
    for index, data in enumerate(chunks):
        print(
            "REL_KNOWLEDGE_ARTIFACT_CHUNK_JSON="
            + json.dumps(
                {"index": index, "total": len(chunks), "data": data},
                separators=(",", ":"),
            )
        )
    print(
        "REL_KNOWLEDGE_SUMMARY_JSON="
        + json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    )
    return 0 if summary["technical_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
