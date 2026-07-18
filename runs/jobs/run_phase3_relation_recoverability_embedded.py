# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "accelerate==1.8.1",
#   "torch==2.7.1",
#   "transformers==4.53.3",
# ]
# ///
"""Run the frozen Phase 3 relation-recoverability audit on HF Jobs."""

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
        == "__PHASE3_RELATION_RECOVERABILITY_BUNDLE_B64_ZLIB__"
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
        return "recoverable_relation_conflict"
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
        if row["source_group"] == "phase3_conflict"
    )
    canonical = [
        row for row in record_rows if row["source_group"] == "canonical_positive"
    ]
    exact = [
        row
        for row in record_rows
        if row["source_group"] == "phase3_generated_exact"
    ]
    minimum = int(bundle["scoring"]["minimum_reference_generations"])
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
        "phase3_exact_recovered_count": exact_recovered,
        "phase3_exact_control_count": len(exact),
        "phase3_exact_recovery_fraction": (
            exact_recovered / len(exact) if exact else 0.0
        ),
        "conflict_category_counts": dict(sorted(category_counts.items())),
        "recoverable_relation_conflict_record_ids": [
            row["record_id"]
            for row in record_rows
            if row["audit_category"] == "recoverable_relation_conflict"
        ],
        "technical_gate": technical_gate,
        "assay_validity_gate": assay_validity_gate,
    }


def continuation_result(
    summary: dict[str, Any], bundle: dict[str, Any]
) -> dict[str, Any]:
    minimum_catalog_rows = int(
        bundle["pilot_continuation_gate"]["minimum_catalog_rows"]
    )
    minimum_conflicts = int(
        bundle["pilot_continuation_gate"][
            "minimum_unique_recoverable_conflict_title_clusters"
        ]
    )
    recoverable_count = len(
        summary["recoverable_relation_conflict_record_ids"]
    )
    gate = (
        summary["assay_validity_gate"]
        and int(bundle["source_catalog_row_count"]) >= minimum_catalog_rows
        and recoverable_count >= minimum_conflicts
    )
    return {
        "minimum_recoverable_conflict_clusters": minimum_conflicts,
        "pilot_continuation_gate": gate,
        "pilot_decision": (
            bundle["pilot_continuation_gate"]["pass_action"]
            if gate
            else bundle["pilot_continuation_gate"]["failure_action"]
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
        torch_dtype=torch.float16,
        trust_remote_code=False,
    ).to(device)
    model.eval()
    observed_layers = len(model.model.layers)
    if observed_layers != int(model_spec["expected_num_layers"]):
        raise ValueError("loaded model does not match the frozen architecture")

    prompt_rows = []
    record_rows = []
    minimum = int(bundle["scoring"]["minimum_reference_generations"])
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
        if record["source_group"] == "phase3_conflict":
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
            "protocol_id": bundle["protocol_id"],
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
            "source_catalog_sha256": bundle["source_catalog_sha256"],
            "source_catalog_row_count": bundle["source_catalog_row_count"],
            "selected_conflict_cluster_count": bundle[
                "selected_conflict_cluster_count"
            ],
            "selected_exact_cluster_count": bundle[
                "selected_exact_cluster_count"
            ],
            **continuation_result(summary, bundle),
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
    }
    for row in record_rows:
        print(
            "PHASE3_REL_KNOWLEDGE_ROW_JSON="
            + json.dumps(row, ensure_ascii=True, separators=(",", ":"))
        )
    chunks = encode_artifact_chunks(artifact)
    for index, data in enumerate(chunks):
        print(
            "PHASE3_REL_KNOWLEDGE_ARTIFACT_CHUNK_JSON="
            + json.dumps(
                {"index": index, "total": len(chunks), "data": data},
                separators=(",", ":"),
            )
        )
    print(
        "PHASE3_REL_KNOWLEDGE_SUMMARY_JSON="
        + json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    )
    return 0 if summary["technical_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
