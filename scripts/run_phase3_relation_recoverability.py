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


EMBEDDED_BUNDLE_B64_ZLIB = "__PHASE3_RELATION_RECOVERABILITY_BUNDLE_B64_ZLIB__"


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
