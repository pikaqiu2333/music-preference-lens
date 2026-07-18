# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "accelerate==1.8.1",
#   "torch==2.7.1",
#   "transformers==4.53.3",
# ]
# ///
"""Run frozen Phase 2 mechanism diagnosis or candidate-free correction."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import statistics
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EMBEDDED_BUNDLE_B64 = "__PHASE2_MECHANISM_BUNDLE_ZLIB_B64__"
ARTIST_RESPONSE = re.compile(r"\s*Artist:\s*([^\r\n]+?)\s*", re.IGNORECASE)
DEFAULT_BATCH_SIZE = 1
ARTIFACT_FORMAT_VERSION = "phase2_mechanism_runner_shard_v1"


def load_bundle(path: Path | None = None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    if EMBEDDED_BUNDLE_B64 == "__PHASE2_MECHANISM_BUNDLE_ZLIB_B64__":
        raise ValueError("pass --bundle or embed a compressed Phase 2 bundle")
    payload = zlib.decompress(base64.b64decode(EMBEDDED_BUNDLE_B64))
    return json.loads(payload.decode("utf-8"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def encode_artifact_chunks(value: Any, maximum_chars: int = 7500) -> list[str]:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    encoded = base64.b64encode(zlib.compress(payload, level=9)).decode("ascii")
    return [
        encoded[start : start + maximum_chars]
        for start in range(0, len(encoded), maximum_chars)
    ]


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(payload)


def bundle_mode(bundle: dict[str, Any]) -> tuple[str, str]:
    version = bundle.get("bundle_version")
    if version == "phase2_mechanism_diagnosis_v1":
        return "diagnosis", "PHASE2_DIAG"
    if version == "phase2_candidate_free_correction_v1":
        return "correction", "PHASE2_CORRECTION"
    raise ValueError(f"unsupported bundle version: {version}")


def record_ids(records: list[dict[str, Any]], label: str = "records") -> list[str]:
    ids = []
    for index, row in enumerate(records):
        if not isinstance(row, dict) or "record_id" not in row:
            raise ValueError(f"{label} row {index} has no record_id")
        value = row["record_id"]
        if value is None or isinstance(value, bool) or not str(value):
            raise ValueError(f"{label} row {index} has an invalid record_id")
        ids.append(str(value))
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} contains duplicate record IDs")
    return ids


def resolve_record_range(
    record_count: int, record_start: int, record_stop: int | None
) -> tuple[int, int]:
    if isinstance(record_count, bool) or not isinstance(record_count, int):
        raise ValueError("record count must be an integer")
    if isinstance(record_start, bool) or not isinstance(record_start, int):
        raise ValueError("record start must be an integer")
    if record_stop is not None and (
        isinstance(record_stop, bool) or not isinstance(record_stop, int)
    ):
        raise ValueError("record stop must be an integer")
    stop = record_count if record_stop is None else record_stop
    if record_start < 0:
        raise ValueError("record start must be non-negative")
    if stop < 0:
        raise ValueError("record stop must be non-negative")
    if record_start >= stop:
        raise ValueError("record range must be non-empty and use start < stop")
    if stop > record_count:
        raise ValueError(
            f"record stop {stop} exceeds full record count {record_count}"
        )
    return record_start, stop


def batch_ranges(
    record_start: int, record_stop: int, batch_size: int
) -> list[tuple[int, int]]:
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (record_start, record_stop, batch_size)
    ):
        raise ValueError("batch range values must be integers")
    if record_start < 0 or record_start >= record_stop:
        raise ValueError("batch range must be non-empty and use start < stop")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    return [
        (start, min(start + batch_size, record_stop))
        for start in range(record_start, record_stop, batch_size)
    ]


def build_receipt(
    bundle: dict[str, Any],
    *,
    mode: str,
    run_id: str,
    script_sha256: str,
    bundle_sha256: str,
    record_start: int,
    record_stop: int,
    batch_size: int,
    technical: dict[str, Any],
) -> dict[str, Any]:
    if bundle_sha256 != canonical_json_sha256(bundle):
        raise ValueError("receipt canonical bundle SHA does not match the full bundle")
    return {
        "receipt_version": "phase2_mechanism_runner_receipt_v1",
        "run_id": run_id,
        "mode": mode,
        "protocol_id": bundle["protocol_id"],
        "protocol_hashes": bundle["protocol_hashes"],
        "catalog_asset_hashes": bundle.get("catalog_asset_hashes", {}),
        "model_id": bundle["model_id"],
        "model_revision": bundle["model_revision"],
        "expected_num_layers": int(bundle["expected_num_layers"]),
        "submitted_script_sha256": script_sha256,
        "canonical_bundle_sha256": bundle_sha256,
        "canonical_bundle_scope": "full_unsliced_bundle",
        "full_record_count": len(bundle["records"]),
        "record_start": record_start,
        "record_stop": record_stop,
        "record_count": record_stop - record_start,
        "record_range_semantics": "zero_based_half_open",
        "batch_size": batch_size,
        "checkpoint_count": len(batch_ranges(record_start, record_stop, batch_size)),
        **technical,
    }


def build_artifact_summary(
    *,
    receipt: dict[str, Any],
    technical: dict[str, Any],
    artifact_kind: str,
    records: list[dict[str, Any]],
    expected_records: list[dict[str, Any]],
    record_start: int,
    record_stop: int,
    maximum_gpu_memory_bytes: int,
    started_at: str,
    finished_at: str,
    batch_index: int | None = None,
) -> dict[str, Any]:
    if artifact_kind not in {"checkpoint", "shard"}:
        raise ValueError("runner artifact kind must be checkpoint or shard")
    if record_start < 0 or record_start >= record_stop:
        raise ValueError("artifact range must be non-empty and use start < stop")
    if not (
        int(receipt["record_start"]) <= record_start
        and record_stop <= int(receipt["record_stop"])
    ):
        raise ValueError("artifact range falls outside the receipt range")
    if artifact_kind == "shard" and (
        record_start != int(receipt["record_start"])
        or record_stop != int(receipt["record_stop"])
    ):
        raise ValueError("final shard range must equal the receipt range")
    expected_ids = record_ids(expected_records, "expected records")
    if len(expected_ids) != record_stop - record_start:
        raise ValueError("expected records do not fill the artifact range")
    observed_ids = record_ids(records, "artifact records")
    record_gate = observed_ids == expected_ids
    technical_gate = bool(
        technical.get("architecture_gate")
        and technical.get("endpoint_gate")
        and record_gate
    )
    summary = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "artifact_kind": artifact_kind,
        "run_id": receipt["run_id"],
        "mode": receipt["mode"],
        "protocol_id": receipt["protocol_id"],
        "protocol_hashes": receipt["protocol_hashes"],
        "catalog_asset_hashes": receipt["catalog_asset_hashes"],
        "submitted_script_sha256": receipt["submitted_script_sha256"],
        "canonical_bundle_sha256": receipt["canonical_bundle_sha256"],
        "canonical_bundle_scope": receipt["canonical_bundle_scope"],
        "model_id": receipt["model_id"],
        "model_revision": receipt["model_revision"],
        "expected_num_layers": receipt["expected_num_layers"],
        "full_record_count": receipt["full_record_count"],
        "requested_record_start": receipt["record_start"],
        "requested_record_stop": receipt["record_stop"],
        "record_start": record_start,
        "record_stop": record_stop,
        "record_range_semantics": "zero_based_half_open",
        "expected_record_count": len(expected_ids),
        "observed_record_count": len(observed_ids),
        "record_ids_sha256": canonical_json_sha256(observed_ids),
        "records_sha256": canonical_json_sha256(records),
        "receipt_sha256": canonical_json_sha256(receipt),
        **technical,
        "technical_gate": technical_gate,
        "maximum_gpu_memory_bytes": maximum_gpu_memory_bytes,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    if batch_index is not None:
        summary["batch_index"] = batch_index
    return summary


def emit_artifact_chunks(
    marker: str,
    artifact: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
    maximum_chars: int = 7500,
    emit: Any = print,
) -> None:
    chunks = encode_artifact_chunks(artifact, maximum_chars=maximum_chars)
    for index, data in enumerate(chunks):
        envelope = {
            **(metadata or {}),
            "index": index,
            "total": len(chunks),
            "data": data,
        }
        emit(marker + json.dumps(envelope, separators=(",", ":")))


def append_continuation(prefix: str, continuation: str) -> tuple[str, list[int]]:
    clean_prefix = prefix.rstrip()
    text = clean_prefix + " " + continuation
    return text, [len(clean_prefix) + 1, len(text)]


def overlapping_positions(
    offsets: list[tuple[int, int]], span: list[int]
) -> list[int]:
    start, end = span
    positions = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > start and token_start < end and token_end > token_start
    ]
    if not positions:
        raise ValueError(f"no tokens overlap continuation span {span}")
    return positions


def normalize_entity(value: str) -> str:
    import unicodedata

    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in value if character.isalnum())


def parse_candidate_free_response(response: str) -> dict[str, Any]:
    stripped = response.strip()
    if stripped.casefold() == "abstain":
        return {"status": "abstained", "artist": None, "normalized_artist": None}
    if len(re.findall(r"\bArtist\s*:", response, flags=re.IGNORECASE)) != 1:
        return {"status": "invalid", "artist": None, "normalized_artist": None}
    match = ARTIST_RESPONSE.fullmatch(response)
    if match is None:
        return {"status": "invalid", "artist": None, "normalized_artist": None}
    artist = match.group(1).strip()
    normalized = normalize_entity(artist)
    if not normalized:
        return {"status": "invalid", "artist": None, "normalized_artist": None}
    return {"status": "parsed", "artist": artist, "normalized_artist": normalized}


def contains_key(value: Any, forbidden_key: str) -> bool:
    if isinstance(value, dict):
        return forbidden_key in value or any(
            contains_key(item, forbidden_key) for item in value.values()
        )
    if isinstance(value, list):
        return any(contains_key(item, forbidden_key) for item in value)
    return False


def patch_relation_shift(
    patched_reference: float,
    patched_emitted: float,
    neutral_reference: float,
    neutral_emitted: float,
) -> float:
    return (patched_reference - patched_emitted) - (
        neutral_reference - neutral_emitted
    )


def endpoint_reproduction(model: Any, tokenizer: Any, device: str) -> tuple[float, bool]:
    import torch

    inputs = tokenizer("Title: Teardrop\nArtist:", return_tensors="pt").to(device)
    captured: list[Any] = []

    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        captured.append(output[0] if isinstance(output, tuple) else output)

    handle = model.model.layers[-1].register_forward_hook(capture)
    try:
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("failed to capture the final residual endpoint")
    hidden = model.model.norm(captured[0])
    reproduced = model.lm_head(hidden)
    scaling = float(getattr(model.config, "logits_scaling", 1.0))
    if scaling != 1.0:
        reproduced = reproduced / scaling
    difference = (reproduced - outputs.logits).abs()
    return float(difference.max().item()), bool(torch.isfinite(difference).all().item())


def load_model(bundle: dict[str, Any]) -> tuple[Any, Any, str, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Phase 2 mechanism experiments require a CUDA GPU")
    device = "cuda"
    source = {"revision": bundle["model_revision"]}
    tokenizer = AutoTokenizer.from_pretrained(
        bundle["model_id"], trust_remote_code=False, **source
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        bundle["model_id"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        **source,
    ).to(device)
    model.eval()
    observed_layers = len(model.model.layers)
    architecture_gate = observed_layers == int(bundle["expected_num_layers"])
    if not architecture_gate:
        raise ValueError(
            f"expected {bundle['expected_num_layers']} layers, found {observed_layers}"
        )
    endpoint_error, endpoint_finite = endpoint_reproduction(model, tokenizer, device)
    endpoint_gate = endpoint_finite and endpoint_error <= float(
        bundle["endpoint_tolerance"]
    )
    if not endpoint_gate:
        raise ValueError(f"endpoint reproduction failed: {endpoint_error}")
    return model, tokenizer, device, {
        "architecture_gate": architecture_gate,
        "observed_num_layers": observed_layers,
        "endpoint_gate": endpoint_gate,
        "endpoint_max_logit_error": endpoint_error,
    }


def encode_continuations(
    tokenizer: Any,
    device: str,
    texts: list[str],
    spans: list[list[int]],
) -> tuple[Any, list[list[int]], list[list[int]]]:
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        return_offsets_mapping=True,
        padding=True,
    )
    offsets_batch = encoded.pop("offset_mapping").tolist()
    prediction_positions = []
    target_ids = []
    for index, raw_offsets in enumerate(offsets_batch):
        positions = overlapping_positions(
            [tuple(item) for item in raw_offsets], spans[index]
        )
        prediction_positions.append([position - 1 for position in positions])
        target_ids.append(
            [int(encoded["input_ids"][index, position]) for position in positions]
        )
    return encoded.to(device), prediction_positions, target_ids


def sequence_scores(
    logits: Any,
    positions: list[list[int]],
    target_ids: list[list[int]],
    device: str,
) -> list[float]:
    import torch

    scores = []
    for index, prediction_positions in enumerate(positions):
        step_logits = logits[index, prediction_positions].to(dtype=torch.float32)
        ids = torch.tensor(target_ids[index], device=device)
        token_logits = step_logits[
            torch.arange(len(prediction_positions), device=device), ids
        ]
        token_logps = token_logits - torch.logsumexp(step_logits, dim=-1)
        scores.append(float(token_logps.mean().item()))
    return scores


def run_template_diagnosis(
    model: Any,
    tokenizer: Any,
    device: str,
    records: list[dict[str, Any]],
    prompt_template: str,
    layers: list[int],
) -> list[dict[str, Any]]:
    import torch

    specs = []
    source_texts = []
    source_spans = []
    target_texts = []
    target_spans = []
    for row_index, row in enumerate(records):
        if row.get("technical_failure") or len(row["diagnostic_controls"]) != 3:
            continue
        for control_index, control in enumerate(row["diagnostic_controls"]):
            for role in ("emitted", "reference"):
                artist = row[f"{role}_artist"]
                source_text, source_span = append_continuation(
                    prompt_template.format(title=row["title"]), artist
                )
                target_text, target_span = append_continuation(
                    prompt_template.format(title=control["title"]), artist
                )
                specs.append(
                    {
                        "row_index": row_index,
                        "control_index": control_index,
                        "role": role,
                    }
                )
                source_texts.append(source_text)
                source_spans.append(source_span)
                target_texts.append(target_text)
                target_spans.append(target_span)

    output = [
        {
            "record_id": row["record_id"],
            "technical_failure": row.get("technical_failure"),
            "controls": [],
            "layer_median_shifts": {},
        }
        for row in records
    ]
    if not specs:
        return output

    source_inputs, source_positions, source_ids = encode_continuations(
        tokenizer, device, source_texts, source_spans
    )
    target_inputs, target_positions, target_ids = encode_continuations(
        tokenizer, device, target_texts, target_spans
    )
    aligned = [
        source_ids[index] == target_ids[index]
        and len(source_positions[index]) == len(target_positions[index])
        for index in range(len(specs))
    ]

    def output_hidden(value: Any) -> Any:
        return value[0] if isinstance(value, tuple) else value

    source_vectors: dict[int, list[Any]] = {}
    handles = []
    for layer_number in layers:
        module = model.model.layers[layer_number - 1]

        def capture(
            _module: Any,
            _inputs: Any,
            layer_output: Any,
            capture_layer: int = layer_number,
        ) -> None:
            hidden = output_hidden(layer_output)
            source_vectors[capture_layer] = [
                hidden[index, positions].detach().clone()
                for index, positions in enumerate(source_positions)
            ]

        handles.append(module.register_forward_hook(capture))
    try:
        with torch.no_grad():
            source_outputs = model(**source_inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    source_scores = sequence_scores(
        source_outputs.logits, source_positions, source_ids, device
    )
    del source_outputs
    with torch.no_grad():
        target_outputs = model(**target_inputs, use_cache=False)
    target_scores = sequence_scores(
        target_outputs.logits, target_positions, target_ids, device
    )
    del target_outputs

    patched_scores: dict[int, list[float]] = {}
    for layer_number in layers:
        module = model.model.layers[layer_number - 1]
        values = source_vectors[layer_number]

        def patch(
            _module: Any,
            _inputs: Any,
            layer_output: Any,
            source_values: list[Any] = values,
        ) -> Any:
            hidden = output_hidden(layer_output)
            patched = hidden.clone()
            for index, positions in enumerate(target_positions):
                if aligned[index]:
                    patched[index, positions] = source_values[index]
            if isinstance(layer_output, tuple):
                return (patched, *layer_output[1:])
            return patched

        handle = module.register_forward_hook(patch)
        try:
            with torch.no_grad():
                patched_outputs = model(**target_inputs, use_cache=False)
        finally:
            handle.remove()
        patched_scores[layer_number] = sequence_scores(
            patched_outputs.logits, target_positions, target_ids, device
        )
        del patched_outputs

    index_by_spec = {
        (spec["row_index"], spec["control_index"], spec["role"]): index
        for index, spec in enumerate(specs)
    }
    for row_index, row in enumerate(records):
        result = output[row_index]
        if row.get("technical_failure") or len(row["diagnostic_controls"]) != 3:
            result["technical_failure"] = (
                row.get("technical_failure") or "diagnostic_control_count_not_three"
            )
            continue
        controls = []
        layer_shifts: dict[int, list[float]] = {layer: [] for layer in layers}
        row_failed = False
        for control_index, control in enumerate(row["diagnostic_controls"]):
            emitted_index = index_by_spec[(row_index, control_index, "emitted")]
            reference_index = index_by_spec[(row_index, control_index, "reference")]
            if not aligned[emitted_index] or not aligned[reference_index]:
                row_failed = True
            control_result = {
                "normalized_title": control["normalized_title"],
                "neutral_emitted_score": target_scores[emitted_index],
                "neutral_reference_score": target_scores[reference_index],
                "source_emitted_score": source_scores[emitted_index],
                "source_reference_score": source_scores[reference_index],
                "layer_shifts": {},
            }
            for layer_number in layers:
                shift = patch_relation_shift(
                    patched_scores[layer_number][reference_index],
                    patched_scores[layer_number][emitted_index],
                    target_scores[reference_index],
                    target_scores[emitted_index],
                )
                control_result["layer_shifts"][str(layer_number)] = shift
                layer_shifts[layer_number].append(shift)
                if not math.isfinite(shift):
                    row_failed = True
            controls.append(control_result)
        result["controls"] = controls
        result["layer_median_shifts"] = {
            str(layer): statistics.median(shifts)
            for layer, shifts in layer_shifts.items()
        }
        if row_failed:
            result["technical_failure"] = "token_alignment_or_nonfinite_shift"
    return output


def score_manipulation_checks(
    model: Any,
    tokenizer: Any,
    device: str,
    records: list[dict[str, Any]],
    prompt_template: str,
) -> list[dict[str, Any]]:
    import torch

    specs = []
    texts = []
    spans = []
    for row_index, row in enumerate(records):
        if row.get("technical_failure") or len(row["manipulation_controls"]) != 3:
            continue
        titles = [("factual", row["title"])] + [
            (f"neutral_{index}", control["title"])
            for index, control in enumerate(row["manipulation_controls"])
        ]
        for title_role, title in titles:
            for candidate_role in ("emitted", "reference"):
                text, span = append_continuation(
                    prompt_template.format(title=title), row[f"{candidate_role}_artist"]
                )
                specs.append(
                    {
                        "row_index": row_index,
                        "title_role": title_role,
                        "candidate_role": candidate_role,
                    }
                )
                texts.append(text)
                spans.append(span)
    output = [
        {"record_id": row["record_id"], "technical_failure": row.get("technical_failure")}
        for row in records
    ]
    if not specs:
        return output
    inputs, positions, ids = encode_continuations(tokenizer, device, texts, spans)
    with torch.no_grad():
        model_outputs = model(**inputs, use_cache=False)
    scores = sequence_scores(model_outputs.logits, positions, ids, device)
    del model_outputs
    lookup = {
        (spec["row_index"], spec["title_role"], spec["candidate_role"]): scores[index]
        for index, spec in enumerate(specs)
    }
    for row_index, row in enumerate(records):
        result = output[row_index]
        if row.get("technical_failure") or len(row["manipulation_controls"]) != 3:
            result["technical_failure"] = (
                row.get("technical_failure") or "manipulation_control_count_not_three"
            )
            continue
        candidate_results = {}
        for role in ("emitted", "reference"):
            factual = lookup[(row_index, "factual", role)]
            neutrals = [
                lookup[(row_index, f"neutral_{index}", role)] for index in range(3)
            ]
            candidate_results[role] = {
                "factual_score": factual,
                "neutral_scores": neutrals,
                "prior_subtracted_score": factual - statistics.mean(neutrals),
                "placebo_score": neutrals[0] - statistics.mean(neutrals[1:]),
            }
        result["candidate_scores"] = candidate_results
        result["prior_subtracted_reference_minus_emitted"] = (
            candidate_results["reference"]["prior_subtracted_score"]
            - candidate_results["emitted"]["prior_subtracted_score"]
        )
        result["placebo_reference_minus_emitted"] = (
            candidate_results["reference"]["placebo_score"]
            - candidate_results["emitted"]["placebo_score"]
        )
    return output


def run_diagnosis(
    bundle: dict[str, Any], model: Any, tokenizer: Any, device: str
) -> dict[str, Any]:
    records = bundle["records"]
    templates = []
    for template_index, template in enumerate(bundle["prompt_templates"]):
        templates.append(
            {
                "template_index": template_index,
                "rows": run_template_diagnosis(
                    model,
                    tokenizer,
                    device,
                    records,
                    template,
                    [int(layer) for layer in bundle["layers"]],
                ),
            }
        )
    manipulation = score_manipulation_checks(
        model,
        tokenizer,
        device,
        records,
        bundle["closed_set_manipulation_prompt_template"],
    )
    return {
        "records": [
            {
                "record_id": row["record_id"],
                "normalized_title": row["normalized_title"],
                "template_results": [
                    template["rows"][index] for template in templates
                ],
                "manipulation_check": manipulation[index],
            }
            for index, row in enumerate(records)
        ]
    }


def run_correction(
    bundle: dict[str, Any], model: Any, tokenizer: Any, device: str
) -> dict[str, Any]:
    import torch

    if contains_key(bundle, "reference_artist"):
        raise ValueError("candidate-free correction bundle contains a reference field")
    rows = []
    for record in bundle["records"]:
        conditions = {}
        for condition, prompts in record["prompts"].items():
            if len(prompts) != 2:
                raise ValueError("each correction condition requires two paraphrases")
            condition_rows = []
            for prompt_index, prompt in enumerate(prompts):
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        do_sample=bool(bundle["generation"]["do_sample"]),
                        max_new_tokens=int(bundle["generation"]["max_new_tokens"]),
                        pad_token_id=tokenizer.pad_token_id,
                    )
                completion_ids = output_ids[0, inputs["input_ids"].shape[1] :]
                response = tokenizer.decode(completion_ids, skip_special_tokens=True)
                condition_rows.append(
                    {
                        "prompt_index": prompt_index,
                        "response": response,
                        "parsed": parse_candidate_free_response(response),
                    }
                )
            conditions[condition] = condition_rows
        rows.append(
            {
                "record_id": record["record_id"],
                "record_type": record["record_type"],
                "conditions": conditions,
            }
        )
    return {"records": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    parser.add_argument(
        "--record-start",
        type=int,
        default=0,
        help="zero-based inclusive start in canonical bundle order",
    )
    parser.add_argument(
        "--record-stop",
        type=int,
        help="zero-based exclusive stop in canonical bundle order",
    )
    parser.add_argument(
        "--batch-size",
        "--checkpoint-batch-size",
        dest="batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="records computed before each durable log checkpoint",
    )
    args = parser.parse_args()

    import torch

    started = datetime.now(timezone.utc)
    bundle = load_bundle(args.bundle)
    mode, prefix = bundle_mode(bundle)
    full_records = bundle["records"]
    record_ids(full_records, "canonical bundle records")
    record_start, record_stop = resolve_record_range(
        len(full_records), args.record_start, args.record_stop
    )
    ranges = batch_ranges(record_start, record_stop, args.batch_size)
    if mode == "correction" and contains_key(bundle, "reference_artist"):
        raise ValueError("candidate-free correction bundle contains a reference field")
    script_sha256 = sha256_bytes(Path(__file__).read_bytes())
    bundle_sha256 = canonical_json_sha256(bundle)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + f"_phase2_{mode}"
    model, tokenizer, device, technical = load_model(bundle)
    receipt = build_receipt(
        bundle,
        mode=mode,
        run_id=run_id,
        script_sha256=script_sha256,
        bundle_sha256=bundle_sha256,
        record_start=record_start,
        record_stop=record_stop,
        batch_size=args.batch_size,
        technical=technical,
    )
    print(
        f"{prefix}_RECEIPT_JSON="
        + json.dumps(receipt, ensure_ascii=True, separators=(",", ":")),
        flush=True,
    )

    completed_records = []
    for batch_index, (batch_start, batch_stop) in enumerate(ranges):
        batch_bundle = {**bundle, "records": full_records[batch_start:batch_stop]}
        if mode == "diagnosis":
            batch_artifact = run_diagnosis(batch_bundle, model, tokenizer, device)
        else:
            batch_artifact = run_correction(batch_bundle, model, tokenizer, device)
        batch_records = batch_artifact["records"]
        checkpoint_summary = build_artifact_summary(
            receipt=receipt,
            technical=technical,
            artifact_kind="checkpoint",
            records=batch_records,
            expected_records=full_records[batch_start:batch_stop],
            record_start=batch_start,
            record_stop=batch_stop,
            maximum_gpu_memory_bytes=int(torch.cuda.max_memory_allocated()),
            started_at=started.isoformat(),
            finished_at=datetime.now(timezone.utc).isoformat(),
            batch_index=batch_index,
        )
        checkpoint = {"records": batch_records, "summary": checkpoint_summary}
        checkpoint_id = f"{batch_start:06d}-{batch_stop:06d}"
        emit_artifact_chunks(
            f"{prefix}_CHECKPOINT_ARTIFACT_CHUNK_JSON=",
            checkpoint,
            metadata={
                "checkpoint_id": checkpoint_id,
                "batch_index": batch_index,
                "record_start": batch_start,
                "record_stop": batch_stop,
            },
            emit=lambda line: print(line, flush=True),
        )
        if not checkpoint_summary["technical_gate"]:
            raise ValueError(f"checkpoint {checkpoint_id} failed its technical gate")
        completed_records.extend(batch_records)

    summary = build_artifact_summary(
        receipt=receipt,
        technical=technical,
        artifact_kind="shard",
        records=completed_records,
        expected_records=full_records[record_start:record_stop],
        record_start=record_start,
        record_stop=record_stop,
        maximum_gpu_memory_bytes=int(torch.cuda.max_memory_allocated()),
        started_at=started.isoformat(),
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    artifact = {"records": completed_records, "summary": summary}
    emit_artifact_chunks(
        f"{prefix}_ARTIFACT_CHUNK_JSON=",
        artifact,
        metadata={
            "artifact_kind": "shard",
            "record_start": record_start,
            "record_stop": record_stop,
        },
        emit=lambda line: print(line, flush=True),
    )
    print(
        f"{prefix}_SUMMARY_JSON="
        + json.dumps(summary, ensure_ascii=True, separators=(",", ":")),
        flush=True,
    )
    return 0 if summary["technical_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
