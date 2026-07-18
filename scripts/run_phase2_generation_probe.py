# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "accelerate==1.8.1",
#   "huggingface-hub>=0.33.2,<1.0",
#   "torch==2.7.1",
#   "transformers==4.53.3",
# ]
# ///
"""Run a frozen Phase 2 free-generation batch on Hugging Face Jobs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import tempfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EMBEDDED_BUNDLE_B64 = "__PHASE2_GENERATION_BUNDLE_B64__"

FIELD_PATTERN = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?(?:\*\*)?"
    r"(?P<field>title|artist|reason)(?:\*\*)?\s*:\s*(?P<value>.*?)\s*$",
    re.IGNORECASE,
)
INLINE_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?(?:\*\*)?title(?:\*\*)?\s*:\s*"
    r"(?P<title>.*?)\s+(?:\*\*)?artist(?:\*\*)?\s*:\s*"
    r"(?P<artist>.*?)\s+(?:\*\*)?reason(?:\*\*)?\s*:\s*"
    r"(?P<reason>.*?)(?=\n\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?"
    r"(?:\*\*)?title(?:\*\*)?\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_bundle(path: Path | None = None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    if EMBEDDED_BUNDLE_B64 == "__PHASE2_GENERATION_BUNDLE_B64__":
        raise ValueError("pass --bundle or embed the Phase 2 generation bundle")
    return json.loads(base64.b64decode(EMBEDDED_BUNDLE_B64).decode("utf-8"))


def validate_protocol_payloads(bundle: dict[str, Any]) -> dict[str, str]:
    observed = {}
    for key in ("json", "markdown"):
        payload = base64.b64decode(bundle["protocol_payloads_b64"][key])
        observed[f"{key}_sha256"] = sha256_bytes(payload)
        if observed[f"{key}_sha256"] != bundle["protocol_hashes"][f"{key}_sha256"]:
            raise ValueError(f"embedded protocol {key} hash mismatch")
    return observed


def clean_field_value(value: str) -> str:
    return value.strip().strip("| ").strip('"\'').strip()


def parse_playlist(text: str) -> list[dict[str, str]]:
    inline = [
        {field: clean_field_value(match.group(field)) for field in ("title", "artist", "reason")}
        for match in INLINE_PATTERN.finditer(text)
    ]
    if inline:
        return [row for row in inline if all(row.values())][:5]

    rows = []
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
    return rows[:5]


def output_hidden(output: Any) -> Any:
    return output[0] if isinstance(output, tuple) else output


def endpoint_reproduction(model: Any, tokenizer: Any, device: str) -> tuple[float, bool]:
    import torch

    captured: list[Any] = []

    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        captured.append(output_hidden(output).detach())

    layers = model.model.layers
    handle = layers[-1].register_forward_hook(capture)
    try:
        inputs = tokenizer("Technical endpoint check.", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("final layer hook did not capture exactly one tensor")
    hidden = model.model.norm(captured[0])
    reproduced = model.lm_head(hidden)
    scaling = float(getattr(model.config, "logits_scaling", 1.0))
    if scaling != 1.0:
        reproduced = reproduced / scaling
    error = float((reproduced.float() - outputs.logits.float()).abs().max().item())
    return error, bool(torch.isfinite(reproduced).all().item())


def encode_artifact_chunks(value: Any, maximum_chars: int = 7500) -> list[str]:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded = base64.b64encode(zlib.compress(payload, level=9)).decode("ascii")
    return [
        encoded[start : start + maximum_chars]
        for start in range(0, len(encoded), maximum_chars)
    ]


def upload_artifact(bundle: dict[str, Any], run_id: str, artifact: dict[str, Any]) -> str | None:
    repo_id = bundle.get("output_repo")
    if not repo_id:
        return None
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is required when output_repo is configured")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=bool(bundle.get("output_repo_private", False)),
        exist_ok=True,
    )
    path_in_repo = f"raw_generation/{run_id}.json"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "artifact.json"
        path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Archive {run_id}",
        )
    return f"https://huggingface.co/datasets/{repo_id}/blob/main/{path_in_repo}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = datetime.now(timezone.utc)
    bundle = load_bundle(args.bundle)
    if bundle["mode"] == "smoke" and bundle.get("output_repo"):
        raise ValueError("technical smoke is forbidden from uploading raw artifacts")
    protocol_hashes = validate_protocol_payloads(bundle)
    script_sha256 = sha256_bytes(Path(__file__).read_bytes())
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + f"_phase2_{bundle['mode']}_generation"
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 2 generation requires a CUDA GPU")
    device = "cuda"
    model_source = {"revision": bundle["model_revision"]}
    tokenizer = AutoTokenizer.from_pretrained(
        bundle["model_id"], trust_remote_code=False, **model_source
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        bundle["model_id"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        **model_source,
    ).to(device)
    model.eval()
    architecture_gate = (
        hasattr(model, "model")
        and hasattr(model.model, "layers")
        and len(model.model.layers) == int(bundle["expected_num_layers"])
    )
    if not architecture_gate:
        raise ValueError("loaded model does not match the frozen decoder architecture")
    endpoint_error, endpoint_finite = endpoint_reproduction(model, tokenizer, device)
    endpoint_gate = endpoint_finite and endpoint_error <= float(bundle["endpoint_tolerance"])

    rows = []
    total_parsed_tracks = 0
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
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            parsed_tracks = parse_playlist(prompt + completion)
            total_parsed_tracks += len(parsed_tracks)
            row = {
                "generation_id": f"{context['context_id']}__seed{seed}",
                "context_id": context["context_id"],
                "seed": int(seed),
                "completion": completion,
                "parsed_track_count": len(parsed_tracks),
            }
            rows.append(row)

    expected = int(bundle["expected_generation_count"])
    nonempty_count = sum(bool(row["completion"].strip()) for row in rows)
    technical_gate = (
        architecture_gate
        and endpoint_gate
        and len(rows) == expected
        and len({row["generation_id"] for row in rows}) == expected
        and nonempty_count == expected
        and total_parsed_tracks >= int(bundle["minimum_parsed_tracks"])
    )
    summary = {
        "run_id": run_id,
        "protocol_id": bundle["protocol_id"],
        "protocol_hashes": protocol_hashes,
        "submitted_script_sha256": script_sha256,
        "mode": bundle["mode"],
        "prompt_template_id": bundle["prompt_template_id"],
        "model_role": bundle["model_role"],
        "model_id": bundle["model_id"],
        "model_revision": bundle["model_revision"],
        "expected_num_layers": bundle["expected_num_layers"],
        "observed_num_layers": len(model.model.layers),
        "generation_count": len(rows),
        "expected_generation_count": expected,
        "nonempty_completion_count": nonempty_count,
        "parsed_track_count": total_parsed_tracks,
        "minimum_parsed_tracks": bundle["minimum_parsed_tracks"],
        "endpoint_max_logit_error": endpoint_error,
        "endpoint_tolerance": bundle["endpoint_tolerance"],
        "architecture_gate": architecture_gate,
        "endpoint_gate": endpoint_gate,
        "technical_gate": technical_gate,
        "maximum_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "completion_lengths": [len(row["completion"]) for row in rows],
        "parsed_track_counts": [row["parsed_track_count"] for row in rows],
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    artifact = {"rows": rows, "summary": summary}
    artifact_url = upload_artifact(bundle, run_id, artifact)
    if artifact_url:
        summary["artifact_url"] = artifact_url

    if not bool(bundle["redact_completions"]):
        chunks = encode_artifact_chunks(artifact)
        for index, data in enumerate(chunks):
            print(
                "PHASE2_GENERATION_ARTIFACT_CHUNK_JSON="
                + json.dumps(
                    {"index": index, "total": len(chunks), "data": data},
                    separators=(",", ":"),
                )
            )
    print(
        "PHASE2_GENERATION_SUMMARY_JSON="
        + json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    )
    return 0 if technical_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
