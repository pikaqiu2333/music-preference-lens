# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "accelerate==1.8.1",
#   "torch==2.7.1",
#   "transformers==4.53.2",
# ]
# ///
"""Generate the frozen independent music-recommendation holdout."""

from __future__ import annotations

import argparse
import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EMBEDDED_BUNDLE_B64 = "__EXPERIMENT_BUNDLE_B64__"


def load_bundle(path: Path | None = None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    if EMBEDDED_BUNDLE_B64 == "__EXPERIMENT_BUNDLE_B64__":
        raise ValueError("pass --bundle or embed the experiment bundle")
    return json.loads(base64.b64decode(EMBEDDED_BUNDLE_B64).decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--model-revision")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + "_holdout"
    bundle = load_bundle(args.bundle)
    model_revision = (
        args.model_revision
        or os.environ.get("MODEL_REVISION")
        or bundle.get("model_revision")
    )
    model_source = {"revision": model_revision} if model_revision else {}
    if not torch.cuda.is_available():
        raise RuntimeError("this generator requires a CUDA GPU")
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(
        bundle["model_id"], trust_remote_code=True, **model_source
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        bundle["model_id"],
        torch_dtype=torch.float16,
        trust_remote_code=True,
        **model_source,
    ).to(device)
    model.eval()

    rows = []
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
            row = {
                "generation_id": f"{context['context_id']}__seed{seed}",
                "context_id": context["context_id"],
                "seed": int(seed),
                "completion": completion,
            }
            rows.append(row)
            print(
                "HOLDOUT_RAW_GENERATION_JSON="
                + json.dumps(row, ensure_ascii=True, separators=(",", ":"))
            )

    expected = int(bundle["expected_generation_count"])
    summary = {
        "run_id": run_id,
        "model_id": bundle["model_id"],
        "model_revision": model_revision,
        "generation_count": len(rows),
        "expected_generation_count": expected,
        "unique_generation_count": len({row["generation_id"] for row in rows}),
        "technical_gate": (
            len(rows) == expected
            and len({row["generation_id"] for row in rows}) == expected
            and all(row["completion"].strip() for row in rows)
        ),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    print(
        "HOLDOUT_GENERATION_SUMMARY_JSON="
        + json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
