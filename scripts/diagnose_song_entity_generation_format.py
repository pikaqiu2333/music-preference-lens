# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.8.0",
#   "huggingface-hub>=0.33.0,<1.0",
#   "safetensors>=0.5.0",
#   "torch==2.7.1",
#   "transformers==4.53.2",
# ]
# ///
"""Generate all planned playlists and print raw outputs for parser diagnosis."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


EMBEDDED_BUNDLE_B64 = "__EXPERIMENT_BUNDLE_B64__"


def load_bundle(path: Path | None) -> dict:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    if EMBEDDED_BUNDLE_B64 == "__EXPERIMENT_BUNDLE_B64__":
        raise ValueError("no experiment bundle supplied")
    return json.loads(base64.b64decode(EMBEDDED_BUNDLE_B64).decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    bundle = load_bundle(args.bundle)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    tokenizer = AutoTokenizer.from_pretrained(bundle["model_id"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        bundle["model_id"],
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda")
    model.eval()

    for context in bundle["contexts"]:
        prompt = context["generation_prompt"].rstrip()
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        for seed in bundle["seeds"]:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
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
            print(
                "GEN_FORMAT_JSON="
                + json.dumps(
                    {
                        "generation_id": f"{context['context_id']}__seed{seed}",
                        "completion": completion,
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
