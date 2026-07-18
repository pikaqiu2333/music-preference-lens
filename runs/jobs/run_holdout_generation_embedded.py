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


EMBEDDED_BUNDLE_B64 = "eyJidW5kbGVfdmVyc2lvbiI6ImluZGVwZW5kZW50X2hvbGRvdXRfZ2VuZXJhdGlvbl92MSIsIm1vZGVsX2lkIjoiUXdlbi9Rd2VuMy0xLjdCLUJhc2UiLCJjb250ZXh0cyI6W3siY29udGV4dF9pZCI6InF1aWV0X25pZ2h0X2RyaXZlIiwiZGltZW5zaW9uIjoicXVpZXRfbmlnaHRfZHJpdmUiLCJwcm9maWxlIjoidXNlciBsaWtlcyBlbGVjdHJvbmljLCBhbWJpZW50LCBDaXR5IFBvcCwgYW5kIHJoeXRobWljIGluZGllIHBvcDsgZGlzbGlrZXMgbm9pc3kgcm9jayBhbmQgaGlnaC1wcmVzc3VyZSBFRE0gaW4gZGFpbHkgbGlzdGVuaW5nLiIsImN1cnJlbnRfbmVlZCI6InJhaW55IG5pZ2h0IGRyaXZpbmc7IHdhbnRzIHF1aWV0IG5pZ2h0IGRyaXZlLCByZXN0cmFpbmVkIGJlYXQsIGNhbG0gdXJiYW4gbW9vZCwgYW5kIG5vdCB0b28gbG91ZC4iLCJnZW5lcmF0aW9uX3Byb21wdCI6Ik11c2ljIHBsYXlsaXN0IHJlcXVlc3QuXG5cblVzZXIgcHJvZmlsZTogdXNlciBsaWtlcyBlbGVjdHJvbmljLCBhbWJpZW50LCBDaXR5IFBvcCwgYW5kIHJoeXRobWljIGluZGllIHBvcDsgZGlzbGlrZXMgbm9pc3kgcm9jayBhbmQgaGlnaC1wcmVzc3VyZSBFRE0gaW4gZGFpbHkgbGlzdGVuaW5nLlxuXG5DdXJyZW50IG5lZWQ6IHJhaW55IG5pZ2h0IGRyaXZpbmc7IHdhbnRzIHF1aWV0IG5pZ2h0IGRyaXZlLCByZXN0cmFpbmVkIGJlYXQsIGNhbG0gdXJiYW4gbW9vZCwgYW5kIG5vdCB0b28gbG91ZC5cblxuUmVjb21tZW5kIDUgcmVhbCwgZXhpc3RpbmcgdHJhY2tzLiBGb3IgZWFjaCB0cmFjaywgcHJvdmlkZSBpdHMgdGl0bGUsIGFydGlzdCxcbmFuZCBvbmUgc2hvcnQgcmVhc29uIHRoYXQgZml0cyB0aGUgY3VycmVudCBuZWVkLlxuXG5QbGF5bGlzdDpcbjEuIFRpdGxlOlxuIn0seyJjb250ZXh0X2lkIjoicGVha190aW1lX3JhdmUiLCJkaW1lbnNpb24iOiJoaWdoX2VuZXJneV9yYXZlIiwicHJvZmlsZSI6InVzZXIgbGlrZXMgZWxlY3Ryb25pYywgZnVuaywgcmV0cm8gZGlzY28sIGFuZCBuaWNoZSBkaXNjb3Zlcnk7IGRpc2xpa2VzIGVudGVyaW5nIHZlcnkgaGFyZCBkYW5jZSBtdXNpYyB0b28gZWFybHkuIiwiY3VycmVudF9uZWVkIjoiMmFtIHBlYWstdGltZSBkYW5jZSBmbG9vcjsgd2FudHMgaGFyZGVyIGVuZXJneSwgcmF2ZS1saWtlIHByZXNzdXJlLCBpbnRlbnNlIGRydW1zLCBhbmQgbWF4aW11bSBtb21lbnR1bS4iLCJnZW5lcmF0aW9uX3Byb21wdCI6Ik11c2ljIHBsYXlsaXN0IHJlcXVlc3QuXG5cblVzZXIgcHJvZmlsZTogdXNlciBsaWtlcyBlbGVjdHJvbmljLCBmdW5rLCByZXRybyBkaXNjbywgYW5kIG5pY2hlIGRpc2NvdmVyeTsgZGlzbGlrZXMgZW50ZXJpbmcgdmVyeSBoYXJkIGRhbmNlIG11c2ljIHRvbyBlYXJseS5cblxuQ3VycmVudCBuZWVkOiAyYW0gcGVhay10aW1lIGRhbmNlIGZsb29yOyB3YW50cyBoYXJkZXIgZW5lcmd5LCByYXZlLWxpa2UgcHJlc3N1cmUsIGludGVuc2UgZHJ1bXMsIGFuZCBtYXhpbXVtIG1vbWVudHVtLlxuXG5SZWNvbW1lbmQgNSByZWFsLCBleGlzdGluZyB0cmFja3MuIEZvciBlYWNoIHRyYWNrLCBwcm92aWRlIGl0cyB0aXRsZSwgYXJ0aXN0LFxuYW5kIG9uZSBzaG9ydCByZWFzb24gdGhhdCBmaXRzIHRoZSBjdXJyZW50IG5lZWQuXG5cblBsYXlsaXN0OlxuMS4gVGl0bGU6XG4ifSx7ImNvbnRleHRfaWQiOiJlbW90aW9uYWxfdm9jYWwiLCJkaW1lbnNpb24iOiJlbW90aW9uYWxfbHlyaWNzIiwicHJvZmlsZSI6InVzZXIgbGlrZXMgQ2hpbmVzZSBmZW1hbGUgdm9jYWxzLCBzb2Z0IGVsZWN0cm9uaWMgbXVzaWMsIGFuZCBxdWlldCBtZWxvZGljIHNvbmdzOyBkaXNsaWtlcyBseXJpY3MgdGhhdCBncmFiIHRvbyBtdWNoIGF0dGVudGlvbiB3aGlsZSB3b3JraW5nLiIsImN1cnJlbnRfbmVlZCI6ImxhdGUtbmlnaHQgZW1vdGlvbmFsIGxpc3RlbmluZzsgd2FudHMgaGVhcnRmZWx0IHZvY2FsLCBseXJpY2FsIHN0b3J5LCBkcmFtYXRpYyBjaG9ydXMsIGFuZCBjbGVhciBlbW90aW9uYWwgcmVsZWFzZS4iLCJnZW5lcmF0aW9uX3Byb21wdCI6Ik11c2ljIHBsYXlsaXN0IHJlcXVlc3QuXG5cblVzZXIgcHJvZmlsZTogdXNlciBsaWtlcyBDaGluZXNlIGZlbWFsZSB2b2NhbHMsIHNvZnQgZWxlY3Ryb25pYyBtdXNpYywgYW5kIHF1aWV0IG1lbG9kaWMgc29uZ3M7IGRpc2xpa2VzIGx5cmljcyB0aGF0IGdyYWIgdG9vIG11Y2ggYXR0ZW50aW9uIHdoaWxlIHdvcmtpbmcuXG5cbkN1cnJlbnQgbmVlZDogbGF0ZS1uaWdodCBlbW90aW9uYWwgbGlzdGVuaW5nOyB3YW50cyBoZWFydGZlbHQgdm9jYWwsIGx5cmljYWwgc3RvcnksIGRyYW1hdGljIGNob3J1cywgYW5kIGNsZWFyIGVtb3Rpb25hbCByZWxlYXNlLlxuXG5SZWNvbW1lbmQgNSByZWFsLCBleGlzdGluZyB0cmFja3MuIEZvciBlYWNoIHRyYWNrLCBwcm92aWRlIGl0cyB0aXRsZSwgYXJ0aXN0LFxuYW5kIG9uZSBzaG9ydCByZWFzb24gdGhhdCBmaXRzIHRoZSBjdXJyZW50IG5lZWQuXG5cblBsYXlsaXN0OlxuMS4gVGl0bGU6XG4ifSx7ImNvbnRleHRfaWQiOiJzdHJpY3Rfbm9fdm9jYWxzIiwiZGltZW5zaW9uIjoibm9fdm9jYWxzX2luc3RydW1lbnRhbCIsInByb2ZpbGUiOiJ1c2VyIGxpa2VzIENoaW5lc2UgZmVtYWxlIHZvY2Fscywgc29mdCBlbGVjdHJvbmljIG11c2ljLCBhbmQgcXVpZXQgbWVsb2RpYyBzb25nczsgZGlzbGlrZXMgbHlyaWNzIHRoYXQgZ3JhYiB0b28gbXVjaCBhdHRlbnRpb24gd2hpbGUgd29ya2luZy4iLCJjdXJyZW50X25lZWQiOiJzdHJpY3Qgd3JpdGluZyBtb2RlOyB3YW50cyBubyB2b2NhbHMsIGluc3RydW1lbnRhbCB0ZXh0dXJlLCBubyBseXJpYyBhdHRlbnRpb24sIGFuZCB3aGl0ZSBub2lzZSBmb2N1cy4iLCJnZW5lcmF0aW9uX3Byb21wdCI6Ik11c2ljIHBsYXlsaXN0IHJlcXVlc3QuXG5cblVzZXIgcHJvZmlsZTogdXNlciBsaWtlcyBDaGluZXNlIGZlbWFsZSB2b2NhbHMsIHNvZnQgZWxlY3Ryb25pYyBtdXNpYywgYW5kIHF1aWV0IG1lbG9kaWMgc29uZ3M7IGRpc2xpa2VzIGx5cmljcyB0aGF0IGdyYWIgdG9vIG11Y2ggYXR0ZW50aW9uIHdoaWxlIHdvcmtpbmcuXG5cbkN1cnJlbnQgbmVlZDogc3RyaWN0IHdyaXRpbmcgbW9kZTsgd2FudHMgbm8gdm9jYWxzLCBpbnN0cnVtZW50YWwgdGV4dHVyZSwgbm8gbHlyaWMgYXR0ZW50aW9uLCBhbmQgd2hpdGUgbm9pc2UgZm9jdXMuXG5cblJlY29tbWVuZCA1IHJlYWwsIGV4aXN0aW5nIHRyYWNrcy4gRm9yIGVhY2ggdHJhY2ssIHByb3ZpZGUgaXRzIHRpdGxlLCBhcnRpc3QsXG5hbmQgb25lIHNob3J0IHJlYXNvbiB0aGF0IGZpdHMgdGhlIGN1cnJlbnQgbmVlZC5cblxuUGxheWxpc3Q6XG4xLiBUaXRsZTpcbiJ9XSwic2VlZHMiOls1OSw3MSw4MywxMDEsMTI3XSwiZ2VuZXJhdGlvbiI6eyJ0ZW1wZXJhdHVyZSI6MC43LCJ0b3BfcCI6MC45LCJtYXhfbmV3X3Rva2VucyI6Mzg0fSwiZXhwZWN0ZWRfZ2VuZXJhdGlvbl9jb3VudCI6MjAsIm1pbmltdW1fcGFyc2VkX2dlbmVyYXRpb25fY291bnQiOjE2LCJtaW5pbXVtX3BhcnNlZF9wYWlyX2NvdW50Ijo3NSwiZGlzY292ZXJ5X2dlbmVyYXRpb25fcnVuIjoiMjAyNjA3MTBUMDM1MjI0Wl9mdWxsIn0="


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
