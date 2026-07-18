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
"""Validate calibrated exact-pair verification on the 40 catalog controls."""

from __future__ import annotations

import argparse
import base64
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


EMBEDDED_BUNDLE_B64 = "__EXPERIMENT_BUNDLE_B64__"


def load_bundle(path: Path | None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    if EMBEDDED_BUNDLE_B64 == "__EXPERIMENT_BUNDLE_B64__":
        raise ValueError("no experiment bundle supplied")
    return json.loads(base64.b64decode(EMBEDDED_BUNDLE_B64).decode("utf-8"))


def render_prompt(
    template: str,
    *,
    title: str,
    artist: str,
    source_condition: str,
    options: list[tuple[str, str]],
) -> tuple[str, dict[str, str]]:
    source_context = ""
    if source_condition == "self_attributed":
        source_context = (
            "Earlier you generated this exact title-artist pair as a music "
            "recommendation."
        )
    mapping: dict[str, str] = {}
    option_lines = []
    for index, (kind, text) in enumerate(options):
        letter = chr(ord("A") + index)
        mapping[letter] = kind
        option_lines.append(f"{letter}. {text}")
    return (
        template.replace("{{source_context}}", source_context)
        .replace("{{title}}", title)
        .replace("{{artist}}", artist)
        .replace("{{options}}", "\n".join(option_lines)),
        mapping,
    )


def balanced_accuracy(labels: list[int], predictions: list[int]) -> float:
    recalls = []
    for label in (0, 1):
        positions = [index for index, value in enumerate(labels) if value == label]
        recalls.append(sum(predictions[index] == label for index in positions) / len(positions))
    return sum(recalls) / 2


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
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        bundle["model_id"],
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda")
    model.eval()

    letter_ids = {}
    for letter in ("A", "B"):
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{letter} is not a single token: {ids}")
        letter_ids[letter] = ids[0]

    option_text = bundle["verification_options"]
    orders = [
        [("known_exact", option_text["known_exact"]), ("unknown", option_text["unknown"])],
        [("unknown", option_text["unknown"]), ("known_exact", option_text["known_exact"])],
    ]
    specs: list[dict[str, Any]] = []
    null_specs: list[dict[str, Any]] = []
    for source_condition in ("neutral", "self_attributed"):
        for order_index, options in enumerate(orders):
            prompt, mapping = render_prompt(
                bundle["verification_template"],
                title="Sample Catalog Title",
                artist="Sample Catalog Artist",
                source_condition=source_condition,
                options=options,
            )
            null_specs.append(
                {
                    "source_condition": source_condition,
                    "order_index": order_index,
                    "prompt": prompt,
                    "mapping": mapping,
                }
            )
            for control in bundle["controls"]:
                prompt, mapping = render_prompt(
                    bundle["verification_template"],
                    title=control["title"],
                    artist=control["artist"],
                    source_condition=source_condition,
                    options=options,
                )
                specs.append(
                    {
                        "pair_id": control["pair_id"],
                        "group": control["group"],
                        "source_condition": source_condition,
                        "order_index": order_index,
                        "prompt": prompt,
                        "mapping": mapping,
                    }
                )

    def score_batches(rows: list[dict[str, Any]], batch_size: int = 8) -> None:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = tokenizer(
                [row["prompt"] for row in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to("cuda")
            with torch.no_grad():
                logits = model(**encoded).logits
            lengths = encoded["attention_mask"].sum(dim=1)
            for index, row in enumerate(batch):
                final = logits[index, int(lengths[index].item()) - 1]
                row["letter_logits"] = {
                    letter: float(final[token_id].item())
                    for letter, token_id in letter_ids.items()
                }

    score_batches(null_specs, batch_size=4)
    score_batches(specs)
    priors = {}
    for row in null_specs:
        unknown_letter = next(k for k, v in row["mapping"].items() if v == "unknown")
        known_letter = next(k for k, v in row["mapping"].items() if v == "known_exact")
        priors[(row["source_condition"], row["order_index"])] = (
            row["letter_logits"][unknown_letter] - row["letter_logits"][known_letter]
        )

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    group_by_pair = {control["pair_id"]: control["group"] for control in bundle["controls"]}
    for row in specs:
        unknown_letter = next(k for k, v in row["mapping"].items() if v == "unknown")
        known_letter = next(k for k, v in row["mapping"].items() if v == "known_exact")
        raw = row["letter_logits"][unknown_letter] - row["letter_logits"][known_letter]
        grouped[(row["pair_id"], row["source_condition"])].append(
            raw - priors[(row["source_condition"], row["order_index"])]
        )

    rows = []
    for pair_id, group in group_by_pair.items():
        neutral = sum(grouped[(pair_id, "neutral")]) / 2
        self_score = sum(grouped[(pair_id, "self_attributed")]) / 2
        rows.append(
            {
                "pair_id": pair_id,
                "group": group,
                "neutral_unknown_logit": neutral,
                "self_attributed_unknown_logit": self_score,
            }
        )

    summary: dict[str, Any] = {"letter_token_ids": letter_ids, "groups": {}}
    for group in ("known_exact", "artist_mismatch", "synthetic_pair"):
        subset = [row for row in rows if row["group"] == group]
        summary["groups"][group] = {
            "n": len(subset),
            "neutral_mean": sum(row["neutral_unknown_logit"] for row in subset) / len(subset),
            "neutral_unknown_count": sum(row["neutral_unknown_logit"] > 0 for row in subset),
            "self_mean": sum(row["self_attributed_unknown_logit"] for row in subset) / len(subset),
            "self_unknown_count": sum(row["self_attributed_unknown_logit"] > 0 for row in subset),
        }
    labels = [int(row["group"] != "known_exact") for row in rows]
    summary["neutral_balanced_accuracy"] = balanced_accuracy(
        labels,
        [int(row["neutral_unknown_logit"] > 0) for row in rows],
    )
    summary["self_balanced_accuracy"] = balanced_accuracy(
        labels,
        [int(row["self_attributed_unknown_logit"] > 0) for row in rows],
    )
    summary["mean_self_minus_neutral"] = sum(
        row["self_attributed_unknown_logit"] - row["neutral_unknown_logit"]
        for row in rows
    ) / len(rows)
    print("PAIR_VERIFY_CONTROL_ROWS_JSON=" + json.dumps(rows, ensure_ascii=True, separators=(",", ":")))
    print("PAIR_VERIFY_CONTROL_SUMMARY_JSON=" + json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
