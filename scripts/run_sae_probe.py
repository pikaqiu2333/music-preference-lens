"""Run a minimal SAE feature-delta probe for the mechanistic pilot.

This script has three modes:

- plan: print the model/resource plan without importing ML packages.
- dry-run: write placeholder rows so downstream analysis can be tested.
- run: load a Hugging Face causal LM plus an SAE and compute top feature deltas.

The actual run path is intentionally conservative. It uses hidden states from
the selected transformer layer, encodes them through the SAE, aggregates mean
feature activation over prompt tokens, and compares original vs counterfactual
prompts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def load_resource_config(path: Path, model_key: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        return data["resources"][model_key]
    except KeyError as exc:
        valid = ", ".join(sorted(data["resources"]))
        raise ValueError(f"unknown model key {model_key!r}; valid keys: {valid}") from exc


def infer_layer_index(cfg: dict[str, Any], explicit_layer: int | None) -> int:
    if explicit_layer is not None:
        return explicit_layer
    sae_id = cfg.get("sae_id") or ""
    match = re.search(r"layer_(\d+)", sae_id)
    if match:
        return int(match.group(1))
    return 12


def observation_direction(delta: float, eps: float = 1e-6) -> str:
    if delta > eps:
        return "increase"
    if delta < -eps:
        return "decrease"
    return "flat"


def spec_original_text(spec: dict[str, Any]) -> str:
    if "original_text" in spec:
        return spec["original_text"]
    if "routine_text" in spec:
        return spec["routine_text"]
    raise KeyError(f"{spec.get('probe_id', '<missing>')}: no original/routine text")


def spec_counterfactual_text(spec: dict[str, Any]) -> str:
    if "counterfactual_text" in spec:
        return spec["counterfactual_text"]
    if "complex_text" in spec:
        return spec["complex_text"]
    raise KeyError(
        f"{spec.get('probe_id', '<missing>')}: no counterfactual/complex text"
    )


def spec_target_dimensions(spec: dict[str, Any]) -> list[str]:
    if "target_dimensions" in spec:
        return list(spec["target_dimensions"])
    signals = set(spec.get("shared_signals", [])) | set(spec.get("added_complex_signals", []))
    if not signals:
        signals = set(spec.get("routine_signals", [])) | set(spec.get("complex_signals", []))
    return sorted(signals)


def spec_added_dimensions(spec: dict[str, Any]) -> list[str]:
    if "added_dimensions" in spec:
        return list(spec["added_dimensions"])
    return list(spec.get("added_complex_signals", []))


def spec_removed_dimensions(spec: dict[str, Any]) -> list[str]:
    if "removed_dimensions" in spec:
        return list(spec["removed_dimensions"])
    return []


def spec_expected_effect(spec: dict[str, Any]) -> str:
    return spec.get("expected_effect") or spec.get("hypothesis") or ""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plan_text(cfg: dict[str, Any], layer_index: int, specs: list[dict[str, Any]]) -> str:
    lines = [
        "# SAE Probe Run Plan",
        "",
        f"- base_model: `{cfg['base_model']}`",
        f"- sae_repo: `{cfg['sae_repo']}`",
        f"- sae_release: `{cfg.get('sae_release')}`",
        f"- sae_id: `{cfg.get('sae_id')}`",
        f"- layer_index: `{layer_index}`",
        f"- probes: `{len(specs)}`",
        "",
        "## Probes",
        "",
    ]
    for spec in specs:
        lines.append(f"### `{spec['probe_id']}`")
        lines.append(f"- added: {', '.join(spec_added_dimensions(spec)) or 'none'}")
        lines.append(f"- removed: {', '.join(spec_removed_dimensions(spec)) or 'none'}")
        lines.append(f"- expected: {spec_expected_effect(spec)}")
        lines.append("")
    return "\n".join(lines)


def dry_run_rows(
    specs: list[dict[str, Any]],
    cfg: dict[str, Any],
    model_key: str,
    layer_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fake_feature = 1000
    for spec in specs:
        added = spec_added_dimensions(spec)
        removed = spec_removed_dimensions(spec)
        targets = spec_target_dimensions(spec)
        for dimension in targets:
            fake_feature += 1
            expected = "inspect"
            if dimension in added:
                expected = "increase"
            elif dimension in removed:
                expected = "decrease"
            rows.append(
                {
                    "probe_id": spec["probe_id"],
                    "model_key": model_key,
                    "base_model": cfg["base_model"],
                    "sae_repo": cfg["sae_repo"],
                    "layer_index": layer_index,
                    "feature_id": fake_feature,
                    "feature_rank": "",
                    "candidate_dimension": dimension,
                    "expected_direction": expected,
                    "original_activation": "",
                    "counterfactual_activation": "",
                    "delta": "",
                    "observed_direction": "",
                    "target_dimensions": ", ".join(targets),
                    "added_dimensions": ", ".join(added),
                    "removed_dimensions": ", ".join(removed),
                    "note": "dry-run placeholder; replace with real SAE observations",
                }
            )
    return rows


def load_sae(cfg: dict[str, Any], device: str):
    from sae_lens import SAE

    release = cfg.get("sae_release")
    sae_id = cfg.get("sae_id")
    if not release or not sae_id:
        raise ValueError(
            "This runner currently expects SAELens release/sae_id. "
            "Use Gemma Scope first, or add a Qwen-Scope loader."
        )
    loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    if isinstance(loaded, tuple):
        return loaded[0]
    return loaded


def encode_prompt_features(
    *,
    text: str,
    tokenizer: Any,
    model: Any,
    sae: Any,
    layer_index: int,
    device: str,
    max_tokens: int,
):
    import torch

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    # hidden_states[0] is embeddings; layer N post block is usually N+1.
    hidden = hidden_states[layer_index + 1][0].float()
    sae_device = next(sae.parameters()).device
    hidden = hidden.to(sae_device)
    with torch.no_grad():
        feature_acts = sae.encode(hidden)
    if feature_acts.ndim == 3:
        feature_acts = feature_acts[0]
    return feature_acts.float().mean(dim=0).cpu()


def real_run_rows(
    *,
    specs: list[dict[str, Any]],
    cfg: dict[str, Any],
    model_key: str,
    layer_index: int,
    top_k: int,
    device: str,
    max_tokens: int,
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    sae = load_sae(cfg, device=device)
    sae.eval()

    rows: list[dict[str, Any]] = []
    for spec in specs:
        original = encode_prompt_features(
            text=spec_original_text(spec),
            tokenizer=tokenizer,
            model=model,
            sae=sae,
            layer_index=layer_index,
            device=device,
            max_tokens=max_tokens,
        )
        counterfactual = encode_prompt_features(
            text=spec_counterfactual_text(spec),
            tokenizer=tokenizer,
            model=model,
            sae=sae,
            layer_index=layer_index,
            device=device,
            max_tokens=max_tokens,
        )
        delta = counterfactual - original
        top_values, top_indices = delta.abs().topk(min(top_k, delta.numel()))
        for rank, (abs_value, feature_idx) in enumerate(zip(top_values, top_indices), 1):
            idx = int(feature_idx.item())
            original_value = float(original[idx].item())
            counterfactual_value = float(counterfactual[idx].item())
            delta_value = float(delta[idx].item())
            rows.append(
                {
                    "probe_id": spec["probe_id"],
                    "model_key": model_key,
                    "base_model": cfg["base_model"],
                    "sae_repo": cfg["sae_repo"],
                    "layer_index": layer_index,
                    "feature_id": idx,
                    "feature_rank": rank,
                    "candidate_dimension": "",
                    "expected_direction": "",
                    "original_activation": original_value,
                    "counterfactual_activation": counterfactual_value,
                    "delta": delta_value,
                    "observed_direction": observation_direction(delta_value),
                    "target_dimensions": ", ".join(spec_target_dimensions(spec)),
                    "added_dimensions": ", ".join(spec_added_dimensions(spec)),
                    "removed_dimensions": ", ".join(spec_removed_dimensions(spec)),
                    "note": f"top abs delta={float(abs_value.item()):.6g}",
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["plan", "dry-run", "run"],
        default="plan",
    )
    parser.add_argument(
        "--specs",
        type=Path,
        default=PROJECT_ROOT / "data" / "mechanistic_pilot_specs.jsonl",
    )
    parser.add_argument(
        "--resources",
        type=Path,
        default=PROJECT_ROOT / "config" / "model_resources.json",
    )
    parser.add_argument("--model-key", default="gemma_scope_2_270m_it")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "sae_probe_observations.csv",
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "sae_probe_run_plan.md",
    )
    args = parser.parse_args()

    specs = load_jsonl(args.specs)
    cfg = load_resource_config(args.resources, args.model_key)
    layer_index = infer_layer_index(cfg, args.layer)

    if args.mode == "plan":
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(plan_text(cfg, layer_index, specs), encoding="utf-8")
        print(f"Wrote {args.plan_output}")
        return 0

    if args.mode == "dry-run":
        rows = dry_run_rows(specs, cfg, args.model_key, layer_index)
        write_csv(args.output, rows)
        print(f"Wrote dry-run rows to {args.output}")
        return 0

    import torch

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    rows = real_run_rows(
        specs=specs,
        cfg=cfg,
        model_key=args.model_key,
        layer_index=layer_index,
        top_k=args.top_k,
        device=device,
        max_tokens=args.max_tokens,
    )
    write_csv(args.output, rows)
    print(f"Wrote real SAE probe rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
