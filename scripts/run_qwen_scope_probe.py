"""Run Qwen-Scope TopK SAE probes for routine-vs-complex specs.

This runner is separate from ``run_sae_probe.py`` because Qwen-Scope publishes
per-layer ``layerN.sae.pt`` checkpoints instead of SAELens release/sae_id names.
It supports the same plan/dry-run/run shape so the expensive path can happen on
Hugging Face Jobs or another GPU environment.
"""

from __future__ import annotations

import argparse
import csv
import json
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


def spec_routine_text(spec: dict[str, Any]) -> str:
    if "routine_text" in spec:
        return spec["routine_text"]
    if "original_text" in spec:
        return spec["original_text"]
    raise KeyError(f"{spec.get('probe_id', '<missing>')}: no routine/original text")


def spec_complex_text(spec: dict[str, Any]) -> str:
    if "complex_text" in spec:
        return spec["complex_text"]
    if "counterfactual_text" in spec:
        return spec["counterfactual_text"]
    raise KeyError(f"{spec.get('probe_id', '<missing>')}: no complex/counterfactual text")


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plan_text(cfg: dict[str, Any], layer_index: int, specs: list[dict[str, Any]]) -> str:
    lines = [
        "# Qwen-Scope Probe Run Plan",
        "",
        f"- base_model: `{cfg['base_model']}`",
        f"- sae_repo: `{cfg['sae_repo']}`",
        f"- sae_file: `layer{layer_index}.sae.pt`",
        f"- layer_index: `{layer_index}`",
        f"- probes: `{len(specs)}`",
        "",
        "## Probes",
        "",
    ]
    for spec in specs:
        lines.append(f"### `{spec['probe_id']}`")
        lines.append(f"- added: {', '.join(spec_added_dimensions(spec)) or 'none'}")
        lines.append(f"- targets: {', '.join(spec_target_dimensions(spec)) or 'none'}")
        lines.append("")
    return "\n".join(lines)


def dry_run_rows(
    *,
    specs: list[dict[str, Any]],
    cfg: dict[str, Any],
    model_key: str,
    layer_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fake_feature = 30000
    for spec in specs:
        for dimension in spec_target_dimensions(spec):
            fake_feature += 1
            rows.append(
                {
                    "probe_id": spec["probe_id"],
                    "model_key": model_key,
                    "base_model": cfg["base_model"],
                    "sae_repo": cfg["sae_repo"],
                    "sae_file": f"layer{layer_index}.sae.pt",
                    "layer_index": layer_index,
                    "feature_id": fake_feature,
                    "feature_rank": "",
                    "candidate_dimension": dimension,
                    "routine_activation": "",
                    "complex_activation": "",
                    "delta": "",
                    "abs_delta": "",
                    "target_dimensions": ", ".join(spec_target_dimensions(spec)),
                    "added_dimensions": ", ".join(spec_added_dimensions(spec)),
                    "note": "dry-run placeholder; replace with real Qwen-Scope observations",
                }
            )
    return rows


def real_run_rows(
    *,
    specs: list[dict[str, Any]],
    cfg: dict[str, Any],
    model_key: str,
    layer_index: int,
    sae_top_k: int,
    report_top_k: int,
    max_tokens: int,
    device: str,
) -> list[dict[str, Any]]:
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    sae_path = hf_hub_download(
        repo_id=cfg["sae_repo"],
        filename=f"layer{layer_index}.sae.pt",
    )
    try:
        sae = torch.load(sae_path, map_location=device, weights_only=True)
    except TypeError:
        sae = torch.load(sae_path, map_location=device)

    w_enc = sae["W_enc"].to(device=device, dtype=torch.float32)
    b_enc = sae["b_enc"].to(device=device, dtype=torch.float32)
    del sae

    def encode(text: str):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        captured: dict[str, Any] = {}

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = hidden.detach()

        handle = model.model.layers[layer_index].register_forward_hook(hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        hidden = captured["hidden"][0].to(device=device, dtype=torch.float32)
        pre_acts = hidden @ w_enc.T + b_enc
        values, indices = torch.relu(pre_acts).topk(sae_top_k, dim=-1)
        mean_acts = torch.zeros(w_enc.shape[0], device=device, dtype=torch.float32)
        mean_acts.scatter_add_(0, indices.reshape(-1), values.reshape(-1))
        mean_acts /= max(1, hidden.shape[0])
        return mean_acts.cpu()

    rows: list[dict[str, Any]] = []
    for spec in specs:
        routine = encode(spec_routine_text(spec))
        complex_ = encode(spec_complex_text(spec))
        delta = complex_ - routine
        top_values, top_indices = delta.abs().topk(min(report_top_k, delta.numel()))
        for rank, (abs_value, feature_idx) in enumerate(zip(top_values, top_indices), 1):
            idx = int(feature_idx.item())
            rows.append(
                {
                    "probe_id": spec["probe_id"],
                    "model_key": model_key,
                    "base_model": cfg["base_model"],
                    "sae_repo": cfg["sae_repo"],
                    "sae_file": f"layer{layer_index}.sae.pt",
                    "layer_index": layer_index,
                    "feature_id": idx,
                    "feature_rank": rank,
                    "candidate_dimension": "",
                    "routine_activation": float(routine[idx].item()),
                    "complex_activation": float(complex_[idx].item()),
                    "delta": float(delta[idx].item()),
                    "abs_delta": float(abs_value.item()),
                    "target_dimensions": ", ".join(spec_target_dimensions(spec)),
                    "added_dimensions": ", ".join(spec_added_dimensions(spec)),
                    "note": "top absolute Qwen-Scope activation delta",
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["plan", "dry-run", "run"], default="plan")
    parser.add_argument(
        "--specs",
        type=Path,
        default=PROJECT_ROOT / "runs" / "identity_probe_specs.jsonl",
    )
    parser.add_argument(
        "--resources",
        type=Path,
        default=PROJECT_ROOT / "config" / "model_resources.json",
    )
    parser.add_argument("--model-key", default="qwen_scope_qwen3_1_7b_base")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--sae-top-k", type=int, default=50)
    parser.add_argument("--report-top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "qwen_scope_probe_observations.csv",
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "qwen_scope_probe_run_plan.md",
    )
    args = parser.parse_args()

    specs = load_jsonl(args.specs)
    cfg = load_resource_config(args.resources, args.model_key)
    if not cfg.get("sae_repo"):
        raise ValueError(f"{args.model_key} does not define sae_repo")

    if args.mode == "plan":
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(plan_text(cfg, args.layer, specs), encoding="utf-8")
        print(f"Wrote {args.plan_output}")
        return 0

    if args.mode == "dry-run":
        rows = dry_run_rows(
            specs=specs,
            cfg=cfg,
            model_key=args.model_key,
            layer_index=args.layer,
        )
        write_csv(args.output, rows)
        print(f"Wrote dry-run rows to {args.output}")
        return 0

    import torch

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    rows = real_run_rows(
        specs=specs,
        cfg=cfg,
        model_key=args.model_key,
        layer_index=args.layer,
        sae_top_k=args.sae_top_k,
        report_top_k=args.report_top_k,
        max_tokens=args.max_tokens,
        device=device,
    )
    write_csv(args.output, rows)
    print(f"Wrote real Qwen-Scope probe rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
