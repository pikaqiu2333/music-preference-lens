"""Check local readiness for the mechanistic pilot without downloading models."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resources",
        type=Path,
        default=PROJECT_ROOT / "config" / "model_resources.json",
    )
    args = parser.parse_args()

    resources = json.loads(args.resources.read_text(encoding="utf-8"))
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("")
    print("Python packages:")
    for module in ["torch", "transformers", "sae_lens", "transformer_lens", "huggingface_hub", "pandas"]:
        print(f"- {module}: {'ok' if has_module(module) else 'missing'}")
    print("")
    print("Configured model resources:")
    for key in resources["recommended_sequence"]:
        cfg = resources["resources"][key]
        print(f"- {key}")
        print(f"  base_model: {cfg['base_model']}")
        print(f"  sae_repo: {cfg['sae_repo']}")
        print(f"  sae_release: {cfg.get('sae_release')}")
        print(f"  sae_id: {cfg.get('sae_id')}")

    print("")
    print("Note: this check intentionally does not download model weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

