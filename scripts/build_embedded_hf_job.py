"""Embed a local experiment bundle into a standalone Hugging Face Job script."""

from __future__ import annotations

import argparse
import base64
import zlib
from pathlib import Path


def embed_bundle(
    runner: str,
    payload: bytes,
    placeholder: str,
    compression: str,
) -> str:
    if runner.count(placeholder) < 2:
        raise ValueError("runner must contain assignment and sentinel placeholders")
    if compression == "zlib":
        payload = zlib.compress(payload, level=9)
    elif compression != "none":
        raise ValueError(f"unknown compression: {compression}")
    encoded = base64.b64encode(payload).decode("ascii")
    output = runner.replace(placeholder, encoded, 1)
    if output.count(placeholder) != runner.count(placeholder) - 1:
        raise ValueError("placeholder injection changed an unexpected occurrence count")
    return output


def extract_embedded_bundle(
    runner: str,
    embedded_runner: str,
    placeholder: str,
    compression: str,
) -> bytes:
    """Recover a bundle while verifying the surrounding runner is unchanged."""
    if runner.count(placeholder) < 2:
        raise ValueError("runner must contain assignment and sentinel placeholders")
    prefix, suffix = runner.split(placeholder, 1)
    if not embedded_runner.startswith(prefix) or not embedded_runner.endswith(suffix):
        raise ValueError("embedded runner differs outside the payload placeholder")
    payload_end = len(embedded_runner) - len(suffix) if suffix else None
    encoded = embedded_runner[len(prefix) : payload_end]
    payload = base64.b64decode(encoded, validate=True)
    if compression == "zlib":
        payload = zlib.decompress(payload)
    elif compression != "none":
        raise ValueError(f"unknown compression: {compression}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--placeholder", required=True)
    parser.add_argument("--compression", choices=("none", "zlib"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = embed_bundle(
        args.runner.read_text(encoding="utf-8"),
        args.bundle.read_bytes().strip(),
        args.placeholder,
        args.compression,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
