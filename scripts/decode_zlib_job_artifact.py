"""Decode a zlib/base64 HF Job artifact into row and summary files."""

from __future__ import annotations

import argparse
import base64
import json
import zlib
from pathlib import Path
from typing import Any


def decode_artifact(encoded: str) -> dict[str, Any]:
    payload = zlib.decompress(base64.b64decode(encoded.strip()))
    artifact = json.loads(payload.decode("utf-8"))
    if not isinstance(artifact.get("rows"), list) or not isinstance(
        artifact.get("summary"), dict
    ):
        raise ValueError("artifact must contain rows and summary")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoded", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    artifact = decode_artifact(args.encoded.read_text(encoding="ascii"))
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    args.rows.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in artifact["rows"]
        ),
        encoding="utf-8",
    )
    args.summary.write_text(
        json.dumps(artifact["summary"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Decoded {len(artifact['rows'])} rows")
    print(f"- {args.rows}")
    print(f"- {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
