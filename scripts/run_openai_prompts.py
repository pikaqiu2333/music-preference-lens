"""Run prompt packs with the OpenAI Responses API.

Environment:

- OPENAI_API_KEY: required unless --dry-run is used.
- MUSIC_PREF_MODEL: optional model override, defaults to gpt-5.5.
- OPENAI_BASE_URL: optional, defaults to https://api.openai.com/v1.

The script writes one JSONL row per prompt. Each row keeps the prompt metadata,
the parsed model output when possible, the raw response text, and any error.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_output_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]

    chunks: list[str] = []
    for item in response.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "\n".join(chunks).strip()


def parse_model_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, str(exc)

    if not isinstance(parsed, dict):
        return None, "model output is not a JSON object"
    return parsed, None


def call_responses_api(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    schema: dict[str, Any],
    reasoning_effort: str,
    timeout: int,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/responses"
    body = {
        "model": model,
        "input": prompt,
        "reasoning": {"effort": reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "music_rerank_output",
                "schema": schema,
                "strict": True,
            }
        },
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc}") from exc


def build_result_row(
    prompt_row: dict[str, Any],
    *,
    model: str,
    raw_text: str | None = None,
    response_json: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    parsed: dict[str, Any] | None = None
    parse_error: str | None = None
    if raw_text:
        parsed, parse_error = parse_model_json(raw_text)

    return {
        "prompt_id": prompt_row["prompt_id"],
        "case_id": prompt_row["case_id"],
        "source_case_id": prompt_row["source_case_id"],
        "variant_type": prompt_row["variant_type"],
        "expected_effect": prompt_row["expected_effect"],
        "candidate_track_ids": prompt_row["candidate_track_ids"],
        "model": model,
        "model_output": parsed,
        "raw_text": raw_text,
        "parse_error": parse_error,
        "error": error,
        "response_id": response_json.get("id") if response_json else None,
        "usage": response_json.get("usage") if response_json else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-pack",
        type=Path,
        default=PROJECT_ROOT / "runs" / "prompt_pack.jsonl",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=PROJECT_ROOT / "schemas" / "model_output.schema.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "openai_results.jsonl",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MUSIC_PREF_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get("MUSIC_PREF_REASONING_EFFORT", "low"),
        choices=["none", "low", "medium", "high", "xhigh"],
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prompt_rows = load_jsonl(args.prompt_pack)
    if args.limit:
        prompt_rows = prompt_rows[: args.limit]
    schema = json.loads(args.schema.read_text(encoding="utf-8"))

    if args.dry_run:
        rows = [
            build_result_row(row, model=args.model, error="dry_run")
            for row in prompt_rows
        ]
        write_jsonl(args.output, rows)
        print(f"Dry run wrote {len(rows)} rows to {args.output}")
        return 0

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is required unless --dry-run is used.")
        return 2

    results: list[dict[str, Any]] = []
    for index, prompt_row in enumerate(prompt_rows, 1):
        print(f"[{index}/{len(prompt_rows)}] {prompt_row['prompt_id']}")
        try:
            response_json = call_responses_api(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                prompt=prompt_row["prompt"],
                schema=schema,
                reasoning_effort=args.reasoning_effort,
                timeout=args.timeout,
            )
            raw_text = extract_output_text(response_json)
            result = build_result_row(
                prompt_row,
                model=args.model,
                raw_text=raw_text,
                response_json=response_json,
            )
        except Exception as exc:  # noqa: BLE001 - keep run output inspectable.
            result = build_result_row(prompt_row, model=args.model, error=str(exc))
        results.append(result)
        write_jsonl(args.output, results)
        if args.sleep:
            time.sleep(args.sleep)

    print(f"Wrote {len(results)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

