"""Build paired routine-vs-complex probe specs for workspace-pressure tests."""

from __future__ import annotations

import argparse
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


def build_specs(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {task["task_id"]: task for task in tasks}
    specs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for task in tasks:
        paired_id = task.get("paired_with")
        if not paired_id:
            continue
        if paired_id not in by_id:
            raise ValueError(f"{task['task_id']} paired_with missing task {paired_id}")
        other = by_id[paired_id]
        low = task if task["complexity_level"] <= other["complexity_level"] else other
        high = other if low is task else task
        key = (low["task_id"], high["task_id"])
        if key in seen:
            continue
        seen.add(key)
        specs.append(
            {
                "probe_id": f"workspace_pressure__{low['task_id']}__vs__{high['task_id']}",
                "task_family": low["task_family"],
                "low_task_id": low["task_id"],
                "high_task_id": high["task_id"],
                "low_complexity_level": low["complexity_level"],
                "high_complexity_level": high["complexity_level"],
                "low_workspace_pressure": low["workspace_pressure"],
                "high_workspace_pressure": high["workspace_pressure"],
                "routine_text": low["input_text"],
                "complex_text": high["input_text"],
                "shared_signals": sorted(
                    set(low["target_signals"]) & set(high["target_signals"])
                ),
                "added_complex_signals": sorted(
                    set(high["target_signals"]) - set(low["target_signals"])
                ),
                "routine_signals": low["target_signals"],
                "complex_signals": high["target_signals"],
                "hypothesis": high["hypothesis"],
                "success_checks": high["success_checks"],
            }
        )
    return specs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=Path,
        default=PROJECT_ROOT / "data" / "j_space_complexity_tasks.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "j_space_probe_specs.jsonl",
    )
    args = parser.parse_args()

    specs = build_specs(load_jsonl(args.tasks))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(json.dumps(spec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(specs)} workspace-pressure probe specs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

