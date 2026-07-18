"""Export J-space complexity tasks into prompt packs and summary tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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


def render_prompt(template: str, task: dict[str, Any]) -> str:
    return template.replace("{{input_text}}", task["input_text"])


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, tasks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "complexity_level",
        "task_family",
        "workspace_pressure",
        "paired_with",
        "target_signals",
        "hypothesis",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "task_id": task["task_id"],
                    "complexity_level": task["complexity_level"],
                    "task_family": task["task_family"],
                    "workspace_pressure": task["workspace_pressure"],
                    "paired_with": task["paired_with"],
                    "target_signals": ", ".join(task["target_signals"]),
                    "hypothesis": task["hypothesis"],
                }
            )


def write_markdown(path: Path, tasks: list[dict[str, Any]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[task["task_family"]].append(task)

    lines = [
        f"# {title}",
        "",
        "| Family | Task | Level | Pressure | Pair | Target Signals |",
        "|---|---|---:|---|---|---|",
    ]
    for family, family_tasks in sorted(grouped.items()):
        for task in sorted(family_tasks, key=lambda item: item["complexity_level"]):
            lines.append(
                "| "
                + " | ".join(
                    [
                        family,
                        f"`{task['task_id']}`",
                        str(task["complexity_level"]),
                        task["workspace_pressure"],
                        f"`{task['paired_with']}`" if task["paired_with"] else "",
                        ", ".join(f"`{signal}`" for signal in task["target_signals"]),
                    ]
                )
                + " |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_pairs(tasks: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    by_id = {task["task_id"]: task for task in tasks}
    for task in tasks:
        paired = task.get("paired_with")
        if paired and paired not in by_id:
            errors.append(f"{task['task_id']}: paired_with target missing: {paired}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=Path,
        default=PROJECT_ROOT / "data" / "j_space_complexity_tasks.jsonl",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=PROJECT_ROOT / "prompts" / "j_space_task_prompt.md",
    )
    parser.add_argument(
        "--prompt-pack",
        type=Path,
        default=PROJECT_ROOT / "runs" / "j_space_complexity_prompt_pack.jsonl",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "runs" / "j_space_complexity_tasks.csv",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=PROJECT_ROOT / "reports" / "j_space_complexity_task_summary.md",
    )
    parser.add_argument("--title", default="J-Space Complexity Task Summary")
    args = parser.parse_args()

    tasks = load_jsonl(args.tasks)
    errors = validate_pairs(tasks)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    template = args.template.read_text(encoding="utf-8")
    prompt_rows = []
    for task in tasks:
        prompt_rows.append(
            {
                "prompt_id": task["task_id"],
                "task_id": task["task_id"],
                "complexity_level": task["complexity_level"],
                "task_family": task["task_family"],
                "workspace_pressure": task["workspace_pressure"],
                "paired_with": task["paired_with"],
                "target_signals": task["target_signals"],
                "success_checks": task["success_checks"],
                "prompt": render_prompt(template, task),
            }
        )

    write_jsonl(args.prompt_pack, prompt_rows)
    write_csv(args.csv, tasks)
    write_markdown(args.markdown, tasks, args.title)
    print(f"Wrote {len(prompt_rows)} J-space complexity prompts")
    print(f"- {args.prompt_pack}")
    print(f"- {args.csv}")
    print(f"- {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
