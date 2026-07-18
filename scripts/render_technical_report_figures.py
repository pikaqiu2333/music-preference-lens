"""Render reproducible SVG figures for the technical report without dependencies."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def text_element(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 14,
    anchor: str = "middle",
    weight: int = 400,
    fill: str = "#1f2933",
    rotate: float | None = None,
) -> str:
    transform = (
        f' transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"'
        if rotate is not None
        else ""
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}"{transform}>'
        f'{html.escape(value)}</text>'
    )


def render_verifier_metrics(
    qwen_17: dict[str, Any], qwen_4: dict[str, Any]
) -> str:
    series = []
    for model_label, summary in (("Qwen3 1.7B", qwen_17), ("Qwen3 4B", qwen_4)):
        for path_label, key in (
            ("Choice", "choice_metrics"),
            ("Sequence", "catalog_sequence_metrics"),
            ("Frozen OR", "primary_metrics"),
        ):
            metrics = summary[key]
            series.append(
                {
                    "model": model_label,
                    "path": path_label,
                    "specificity": metrics["exact_specificity"],
                    "sensitivity": metrics["conflict_sensitivity"],
                    "balanced": metrics["balanced_accuracy"],
                }
            )

    width, height = 1120, 620
    left, right, top, bottom = 90, 30, 80, 150
    plot_width = width - left - right
    plot_height = height - top - bottom
    group_width = plot_width / len(series)
    bar_width = 28
    colors = {
        "specificity": "#2A9D8F",
        "sensitivity": "#E76F51",
        "balanced": "#264653",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<title>Independent holdout verifier metrics</title>',
        '<desc>Exact specificity, conflict sensitivity, and balanced accuracy for Qwen3 1.7B and 4B verification paths.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text_element(width / 2, 34, "Independent holdout: verifier performance", size=22, weight=700),
        text_element(width / 2, 58, "9 exact and 9 catalog-conflict events; frozen zero thresholds", size=14, fill="#52606d"),
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_height * (1 - tick)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d9e2ec" stroke-width="1"/>'
        )
        parts.append(text_element(left - 14, y + 5, f"{tick:.2f}", anchor="end", size=12, fill="#52606d"))
    threshold_y = top + plot_height * 0.25
    parts.append(
        f'<line x1="{left}" y1="{threshold_y:.1f}" x2="{width-right}" y2="{threshold_y:.1f}" stroke="#9b5de5" stroke-width="2" stroke-dasharray="7 6"/>'
    )
    parts.append(text_element(width - right, threshold_y - 8, "pre-specified 0.75 gate", anchor="end", size=12, fill="#7b2cbf"))

    for index, item in enumerate(series):
        center = left + group_width * (index + 0.5)
        for offset, metric in zip((-bar_width, 0, bar_width), colors):
            value = float(item[metric])
            bar_height = plot_height * value
            x = center + offset - bar_width / 2 + 2
            y = top + plot_height - bar_height
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width-4}" height="{bar_height:.1f}" fill="{colors[metric]}" rx="2"/>'
            )
            parts.append(text_element(x + (bar_width - 4) / 2, y - 6, f"{value:.2f}", size=10, fill="#334e68"))
        parts.append(text_element(center, top + plot_height + 28, item["path"], size=13, weight=600))
        parts.append(text_element(center, top + plot_height + 48, item["model"], size=12, fill="#52606d"))
        if index == 2:
            separator = left + group_width * 3
            parts.append(
                f'<line x1="{separator:.1f}" y1="{top}" x2="{separator:.1f}" y2="{top+plot_height+58}" stroke="#bcccdc" stroke-width="1"/>'
            )

    legend_y = height - 40
    legend_items = [
        ("specificity", "Exact specificity"),
        ("sensitivity", "Conflict sensitivity"),
        ("balanced", "Balanced accuracy"),
    ]
    start_x = width / 2 - 260
    for index, (metric, label) in enumerate(legend_items):
        x = start_x + index * 205
        parts.append(f'<rect x="{x:.1f}" y="{legend_y-13}" width="16" height="16" fill="{colors[metric]}" rx="2"/>')
        parts.append(text_element(x + 24, legend_y, label, anchor="start", size=13))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_causal_trace(summary: dict[str, Any]) -> str:
    group_specs = [
        ("focus:exact_clean_control", "Clean exact", "#2A9D8F"),
        ("focus:sequence_exact_false_positive", "Exact, prior-masked", "#3A86FF"),
        ("focus:choice_conflict_hit", "Detected conflict", "#F4A261"),
        ("focus:choice_conflict_miss", "Missed conflict", "#D62828"),
    ]
    width, height = 1120, 650
    left, right, top, bottom = 90, 40, 80, 110
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_min, y_max = -0.3, 1.3

    def x_pos(layer: int) -> float:
        return left + plot_width * (layer - 1) / 27

    def y_pos(value: float) -> float:
        return top + plot_height * (y_max - value) / (y_max - y_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<title>Layerwise causal recovery of factual-title effects</title>',
        '<desc>Mean normalized recovery from control-title states toward factual-title complete-artist scoring states after full residual patching.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text_element(width / 2, 34, "Where factual-title effects enter the residual stream", size=22, weight=700),
        text_element(width / 2, 58, "Full-sequence causal patching at all artist prediction positions", size=14, fill="#52606d"),
    ]
    for tick in (-0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
        y = y_pos(tick)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d9e2ec" stroke-width="1"/>'
        )
        parts.append(text_element(left - 14, y + 5, f"{tick:.2f}", anchor="end", size=12, fill="#52606d"))
    for layer in (1, 4, 8, 12, 16, 20, 24, 28):
        x = x_pos(layer)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_height}" stroke="#edf2f7" stroke-width="1"/>'
        )
        parts.append(text_element(x, top + plot_height + 26, str(layer), size=12, fill="#52606d"))
    parts.append(
        text_element(
            28,
            top + plot_height / 2,
            "Mean normalized recovery",
            anchor="middle",
            size=14,
            weight=600,
            rotate=-90,
        )
    )
    parts.append(text_element(left + plot_width / 2, top + plot_height + 52, "Qwen3-1.7B layer", size=14, weight=600))

    for group_key, label, color in group_specs:
        curve = [
            point
            for point in summary["group_curves"][group_key]["curve"]
            if point["component"] == "full_residual"
        ]
        coordinates = " ".join(
            f"{x_pos(int(point['layer'])):.1f},{y_pos(float(point['mean_recovery'])):.1f}"
            for point in curve
        )
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for point in curve:
            if int(point["layer"]) in (12, 16, 18, 21, 24, 28):
                parts.append(
                    f'<circle cx="{x_pos(int(point["layer"])):.1f}" cy="{y_pos(float(point["mean_recovery"])):.1f}" r="3.5" fill="{color}"/>'
                )

    legend_y = height - 28
    start_x = 120
    for index, (_, label, color) in enumerate(group_specs):
        x = start_x + index * 245
        parts.append(f'<line x1="{x}" y1="{legend_y-5}" x2="{x+24}" y2="{legend_y-5}" stroke="{color}" stroke-width="4"/>')
        parts.append(text_element(x + 32, legend_y, label, anchor="start", size=13))
    parts.append(
        text_element(
            width - right,
            top + 18,
            "Recovery follows the factual-title state; it is not necessarily a correction.",
            anchor="end",
            size=12,
            fill="#7b2c2c",
        )
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> int:
    qwen_17 = load_json(
        PROJECT_ROOT / "runs" / "independent_holdout_verifier_summary.json"
    )
    qwen_4 = load_json(
        PROJECT_ROOT / "runs" / "qwen3_4b_cross_model_verifier_summary.json"
    )
    causal = load_json(
        PROJECT_ROOT / "runs" / "holdout_sequence_causal_trace_summary.json"
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        FIGURE_DIR / "holdout_verifier_metrics.svg": render_verifier_metrics(
            qwen_17, qwen_4
        ),
        FIGURE_DIR / "causal_trace_residual.svg": render_causal_trace(causal),
    }
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
