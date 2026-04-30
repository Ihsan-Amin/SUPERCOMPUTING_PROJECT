#!/usr/bin/env python3
"""
Create a basic SVG chart from output/model_comparison.csv.

The chart compares top-k test accuracy, training time, and parameter count
across the trained Fruits-262 models without requiring extra plotting packages.
"""

import argparse
import csv
import html
import os


DEFAULT_CSV = "output/model_comparison.csv"
DEFAULT_OUT = "output/model_comparison_graph.svg"


COLORS = {
    "top1": "#1f77b4",
    "top5": "#2ca02c",
    "top10": "#ff7f0e",
    "hours": "#7f3c8d",
    "params": "#666666",
    "axis": "#333333",
    "grid": "#dddddd",
    "text": "#222222",
}


def read_results(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        row["test_top1"] = float(row["test_top1"])
        row["test_top5"] = float(row["test_top5"])
        row["test_top10"] = float(row["test_top10"])
        row["params"] = int(row["params"])
        row["hours"] = float(row["hours"])
        row["epochs"] = int(row["epochs"])
        row["batch_size"] = int(row["batch_size"])

    return rows


def sx(x):
    return f"{x:.1f}"


def label(text):
    return html.escape(str(text), quote=True)


def add_text(parts, x, y, text, size=16, weight="400", anchor="middle", fill=None):
    fill = fill or COLORS["text"]
    parts.append(
        f'<text x="{sx(x)}" y="{sx(y)}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
        f"{label(text)}</text>"
    )


def add_bar(parts, x, y, width, height, color):
    parts.append(
        f'<rect x="{sx(x)}" y="{sx(y)}" width="{sx(width)}" '
        f'height="{sx(height)}" fill="{color}" rx="3" />'
    )


def build_svg(rows):
    width = 1200
    height = 760
    pad = 60
    panel_gap = 70

    acc = {
        "x": pad,
        "y": 120,
        "w": width - 2 * pad,
        "h": 310,
    }
    time = {
        "x": pad,
        "y": acc["y"] + acc["h"] + panel_gap,
        "w": width - 2 * pad,
        "h": 180,
    }

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
    ]

    add_text(parts, width / 2, 44, "Fruits-262 Model Comparison", 28, "700")
    add_text(
        parts,
        width / 2,
        74,
        "Accuracy, training time, and parameter count from output/model_comparison.csv",
        15,
        "400",
        fill="#555555",
    )

    draw_accuracy_panel(parts, rows, acc)
    draw_time_panel(parts, rows, time)
    draw_notes(parts, rows, time["y"] + time["h"] + 46)

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def draw_accuracy_panel(parts, rows, panel):
    x0, y0, w, h = panel["x"], panel["y"], panel["w"], panel["h"]
    chart_left = x0 + 58
    chart_right = x0 + w - 18
    chart_top = y0 + 35
    chart_bottom = y0 + h - 54
    chart_w = chart_right - chart_left
    chart_h = chart_bottom - chart_top

    add_text(parts, x0, y0 + 5, "Test Accuracy (%)", 18, "700", anchor="start")

    for tick in range(0, 101, 20):
        y = chart_bottom - chart_h * tick / 100
        parts.append(
            f'<line x1="{sx(chart_left)}" y1="{sx(y)}" '
            f'x2="{sx(chart_right)}" y2="{sx(y)}" '
            f'stroke="{COLORS["grid"]}" stroke-width="1" />'
        )
        add_text(parts, chart_left - 12, y + 5, tick, 13, anchor="end", fill="#555555")

    parts.append(
        f'<line x1="{sx(chart_left)}" y1="{sx(chart_bottom)}" '
        f'x2="{sx(chart_right)}" y2="{sx(chart_bottom)}" '
        f'stroke="{COLORS["axis"]}" stroke-width="1.5" />'
    )
    parts.append(
        f'<line x1="{sx(chart_left)}" y1="{sx(chart_top)}" '
        f'x2="{sx(chart_left)}" y2="{sx(chart_bottom)}" '
        f'stroke="{COLORS["axis"]}" stroke-width="1.5" />'
    )

    group_w = chart_w / len(rows)
    bar_w = min(58, group_w / 6)
    offsets = [-bar_w * 1.2, 0, bar_w * 1.2]
    metrics = [
        ("test_top1", "Top-1", COLORS["top1"]),
        ("test_top5", "Top-5", COLORS["top5"]),
        ("test_top10", "Top-10", COLORS["top10"]),
    ]

    for i, row in enumerate(rows):
        cx = chart_left + group_w * (i + 0.5)
        for offset, (key, name, color) in zip(offsets, metrics):
            value = row[key]
            bh = chart_h * value / 100
            bx = cx + offset - bar_w / 2
            by = chart_bottom - bh
            add_bar(parts, bx, by, bar_w, bh, color)
            add_text(parts, bx + bar_w / 2, by - 8, f"{value:.1f}", 12)
        add_text(parts, cx, chart_bottom + 28, row["model"], 14, "700")
        add_text(parts, cx, chart_bottom + 47, row["resolution"], 12, fill="#666666")

    legend_x = chart_right - 250
    legend_y = y0 + 8
    for i, (_, name, color) in enumerate(metrics):
        lx = legend_x + i * 88
        parts.append(
            f'<rect x="{sx(lx)}" y="{sx(legend_y - 11)}" width="14" '
            f'height="14" fill="{color}" rx="2" />'
        )
        add_text(parts, lx + 20, legend_y + 1, name, 13, anchor="start", fill="#444444")


def draw_time_panel(parts, rows, panel):
    x0, y0, w, h = panel["x"], panel["y"], panel["w"], panel["h"]
    chart_left = x0 + 58
    chart_right = x0 + w - 18
    chart_top = y0 + 36
    chart_bottom = y0 + h - 42
    chart_w = chart_right - chart_left
    chart_h = chart_bottom - chart_top
    max_hours = max(row["hours"] for row in rows)

    add_text(parts, x0, y0 + 5, "Training Time (hours)", 18, "700", anchor="start")

    for tick in range(0, int(max_hours) + 2):
        y = chart_bottom - chart_h * tick / (int(max_hours) + 1)
        parts.append(
            f'<line x1="{sx(chart_left)}" y1="{sx(y)}" '
            f'x2="{sx(chart_right)}" y2="{sx(y)}" '
            f'stroke="{COLORS["grid"]}" stroke-width="1" />'
        )
        add_text(parts, chart_left - 12, y + 5, tick, 13, anchor="end", fill="#555555")

    parts.append(
        f'<line x1="{sx(chart_left)}" y1="{sx(chart_bottom)}" '
        f'x2="{sx(chart_right)}" y2="{sx(chart_bottom)}" '
        f'stroke="{COLORS["axis"]}" stroke-width="1.5" />'
    )
    parts.append(
        f'<line x1="{sx(chart_left)}" y1="{sx(chart_top)}" '
        f'x2="{sx(chart_left)}" y2="{sx(chart_bottom)}" '
        f'stroke="{COLORS["axis"]}" stroke-width="1.5" />'
    )

    group_w = chart_w / len(rows)
    bar_w = min(94, group_w * 0.36)
    scale_max = int(max_hours) + 1

    for i, row in enumerate(rows):
        cx = chart_left + group_w * (i + 0.5)
        bh = chart_h * row["hours"] / scale_max
        bx = cx - bar_w / 2
        by = chart_bottom - bh
        add_bar(parts, bx, by, bar_w, bh, COLORS["hours"])
        add_text(parts, cx, by - 8, f"{row['hours']:.2f}h", 13, "700")
        add_text(parts, cx, chart_bottom + 28, row["model"], 14, "700")


def draw_notes(parts, rows, y):
    add_text(parts, 60, y, "Parameter Count", 16, "700", anchor="start")
    x = 210
    for row in rows:
        params_m = row["params"] / 1_000_000
        text = (
            f"{row['model']}: {params_m:.1f}M params, "
            f"{row['epochs']} epochs, batch {row['batch_size']}"
        )
        add_text(parts, x, y, text, 13, anchor="start", fill="#444444")
        x += 305


def main():
    parser = argparse.ArgumentParser(description="Plot model comparison CSV as SVG")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to model_comparison.csv")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Path to output SVG")
    args = parser.parse_args()

    rows = read_results(args.csv)
    svg = build_svg(rows)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(svg)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
