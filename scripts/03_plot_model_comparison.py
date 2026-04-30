#!/usr/bin/env python3
"""
Create a basic SVG chart from model_comparison.csv.

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


def fmt(value):
    return f"{value:.1f}"


def esc(value):
    return html.escape(str(value), quote=True)


def text(parts, x, y, value, size=14, weight="400", anchor="middle", fill=None):
    fill = fill or COLORS["text"]
    parts.append(
        f'<text x="{fmt(x)}" y="{fmt(y)}" font-size="{size}" '
        f'font-family="Arial, Helvetica, sans-serif" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{fill}">{esc(value)}</text>'
    )


def rect(parts, x, y, width, height, fill):
    parts.append(
        f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(width)}" '
        f'height="{fmt(height)}" fill="{fill}" rx="3" />'
    )


def line(parts, x1, y1, x2, y2, color=None, stroke_width=1):
    color = color or COLORS["grid"]
    parts.append(
        f'<line x1="{fmt(x1)}" y1="{fmt(y1)}" x2="{fmt(x2)}" y2="{fmt(y2)}" '
        f'stroke="{color}" stroke-width="{stroke_width}" />'
    )


def draw_accuracy(parts, rows, x, y, width, height):
    left = x + 60
    right = x + width - 20
    top = y + 45
    bottom = y + height - 60
    chart_w = right - left
    chart_h = bottom - top

    text(parts, x, y + 12, "Test Accuracy (%)", 18, "700", "start")

    for tick in range(0, 101, 20):
        ty = bottom - chart_h * tick / 100
        line(parts, left, ty, right, ty)
        text(parts, left - 12, ty + 5, tick, 13, anchor="end", fill="#555555")

    line(parts, left, bottom, right, bottom, COLORS["axis"], 1.5)
    line(parts, left, top, left, bottom, COLORS["axis"], 1.5)

    metrics = [
        ("test_top1", "Top-1", COLORS["top1"]),
        ("test_top5", "Top-5", COLORS["top5"]),
        ("test_top10", "Top-10", COLORS["top10"]),
    ]

    group_w = chart_w / len(rows)
    bar_w = min(54, group_w / 6)
    offsets = [-bar_w * 1.2, 0, bar_w * 1.2]

    for i, row in enumerate(rows):
        cx = left + group_w * (i + 0.5)
        for offset, (key, _, color) in zip(offsets, metrics):
            value = row[key]
            bar_h = chart_h * value / 100
            bx = cx + offset - bar_w / 2
            by = bottom - bar_h
            rect(parts, bx, by, bar_w, bar_h, color)
            text(parts, bx + bar_w / 2, by - 8, f"{value:.1f}", 12)

        text(parts, cx, bottom + 28, row["model"], 14, "700")
        text(parts, cx, bottom + 48, row["resolution"], 12, fill="#666666")

    legend_x = right - 250
    legend_y = y + 12
    for i, (_, label, color) in enumerate(metrics):
        lx = legend_x + i * 88
        rect(parts, lx, legend_y - 12, 14, 14, color)
        text(parts, lx + 20, legend_y, label, 13, anchor="start", fill="#444444")


def draw_training_time(parts, rows, x, y, width, height):
    left = x + 60
    right = x + width - 20
    top = y + 45
    bottom = y + height - 55
    chart_w = right - left
    chart_h = bottom - top
    max_hours = max(row["hours"] for row in rows)
    scale_max = int(max_hours) + 1

    text(parts, x, y + 12, "Training Time (hours)", 18, "700", "start")

    for tick in range(scale_max + 1):
        ty = bottom - chart_h * tick / scale_max
        line(parts, left, ty, right, ty)
        text(parts, left - 12, ty + 5, tick, 13, anchor="end", fill="#555555")

    line(parts, left, bottom, right, bottom, COLORS["axis"], 1.5)
    line(parts, left, top, left, bottom, COLORS["axis"], 1.5)

    group_w = chart_w / len(rows)
    bar_w = min(90, group_w * 0.34)

    for i, row in enumerate(rows):
        cx = left + group_w * (i + 0.5)
        bar_h = chart_h * row["hours"] / scale_max
        bx = cx - bar_w / 2
        by = bottom - bar_h
        rect(parts, bx, by, bar_w, bar_h, COLORS["hours"])
        text(parts, cx, by - 8, f"{row['hours']:.2f}h", 13, "700")
        text(parts, cx, bottom + 30, row["model"], 14, "700")


def draw_params(parts, rows, y):
    text(parts, 60, y, "Parameter Count", 16, "700", "start")
    x = 210
    for row in rows:
        params_m = row["params"] / 1_000_000
        summary = (
            f"{row['model']}: {params_m:.1f}M params, "
            f"{row['epochs']} epochs, batch {row['batch_size']}"
        )
        text(parts, x, y, summary, 13, anchor="start", fill="#444444")
        x += 305


def build_svg(rows):
    width = 1200
    height = 760
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
    ]

    text(parts, width / 2, 44, "Fruits-262 Model Comparison", 28, "700")
    text(
        parts,
        width / 2,
        74,
        "Accuracy, training time, and parameter count from model_comparison.csv",
        15,
        fill="#555555",
    )

    draw_accuracy(parts, rows, 60, 115, 1080, 320)
    draw_training_time(parts, rows, 60, 480, 1080, 185)
    draw_params(parts, rows, 725)

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Create a model comparison SVG chart")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to model_comparison.csv")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output SVG path")
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
