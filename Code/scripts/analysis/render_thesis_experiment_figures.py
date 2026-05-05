#!/usr/bin/env python3
"""Render lightweight SVG figures for the thesis experiment chapter."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[3]
ASSET_DIR = ROOT / "docs" / "active" / "Assets"


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    rows: list[tuple[str, float, str]],
    width: int = 960,
    height: int = 520,
) -> None:
    margin_left = 240
    margin_right = 90
    margin_top = 92
    row_h = 46
    plot_w = width - margin_left - margin_right
    max_value = max(v for _, v, _ in rows) * 1.08
    chart_h = len(rows) * row_h
    height = max(height, margin_top + chart_h + 70)

    palette = ["#376996", "#2a9d8f", "#e76f51", "#8a5a44", "#6d597a", "#b56576"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="40" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#1f2933">{svg_escape(title)}</text>',
        f'<text x="40" y="68" font-family="Arial, sans-serif" font-size="14" fill="#52616b">{svg_escape(subtitle)}</text>',
    ]
    for i, (label, value, note) in enumerate(rows):
        y = margin_top + i * row_h
        bar_w = value / max_value * plot_w
        color = palette[i % len(palette)]
        parts.extend(
            [
                f'<text x="{margin_left - 16}" y="{y + 22}" text-anchor="end" font-family="Arial, sans-serif" font-size="15" fill="#1f2933">{svg_escape(label)}</text>',
                f'<rect x="{margin_left}" y="{y}" width="{bar_w:.1f}" height="28" rx="3" fill="{color}"/>',
                f'<text x="{margin_left + bar_w + 10:.1f}" y="{y + 20}" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#1f2933">{value:.3f}%</text>',
                f'<text x="{margin_left}" y="{y + 43}" font-family="Arial, sans-serif" font-size="12" fill="#52616b">{svg_escape(note)}</text>',
            ]
        )
    axis_y = margin_top + chart_h + 18
    parts.append(
        f'<line x1="{margin_left}" y1="{axis_y}" x2="{margin_left + plot_w}" y2="{axis_y}" stroke="#c9d1d9" stroke-width="1"/>'
    )
    for tick in range(0, int(max_value) + 1, 20):
        x = margin_left + tick / max_value * plot_w
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{axis_y}" x2="{x:.1f}" y2="{axis_y + 5}" stroke="#c9d1d9" stroke-width="1"/>',
                f'<text x="{x:.1f}" y="{axis_y + 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#52616b">{tick}</text>',
            ]
        )
    parts.append(
        f'<text x="{margin_left + plot_w}" y="{axis_y + 45}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#52616b">relative TTC error, lower is better</text>'
    )
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_line_chart(
    path: Path,
    title: str,
    subtitle: str,
    series: list[tuple[str, list[tuple[float, float]], str]],
    width: int = 960,
    height: int = 520,
) -> None:
    margin_left = 80
    margin_right = 210
    margin_top = 92
    margin_bottom = 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    xs = [x for _, pts, _ in series for x, _y in pts]
    ys = [y for _, pts, _ in series for _x, y in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = 0.0, max(ys) * 1.12

    def sx(x: float) -> float:
        return margin_left + (x - min_x) / (max_x - min_x or 1.0) * plot_w

    def sy(y: float) -> float:
        return margin_top + plot_h - (y - min_y) / (max_y - min_y or 1.0) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="40" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#1f2933">{svg_escape(title)}</text>',
        f'<text x="40" y="68" font-family="Arial, sans-serif" font-size="14" fill="#52616b">{svg_escape(subtitle)}</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="#f7f9fb" stroke="#d0d7de"/>',
    ]
    for i in range(6):
        yv = max_y * i / 5
        y = sy(yv)
        parts.extend(
            [
                f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" stroke="#e5eaf0" stroke-width="1"/>',
                f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#52616b">{yv:.2f}</text>',
            ]
        )
    for i in range(0, int(max_x) + 1, max(1, int(max_x // 5) or 1)):
        x = sx(i)
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{margin_top + plot_h}" x2="{x:.1f}" y2="{margin_top + plot_h + 5}" stroke="#c9d1d9" stroke-width="1"/>',
                f'<text x="{x:.1f}" y="{margin_top + plot_h + 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#52616b">{i}</text>',
            ]
        )
    for label, pts, color in series:
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
    legend_x = margin_left + plot_w + 28
    for idx, (label, _pts, color) in enumerate(series):
        y = margin_top + idx * 28 + 8
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 38}" y="{y + 5}" font-family="Arial, sans-serif" font-size="12" fill="#1f2933">{svg_escape(label)}</text>',
            ]
        )
    parts.extend(
        [
            f'<text x="{margin_left + plot_w / 2}" y="{height - 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#52616b">epoch</text>',
            f'<text x="28" y="{margin_top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 28 {margin_top + plot_h / 2})" font-family="Arial, sans-serif" font-size="12" fill="#52616b">loss</text>',
            "</svg>\n",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def read_train_mean(run_dir: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    by_epoch: dict[int, list[tuple[float, float]]] = {}
    for csv_path in sorted(run_dir.glob("*/TrainHistory.csv")):
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                epoch = int(row["epoch"])
                by_epoch.setdefault(epoch, []).append(
                    (float(row["train_loss"]), float(row["val_loss"]))
                )
    train = []
    val = []
    for epoch in sorted(by_epoch):
        train.append((epoch, mean(v[0] for v in by_epoch[epoch])))
        val.append((epoch, mean(v[1] for v in by_epoch[epoch])))
    return train, val


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    write_bar_chart(
        ASSET_DIR / "ExperimentMainResults.svg",
        "Main formal comparison",
        "32-sequence formal protocol; relative TTC error on valid estimates",
        [
            ("BBoxLoomingV13", 14.410, "high-confidence traditional result"),
            ("BBoxLoomingV10", 18.064, "high-coverage traditional result"),
            ("TorchTinyCnnBBoxAux", 18.117, "strongest lightweight ML comparison"),
            ("TorchTinyCnnImageOnly", 36.059, "image-only ML comparison"),
            ("STRTTC Python", 88.054, "legacy direct formal rerun"),
            ("CMax direct rerun", 103.439, "legacy direct formal rerun"),
            ("HeuristicCueV1", 126.894, "early local-event baseline"),
        ],
        height=560,
    )

    write_line_chart(
        ASSET_DIR / "BBoxAblationTrend.svg",
        "BBox looming method evolution",
        "Problem-driven stages from local cues to quality-aware bbox looming",
        [
            (
                "relative TTC error",
                [
                    (1, 126.894),
                    (2, 19.319),
                    (3, 18.064),
                    (4, 15.855),
                    (5, 14.765),
                    (6, 14.410),
                ],
                "#376996",
            )
        ],
        height=500,
    )

    write_bar_chart(
        ASSET_DIR / "LegacyCleanRerunComparison.svg",
        "Legacy clean reruns",
        "CMax and STRTTC under the current formal protocol",
        [
            ("STRTTC Python", 88.054, "2407 valid estimates; 1585 failures"),
            ("STRTTC quality-aware", 92.106, "2347 valid estimates; 1645 failures"),
            ("CMax direct rerun", 103.439, "3592 valid estimates; 400 failures"),
        ],
        height=360,
    )

    image_train, image_val = read_train_mean(
        ROOT
        / "Code"
        / "Experiments"
        / "MlBaselineTorch"
        / "TorchTinyCnnImageOnlyV1_AllData_MultiTauPolSplitV1_Mps"
    )
    bbox_train, bbox_val = read_train_mean(
        ROOT
        / "Code"
        / "Experiments"
        / "MlBaselineTorch"
        / "TorchTinyCnnBBoxAuxV1_AllData_MultiTauPolSplitV1_Mps"
    )
    write_line_chart(
        ASSET_DIR / "MLTrainingCurves.svg",
        "Lightweight ML baseline training curves",
        "Mean train/validation loss across sequence-held-out runs",
        [
            ("image-only train", image_train, "#376996"),
            ("image-only val", image_val, "#5f9ea0"),
            ("bbox-aux train", bbox_train, "#e76f51"),
            ("bbox-aux val", bbox_val, "#f4a261"),
        ],
        height=520,
    )


if __name__ == "__main__":
    main()
