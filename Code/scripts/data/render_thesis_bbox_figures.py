#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import math
import sys
from datetime import datetime
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evttc.native_adapter import NativeSampleAdapter


DEFAULT_METRICS_CSV = (
    CODE_ROOT
    / "Experiments"
    / "SparsityAwareLTS"
    / "BBoxLoomingV13_20260427_212537"
    / "IntervalMetrics.csv"
)
DEFAULT_FAILURE_CSV = (
    CODE_ROOT
    / "Experiments"
    / "SparsityAwareLTSDiagnostics"
    / "BBoxFailureTypeReport_20260427_212614"
    / "IntervalFailureTypes.csv"
)
DEFAULT_CANDIDATE_CSV = (
    CODE_ROOT
    / "Experiments"
    / "SparsityAwareLTSDiagnostics"
    / "BBoxHighConfGeometryReport_20260427_212710"
    / "IntervalCandidateQuantiles.csv"
)
DEFAULT_SAMPLES = (
    "CCRm-medium-100%_0046",
    "CPLA-low_0066",
    "CPNAO-low_0032",
    "CCRm-medium-0%_0188",
    "CPNAO-medium_0008",
    "CCRs-1-high-50%_0029",
)
QUANTILE_KEYS = ("q05", "q10", "q15", "q25", "q35", "q50", "q65", "q75", "q90")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render thesis candidate SVGs for BBoxLooming samples.",
    )
    parser.add_argument("--formal-root", type=Path, default=CODE_ROOT / "DatasetFormal")
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--failure-csv", type=Path, default=DEFAULT_FAILURE_CSV)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CODE_ROOT / "Experiments" / "SparsityAwareLTSDiagnostics",
    )
    parser.add_argument("--samples", nargs="*", default=list(DEFAULT_SAMPLES))
    parser.add_argument("--history-back", type=int, default=12)
    return parser.parse_args()


def read_csv_by_sample(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def fnum(value: object, digits: int = 3) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        value_f = float(text)
    except ValueError:
        return text
    if not math.isfinite(value_f):
        return ""
    return f"{value_f:.{digits}f}"


def svg_text(x: float, y: float, text: object, *, size: int = 13, weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="#111827">'
        f"{html.escape(str(text))}</text>"
    )


def scale(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    if abs(in_max - in_min) < 1e-9:
        return (out_min + out_max) / 2.0
    ratio = (value - in_min) / (in_max - in_min)
    return out_min + ratio * (out_max - out_min)


def bbox_area(record: object) -> float:
    return max(0.0, float((record.x2 - record.x1) * (record.y2 - record.y1)))


def polyline(points: list[tuple[float, float]], color: str, width: float = 2.0) -> str:
    if not points:
        return ""
    packed = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{packed}" fill="none" stroke="{color}" stroke-width="{width:.1f}"/>'


def render_area_panel(records: list[object], current_label: int, x0: float, y0: float, w: float, h: float) -> str:
    areas = [bbox_area(record) for record in records]
    labels = [int(record.label_id) for record in records]
    if not records or not areas:
        return svg_text(x0, y0 + 20, "No bbox records", size=12)

    min_label = min(labels)
    max_label = max(labels)
    max_area = max(max(areas), 1.0)
    min_area = 0.0
    points = [
        (
            scale(label, min_label, max_label, x0, x0 + w),
            scale(area, min_area, max_area, y0 + h, y0),
        )
        for label, area in zip(labels, areas)
    ]

    current_x = scale(current_label, min_label, max_label, x0, x0 + w)
    parts = [
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" fill="#f9fafb" stroke="#d1d5db"/>',
        polyline(points, "#2563eb", width=2.4),
        f'<line x1="{current_x:.1f}" y1="{y0:.1f}" x2="{current_x:.1f}" y2="{(y0+h):.1f}" stroke="#dc2626" stroke-width="1.8" stroke-dasharray="5 4"/>',
    ]
    for label, area in zip(labels, areas):
        x = scale(label, min_label, max_label, x0, x0 + w)
        y = scale(area, min_area, max_area, y0 + h, y0)
        fill = "#dc2626" if label == current_label else "#1d4ed8"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{fill}"/>')
    parts.extend(
        [
            svg_text(x0, y0 - 8, "BBox area history", size=13, weight="700"),
            svg_text(x0, y0 + h + 18, f"label {min_label} -> {max_label}", size=11),
            svg_text(x0 + w - 135, y0 + h + 18, f"max area {max_area:.0f}", size=11),
        ]
    )
    return "\n".join(parts)


def render_quantile_panel(row: dict[str, str], x0: float, y0: float, w: float, h: float) -> str:
    parts = [
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" fill="#f9fafb" stroke="#d1d5db"/>',
        svg_text(x0, y0 - 8, "Candidate quantile error", size=13, weight="700"),
    ]
    values: list[tuple[str, float]] = []
    for key in QUANTILE_KEYS:
        text = row.get(f"candidate_{key}_e_ttc_pct", "")
        if text:
            values.append((key, float(text)))
    if not values:
        parts.append(svg_text(x0 + 16, y0 + 34, "No candidate errors available", size=12))
        return "\n".join(parts)

    max_value = max(max(value for _, value in values), 1.0)
    chart_top = y0 + 54.0
    chart_bottom = y0 + h - 36.0
    chart_h = chart_bottom - chart_top
    bar_gap = 9.0
    bar_w = (w - 48.0 - bar_gap * (len(values) - 1)) / len(values)
    for idx, (key, value) in enumerate(values):
        bx = x0 + 24.0 + idx * (bar_w + bar_gap)
        bh = scale(value, 0.0, max_value, 0.0, chart_h)
        by = chart_bottom - bh
        fill = "#059669" if key == row.get("best_quantile_key") else "#6b7280"
        if key == "q25":
            fill = "#dc2626"
        parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{fill}"/>')
        label_x = bx + bar_w / 2.0
        parts.append(
            f'<text x="{label_x:.1f}" y="{(y0 + h - 12.0):.1f}" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="9" fill="#111827">'
            f"{html.escape(key)}</text>"
        )
    parts.append(svg_text(x0 + 18, y0 + 24, f"best diagnostic quantile: {row.get('best_quantile_key', '')}", size=10))
    parts.append(svg_text(x0 + w - 128, y0 + 24, f"max error {max_value:.1f}%", size=10))
    parts.append(
        f'<line x1="{x0 + 18:.1f}" y1="{chart_bottom:.1f}" x2="{x0 + w - 18:.1f}" y2="{chart_bottom:.1f}" '
        'stroke="#d1d5db" stroke-width="1"/>'
    )
    return "\n".join(parts)


def render_status_panel(
    *,
    metrics_row: dict[str, str],
    failure_row: dict[str, str],
    x0: float,
    y0: float,
    w: float,
    h: float,
) -> str:
    status = metrics_row.get("status", failure_row.get("status", ""))
    failure_type = failure_row.get("primary_failure_type", "")
    gt = fnum(metrics_row.get("gt_ttc_s", failure_row.get("gt_ttc_s")))
    pred = fnum(metrics_row.get("ttc_est_s", failure_row.get("pred_ttc_s")))
    err = fnum(metrics_row.get("e_ttc_pct", failure_row.get("e_ttc_pct")))
    parts = [
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" fill="#f9fafb" stroke="#d1d5db"/>',
        svg_text(x0, y0 - 8, "Prediction status", size=13, weight="700"),
        svg_text(x0 + 18, y0 + 34, f"status: {status}", size=12),
        svg_text(x0 + 18, y0 + 60, f"failure type: {failure_type or 'not classified'}", size=12),
        svg_text(x0 + 18, y0 + 100, f"GT TTC: {gt}s", size=12),
        svg_text(x0 + 18, y0 + 126, f"prediction: {pred}s", size=12),
        svg_text(x0 + 18, y0 + 152, f"relative error: {err}%", size=12),
        svg_text(x0 + 18, y0 + h - 46, "No diagnostic quantile row is available.", size=11),
        svg_text(x0 + 18, y0 + h - 24, "This panel therefore reports status, not empty bars.", size=11),
    ]
    return "\n".join(parts)


def render_svg(
    *,
    sample_id: str,
    sample: object,
    local_records: list[object],
    metrics_row: dict[str, str],
    failure_row: dict[str, str],
    candidate_row: dict[str, str] | None,
) -> str:
    width = 980
    height = 520
    title = f"{sample_id} / {metrics_row.get('status', failure_row.get('status', ''))}"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(28, 34, title, size=18, weight="700"),
        svg_text(
            28,
            60,
            (
                f"GT={fnum(metrics_row.get('gt_ttc_s', failure_row.get('gt_ttc_s')))}s  "
                f"Pred={fnum(metrics_row.get('ttc_est_s', failure_row.get('pred_ttc_s')))}s  "
                f"e_ttc={fnum(metrics_row.get('e_ttc_pct', failure_row.get('e_ttc_pct')))}%  "
                f"type={failure_row.get('primary_failure_type', '')}"
            ),
            size=13,
        ),
        render_area_panel(local_records, int(sample.start_label_id), 45, 108, 400, 230),
    ]
    if candidate_row:
        parts.append(render_quantile_panel(candidate_row, 525, 108, 400, 230))
    else:
        parts.append(
            render_status_panel(
                metrics_row=metrics_row,
                failure_row=failure_row,
                x0=525,
                y0=108,
                w=400,
                h=230,
            )
        )
    notes = [
        f"sequence: {sample.sequence_id}",
        f"interval_idx: {sample.interval_idx}",
        f"start_label_id: {sample.start_label_id}",
        f"event_count: {sample.event_count}",
    ]
    parts.append(svg_text(45, 390, "Sample metadata", size=14, weight="700"))
    for idx, note in enumerate(notes):
        parts.append(svg_text(45, 415 + idx * 22, note, size=12))
    parts.append(svg_text(525, 390, "Interpretation", size=14, weight="700"))
    if candidate_row:
        interp = [
            "Red bar is current Q25 diagnostic error.",
            "Green bar is the best diagnostic quantile for this sample.",
            "This is diagnostic only; GT-derived oracle is not part of the method.",
        ]
    else:
        interp = [
            "The right panel reports V13 status and error.",
            "It is used when candidate quantiles are unavailable.",
            "Use this sample for residual failure illustration.",
        ]
    for idx, note in enumerate(interp):
        parts.append(svg_text(525, 415 + idx * 22, note, size=12))
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    run_ts = datetime.now().strftime("ThesisBBoxFigures_%Y%m%d_%H%M%S")
    run_dir = args.output_root.resolve() / run_ts
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_sample = read_csv_by_sample(args.metrics_csv.resolve())
    failure_by_sample = read_csv_by_sample(args.failure_csv.resolve())
    candidate_by_sample = read_csv_by_sample(args.candidate_csv.resolve())
    adapters: dict[str, NativeSampleAdapter] = {}
    manifest_rows: list[dict[str, str]] = []

    try:
        for sample_id in args.samples:
            metrics_row = metrics_by_sample.get(sample_id)
            failure_row = failure_by_sample.get(sample_id, {})
            row_for_sequence = metrics_row or failure_row
            if not row_for_sequence:
                print(f"[Warn] missing sample in metrics/failure CSV: {sample_id}", file=sys.stderr)
                continue
            sequence_id = row_for_sequence["sequence_id"]
            if sequence_id not in adapters:
                adapters[sequence_id] = NativeSampleAdapter(
                    sequence_id,
                    formal_root=args.formal_root.resolve(),
                )
            adapter = adapters[sequence_id]
            bundle = adapter.get_sample(sample_id=sample_id, include_events=False)
            sample = bundle.interval_sample
            lo = sample.start_label_id - args.history_back
            hi = sample.start_label_id + 2
            local_records = [
                record for record in adapter.bbox_records if lo <= record.label_id <= hi
            ]
            svg = render_svg(
                sample_id=sample_id,
                sample=sample,
                local_records=local_records,
                metrics_row=metrics_row or {},
                failure_row=failure_row,
                candidate_row=candidate_by_sample.get(sample_id),
            )
            output_name = f"{sample_id}.svg".replace("/", "_")
            output_path = plots_dir / output_name
            output_path.write_text(svg, encoding="utf-8")
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "sequence_id": sequence_id,
                    "svg": str(output_path.relative_to(run_dir)),
                    "status": (metrics_row or failure_row).get("status", ""),
                    "primary_failure_type": failure_row.get("primary_failure_type", ""),
                    "e_ttc_pct": (metrics_row or failure_row).get("e_ttc_pct", ""),
                }
            )
    finally:
        for adapter in adapters.values():
            adapter.close()

    with (run_dir / "FigureManifest.csv").open("w", encoding="utf-8", newline="\n") as handle:
        fieldnames = ["sample_id", "sequence_id", "svg", "status", "primary_failure_type", "e_ttc_pct"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_lines = [
        "# Thesis BBox Figure Candidates",
        "",
        f"- run_dir: `{run_dir}`",
        f"- metrics_csv: `{args.metrics_csv.resolve()}`",
        f"- failure_csv: `{args.failure_csv.resolve()}`",
        f"- candidate_csv: `{args.candidate_csv.resolve()}`",
        f"- rendered_count: `{len(manifest_rows)}`",
        "",
        "| sample | type | e_ttc_pct | svg |",
        "|---|---|---:|---|",
    ]
    for row in manifest_rows:
        summary_lines.append(
            f"| `{row['sample_id']}` | `{row['primary_failure_type']}` | "
            f"{fnum(row['e_ttc_pct'])} | `{row['svg']}` |"
        )
    (run_dir / "Summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"[Done] run_dir: {run_dir}")
    print(f"[Done] summary: {run_dir / 'Summary.md'}")


if __name__ == "__main__":
    main()
