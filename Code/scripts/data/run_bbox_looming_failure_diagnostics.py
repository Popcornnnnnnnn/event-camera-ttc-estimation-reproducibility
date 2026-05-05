#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Iterable
from xml.sax.saxutils import escape

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evttc import NativeSampleAdapter, list_formal_sequences
from evttc.formal_loader import BoundingBoxRecord
from evttc.sparsity_aware_lts import (
    BBoxLoomingConfigV1,
    _weighted_fit_residual,
    _weighted_linear_fit,
)


DEFAULT_FOCUS_SEQUENCES = ("CPNAO-medium", "CPNA-low", "CPNAO-low", "CPLA-medium")


@dataclass(frozen=True)
class CandidateRow:
    sample_id: str
    sequence_id: str
    interval_idx: int
    cue_name: str
    history_back: int
    fit_points: int
    ttc_s: float
    slope_px_per_s: float
    ref_px: float
    normalized_residual: float
    relative_slope: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose high-error BBoxLoomingV4 intervals with bbox trajectories and candidate distributions.",
    )
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=CODE_ROOT / "DatasetFormal",
        help="Root directory of formal sequence artifacts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CODE_ROOT / "Experiments" / "SparsityAwareLTSDiagnostics",
        help="Root directory for diagnostics outputs.",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional sequence IDs. Default: current high-error focus sequences.",
    )
    parser.add_argument(
        "--all-sequences",
        action="store_true",
        help="Run diagnostics on all formal sequences instead of the focus set.",
    )
    parser.add_argument("--top-n", type=int, default=80, help="Number of worst intervals to write.")
    parser.add_argument("--plot-top-n", type=int, default=24, help="Number of worst intervals to render as SVG.")
    parser.add_argument("--history-back", type=int, default=12, help="Maximum previous labels for V4 candidates.")
    parser.add_argument("--history-min-back", type=int, default=2, help="Minimum previous labels for V4 candidates.")
    parser.add_argument("--candidate-quantile", type=float, default=0.25, help="Quantile used by BBoxLoomingV4.")
    parser.add_argument("--max-ttc", type=float, default=16.0, help="Maximum TTC candidate horizon.")
    parser.add_argument("--min-width-px", type=float, default=3.0, help="Minimum reference dimension.")
    parser.add_argument("--min-slope-px-per-s", type=float, default=1.0, help="Minimum positive slope.")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: float | None, digits: int = 6) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def dims_for_records(
    records: list[BoundingBoxRecord],
    *,
    bottom_reference_y: float,
) -> dict[str, np.ndarray]:
    widths = np.asarray([record.x2 - record.x1 for record in records], dtype=np.float64)
    heights = np.asarray([record.y2 - record.y1 for record in records], dtype=np.float64)
    bottom_y = np.asarray([record.y2 for record in records], dtype=np.float64)
    return {
        "width": widths,
        "height": heights,
        "diag": np.sqrt(np.square(widths) + np.square(heights)),
        "sqrt_area": np.sqrt(np.maximum(widths * heights, 1e-6)),
        "bottom_offset": bottom_y - float(bottom_reference_y),
    }


def collect_candidates(
    *,
    sample_id: str,
    sequence_id: str,
    interval_idx: int,
    history_back: int,
    query_t_s: float,
    records: list[BoundingBoxRecord],
    bottom_reference_y: float,
    config: BBoxLoomingConfigV1,
) -> list[CandidateRow]:
    if len(records) < config.min_fit_points:
        return []

    times = np.asarray([record.timestamp_s for record in records], dtype=np.float64)
    weights = np.ones_like(times, dtype=np.float64)
    rows: list[CandidateRow] = []

    for cue_name, values in dims_for_records(records, bottom_reference_y=bottom_reference_y).items():
        fit = _weighted_linear_fit(times, values, weights)
        if fit is None:
            continue
        slope_px_per_s, intercept_px = fit
        ref_px = float(slope_px_per_s * query_t_s + intercept_px)
        if ref_px <= config.min_width_px or slope_px_per_s <= config.min_slope_px_per_s:
            continue
        ttc_s = float(ref_px / slope_px_per_s)
        if not math.isfinite(ttc_s) or ttc_s < config.min_ttc_s or ttc_s > config.max_ttc_s:
            continue
        residual = _weighted_fit_residual(times, values, weights, slope_px_per_s, intercept_px)
        normalized_residual = float((residual or 0.0) / max(abs(ref_px), 1e-6))
        relative_slope = float(slope_px_per_s / max(abs(ref_px), 1e-6))
        rows.append(
            CandidateRow(
                sample_id=sample_id,
                sequence_id=sequence_id,
                interval_idx=interval_idx,
                cue_name=cue_name,
                history_back=history_back,
                fit_points=len(records),
                ttc_s=ttc_s,
                slope_px_per_s=float(slope_px_per_s),
                ref_px=ref_px,
                normalized_residual=normalized_residual,
                relative_slope=relative_slope,
            )
        )
    return rows


def collect_multiwindow_candidates(
    *,
    bundle_records: tuple[BoundingBoxRecord, ...],
    start_label_id: int,
    sample_id: str,
    sequence_id: str,
    interval_idx: int,
    query_t_s: float,
    bottom_reference_y: float,
    config: BBoxLoomingConfigV1,
) -> tuple[list[CandidateRow], str]:
    max_history_back = max(int(config.history_back_labels), int(config.history_min_back_labels))
    lo = start_label_id - max_history_back
    hi = start_label_id
    records = sorted(
        [record for record in bundle_records if lo <= record.label_id <= hi],
        key=lambda item: item.timestamp_s,
    )
    if len(records) < config.min_fit_points:
        return [], "BBOX_HISTORY_TOO_SHORT"

    candidate_rows: list[CandidateRow] = []
    min_history_back = max(1, int(config.history_min_back_labels))
    for history_back in range(min_history_back, max_history_back + 1):
        window_lo = start_label_id - history_back
        window_records = [record for record in records if window_lo <= record.label_id <= hi]
        candidate_rows.extend(
            collect_candidates(
                sample_id=sample_id,
                sequence_id=sequence_id,
                interval_idx=interval_idx,
                history_back=history_back,
                query_t_s=query_t_s,
                records=window_records,
                bottom_reference_y=bottom_reference_y,
                config=config,
            )
        )
    if not candidate_rows:
        return [], "BBOX_FIT_FAIL"
    return candidate_rows, "OK_BBOX_MULTI_Q25"


def interval_diagnostics_row(
    *,
    sample_id: str,
    sequence_id: str,
    interval_idx: int,
    gt_ttc_s: float | None,
    status_gt: str,
    pred_ttc_s: float | None,
    status: str,
    current_bbox: BoundingBoxRecord | None,
    bottom_reference_y: float,
    candidates: list[CandidateRow],
) -> dict[str, object]:
    ttc_values = [candidate.ttc_s for candidate in candidates]
    cue_counts = Counter(candidate.cue_name for candidate in candidates)
    max_fit_points = max((candidate.fit_points for candidate in candidates), default=0)
    min_history_back = min((candidate.history_back for candidate in candidates), default=0)
    max_history_back = max((candidate.history_back for candidate in candidates), default=0)
    median_ttc = quantile(ttc_values, 0.50)
    q25_ttc = quantile(ttc_values, 0.25)
    q75_ttc = quantile(ttc_values, 0.75)
    p10_ttc = quantile(ttc_values, 0.10)
    p90_ttc = quantile(ttc_values, 0.90)
    iqr_s = None if q25_ttc is None or q75_ttc is None else q75_ttc - q25_ttc
    iqr_over_median = None
    if iqr_s is not None and median_ttc is not None:
        iqr_over_median = iqr_s / max(abs(median_ttc), 1e-6)
    spread_p90_p10_over_median = None
    if p10_ttc is not None and p90_ttc is not None and median_ttc is not None:
        spread_p90_p10_over_median = (p90_ttc - p10_ttc) / max(abs(median_ttc), 1e-6)

    e_ttc_pct = None
    abs_error_s = None
    pred_over_gt = None
    if pred_ttc_s is not None and gt_ttc_s is not None and gt_ttc_s > 0:
        abs_error_s = abs(pred_ttc_s - gt_ttc_s)
        e_ttc_pct = abs_error_s / gt_ttc_s * 100.0
        pred_over_gt = pred_ttc_s / gt_ttc_s

    width = None
    height = None
    bottom_offset = None
    if current_bbox is not None:
        width = float(current_bbox.x2 - current_bbox.x1)
        height = float(current_bbox.y2 - current_bbox.y1)
        bottom_offset = float(current_bbox.y2 - bottom_reference_y)

    return {
        "sample_id": sample_id,
        "sequence_id": sequence_id,
        "interval_idx": interval_idx,
        "status_gt": status_gt,
        "status": status,
        "gt_ttc_s": format_number(gt_ttc_s),
        "pred_ttc_s": format_number(pred_ttc_s),
        "abs_error_s": format_number(abs_error_s),
        "e_ttc_pct": format_number(e_ttc_pct),
        "pred_over_gt": format_number(pred_over_gt),
        "bbox_width_px": format_number(width),
        "bbox_height_px": format_number(height),
        "bbox_bottom_offset_px": format_number(bottom_offset),
        "candidate_count": len(candidates),
        "candidate_max_fit_points": max_fit_points,
        "candidate_min_history_back": min_history_back,
        "candidate_max_history_back": max_history_back,
        "candidate_min_s": format_number(min(ttc_values) if ttc_values else None),
        "candidate_p10_s": format_number(p10_ttc),
        "candidate_q25_s": format_number(q25_ttc),
        "candidate_median_s": format_number(median_ttc),
        "candidate_q75_s": format_number(q75_ttc),
        "candidate_p90_s": format_number(p90_ttc),
        "candidate_max_s": format_number(max(ttc_values) if ttc_values else None),
        "candidate_iqr_over_median": format_number(iqr_over_median),
        "candidate_p90_p10_over_median": format_number(spread_p90_p10_over_median),
        "candidate_width_count": cue_counts["width"],
        "candidate_height_count": cue_counts["height"],
        "candidate_diag_count": cue_counts["diag"],
        "candidate_sqrt_area_count": cue_counts["sqrt_area"],
        "candidate_bottom_offset_count": cue_counts["bottom_offset"],
    }


def candidate_rows_to_dicts(candidates: Iterable[CandidateRow]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        row = asdict(candidate)
        for key in ("ttc_s", "slope_px_per_s", "ref_px", "normalized_residual", "relative_slope"):
            row[key] = format_number(float(row[key]))
        rows.append(row)
    return rows


def svg_polyline(points: list[tuple[float, float]], *, color: str, width: int = 2) -> str:
    if not points:
        return ""
    payload = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{payload}" fill="none" stroke="{color}" stroke-width="{width}" />'


def scale_series(values: list[float], *, x0: float, y0: float, w: float, h: float) -> list[tuple[float, float]]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0
    points = []
    count = max(1, len(values) - 1)
    for idx, value in enumerate(values):
        x = x0 + w * idx / count
        y = y0 + h - h * (value - vmin) / (vmax - vmin)
        points.append((x, y))
    return points


def render_interval_svg(
    *,
    path: Path,
    row: dict[str, object],
    history_records: list[BoundingBoxRecord],
    candidates: list[CandidateRow],
    bottom_reference_y: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 920
    height = 520
    plot_x = 70
    plot_w = 760
    top_y = 70
    top_h = 170
    bottom_y = 320
    bottom_h = 95

    bbox_widths = [record.x2 - record.x1 for record in history_records]
    bbox_heights = [record.y2 - record.y1 for record in history_records]
    bbox_bottom_offsets = [record.y2 - bottom_reference_y for record in history_records]
    bbox_diags = [math.sqrt((record.x2 - record.x1) ** 2 + (record.y2 - record.y1) ** 2) for record in history_records]

    max_ttc_axis = max(
        1.0,
        float(row["gt_ttc_s"] or 0) if row["gt_ttc_s"] else 0.0,
        float(row["pred_ttc_s"] or 0) if row["pred_ttc_s"] else 0.0,
        max((candidate.ttc_s for candidate in candidates), default=0.0),
    )
    max_ttc_axis = min(20.0, max(4.0, max_ttc_axis * 1.08))
    cue_y = {
        "width": bottom_y + 10,
        "height": bottom_y + 28,
        "diag": bottom_y + 46,
        "sqrt_area": bottom_y + 64,
        "bottom_offset": bottom_y + 82,
    }

    def x_for_ttc(value: float) -> float:
        return plot_x + plot_w * max(0.0, min(value, max_ttc_axis)) / max_ttc_axis

    title = f"{row['sample_id']}  e={row['e_ttc_pct']}%  gt={row['gt_ttc_s']}  pred={row['pred_ttc_s']}"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="30" y="34" font-family="monospace" font-size="17" fill="#111111">{escape(title)}</text>',
        f'<text x="30" y="56" font-family="monospace" font-size="12" fill="#444444">status={escape(str(row["status"]))} candidates={row["candidate_count"]} iqr/median={row["candidate_iqr_over_median"]}</text>',
        f'<rect x="{plot_x}" y="{top_y}" width="{plot_w}" height="{top_h}" fill="#f7f7f7" stroke="#cccccc"/>',
        f'<text x="30" y="{top_y + 8}" font-family="monospace" font-size="12" fill="#333333">bbox history</text>',
        svg_polyline(scale_series(bbox_widths, x0=plot_x, y0=top_y, w=plot_w, h=top_h), color="#1f77b4"),
        svg_polyline(scale_series(bbox_heights, x0=plot_x, y0=top_y, w=plot_w, h=top_h), color="#2ca02c"),
        svg_polyline(scale_series(bbox_diags, x0=plot_x, y0=top_y, w=plot_w, h=top_h), color="#9467bd"),
        svg_polyline(scale_series(bbox_bottom_offsets, x0=plot_x, y0=top_y, w=plot_w, h=top_h), color="#ff7f0e"),
        '<text x="700" y="266" font-family="monospace" font-size="12" fill="#1f77b4">width</text>',
        '<text x="760" y="266" font-family="monospace" font-size="12" fill="#2ca02c">height</text>',
        '<text x="830" y="266" font-family="monospace" font-size="12" fill="#9467bd">diag</text>',
        '<text x="700" y="286" font-family="monospace" font-size="12" fill="#ff7f0e">bottom_offset</text>',
        f'<rect x="{plot_x}" y="{bottom_y}" width="{plot_w}" height="{bottom_h}" fill="#f7f7f7" stroke="#cccccc"/>',
        f'<text x="30" y="{bottom_y + 10}" font-family="monospace" font-size="12" fill="#333333">candidate TTC</text>',
    ]
    for cue_name, y in cue_y.items():
        parts.append(f'<text x="6" y="{y + 4}" font-family="monospace" font-size="10" fill="#555555">{escape(cue_name)}</text>')
        parts.append(f'<line x1="{plot_x}" y1="{y}" x2="{plot_x + plot_w}" y2="{y}" stroke="#dddddd"/>')

    cue_colors = {
        "width": "#1f77b4",
        "height": "#2ca02c",
        "diag": "#9467bd",
        "sqrt_area": "#8c564b",
        "bottom_offset": "#ff7f0e",
    }
    for candidate in candidates:
        x = x_for_ttc(candidate.ttc_s)
        y = cue_y.get(candidate.cue_name, bottom_y + 48)
        color = cue_colors.get(candidate.cue_name, "#333333")
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}" fill-opacity="0.58"/>')

    if row["gt_ttc_s"]:
        x = x_for_ttc(float(row["gt_ttc_s"]))
        parts.append(f'<line x1="{x:.2f}" y1="{bottom_y - 18}" x2="{x:.2f}" y2="{bottom_y + bottom_h}" stroke="#d62728" stroke-width="2"/>')
        parts.append(f'<text x="{x + 4:.2f}" y="{bottom_y - 22}" font-family="monospace" font-size="11" fill="#d62728">GT</text>')
    if row["pred_ttc_s"]:
        x = x_for_ttc(float(row["pred_ttc_s"]))
        parts.append(f'<line x1="{x:.2f}" y1="{bottom_y - 8}" x2="{x:.2f}" y2="{bottom_y + bottom_h}" stroke="#111111" stroke-width="2"/>')
        parts.append(f'<text x="{x + 4:.2f}" y="{bottom_y - 10}" font-family="monospace" font-size="11" fill="#111111">pred</text>')

    parts.extend(
        [
            f'<text x="{plot_x}" y="455" font-family="monospace" font-size="12" fill="#444444">0s</text>',
            f'<text x="{plot_x + plot_w - 50}" y="455" font-family="monospace" font-size="12" fill="#444444">{max_ttc_axis:.1f}s</text>',
            f'<text x="30" y="490" font-family="monospace" font-size="12" fill="#444444">worst-case plot: inspect whether GT is outside candidate mass, or whether candidate spread flags unreliable bbox geometry.</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.all_sequences:
        sequences = list_formal_sequences(args.formal_root)
    else:
        sequences = list(args.sequences or DEFAULT_FOCUS_SEQUENCES)
    if not sequences:
        raise SystemExit("No sequences selected.")

    config = BBoxLoomingConfigV1(
        history_back_labels=args.history_back,
        history_min_back_labels=args.history_min_back,
        max_ttc_s=args.max_ttc,
        fallback_to_lts_v1=False,
        cue_mode="multi_quantile",
        candidate_quantile=args.candidate_quantile,
        include_bottom_cue=True,
    )

    run_ts = datetime.now().strftime("BBoxLoomingV4FailureDiagnostics_%Y%m%d_%H%M%S")
    run_dir = args.output_root.resolve() / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)

    interval_rows: list[dict[str, object]] = []
    candidates_by_sample: dict[str, list[CandidateRow]] = {}
    history_by_sample: dict[str, list[BoundingBoxRecord]] = {}
    bottom_ref_by_sample: dict[str, float] = {}

    for sequence_id in sequences:
        with NativeSampleAdapter(sequence_id, formal_root=args.formal_root.resolve()) as adapter:
            bottom_reference_y = float(adapter.reader.calibration.intrinsics["cy"])
            bbox_by_label = {record.label_id: record for record in adapter.bbox_records}
            for bundle in adapter.iter_samples(include_events=False):
                sample = bundle.interval_sample
                if sample.gt_ttc_s is None and sample.status_gt != "OK":
                    row = interval_diagnostics_row(
                        sample_id=sample.sample_id,
                        sequence_id=sample.sequence_id,
                        interval_idx=sample.interval_idx,
                        gt_ttc_s=sample.gt_ttc_s,
                        status_gt=sample.status_gt,
                        pred_ttc_s=None,
                        status="GT_INVALID",
                        current_bbox=bbox_by_label.get(sample.start_label_id),
                        bottom_reference_y=bottom_reference_y,
                        candidates=[],
                    )
                    interval_rows.append(row)
                    continue

                candidates, status = collect_multiwindow_candidates(
                    bundle_records=bundle.bbox_records,
                    start_label_id=sample.start_label_id,
                    sample_id=sample.sample_id,
                    sequence_id=sample.sequence_id,
                    interval_idx=sample.interval_idx,
                    query_t_s=sample.query_t_s,
                    bottom_reference_y=bottom_reference_y,
                    config=config,
                )
                pred_ttc_s = quantile([candidate.ttc_s for candidate in candidates], config.candidate_quantile)
                candidates_by_sample[sample.sample_id] = candidates
                bottom_ref_by_sample[sample.sample_id] = bottom_reference_y
                history_lo = sample.start_label_id - config.history_back_labels
                history_by_sample[sample.sample_id] = [
                    record
                    for record in bundle.bbox_records
                    if history_lo <= record.label_id <= sample.start_label_id
                ]
                row = interval_diagnostics_row(
                    sample_id=sample.sample_id,
                    sequence_id=sample.sequence_id,
                    interval_idx=sample.interval_idx,
                    gt_ttc_s=sample.gt_ttc_s,
                    status_gt=sample.status_gt,
                    pred_ttc_s=pred_ttc_s,
                    status=status,
                    current_bbox=bbox_by_label.get(sample.start_label_id),
                    bottom_reference_y=bottom_reference_y,
                    candidates=candidates,
                )
                interval_rows.append(row)

    fieldnames = [
        "sample_id",
        "sequence_id",
        "interval_idx",
        "status_gt",
        "status",
        "gt_ttc_s",
        "pred_ttc_s",
        "abs_error_s",
        "e_ttc_pct",
        "pred_over_gt",
        "bbox_width_px",
        "bbox_height_px",
        "bbox_bottom_offset_px",
        "candidate_count",
        "candidate_max_fit_points",
        "candidate_min_history_back",
        "candidate_max_history_back",
        "candidate_min_s",
        "candidate_p10_s",
        "candidate_q25_s",
        "candidate_median_s",
        "candidate_q75_s",
        "candidate_p90_s",
        "candidate_max_s",
        "candidate_iqr_over_median",
        "candidate_p90_p10_over_median",
        "candidate_width_count",
        "candidate_height_count",
        "candidate_diag_count",
        "candidate_sqrt_area_count",
        "candidate_bottom_offset_count",
    ]
    write_csv(run_dir / "IntervalDiagnostics.csv", interval_rows, fieldnames)

    valid_rows = [row for row in interval_rows if row["e_ttc_pct"]]
    worst_rows = sorted(valid_rows, key=lambda item: float(item["e_ttc_pct"]), reverse=True)[: args.top_n]
    write_csv(run_dir / "WorstIntervals.csv", worst_rows, fieldnames)

    candidate_rows: list[dict[str, object]] = []
    for row in worst_rows:
        candidate_rows.extend(candidate_rows_to_dicts(candidates_by_sample.get(str(row["sample_id"]), [])))
    write_csv(
        run_dir / "WorstIntervalCandidates.csv",
        candidate_rows,
        [
            "sample_id",
            "sequence_id",
            "interval_idx",
            "cue_name",
            "history_back",
            "fit_points",
            "ttc_s",
            "slope_px_per_s",
            "ref_px",
            "normalized_residual",
            "relative_slope",
        ],
    )

    summary_rows: list[dict[str, object]] = []
    for sequence_id in sequences:
        seq_rows = [row for row in interval_rows if row["sequence_id"] == sequence_id]
        seq_valid = [row for row in seq_rows if row["e_ttc_pct"]]
        errors = [float(row["e_ttc_pct"]) for row in seq_valid]
        summary_rows.append(
            {
                "sequence_id": sequence_id,
                "interval_count": len(seq_rows),
                "valid_prediction_count": len(seq_valid),
                "mean_e_ttc_pct": format_number(safe_mean(errors)),
                "median_e_ttc_pct": format_number(quantile(errors, 0.50)),
                "p90_e_ttc_pct": format_number(quantile(errors, 0.90)),
                "over50_count": sum(1 for value in errors if value > 50.0),
                "mean_candidate_count": format_number(
                    safe_mean([float(row["candidate_count"]) for row in seq_valid])
                ),
                "mean_candidate_max_fit_points": format_number(
                    safe_mean([float(row["candidate_max_fit_points"]) for row in seq_valid])
                ),
                "mean_candidate_iqr_over_median": format_number(
                    safe_mean(
                        [
                            float(row["candidate_iqr_over_median"])
                            for row in seq_valid
                            if row["candidate_iqr_over_median"]
                        ]
                    )
                ),
            }
        )
    write_csv(
        run_dir / "SequenceDiagnostics.csv",
        summary_rows,
        [
            "sequence_id",
            "interval_count",
            "valid_prediction_count",
            "mean_e_ttc_pct",
            "median_e_ttc_pct",
            "p90_e_ttc_pct",
            "over50_count",
            "mean_candidate_count",
            "mean_candidate_max_fit_points",
            "mean_candidate_iqr_over_median",
        ],
    )

    plot_rows = worst_rows[: args.plot_top_n]
    for row in plot_rows:
        sample_id = str(row["sample_id"])
        render_interval_svg(
            path=run_dir / "plots" / f"{sample_id}.svg",
            row=row,
            history_records=history_by_sample.get(sample_id, []),
            candidates=candidates_by_sample.get(sample_id, []),
            bottom_reference_y=bottom_ref_by_sample.get(sample_id, 0.0),
        )

    overall_errors = [float(row["e_ttc_pct"]) for row in valid_rows]
    payload = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "variant": "BBoxLoomingV4FailureDiagnostics",
        "formal_root": str(args.formal_root.resolve()),
        "sequences": sequences,
        "config": asdict(config),
        "interval_count": len(interval_rows),
        "valid_prediction_count": len(valid_rows),
        "mean_e_ttc_pct": safe_mean(overall_errors),
        "median_e_ttc_pct": quantile(overall_errors, 0.50),
        "p90_e_ttc_pct": quantile(overall_errors, 0.90),
        "top_n": args.top_n,
        "plot_top_n": args.plot_top_n,
    }
    (run_dir / "Config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# BBoxLoomingV4 Failure Diagnostics",
        "",
        f"- run_dir: `{run_dir}`",
        f"- sequences: `{', '.join(sequences)}`",
        f"- interval_count: `{len(interval_rows)}`",
        f"- valid_prediction_count: `{len(valid_rows)}`",
        f"- mean_e_ttc_pct: `{format_number(payload['mean_e_ttc_pct'], digits=3)}`",
        f"- median_e_ttc_pct: `{format_number(payload['median_e_ttc_pct'], digits=3)}`",
        f"- p90_e_ttc_pct: `{format_number(payload['p90_e_ttc_pct'], digits=3)}`",
        "",
        "## Files",
        "",
        "- `IntervalDiagnostics.csv`",
        "- `SequenceDiagnostics.csv`",
        "- `WorstIntervals.csv`",
        "- `WorstIntervalCandidates.csv`",
        "- `plots/*.svg`",
        "",
        "## Worst Intervals",
        "",
        "| sample | sequence | gt | pred | e_ttc_pct | candidates | iqr/median |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in worst_rows[:20]:
        lines.append(
            f"| `{row['sample_id']}` | `{row['sequence_id']}` | {row['gt_ttc_s']} | "
            f"{row['pred_ttc_s']} | {row['e_ttc_pct']} | {row['candidate_count']} | "
            f"{row['candidate_iqr_over_median']} |"
        )
    lines.append("")
    (run_dir / "Summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[Done] run_dir: {run_dir}")
    print(f"[Done] summary: {run_dir / 'Summary.md'}")


if __name__ == "__main__":
    main()
