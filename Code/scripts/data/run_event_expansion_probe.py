from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evttc.formal_loader import list_formal_sequences
from evttc.native_adapter import NativeSampleAdapter, NativeSampleBundle


DEFAULT_FOCUS_SEQUENCES = ["CPNAO-medium", "CPNA-low", "CPNAO-low", "CPLA-medium"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe a naive event-cloud expansion TTC cue against current bbox looming results.",
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
        help="Root directory for diagnostic outputs.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=CODE_ROOT
        / "Experiments"
        / "SparsityAwareLTS"
        / "BBoxLoomingV7_20260427_170415"
        / "IntervalMetrics.csv",
        help="Existing IntervalMetrics.csv to compare against.",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional sequence IDs. Default: focused remaining-failure sequences.",
    )
    parser.add_argument(
        "--all-sequences",
        action="store_true",
        help="Run on all formal sequences instead of the default focus set.",
    )
    parser.add_argument("--bins", type=int, default=4, help="Temporal bins used for event cloud fitting.")
    parser.add_argument("--min-events-per-bin", type=int, default=12, help="Minimum ROI events per bin.")
    return parser.parse_args()


def fit_line(times_s: list[float], values: list[float]) -> tuple[float, float] | None:
    if len(times_s) < 3:
        return None
    x = np.asarray(times_s, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denom = float(np.square(x - x_mean).sum())
    if denom < 1e-12:
        return None
    slope = float(((x - x_mean) * (y - y_mean)).sum() / denom)
    intercept = float(y_mean - slope * x_mean)
    return slope, intercept


def event_expansion_candidates(
    bundle: NativeSampleBundle,
    *,
    bins: int,
    min_events_per_bin: int,
) -> list[dict[str, object]]:
    event_slice = bundle.event_slice
    sample = bundle.interval_sample
    if event_slice is None or event_slice.x.size < bins * min_events_per_bin:
        return []

    x = event_slice.x.astype(np.float64)
    y = event_slice.y.astype(np.float64)
    t_s = event_slice.t_us.astype(np.float64) / 1e6
    inside = (
        (x >= sample.bbox_x1)
        & (x <= sample.bbox_x2)
        & (y >= sample.bbox_y1)
        & (y <= sample.bbox_y2)
    )
    x = x[inside]
    y = y[inside]
    t_s = t_s[inside]
    if x.size < bins * min_events_per_bin:
        return []

    edges = np.linspace(sample.t_start_s, sample.t_end_s, bins + 1)
    cx = 0.5 * (sample.bbox_x1 + sample.bbox_x2)
    cy = 0.5 * (sample.bbox_y1 + sample.bbox_y2)
    bin_rows: list[dict[str, float]] = []
    for idx in range(bins):
        if idx == bins - 1:
            mask = (t_s >= edges[idx]) & (t_s <= edges[idx + 1])
        else:
            mask = (t_s >= edges[idx]) & (t_s < edges[idx + 1])
        if int(mask.sum()) < min_events_per_bin:
            continue
        x_bin = x[mask]
        y_bin = y[mask]
        t_bin = t_s[mask]
        radius = np.sqrt(np.square(x_bin - cx) + np.square(y_bin - cy))
        bin_rows.append(
            {
                "t_s": float(np.median(t_bin)),
                "r70": float(np.percentile(radius, 70)),
                "r80": float(np.percentile(radius, 80)),
                "r90": float(np.percentile(radius, 90)),
                "w90": float(np.percentile(x_bin, 95) - np.percentile(x_bin, 5)),
                "h90": float(np.percentile(y_bin, 95) - np.percentile(y_bin, 5)),
            }
        )
    if len(bin_rows) < 3:
        return []

    candidates: list[dict[str, object]] = []
    times = [row["t_s"] for row in bin_rows]
    for cue_name in ["r70", "r80", "r90", "w90", "h90"]:
        values = [row[cue_name] for row in bin_rows]
        fit = fit_line(times, values)
        if fit is None:
            continue
        slope, intercept = fit
        ref = slope * sample.query_t_s + intercept
        if slope <= 1.0 or ref <= 3.0:
            continue
        ttc_s = ref / slope
        if not math.isfinite(ttc_s) or ttc_s < 0.05 or ttc_s > 16.0:
            continue
        candidates.append(
            {
                "cue_name": cue_name,
                "ttc_s": float(ttc_s),
                "slope_px_per_s": float(slope),
                "ref_px": float(ref),
                "bin_count": len(bin_rows),
                "roi_event_count": int(x.size),
            }
        )
    return candidates


def load_compare_metrics(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_errors(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def main() -> None:
    args = parse_args()
    if args.all_sequences:
        sequences = list_formal_sequences(args.formal_root)
    else:
        sequences = args.sequences or DEFAULT_FOCUS_SEQUENCES
    compare_metrics = load_compare_metrics(args.metrics_csv.resolve())

    run_ts = datetime.now().strftime("EventExpansionProbe_%Y%m%d_%H%M%S")
    run_dir = args.output_root.resolve() / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for sequence_id in sequences:
        with NativeSampleAdapter(sequence_id, formal_root=args.formal_root.resolve()) as adapter:
            for bundle in adapter.iter_samples(include_events=True):
                sample = bundle.interval_sample
                if sample.gt_ttc_s is None or sample.status_gt != "OK":
                    continue
                candidates = event_expansion_candidates(
                    bundle,
                    bins=args.bins,
                    min_events_per_bin=args.min_events_per_bin,
                )
                if not candidates:
                    continue
                event_values = np.asarray([float(item["ttc_s"]) for item in candidates], dtype=np.float64)
                event_pred = float(np.median(event_values))
                event_error = abs(event_pred - sample.gt_ttc_s) / sample.gt_ttc_s * 100.0
                compare_row = compare_metrics.get(bundle.sample_id, {})
                compare_pred = compare_row.get("ttc_est_s") or ""
                compare_error = compare_row.get("e_ttc_pct") or ""
                row = {
                    "sample_id": bundle.sample_id,
                    "sequence_id": sequence_id,
                    "interval_idx": sample.interval_idx,
                    "gt_ttc_s": sample.gt_ttc_s,
                    "event_ttc_s": event_pred,
                    "event_e_ttc_pct": event_error,
                    "event_candidate_count": len(candidates),
                    "event_candidate_min_s": float(event_values.min()),
                    "event_candidate_median_s": float(np.median(event_values)),
                    "event_candidate_max_s": float(event_values.max()),
                    "compare_ttc_s": compare_pred,
                    "compare_e_ttc_pct": compare_error,
                    "compare_status": compare_row.get("status", ""),
                }
                rows.append(row)
                for candidate in candidates:
                    candidate_rows.append(
                        {
                            "sample_id": bundle.sample_id,
                            "sequence_id": sequence_id,
                            "interval_idx": sample.interval_idx,
                            **candidate,
                        }
                    )

    metric_fields = [
        "sample_id",
        "sequence_id",
        "interval_idx",
        "gt_ttc_s",
        "event_ttc_s",
        "event_e_ttc_pct",
        "event_candidate_count",
        "event_candidate_min_s",
        "event_candidate_median_s",
        "event_candidate_max_s",
        "compare_ttc_s",
        "compare_e_ttc_pct",
        "compare_status",
    ]
    write_csv(run_dir / "EventExpansionMetrics.csv", rows, metric_fields)
    write_csv(
        run_dir / "EventExpansionCandidates.csv",
        candidate_rows,
        [
            "sample_id",
            "sequence_id",
            "interval_idx",
            "cue_name",
            "ttc_s",
            "slope_px_per_s",
            "ref_px",
            "bin_count",
            "roi_event_count",
        ],
    )

    event_errors = [float(row["event_e_ttc_pct"]) for row in rows]
    compare_errors = [
        float(row["compare_e_ttc_pct"])
        for row in rows
        if row["compare_e_ttc_pct"] not in ("", None)
    ]
    summary = {
        "run_dir": str(run_dir),
        "sequences": sequences,
        "metrics_csv": str(args.metrics_csv.resolve()),
        "bins": args.bins,
        "min_events_per_bin": args.min_events_per_bin,
        "event": summarize_errors(event_errors),
        "compare_same_samples": summarize_errors(compare_errors),
    }
    (run_dir / "Summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Event Expansion Probe",
        "",
        f"- run_dir: `{run_dir}`",
        f"- sequences: `{', '.join(sequences)}`",
        f"- metrics_csv: `{args.metrics_csv.resolve()}`",
        f"- event_count: `{summary['event']['count']}`",
        f"- event_mean_e_ttc_pct: `{summary['event']['mean']:.3f}`" if summary["event"]["mean"] is not None else "- event_mean_e_ttc_pct: `N/A`",
        f"- event_median_e_ttc_pct: `{summary['event']['median']:.3f}`" if summary["event"]["median"] is not None else "- event_median_e_ttc_pct: `N/A`",
        f"- event_p90_e_ttc_pct: `{summary['event']['p90']:.3f}`" if summary["event"]["p90"] is not None else "- event_p90_e_ttc_pct: `N/A`",
        f"- compare_same_sample_mean_e_ttc_pct: `{summary['compare_same_samples']['mean']:.3f}`"
        if summary["compare_same_samples"]["mean"] is not None
        else "- compare_same_sample_mean_e_ttc_pct: `N/A`",
        "",
        "## Files",
        "",
        "- `EventExpansionMetrics.csv`",
        "- `EventExpansionCandidates.csv`",
        "- `Summary.json`",
        "",
    ]
    (run_dir / "Summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[Done] run_dir: {run_dir}")
    print(f"[Done] summary: {run_dir / 'Summary.md'}")


if __name__ == "__main__":
    main()
