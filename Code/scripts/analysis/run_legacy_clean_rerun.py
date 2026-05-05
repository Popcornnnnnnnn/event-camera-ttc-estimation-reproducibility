#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evttc import NativeSampleAdapter, NativeSampleBundle, list_formal_sequences


DEFAULT_CMAX_CORE = Path(
    os.environ.get(
        "EVTTC_CMAX_CORE_PATH",
        "external/cmax-repro-resume/Code/scripts/baselines/cmax_aligned_core.py",
    )
)
DEFAULT_STRTTC_DIR = Path(
    os.environ.get(
        "EVTTC_STRTTC_BASELINES_DIR",
        "external/strttc-quality-aware/Code/scripts/baselines",
    )
)


@dataclass(frozen=True)
class LegacyBBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center_x(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def center_y(self) -> float:
        return 0.5 * (self.y1 + self.y2)


@dataclass(frozen=True)
class LegacyIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-run legacy CMax/STRTTC variants on the current EvTTC formal protocol. "
            "This runner intentionally uses current formal_loader/native_adapter and only "
            "loads legacy algorithm kernels from read-only worktrees."
        )
    )
    parser.add_argument("--formal-root", type=Path, default=CODE_ROOT / "DatasetFormal")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CODE_ROOT / "Experiments" / "LegacyCleanReruns",
    )
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--max-intervals", type=int, default=5, help="Per-sequence cap; use 0 for all intervals.")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["cmax", "strttc-baseline", "strttc-quality-aware"],
        choices=["cmax", "strttc-baseline", "strttc-quality-aware"],
    )
    parser.add_argument("--cmax-core-path", type=Path, default=DEFAULT_CMAX_CORE)
    parser.add_argument("--strttc-baselines-dir", type=Path, default=DEFAULT_STRTTC_DIR)
    parser.add_argument("--roi-pad-px", type=int, default=24)
    parser.add_argument("--max-events-per-sample", type=int, default=50000)
    parser.add_argument("--min-ttc", type=float, default=0.1)
    parser.add_argument("--max-ttc", type=float, default=3.5)
    parser.add_argument("--cmax-coarse-grid", type=int, default=21)
    parser.add_argument("--cmax-refine", action="store_true")
    parser.add_argument("--strttc-window-size", type=int, default=8)
    parser.add_argument("--strttc-max-points", type=int, default=220)
    parser.add_argument("--strttc-robust-c", type=float, default=0.04)
    parser.add_argument("--strttc-quality-topk", type=int, default=96)
    parser.add_argument("--strttc-quality-near-contact-bbox-area", type=float, default=120000.0)
    parser.add_argument(
        "--strttc-time-direction",
        choices=["forward", "reverse"],
        default="forward",
        help="Pass formal event timestamps as-is or reversed relative time to STRTTC.",
    )
    parser.add_argument(
        "--strttc-polarity-source",
        choices=["as-is", "flipped"],
        default="as-is",
        help="Pass formal polarity as-is or flipped to STRTTC.",
    )
    parser.add_argument(
        "--strttc-negative-ttc-policy",
        choices=["reject", "abs"],
        default="reject",
        help="Reject negative linear TTC or take abs() as a sign-normalized legacy comparison.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_module_from_path(name: str, path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"Missing legacy module: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_strttc_runner(baselines_dir: Path) -> ModuleType:
    if str(baselines_dir) not in sys.path:
        sys.path.insert(0, str(baselines_dir))
    return load_module_from_path(
        "legacy_strttc_quality_runner",
        baselines_dir / "run_p0_aligned_baselines.py",
    )


def format_number(value: float | None, digits: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def clamp_bbox(bundle: NativeSampleBundle, pad_px: int) -> tuple[LegacyBBox, np.ndarray]:
    sample = bundle.interval_sample
    resolution = bundle.calibration.resolution
    width = int(resolution["width"])
    height = int(resolution["height"])
    x1 = max(0, int(math.floor(sample.bbox_x1)) - pad_px)
    y1 = max(0, int(math.floor(sample.bbox_y1)) - pad_px)
    x2 = min(width - 1, int(math.ceil(sample.bbox_x2)) + pad_px)
    y2 = min(height - 1, int(math.ceil(sample.bbox_y2)) + pad_px)
    bbox = LegacyBBox(
        x1=int(round(sample.bbox_x1)),
        y1=int(round(sample.bbox_y1)),
        x2=int(round(sample.bbox_x2)),
        y2=int(round(sample.bbox_y2)),
    )
    if bundle.event_slice is None:
        return bbox, np.zeros((0,), dtype=bool)
    x = bundle.event_slice.x
    y = bundle.event_slice.y
    keep = (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
    return bbox, keep


def intrinsics_from_bundle(bundle: NativeSampleBundle) -> LegacyIntrinsics:
    intrinsics = bundle.calibration.intrinsics
    resolution = bundle.calibration.resolution
    return LegacyIntrinsics(
        fx=float(intrinsics["fx"]),
        fy=float(intrinsics["fy"]),
        cx=float(intrinsics["cx"]),
        cy=float(intrinsics["cy"]),
        width=int(resolution["width"]),
        height=int(resolution["height"]),
    )


def event_arrays(
    bundle: NativeSampleBundle,
    *,
    keep: np.ndarray,
    max_events: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if bundle.event_slice is None or keep.size == 0:
        empty = np.zeros((0,), dtype=np.float64)
        return empty, empty, empty, empty
    idx = np.nonzero(keep)[0]
    if max_events > 0 and idx.size > max_events:
        idx = idx[-max_events:]
    event_slice = bundle.event_slice
    return (
        event_slice.x[idx].astype(np.float64),
        event_slice.y[idx].astype(np.float64),
        event_slice.t_s[idx].astype(np.float64),
        event_slice.p[idx],
    )


def valid_prediction(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0.0)


def build_strttc_args(args: argparse.Namespace, mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        min_ttc=args.min_ttc,
        max_ttc=args.max_ttc,
        strttc_window_size=args.strttc_window_size,
        strttc_max_points=args.strttc_max_points,
        strttc_robust_c=args.strttc_robust_c,
        strttc_event_num_percent=0.5,
        strttc_plane_ransac_iters=160,
        strttc_plane_ransac_threshold=1e-3,
        strttc_plane_ransac_best_inlier_ratio=0.6,
        strttc_flow_threshold=1e-6,
        strttc_linear_ransac_iters=240,
        strttc_linear_ransac_threshold=1e-3,
        strttc_linear_ransac_best_inlier_ratio=0.4,
        strttc_normal_flow_backend="timestamp_grad",
        strttc_normal_flow_grad_threshold=1e-6,
        strttc_normal_flow_min_points=20,
        strttc_event_selection_mode="formal_interval_roi",
        strttc_quality_aware_mode=mode,
        strttc_quality_topk=args.strttc_quality_topk,
        strttc_quality_near_contact_bbox_area=args.strttc_quality_near_contact_bbox_area,
    )


def predict_cmax(
    module: ModuleType,
    bundle: NativeSampleBundle,
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]]:
    bbox, keep = clamp_bbox(bundle, args.roi_pad_px)
    x, y, t_s, p = event_arrays(bundle, keep=keep, max_events=args.max_events_per_sample)
    if x.size < 150:
        return math.nan, {"status_detail": "too_few_roi_events", "roi_event_count": int(x.size)}
    ttc_s, score, model = module.estimate_cmax_ttc_affine_with_model(
        x=x,
        y=y,
        t_s=t_s,
        event_weights=None,
        bbox=bbox,
        intrinsics=intrinsics_from_bundle(bundle),
        min_ttc=args.min_ttc,
        max_ttc=args.max_ttc,
        coarse_grid=args.cmax_coarse_grid,
        patch_pad=args.roi_pad_px,
        candidate_mode="uniform_ttc",
        score_mode="variance",
        roi_mode="full_bbox",
        refine=bool(args.cmax_refine),
        refine_maxiter=20,
    )
    return ttc_s, {
        "status_detail": "ok" if valid_prediction(ttc_s) else "invalid_cmax_estimate",
        "roi_event_count": int(x.size),
        "score": score,
        "model": None if model is None else [float(item) for item in model.tolist()],
    }


def predict_strttc(
    module: ModuleType,
    bundle: NativeSampleBundle,
    args: argparse.Namespace,
    *,
    quality_aware: bool,
    rng: np.random.Generator,
) -> tuple[float, dict[str, Any]]:
    bbox, keep = clamp_bbox(bundle, args.roi_pad_px)
    x, y, t_s, p = event_arrays(bundle, keep=keep, max_events=args.max_events_per_sample)
    if x.size < 300:
        return math.nan, {"status_detail": "too_few_roi_events", "roi_event_count": int(x.size)}
    if args.strttc_time_direction == "reverse":
        t_s = -t_s
    if args.strttc_polarity_source == "flipped":
        p = 1 - p
    mode = "linear" if quality_aware else "off"
    if quality_aware and bundle.interval_sample.bbox_area >= args.strttc_quality_near_contact_bbox_area:
        mode = "off"
    strttc_args = build_strttc_args(args, mode=mode)
    intr = intrinsics_from_bundle(bundle)
    camera_intrinsics = module.CameraIntrinsics(
        fx=intr.fx,
        fy=intr.fy,
        cx=intr.cx,
        cy=intr.cy,
        width=intr.width,
        height=intr.height,
    )
    ttc_s, nflow_points, inlier_ratio, linear_ttc_s, diagnostics = module.estimate_strttc_ttc(
        x=x,
        y=y,
        t_s=t_s,
        p=p,
        intrinsics=camera_intrinsics,
        undistort_map=None,
        args=strttc_args,
        rng=rng,
        normal_flow_backend="timestamp_grad",
        quality_mode_override=mode,
    )
    status_detail = "ok" if valid_prediction(ttc_s) else "invalid_strttc_estimate"
    if not valid_prediction(ttc_s) and args.strttc_negative_ttc_policy == "abs":
        if np.isfinite(linear_ttc_s) and linear_ttc_s != 0.0:
            abs_ttc_s = abs(float(linear_ttc_s))
            if args.min_ttc <= abs_ttc_s <= args.max_ttc:
                ttc_s = abs_ttc_s
                status_detail = "ok_sign_abs"
    return ttc_s, {
        "status_detail": status_detail,
        "roi_event_count": int(x.size),
        "quality_mode": mode,
        "negative_ttc_policy": args.strttc_negative_ttc_policy,
        "nflow_points": int(nflow_points),
        "inlier_ratio": float(inlier_ratio),
        "linear_ttc_s": float(linear_ttc_s),
        "valid_point_count": int(getattr(diagnostics, "valid_point_count", 0)),
        "full_nflow_points": int(getattr(diagnostics, "full_nflow_points", 0)),
        "selected_nflow_points": int(getattr(diagnostics, "selected_nflow_points", 0)),
        "selected_nflow_ratio": float(getattr(diagnostics, "selected_nflow_ratio", math.nan)),
        "selected_flow_quality_mean": float(
            getattr(diagnostics, "selected_flow_quality_mean", math.nan)
        ),
        "full_flow_quality_mean": float(getattr(diagnostics, "full_flow_quality_mean", math.nan)),
    }


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    gt_valid = sum(1 for row in rows if row["gt_ttc_s"] not in (None, ""))
    est_valid_rows = [row for row in rows if row["ttc_est_s"] not in (None, "")]
    failure_count = sum(1 for row in rows if not str(row["status"]).startswith("OK"))
    mae_s = None
    e_ttc_pct = None
    cost_time_s_mean = None
    if est_valid_rows:
        gt = np.asarray([float(row["gt_ttc_s"]) for row in est_valid_rows], dtype=np.float64)
        pred = np.asarray([float(row["ttc_est_s"]) for row in est_valid_rows], dtype=np.float64)
        cost = np.asarray([float(row["cost_time_s"]) for row in est_valid_rows], dtype=np.float64)
        mae_s = float(np.mean(np.abs(pred - gt)))
        e_ttc_pct = float(np.mean(np.abs(pred - gt) / gt * 100.0))
        cost_time_s_mean = float(np.mean(cost))
    return {
        "gt_valid_count": gt_valid,
        "est_valid_count": len(est_valid_rows),
        "failure_count": failure_count,
        "mae_s": mae_s,
        "e_ttc_pct": e_ttc_pct,
        "cost_time_s_mean": cost_time_s_mean,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_method(
    *,
    method: str,
    sequences: list[str],
    args: argparse.Namespace,
    cmax_module: ModuleType | None,
    strttc_module: ModuleType | None,
    run_dir: Path,
) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    per_sequence: list[dict[str, object]] = []

    for sequence_id in sequences:
        seq_rows: list[dict[str, object]] = []
        with NativeSampleAdapter(sequence_id, formal_root=args.formal_root.resolve()) as adapter:
            for idx, bundle in enumerate(adapter.iter_samples(include_events=True)):
                if args.max_intervals is not None and idx >= args.max_intervals:
                    break
                sample = bundle.interval_sample
                ttc_est_s = math.nan
                debug: dict[str, Any] = {}
                status = "GT_INVALID"
                started = time.perf_counter()
                if sample.has_valid_gt:
                    try:
                        if method == "cmax":
                            if cmax_module is None:
                                raise RuntimeError("CMax module was not loaded")
                            ttc_est_s, debug = predict_cmax(cmax_module, bundle, args)
                        elif method == "strttc-baseline":
                            if strttc_module is None:
                                raise RuntimeError("STRTTC module was not loaded")
                            ttc_est_s, debug = predict_strttc(
                                strttc_module,
                                bundle,
                                args,
                                quality_aware=False,
                                rng=rng,
                            )
                        elif method == "strttc-quality-aware":
                            if strttc_module is None:
                                raise RuntimeError("STRTTC module was not loaded")
                            ttc_est_s, debug = predict_strttc(
                                strttc_module,
                                bundle,
                                args,
                                quality_aware=True,
                                rng=rng,
                            )
                        else:
                            raise ValueError(f"Unsupported method: {method}")
                        status = "OK" if valid_prediction(ttc_est_s) else "METHOD_INVALID"
                    except Exception as exc:  # keep smoke/full runs from dying on one interval
                        status = "RUNTIME_FAIL"
                        debug = {"exception": f"{type(exc).__name__}: {exc}"}
                        ttc_est_s = math.nan
                cost_time_s = time.perf_counter() - started
                gt_ttc_s = sample.gt_ttc_s if sample.has_valid_gt else None
                est_valid = valid_prediction(ttc_est_s) and gt_ttc_s is not None
                e_ttc_pct = (
                    abs(float(ttc_est_s) - float(gt_ttc_s)) / float(gt_ttc_s) * 100.0
                    if est_valid
                    else None
                )
                row = {
                    "sample_id": sample.sample_id,
                    "sequence_id": sample.sequence_id,
                    "interval_idx": sample.interval_idx,
                    "timestamp_s": sample.query_t_s,
                    "gt_ttc_s": gt_ttc_s,
                    "ttc_est_s": float(ttc_est_s) if valid_prediction(ttc_est_s) else None,
                    "e_ttc_pct": e_ttc_pct,
                    "status": status,
                    "cost_time_s": cost_time_s,
                    "debug_ref": json.dumps(debug, ensure_ascii=False, sort_keys=True),
                }
                seq_rows.append(row)
                rows.append(row)

        seq_dir = run_dir / method / sequence_id
        seq_summary = {
            "sequence_id": sequence_id,
            "method_family": "LegacyCleanRerun",
            "variant": method,
            **summarize_rows(seq_rows),
        }
        per_sequence.append(seq_summary)
        write_csv(seq_dir / "IntervalMetrics.csv", seq_rows, FIELDNAMES)
        (seq_dir / "Summary.json").write_text(
            json.dumps(seq_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (seq_dir / "Summary.md").write_text(summary_markdown(method, seq_summary), encoding="utf-8")
        print(
            f"[Done] {method} / {sequence_id}: est_valid={seq_summary['est_valid_count']} "
            f"mae={format_number(seq_summary['mae_s'])} "
            f"e_ttc_pct={format_number(seq_summary['e_ttc_pct'], digits=3)}"
        )

    method_summary = {
        "method_family": "LegacyCleanRerun",
        "variant": method,
        "sequence_count": len(sequences),
        "sequence_summaries": per_sequence,
        **summarize_rows(rows),
    }
    method_dir = run_dir / method
    write_csv(method_dir / "IntervalMetrics.csv", rows, FIELDNAMES)
    (method_dir / "Summary.json").write_text(
        json.dumps(method_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (method_dir / "Summary.md").write_text(
        summary_markdown(method, method_summary, per_sequence=per_sequence),
        encoding="utf-8",
    )
    return method_summary


FIELDNAMES = [
    "sample_id",
    "sequence_id",
    "interval_idx",
    "timestamp_s",
    "gt_ttc_s",
    "ttc_est_s",
    "e_ttc_pct",
    "status",
    "cost_time_s",
    "debug_ref",
]


def summary_markdown(
    method: str,
    summary: dict[str, object],
    *,
    per_sequence: list[dict[str, object]] | None = None,
) -> str:
    lines = [
        f"# LegacyCleanRerun / {method}",
        "",
        f"- gt_valid_count: `{summary['gt_valid_count']}`",
        f"- est_valid_count: `{summary['est_valid_count']}`",
        f"- failure_count: `{summary['failure_count']}`",
        f"- mae_s: `{format_number(summary['mae_s'])}`",
        f"- e_ttc_pct: `{format_number(summary['e_ttc_pct'], digits=3)}`",
        f"- cost_time_s_mean: `{format_number(summary['cost_time_s_mean'])}`",
        "",
    ]
    if per_sequence:
        lines.extend(
            [
                "| sequence | gt_valid | est_valid | failures | mae_s | e_ttc_pct(%) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for item in per_sequence:
            lines.append(
                f"| {item['sequence_id']} | {item['gt_valid_count']} | "
                f"{item['est_valid_count']} | {item['failure_count']} | "
                f"{format_number(item['mae_s'])} | {format_number(item['e_ttc_pct'], digits=3)} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.max_intervals is not None and args.max_intervals <= 0:
        args.max_intervals = None
    sequences = args.sequences or list_formal_sequences(args.formal_root)
    if not sequences:
        raise SystemExit(f"No formal sequences found under {args.formal_root}")

    cmax_module = load_module_from_path("legacy_cmax_aligned_core", args.cmax_core_path)
    strttc_module = load_strttc_runner(args.strttc_baselines_dir)

    is_full_run = args.max_intervals is None
    prefix = "LegacyCleanFull" if is_full_run else "LegacyCleanSmoke"
    report_title = "Legacy Clean Full Rerun" if is_full_run else "Legacy Clean Rerun Smoke"
    run_id = datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")
    run_dir = args.output_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for method in args.methods:
        summaries.append(
            run_method(
                method=method,
                sequences=sequences,
                args=args,
                cmax_module=cmax_module,
                strttc_module=strttc_module,
                run_dir=run_dir,
            )
        )

    config = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "formal_root": str(args.formal_root.resolve()),
        "sequences": sequences,
        "max_intervals": args.max_intervals,
        "methods": args.methods,
        "cmax_core_path": str(args.cmax_core_path),
        "strttc_baselines_dir": str(args.strttc_baselines_dir),
        "roi_pad_px": args.roi_pad_px,
        "max_events_per_sample": args.max_events_per_sample,
        "min_ttc": args.min_ttc,
        "max_ttc": args.max_ttc,
        "cmax_coarse_grid": args.cmax_coarse_grid,
        "cmax_refine": args.cmax_refine,
        "strttc_window_size": args.strttc_window_size,
        "strttc_max_points": args.strttc_max_points,
        "strttc_robust_c": args.strttc_robust_c,
        "strttc_quality_topk": args.strttc_quality_topk,
        "strttc_quality_near_contact_bbox_area": args.strttc_quality_near_contact_bbox_area,
        "strttc_time_direction": args.strttc_time_direction,
        "strttc_polarity_source": args.strttc_polarity_source,
        "strttc_negative_ttc_policy": args.strttc_negative_ttc_policy,
        "seed": args.seed,
    }
    (run_dir / "Config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "Summary.json").write_text(
        json.dumps({"config": config, "method_summaries": summaries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [f"# {report_title}", ""]
    lines.append(f"- run_dir: `{run_dir}`")
    lines.append(f"- sequences: `{', '.join(sequences)}`")
    lines.append(f"- max_intervals: `{args.max_intervals}`")
    lines.append("")
    lines.extend(
        [
            "| method | sequence_count | gt_valid | est_valid | failure | mae_s | e_ttc_pct |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summaries:
        lines.append(
            f"| {item['variant']} | {item['sequence_count']} | {item['gt_valid_count']} | "
            f"{item['est_valid_count']} | {item['failure_count']} | "
            f"{format_number(item['mae_s'])} | {format_number(item['e_ttc_pct'], digits=3)} |"
        )
    lines.append("")
    (run_dir / "Summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[Done] run_dir: {run_dir}")
    print(f"[Done] summary: {run_dir / 'Summary.md'}")


if __name__ == "__main__":
    main()
