#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import sys

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evttc import NativeSampleAdapter, list_formal_sequences
from evttc.sparsity_aware_lts import (
    BBoxLoomingConfigV1,
    SparsityAwareLtsConfigV1,
    predict_ttc_bbox_looming_v1,
    predict_ttc_v1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the first end-to-end SparsityAwareLTS evaluation on formal EvTTC sequences.",
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
        default=CODE_ROOT / "Experiments" / "SparsityAwareLTS",
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional sequence IDs. Default: all formal sequences.",
    )
    parser.add_argument("--max-intervals", type=int, default=None, help="Optional cap per sequence.")
    parser.add_argument("--tau-ms", type=float, default=20.0, help="Tau used by LTS.")
    parser.add_argument("--grid-rows", type=int, default=3, help="Rows used by LocalStats.")
    parser.add_argument("--grid-cols", type=int, default=3, help="Cols used by LocalStats.")
    bbox_variants = (
        "BBoxLoomingV1",
        "BBoxLoomingV2",
        "BBoxLoomingV3",
        "BBoxLoomingV4",
        "BBoxLoomingV5",
        "BBoxLoomingV6",
        "BBoxLoomingV7",
        "BBoxLoomingV8",
        "BBoxLoomingV9",
        "BBoxLoomingV10",
        "BBoxLoomingV11",
        "BBoxLoomingV12",
        "BBoxLoomingV13",
    )
    parser.add_argument(
        "--variant",
        choices=("HeuristicCueV1", *bbox_variants),
        default="HeuristicCueV1",
        help="Self-developed traditional algorithm variant to evaluate.",
    )
    parser.add_argument(
        "--bbox-history-back",
        type=int,
        default=None,
        help="Number of previous bbox labels used by BBoxLooming. Default: 7 for V1/V2, 12 for V3+.",
    )
    parser.add_argument(
        "--bbox-history-min-back",
        type=int,
        default=2,
        help="Minimum previous bbox labels used by BBoxLoomingV3 multi-window candidates.",
    )
    parser.add_argument(
        "--bbox-quantile",
        type=float,
        default=0.25,
        help="Candidate TTC quantile used by BBoxLoomingV3.",
    )
    parser.add_argument(
        "--bbox-max-ttc",
        type=float,
        default=None,
        help="Maximum TTC candidate horizon for BBoxLooming. Default: 15s for V3, 16s for V4+, 20s otherwise.",
    )
    parser.add_argument(
        "--disable-lts-fallback",
        action="store_true",
        help="For BBoxLooming variants, do not fall back to HeuristicCueV1 when bbox expansion is invalid.",
    )
    parser.add_argument(
        "--disable-bbox-bottom-cue",
        action="store_true",
        help="For BBoxLoomingV4+, disable the bottom-y motion cue.",
    )
    parser.add_argument(
        "--bbox-quality-min-candidates",
        type=int,
        default=None,
        help="For BBoxLoomingV5+, minimum accepted multi-window candidate count. Default: 30.",
    )
    parser.add_argument(
        "--bbox-quality-min-fit-points",
        type=int,
        default=None,
        help="For BBoxLoomingV5+, minimum maximum real bbox fit points across candidates. Default: 6.",
    )
    parser.add_argument(
        "--bbox-quality-max-iqr",
        type=float,
        default=None,
        help="For BBoxLoomingV5+, maximum candidate IQR/median dispersion. Default: 0.7.",
    )
    parser.add_argument(
        "--bbox-cue-mode",
        choices=("width", "highest_rel_slope", "lowest_residual", "score_rel_slope_over_resid", "multi_quantile"),
        default=None,
        help="BBox looming cue selector. Default: width for V1, highest_rel_slope for V2, multi_quantile for V3+.",
    )
    parser.add_argument(
        "--selection-threshold",
        type=float,
        default=0.55,
        help="Threshold used by LocalConfidenceV1.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def build_config(args: argparse.Namespace) -> SparsityAwareLtsConfigV1:
    cfg = SparsityAwareLtsConfigV1(
        tau_ms=args.tau_ms,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
    )
    bbox_history_back = args.bbox_history_back
    if bbox_history_back is None:
        bbox_history_back = (
            12
            if args.variant
            in {
                "BBoxLoomingV3",
                "BBoxLoomingV4",
                "BBoxLoomingV5",
                "BBoxLoomingV6",
                "BBoxLoomingV7",
                "BBoxLoomingV8",
                "BBoxLoomingV9",
                "BBoxLoomingV10",
                "BBoxLoomingV11",
                "BBoxLoomingV12",
                "BBoxLoomingV13",
            }
            else 7
        )
    bbox_max_ttc = args.bbox_max_ttc
    if bbox_max_ttc is None:
        if args.variant in {
            "BBoxLoomingV4",
            "BBoxLoomingV5",
            "BBoxLoomingV6",
            "BBoxLoomingV7",
            "BBoxLoomingV8",
            "BBoxLoomingV9",
            "BBoxLoomingV10",
            "BBoxLoomingV11",
            "BBoxLoomingV12",
            "BBoxLoomingV13",
        }:
            bbox_max_ttc = 16.0
        elif args.variant == "BBoxLoomingV3":
            bbox_max_ttc = 15.0
        else:
            bbox_max_ttc = cfg.bbox_looming.max_ttc_s
    bbox_cue_mode = args.bbox_cue_mode
    if bbox_cue_mode is None:
        if args.variant in {
            "BBoxLoomingV3",
            "BBoxLoomingV4",
            "BBoxLoomingV5",
            "BBoxLoomingV6",
            "BBoxLoomingV7",
            "BBoxLoomingV8",
            "BBoxLoomingV9",
            "BBoxLoomingV10",
            "BBoxLoomingV11",
            "BBoxLoomingV12",
            "BBoxLoomingV13",
        }:
            bbox_cue_mode = "multi_quantile"
        elif args.variant == "BBoxLoomingV2":
            bbox_cue_mode = "highest_rel_slope"
        else:
            bbox_cue_mode = "width"
    quality_gate_enabled = args.variant in {
        "BBoxLoomingV5",
        "BBoxLoomingV6",
        "BBoxLoomingV7",
        "BBoxLoomingV8",
        "BBoxLoomingV9",
        "BBoxLoomingV10",
        "BBoxLoomingV11",
        "BBoxLoomingV12",
        "BBoxLoomingV13",
    }
    quality_min_candidates = (
        30 if args.bbox_quality_min_candidates is None else args.bbox_quality_min_candidates
    )
    quality_min_fit_points = 6 if args.bbox_quality_min_fit_points is None else args.bbox_quality_min_fit_points
    quality_max_iqr = 0.7 if args.bbox_quality_max_iqr is None else args.bbox_quality_max_iqr
    # dataclass is frozen; use replace-style reconstruction
    return SparsityAwareLtsConfigV1(
        tau_ms=cfg.tau_ms,
        grid_rows=cfg.grid_rows,
        grid_cols=cfg.grid_cols,
        pad_px=cfg.pad_px,
        reference_time=cfg.reference_time,
        confidence=cfg.confidence,
        cue=cfg.cue,
        aggregation=type(cfg.aggregation)(
            selection_threshold=args.selection_threshold,
            min_selected_cells=cfg.aggregation.min_selected_cells,
            min_valid_local_cues=cfg.aggregation.min_valid_local_cues,
            blend_with_fallback=cfg.aggregation.blend_with_fallback,
            fallback_blend_weight=cfg.aggregation.fallback_blend_weight,
        ),
        fallback=cfg.fallback,
        bbox_looming=BBoxLoomingConfigV1(
            history_back_labels=bbox_history_back,
            history_min_back_labels=args.bbox_history_min_back,
            max_ttc_s=bbox_max_ttc,
            fallback_to_lts_v1=not args.disable_lts_fallback,
            cue_mode=bbox_cue_mode,
            candidate_quantile=args.bbox_quantile,
            include_bottom_cue=args.variant in {
                "BBoxLoomingV4",
                "BBoxLoomingV5",
                "BBoxLoomingV6",
                "BBoxLoomingV7",
                "BBoxLoomingV8",
                "BBoxLoomingV9",
                "BBoxLoomingV10",
                "BBoxLoomingV11",
                "BBoxLoomingV12",
                "BBoxLoomingV13",
            }
            and not args.disable_bbox_bottom_cue,
            quality_gate_enabled=quality_gate_enabled,
            quality_min_candidate_count=quality_min_candidates,
            quality_min_fit_points=quality_min_fit_points,
            quality_max_iqr_over_median=quality_max_iqr,
            quality_low_confidence_strategy="adaptive_quantile"
            if args.variant == "BBoxLoomingV6"
            else "adaptive_quantile_v2"
            if args.variant
            in {
                "BBoxLoomingV7",
                "BBoxLoomingV8",
                "BBoxLoomingV9",
                "BBoxLoomingV10",
                "BBoxLoomingV11",
                "BBoxLoomingV12",
                "BBoxLoomingV13",
            }
            else "reject",
            quality_reject_low_confidence_after_adaptive=args.variant
            in {"BBoxLoomingV11", "BBoxLoomingV12", "BBoxLoomingV13"},
            current_bbox_collapse_guard_enabled=args.variant
            in {
                "BBoxLoomingV8",
                "BBoxLoomingV9",
                "BBoxLoomingV10",
                "BBoxLoomingV11",
                "BBoxLoomingV12",
                "BBoxLoomingV13",
            },
            history_bbox_collapse_guard_enabled=args.variant
            in {"BBoxLoomingV9", "BBoxLoomingV10", "BBoxLoomingV11", "BBoxLoomingV12", "BBoxLoomingV13"},
            history_bbox_collapse_guard_require_low_confidence=args.variant
            in {"BBoxLoomingV10", "BBoxLoomingV11", "BBoxLoomingV12", "BBoxLoomingV13"},
            phase_quantile_enabled=args.variant in {"BBoxLoomingV12", "BBoxLoomingV13"},
            phase_growth_quantile_enabled=args.variant == "BBoxLoomingV13",
            phase_stable_quantile_enabled=args.variant == "BBoxLoomingV13",
        ),
        interface_version=cfg.interface_version,
        representation_version=cfg.representation_version,
    )


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


def main() -> None:
    args = parse_args()
    sequences = args.sequences or list_formal_sequences(args.formal_root)
    if not sequences:
        raise SystemExit(f"No formal sequences found under {args.formal_root}")

    config = build_config(args)
    run_ts = datetime.now().strftime(f"{args.variant}_%Y%m%d_%H%M%S")
    run_dir = args.output_root.resolve() / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")

    combined_rows: list[dict[str, object]] = []
    per_sequence_summaries: list[dict[str, object]] = []

    for sequence_id in sequences:
        needs_events = args.variant == "HeuristicCueV1" or (
            args.variant.startswith("BBoxLooming") and config.bbox_looming.fallback_to_lts_v1
        )
        with NativeSampleAdapter(sequence_id, formal_root=args.formal_root.resolve()) as adapter:
            rows: list[dict[str, object]] = []
            for idx, bundle in enumerate(adapter.iter_samples(include_events=needs_events)):
                if args.max_intervals is not None and idx >= args.max_intervals:
                    break
                if args.variant.startswith("BBoxLooming"):
                    record, _ = predict_ttc_bbox_looming_v1(bundle, config=config)
                else:
                    record, _ = predict_ttc_v1(bundle, config=config)
                rows.append(asdict(record))

        seq_dir = run_dir / sequence_id
        seq_summary = {
            "sequence_id": sequence_id,
            "method_family": "SparsityAwareLTS",
            "variant": args.variant,
            "interface_version": config.interface_version,
            "representation_version": config.representation_version,
            **summarize_rows(rows),
        }
        per_sequence_summaries.append(seq_summary)
        combined_rows.extend(rows)

        write_csv(
            seq_dir / "IntervalMetrics.csv",
            rows,
            fieldnames=[
                "sample_id",
                "sequence_id",
                "interval_idx",
                "timestamp_s",
                "gt_ttc_s",
                "ttc_est_s",
                "e_ttc_pct",
                "confidence",
                "status",
                "cost_time_s",
                "local_cue_valid_count",
                "selected_cell_count",
                "fallback_ttc_s",
                "debug_ref",
            ],
        )
        (seq_dir / "Summary.json").write_text(
            json.dumps(seq_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        seq_lines = [
            f"# {sequence_id} / {args.variant}",
            "",
            f"- gt_valid_count: `{seq_summary['gt_valid_count']}`",
            f"- est_valid_count: `{seq_summary['est_valid_count']}`",
            f"- failure_count: `{seq_summary['failure_count']}`",
            f"- mae_s: `{format_number(seq_summary['mae_s'])}`",
            f"- e_ttc_pct: `{format_number(seq_summary['e_ttc_pct'], digits=3)}`",
            f"- cost_time_s_mean: `{format_number(seq_summary['cost_time_s_mean'])}`",
            "",
        ]
        (seq_dir / "Summary.md").write_text("\n".join(seq_lines), encoding="utf-8")
        print(
            f"[Done] {sequence_id}: est_valid={seq_summary['est_valid_count']} "
            f"mae={format_number(seq_summary['mae_s'])} "
            f"e_ttc_pct={format_number(seq_summary['e_ttc_pct'], digits=3)}"
        )

    combined_summary = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "method_family": "SparsityAwareLTS",
        "variant": run_ts,
        "variant_label": args.variant,
        "formal_root": str(args.formal_root.resolve()),
        "interface_version": config.interface_version,
        "representation_version": config.representation_version,
        "config": {
            "tau_ms": config.tau_ms,
            "grid_rows": config.grid_rows,
            "grid_cols": config.grid_cols,
            "selection_threshold": config.aggregation.selection_threshold,
            "fallback_blend_weight": config.aggregation.fallback_blend_weight,
            "bbox_history_back_labels": config.bbox_looming.history_back_labels,
            "bbox_history_min_back_labels": config.bbox_looming.history_min_back_labels,
            "bbox_max_ttc_s": config.bbox_looming.max_ttc_s,
            "bbox_fallback_to_lts_v1": config.bbox_looming.fallback_to_lts_v1,
            "bbox_cue_mode": config.bbox_looming.cue_mode,
            "bbox_candidate_quantile": config.bbox_looming.candidate_quantile,
            "bbox_include_bottom_cue": config.bbox_looming.include_bottom_cue,
            "bbox_bottom_reference": config.bbox_looming.bottom_reference,
            "bbox_bottom_reference_fraction": config.bbox_looming.bottom_reference_fraction,
            "bbox_quality_gate_enabled": config.bbox_looming.quality_gate_enabled,
            "bbox_quality_min_candidate_count": config.bbox_looming.quality_min_candidate_count,
            "bbox_quality_min_fit_points": config.bbox_looming.quality_min_fit_points,
            "bbox_quality_max_iqr_over_median": config.bbox_looming.quality_max_iqr_over_median,
            "bbox_quality_low_confidence_strategy": config.bbox_looming.quality_low_confidence_strategy,
            "bbox_quality_low_candidate_quantile": config.bbox_looming.quality_low_candidate_quantile,
            "bbox_quality_low_fit_points_quantile": config.bbox_looming.quality_low_fit_points_quantile,
            "bbox_quality_high_dispersion_quantile": config.bbox_looming.quality_high_dispersion_quantile,
            "bbox_quality_low_candidate_spike_min_ttc_s": config.bbox_looming.quality_low_candidate_spike_min_ttc_s,
            "bbox_quality_low_candidate_spike_q25_over_min_max": (
                config.bbox_looming.quality_low_candidate_spike_q25_over_min_max
            ),
            "bbox_quality_low_candidate_spike_quantile": config.bbox_looming.quality_low_candidate_spike_quantile,
            "bbox_quality_high_dispersion_short_min_ttc_s": (
                config.bbox_looming.quality_high_dispersion_short_min_ttc_s
            ),
            "bbox_quality_high_dispersion_q25_over_min_min": (
                config.bbox_looming.quality_high_dispersion_q25_over_min_min
            ),
            "bbox_quality_high_dispersion_short_quantile": (
                config.bbox_looming.quality_high_dispersion_short_quantile
            ),
            "bbox_quality_reject_low_confidence_after_adaptive": (
                config.bbox_looming.quality_reject_low_confidence_after_adaptive
            ),
            "bbox_current_bbox_collapse_guard_enabled": (
                config.bbox_looming.current_bbox_collapse_guard_enabled
            ),
            "bbox_current_bbox_collapse_history_back": (
                config.bbox_looming.current_bbox_collapse_history_back
            ),
            "bbox_current_bbox_collapse_min_history": (
                config.bbox_looming.current_bbox_collapse_min_history
            ),
            "bbox_current_bbox_collapse_area_ratio_min": (
                config.bbox_looming.current_bbox_collapse_area_ratio_min
            ),
            "bbox_history_bbox_collapse_guard_enabled": (
                config.bbox_looming.history_bbox_collapse_guard_enabled
            ),
            "bbox_history_bbox_collapse_neighbor_back": (
                config.bbox_looming.history_bbox_collapse_neighbor_back
            ),
            "bbox_history_bbox_collapse_neighbor_forward": (
                config.bbox_looming.history_bbox_collapse_neighbor_forward
            ),
            "bbox_history_bbox_collapse_area_ratio_min": (
                config.bbox_looming.history_bbox_collapse_area_ratio_min
            ),
            "bbox_history_bbox_collapse_guard_require_low_confidence": (
                config.bbox_looming.history_bbox_collapse_guard_require_low_confidence
            ),
            "bbox_phase_quantile_enabled": config.bbox_looming.phase_quantile_enabled,
            "bbox_phase_early_fraction_max": config.bbox_looming.phase_early_fraction_max,
            "bbox_phase_early_iqr_over_median_min": (
                config.bbox_looming.phase_early_iqr_over_median_min
            ),
            "bbox_phase_early_spread_over_median_min": (
                config.bbox_looming.phase_early_spread_over_median_min
            ),
            "bbox_phase_early_quantile": config.bbox_looming.phase_early_quantile,
            "bbox_phase_growth_quantile_enabled": (
                config.bbox_looming.phase_growth_quantile_enabled
            ),
            "bbox_phase_growth_fraction_max": config.bbox_looming.phase_growth_fraction_max,
            "bbox_phase_growth_area_ratio_med5_min": (
                config.bbox_looming.phase_growth_area_ratio_med5_min
            ),
            "bbox_phase_growth_iqr_over_median_min": (
                config.bbox_looming.phase_growth_iqr_over_median_min
            ),
            "bbox_phase_growth_quantile": config.bbox_looming.phase_growth_quantile,
            "bbox_phase_stable_quantile_enabled": (
                config.bbox_looming.phase_stable_quantile_enabled
            ),
            "bbox_phase_stable_fraction_min": config.bbox_looming.phase_stable_fraction_min,
            "bbox_phase_stable_area_ratio_med5_max": (
                config.bbox_looming.phase_stable_area_ratio_med5_max
            ),
            "bbox_phase_stable_iqr_over_median_max": (
                config.bbox_looming.phase_stable_iqr_over_median_max
            ),
            "bbox_phase_stable_quantile": config.bbox_looming.phase_stable_quantile,
            "bbox_phase_late_fraction_min": config.bbox_looming.phase_late_fraction_min,
            "bbox_phase_late_iqr_over_median_max": (
                config.bbox_looming.phase_late_iqr_over_median_max
            ),
            "bbox_phase_late_area_ratio_med5_max": (
                config.bbox_looming.phase_late_area_ratio_med5_max
            ),
            "bbox_phase_late_quantile": config.bbox_looming.phase_late_quantile,
        },
        "sequence_summaries": per_sequence_summaries,
        **summarize_rows(combined_rows),
    }

    write_csv(
        run_dir / "IntervalMetrics.csv",
        combined_rows,
        fieldnames=[
            "sample_id",
            "sequence_id",
            "interval_idx",
            "timestamp_s",
            "gt_ttc_s",
            "ttc_est_s",
            "e_ttc_pct",
            "confidence",
            "status",
            "cost_time_s",
            "local_cue_valid_count",
            "selected_cell_count",
            "fallback_ttc_s",
            "debug_ref",
        ],
    )
    (run_dir / "Config.json").write_text(
        json.dumps(
            {
                "started_at": started_at,
                "formal_root": str(args.formal_root.resolve()),
                "sequences": sequences,
                "max_intervals": args.max_intervals,
                "method_family": "SparsityAwareLTS",
                "variant_label": args.variant,
                "config": combined_summary["config"],
                "interface_version": config.interface_version,
                "representation_version": config.representation_version,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "Summary.json").write_text(
        json.dumps(combined_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        f"# SparsityAwareLTS / {run_ts}",
        "",
        f"- started_at: `{combined_summary['started_at']}`",
        f"- finished_at: `{combined_summary['finished_at']}`",
        f"- gt_valid_count: `{combined_summary['gt_valid_count']}`",
        f"- est_valid_count: `{combined_summary['est_valid_count']}`",
        f"- failure_count: `{combined_summary['failure_count']}`",
        f"- mae_s: `{format_number(combined_summary['mae_s'])}`",
        f"- e_ttc_pct: `{format_number(combined_summary['e_ttc_pct'], digits=3)}`",
        f"- cost_time_s_mean: `{format_number(combined_summary['cost_time_s_mean'])}`",
        "",
        "| sequence | gt_valid | est_valid | failures | mae_s | e_ttc_pct(%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in per_sequence_summaries:
        lines.append(
            f"| {item['sequence_id']} | {item['gt_valid_count']} | {item['est_valid_count']} | "
            f"{item['failure_count']} | {format_number(item['mae_s'])} | "
            f"{format_number(item['e_ttc_pct'], digits=3)} |"
        )
    lines.append("")
    (run_dir / "Summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[Done] run_dir: {run_dir}")
    print(f"[Done] summary: {run_dir / 'Summary.md'}")


if __name__ == "__main__":
    main()
