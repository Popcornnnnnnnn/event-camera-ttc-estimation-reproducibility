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
from evttc.sparsity_aware_lts import SparsityAwareLtsConfigV1, predict_ttc_v1, predict_ttc_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SparsityAwareLTS V1 and V2 on formal EvTTC sequences.",
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
        default=CODE_ROOT / "Experiments" / "SparsityAwareLTSCompare",
        help="Root directory for comparison outputs.",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional sequence IDs. Default: all formal sequences.",
    )
    parser.add_argument(
        "--max-intervals",
        type=int,
        default=20,
        help="Per-sequence interval cap for the comparison run. Use 0 or a negative value for all intervals.",
    )
    args = parser.parse_args()
    if args.max_intervals is not None and args.max_intervals <= 0:
        args.max_intervals = None
    return args


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


def summarize_debug_v1(debug: object) -> dict[str, float | None]:
    local_cues = getattr(debug, "local_cues", ())
    valid_ttc = [float(cue.ttc_s) for cue in local_cues if cue.status == "OK" and cue.ttc_s is not None]
    if not valid_ttc:
        return {
            "cue_ttc_median_s": None,
            "cue_ttc_iqr_s": None,
            "cue_ttc_normalized_iqr": None,
            "cue_ttc_min_s": None,
            "cue_ttc_max_s": None,
        }
    cue_array = np.asarray(valid_ttc, dtype=np.float64)
    cue_median = float(np.median(cue_array))
    cue_iqr = float(np.percentile(cue_array, 75) - np.percentile(cue_array, 25))
    return {
        "cue_ttc_median_s": cue_median,
        "cue_ttc_iqr_s": cue_iqr,
        "cue_ttc_normalized_iqr": float(cue_iqr / max(abs(cue_median), 1e-6)),
        "cue_ttc_min_s": float(cue_array.min()),
        "cue_ttc_max_s": float(cue_array.max()),
    }


def summarize_debug_v2(debug: object) -> dict[str, float | None]:
    local_cues = getattr(debug, "local_cues", ())
    valid_cues = [cue for cue in local_cues if cue.status == "OK" and cue.ttc_s is not None]
    valid_ttc = [float(cue.ttc_s) for cue in valid_cues]
    quality_values = [float(cue.cue_quality) for cue in valid_cues if cue.cue_quality is not None]
    if not valid_ttc:
        return {
            "cue_ttc_median_s": None,
            "cue_ttc_iqr_s": None,
            "cue_ttc_normalized_iqr": None,
            "cue_ttc_min_s": None,
            "cue_ttc_max_s": None,
            "cue_quality_mean": None,
            "cue_quality_iqr": None,
            "cue_quality_min": None,
            "cue_quality_max": None,
        }
    cue_array = np.asarray(valid_ttc, dtype=np.float64)
    cue_median = float(np.median(cue_array))
    cue_iqr = float(np.percentile(cue_array, 75) - np.percentile(cue_array, 25))
    metrics: dict[str, float | None] = {
        "cue_ttc_median_s": cue_median,
        "cue_ttc_iqr_s": cue_iqr,
        "cue_ttc_normalized_iqr": float(cue_iqr / max(abs(cue_median), 1e-6)),
        "cue_ttc_min_s": float(cue_array.min()),
        "cue_ttc_max_s": float(cue_array.max()),
        "cue_quality_mean": None,
        "cue_quality_iqr": None,
        "cue_quality_min": None,
        "cue_quality_max": None,
    }
    if quality_values:
        quality_array = np.asarray(quality_values, dtype=np.float64)
        metrics["cue_quality_mean"] = float(np.mean(quality_array))
        metrics["cue_quality_iqr"] = float(np.percentile(quality_array, 75) - np.percentile(quality_array, 25))
        metrics["cue_quality_min"] = float(quality_array.min())
        metrics["cue_quality_max"] = float(quality_array.max())
    return metrics


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    gt_valid = sum(1 for row in rows if row["gt_ttc_s"] not in (None, ""))
    est_valid_rows = [row for row in rows if row["ttc_est_s"] not in (None, "")]
    fallback_count = sum(1 for row in rows if str(row["status"]) == "OK_FALLBACK")
    blend_count = sum(1 for row in rows if str(row["status"]) == "OK_BLEND")
    cue_only_count = sum(1 for row in rows if str(row["status"]) == "OK_CUE_ONLY")
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
        "fallback_count": fallback_count,
        "blend_count": blend_count,
        "cue_only_count": cue_only_count,
        "mae_s": mae_s,
        "e_ttc_pct": e_ttc_pct,
        "cost_time_s_mean": cost_time_s_mean,
    }


def compare_records(v1: dict[str, object], v2: dict[str, object]) -> dict[str, object]:
    delta_ttc_s = None
    delta_e_ttc_pct = None
    if v1["ttc_est_s"] is not None and v2["ttc_est_s"] is not None:
        delta_ttc_s = float(v2["ttc_est_s"]) - float(v1["ttc_est_s"])
    if v1["e_ttc_pct"] is not None and v2["e_ttc_pct"] is not None:
        delta_e_ttc_pct = float(v2["e_ttc_pct"]) - float(v1["e_ttc_pct"])
    return {
        "sample_id": v1["sample_id"],
        "sequence_id": v1["sequence_id"],
        "interval_idx": v1["interval_idx"],
        "gt_ttc_s": v1["gt_ttc_s"],
        "v1_status": v1["status"],
        "v1_ttc_est_s": v1["ttc_est_s"],
        "v1_e_ttc_pct": v1["e_ttc_pct"],
        "v1_local_cue_valid_count": v1["local_cue_valid_count"],
        "v1_fallback_ttc_s": v1["fallback_ttc_s"],
        "v1_cue_ttc_median_s": v1.get("cue_ttc_median_s"),
        "v1_cue_ttc_iqr_s": v1.get("cue_ttc_iqr_s"),
        "v1_cue_ttc_normalized_iqr": v1.get("cue_ttc_normalized_iqr"),
        "v1_cue_ttc_min_s": v1.get("cue_ttc_min_s"),
        "v1_cue_ttc_max_s": v1.get("cue_ttc_max_s"),
        "v2_status": v2["status"],
        "v2_ttc_est_s": v2["ttc_est_s"],
        "v2_e_ttc_pct": v2["e_ttc_pct"],
        "v2_local_cue_valid_count": v2["local_cue_valid_count"],
        "v2_fallback_ttc_s": v2["fallback_ttc_s"],
        "v2_cue_ttc_median_s": v2.get("cue_ttc_median_s"),
        "v2_cue_ttc_iqr_s": v2.get("cue_ttc_iqr_s"),
        "v2_cue_ttc_normalized_iqr": v2.get("cue_ttc_normalized_iqr"),
        "v2_cue_ttc_min_s": v2.get("cue_ttc_min_s"),
        "v2_cue_ttc_max_s": v2.get("cue_ttc_max_s"),
        "v2_cue_quality_mean": v2.get("cue_quality_mean"),
        "v2_cue_quality_iqr": v2.get("cue_quality_iqr"),
        "v2_cue_quality_min": v2.get("cue_quality_min"),
        "v2_cue_quality_max": v2.get("cue_quality_max"),
        "delta_ttc_s": delta_ttc_s,
        "delta_e_ttc_pct": delta_e_ttc_pct,
    }


def main() -> None:
    args = parse_args()
    sequences = args.sequences or list_formal_sequences(args.formal_root)
    if not sequences:
        raise SystemExit(f"No formal sequences found under {args.formal_root}")

    config = SparsityAwareLtsConfigV1()
    run_ts = datetime.now().strftime("CompareV1V2_%Y%m%d_%H%M%S")
    run_dir = args.output_root.resolve() / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")

    all_v1_rows: list[dict[str, object]] = []
    all_v2_rows: list[dict[str, object]] = []
    all_compare_rows: list[dict[str, object]] = []
    sequence_summaries: list[dict[str, object]] = []

    for sequence_id in sequences:
        v1_rows: list[dict[str, object]] = []
        v2_rows: list[dict[str, object]] = []
        compare_rows: list[dict[str, object]] = []
        with NativeSampleAdapter(sequence_id, formal_root=args.formal_root.resolve()) as adapter:
            for idx, bundle in enumerate(adapter.iter_samples(include_events=True)):
                if args.max_intervals is not None and idx >= args.max_intervals:
                    break
                record_v1, debug_v1 = predict_ttc_v1(bundle, config=config)
                record_v2, debug_v2 = predict_ttc_v2(bundle, config=config)
                row_v1 = asdict(record_v1)
                row_v2 = asdict(record_v2)
                row_v1.update(summarize_debug_v1(debug_v1))
                row_v2.update(summarize_debug_v2(debug_v2))
                v1_rows.append(row_v1)
                v2_rows.append(row_v2)
                compare_rows.append(compare_records(row_v1, row_v2))

        seq_dir = run_dir / sequence_id
        write_csv(
            seq_dir / "V1IntervalMetrics.csv",
            v1_rows,
            fieldnames=list(v1_rows[0].keys()) if v1_rows else [],
        )
        write_csv(
            seq_dir / "V2IntervalMetrics.csv",
            v2_rows,
            fieldnames=list(v2_rows[0].keys()) if v2_rows else [],
        )
        write_csv(
            seq_dir / "CompareIntervalMetrics.csv",
            compare_rows,
            fieldnames=list(compare_rows[0].keys()) if compare_rows else [],
        )

        seq_summary = {
            "sequence_id": sequence_id,
            "v1": summarize_rows(v1_rows),
            "v2": summarize_rows(v2_rows),
        }
        sequence_summaries.append(seq_summary)
        all_v1_rows.extend(v1_rows)
        all_v2_rows.extend(v2_rows)
        all_compare_rows.extend(compare_rows)
        print(
            f"[Done] {sequence_id}: "
            f"v1_fallback={seq_summary['v1']['fallback_count']} "
            f"v2_fallback={seq_summary['v2']['fallback_count']} "
            f"v1_e_ttc_pct={format_number(seq_summary['v1']['e_ttc_pct'], 3)} "
            f"v2_e_ttc_pct={format_number(seq_summary['v2']['e_ttc_pct'], 3)}"
        )

    overall = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "method_family": "SparsityAwareLTSCompare",
        "variant": run_ts,
        "variant_label": "CompareV1V2",
        "formal_root": str(args.formal_root.resolve()),
        "max_intervals": args.max_intervals,
        "sequence_summaries": sequence_summaries,
        "v1": summarize_rows(all_v1_rows),
        "v2": summarize_rows(all_v2_rows),
    }

    write_csv(
        run_dir / "CompareIntervalMetrics.csv",
        all_compare_rows,
        fieldnames=list(all_compare_rows[0].keys()) if all_compare_rows else [],
    )
    (run_dir / "Summary.json").write_text(
        json.dumps(overall, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        f"# SparsityAwareLTS Compare / {run_ts}",
        "",
        f"- started_at: `{overall['started_at']}`",
        f"- finished_at: `{overall['finished_at']}`",
        f"- max_intervals_per_sequence: `{args.max_intervals}`",
        "",
        "| sequence | v1_fallback | v2_fallback | v1_mae_s | v2_mae_s | v1_e_ttc_pct | v2_e_ttc_pct |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sequence_summaries:
        lines.append(
            f"| {item['sequence_id']} | {item['v1']['fallback_count']} | {item['v2']['fallback_count']} | "
            f"{format_number(item['v1']['mae_s'])} | {format_number(item['v2']['mae_s'])} | "
            f"{format_number(item['v1']['e_ttc_pct'], 3)} | {format_number(item['v2']['e_ttc_pct'], 3)} |"
        )
    lines.extend(
        [
            "",
            f"- overall_v1_fallback_count: `{overall['v1']['fallback_count']}`",
            f"- overall_v2_fallback_count: `{overall['v2']['fallback_count']}`",
            f"- overall_v2_blend_count: `{overall['v2']['blend_count']}`",
            f"- overall_v2_cue_only_count: `{overall['v2']['cue_only_count']}`",
            f"- overall_v1_e_ttc_pct: `{format_number(overall['v1']['e_ttc_pct'], 3)}`",
            f"- overall_v2_e_ttc_pct: `{format_number(overall['v2']['e_ttc_pct'], 3)}`",
            "",
        ]
    )
    (run_dir / "Summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[Done] run_dir: {run_dir}")
    print(f"[Done] summary: {run_dir / 'Summary.md'}")


if __name__ == "__main__":
    main()
