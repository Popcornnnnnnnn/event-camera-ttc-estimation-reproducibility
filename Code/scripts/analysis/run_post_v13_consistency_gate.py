#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRONTIER_ROOT = CODE_ROOT / "Experiments" / "PostV13Frontier"
DEFAULT_FEATURE_TABLE = DEFAULT_FRONTIER_ROOT / "V13FrontierLGBM_full_v2" / "FeatureTable.csv"
BASE_MODEL_COLUMNS = (
    "lightgbm_l1_log_direct_pred_ttc_s",
    "lightgbm_l2_log_direct_pred_ttc_s",
    "lightgbm_huber_log_direct_pred_ttc_s",
    "lightgbm_fair_log_direct_pred_ttc_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine post-V13 sequence-consistency prediction smoothing with the "
            "confidence gate to search for higher <7% coverage."
        )
    )
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FRONTIER_ROOT)
    parser.add_argument("--run-name", default="V13ConsistencyGate_centerW5_targetSweep_v1")
    parser.add_argument("--windows", nargs="+", type=int, default=(3, 5))
    parser.add_argument("--modes", nargs="+", default=("center", "causal"))
    parser.add_argument("--targets", nargs="+", type=float, default=(6.7, 6.8, 6.9, 7.0, 7.1))
    parser.add_argument("--keep-frac-min", type=float, default=0.45)
    parser.add_argument("--keep-frac-max", type=float, default=0.90)
    parser.add_argument("--keep-frac-steps", type=int, default=31)
    parser.add_argument("--pred-min-s", type=float, default=0.05)
    parser.add_argument("--pred-max-s", type=float, default=25.0)
    parser.add_argument("--min-train-kept", type=int, default=100)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_float(value: Any) -> float:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def finite_with_median(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float64, copy=True)
    median = float(np.nanmedian(out)) if np.any(np.isfinite(out)) else 0.0
    out[~np.isfinite(out)] = median
    return out


def metric_summary(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    valid = mask & np.isfinite(gt) & (gt > 0.0) & np.isfinite(pred) & (pred > 0.0)
    if not np.any(valid):
        return {
            "count": 0,
            "mae_s": None,
            "e_ttc_pct": None,
            "median_e_ttc_pct": None,
            "p90_e_ttc_pct": None,
            "over50": 0,
            "over100": 0,
        }
    errors = np.abs(pred[valid] - gt[valid]) / gt[valid] * 100.0
    return {
        "count": int(np.sum(valid)),
        "mae_s": float(np.mean(np.abs(pred[valid] - gt[valid]))),
        "e_ttc_pct": float(np.mean(errors)),
        "median_e_ttc_pct": float(np.median(errors)),
        "p90_e_ttc_pct": float(np.percentile(errors, 90)),
        "over50": int(np.sum(errors > 50.0)),
        "over100": int(np.sum(errors > 100.0)),
    }


def interval_key(row: dict[str, str]) -> int:
    value = parse_float(row.get("interval_idx"))
    return int(value) if math.isfinite(value) else 0


def rolling_prediction(
    rows: list[dict[str, str]],
    source_pred: np.ndarray,
    *,
    window: int,
    mode: str,
    pred_min_s: float,
    pred_max_s: float,
) -> np.ndarray:
    by_sequence: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_sequence[row.get("sequence_id", "")].append((idx, row))
    out = np.full(source_pred.shape, np.nan, dtype=np.float64)
    radius = window // 2
    for items in by_sequence.values():
        items.sort(key=lambda item: interval_key(item[1]))
        indices = [idx for idx, _ in items]
        values = source_pred[indices]
        for pos, idx in enumerate(indices):
            if mode == "center":
                lo = max(0, pos - radius)
                hi = min(len(indices), pos + radius + 1)
            elif mode == "causal":
                lo = max(0, pos - window + 1)
                hi = pos + 1
            else:
                raise ValueError(f"Unknown mode: {mode}")
            local = values[lo:hi]
            local = local[np.isfinite(local)]
            out[idx] = float(np.median(local)) if local.size else float("nan")
    return np.clip(out, pred_min_s, pred_max_s)


def select_two_feature_gate(
    *,
    train_mask: np.ndarray,
    errors: np.ndarray,
    feature_a: np.ndarray,
    feature_b: np.ndarray,
    target_e_ttc_pct: float,
    keep_fracs: np.ndarray,
    min_train_kept: int,
) -> dict[str, Any]:
    train_feature_a = feature_a[train_mask]
    train_feature_b = feature_b[train_mask]
    train_errors = errors[train_mask]
    thresholds_a = np.quantile(train_feature_a, keep_fracs)
    thresholds_b = np.quantile(train_feature_b, keep_fracs)
    keep_a = train_feature_a[None, :] <= thresholds_a[:, None]
    keep_b = train_feature_b[None, :] <= thresholds_b[:, None]
    count_grid = keep_a.astype(np.float64) @ keep_b.T.astype(np.float64)
    error_grid = (keep_a.astype(np.float64) * train_errors[None, :]) @ keep_b.T.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        e_grid = error_grid / count_grid
    valid_grid = count_grid >= float(min_train_kept)
    if not np.any(valid_grid):
        raise RuntimeError("No valid gate choice.")

    feasible_grid = valid_grid & (e_grid <= float(target_e_ttc_pct))
    if np.any(feasible_grid):
        score_grid = np.where(feasible_grid, count_grid, -1.0)
        index = np.unravel_index(int(np.argmax(score_grid)), score_grid.shape)
    else:
        score_grid = np.where(valid_grid, e_grid, np.inf)
        index = np.unravel_index(int(np.argmin(score_grid)), score_grid.shape)
    keep_idx_a, keep_idx_b = int(index[0]), int(index[1])
    return {
        "keep_frac_a": float(keep_fracs[keep_idx_a]),
        "keep_frac_b": float(keep_fracs[keep_idx_b]),
        "threshold_a": float(thresholds_a[keep_idx_a]),
        "threshold_b": float(thresholds_b[keep_idx_b]),
        "train_count": int(count_grid[index]),
        "train_e_ttc_pct": float(e_grid[index]),
    }


def run_gate(
    *,
    rows: list[dict[str, str]],
    gt: np.ndarray,
    pred: np.ndarray,
    sequence_ids: np.ndarray,
    candidate_iqr: np.ndarray,
    model_log_std: np.ndarray,
    target: float,
    keep_fracs: np.ndarray,
    min_train_kept: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    errors = np.abs(pred - gt) / np.maximum(gt, 1e-9) * 100.0
    mask = np.zeros(len(rows), dtype=bool)
    fold_rows: list[dict[str, Any]] = []
    for sequence_id in sorted(set(sequence_ids.tolist())):
        test_mask = sequence_ids == sequence_id
        train_mask = ~test_mask
        choice = select_two_feature_gate(
            train_mask=train_mask,
            errors=errors,
            feature_a=candidate_iqr,
            feature_b=model_log_std,
            target_e_ttc_pct=target,
            keep_fracs=keep_fracs,
            min_train_kept=min_train_kept,
        )
        fold_keep = (
            (candidate_iqr[test_mask] <= float(choice["threshold_a"]))
            & (model_log_std[test_mask] <= float(choice["threshold_b"]))
        )
        mask[test_mask] = fold_keep
        fold_summary = metric_summary(gt, pred, mask & test_mask)
        fold_rows.append(
            {
                "sequence_id": sequence_id,
                "test_count": int(np.sum(test_mask)),
                "test_kept": int(np.sum(fold_keep)),
                "test_e_ttc_pct": fold_summary["e_ttc_pct"],
                "test_mae_s": fold_summary["mae_s"],
                **choice,
            }
        )
    return mask, fold_rows


def plot_frontier(output_path: Path, rows: list[dict[str, Any]]) -> Path:
    plot_rows = [row for row in rows if row["policy"] == "iqr_modelstd_gate"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if plt is None:
        svg_path = output_path.with_suffix(".svg")
        write_frontier_svg(svg_path, plot_rows)
        return svg_path
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for pred_name in sorted({row["prediction"] for row in plot_rows}):
        sub = [row for row in plot_rows if row["prediction"] == pred_name]
        sub.sort(key=lambda row: float(row["coverage_of_v13_valid"]))
        ax.plot(
            [float(row["coverage_of_v13_valid"]) * 100.0 for row in sub],
            [float(row["e_ttc_pct"]) for row in sub],
            marker="o",
            linewidth=1.8,
            label=pred_name,
        )
    ax.axhline(7.0, color="#9a1f1f", linestyle="--", linewidth=1.2, label="7% target")
    ax.set_xlabel("Coverage of V13-valid intervals (%)")
    ax.set_ylabel("Mean relative TTC error (%)")
    ax.set_title("Post-V13 consistency gate coverage frontier")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def write_frontier_svg(output_path: Path, rows: list[dict[str, Any]]) -> None:
    width = 900
    height = 560
    left = 95
    right = 30
    top = 55
    bottom = 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    xs = [float(row["coverage_of_v13_valid"]) * 100.0 for row in rows]
    ys = [float(row["e_ttc_pct"]) for row in rows]
    x_min = min(xs) if xs else 40.0
    x_max = max(xs) if xs else 90.0
    y_min = min(min(ys), 6.5) if ys else 6.5
    y_max = max(max(ys), 14.5) if ys else 14.5
    x_pad = max((x_max - x_min) * 0.08, 1.0)
    y_pad = max((y_max - y_min) * 0.10, 0.5)
    x_min -= x_pad
    x_max += x_pad
    y_min = max(0.0, y_min - y_pad)
    y_max += y_pad

    def sx(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1e-9) * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / max(y_max - y_min, 1e-9) * plot_height

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]
    predictions = sorted({row["prediction"] for row in rows})
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="450" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="700">Post-V13 consistency gate coverage frontier</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#fafafa" stroke="#d6d6d6"/>',
    ]
    for tick in range(int(math.floor(x_min / 5) * 5), int(math.ceil(x_max / 5) * 5) + 1, 5):
        x = sx(float(tick))
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#e7e7e7"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">{tick}</text>')
    for tick in range(int(math.floor(y_min)), int(math.ceil(y_max)) + 1):
        y = sy(float(tick))
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e7e7e7"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12">{tick}</text>')
    y_target = sy(7.0)
    parts.append(f'<line x1="{left}" y1="{y_target:.1f}" x2="{left + plot_width}" y2="{y_target:.1f}" stroke="#9a1f1f" stroke-width="2" stroke-dasharray="8 5"/>')
    parts.append(f'<text x="{left + plot_width - 5}" y="{y_target - 8:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#9a1f1f">7% target</text>')

    for pred_idx, pred_name in enumerate(predictions):
        sub = [row for row in rows if row["prediction"] == pred_name]
        sub.sort(key=lambda row: float(row["coverage_of_v13_valid"]))
        color = colors[pred_idx % len(colors)]
        points = " ".join(
            f'{sx(float(row["coverage_of_v13_valid"]) * 100.0):.1f},{sy(float(row["e_ttc_pct"])):.1f}'
            for row in sub
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for row in sub:
            x = sx(float(row["coverage_of_v13_valid"]) * 100.0)
            y = sy(float(row["e_ttc_pct"]))
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}"/>')
    legend_x = left + 12
    legend_y = top + 18
    for pred_idx, pred_name in enumerate(predictions):
        color = colors[pred_idx % len(colors)]
        y = legend_y + pred_idx * 20
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 32}" y="{y + 4}" font-family="Arial, sans-serif" font-size="12">{pred_name}</text>')
    parts.extend(
        [
            f'<text x="{left + plot_width / 2}" y="{height - 25}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Coverage of V13-valid intervals (%)</text>',
            f'<text transform="translate(28 {top + plot_height / 2}) rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Mean relative TTC error (%)</text>',
            "</svg>",
        ]
    )
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.feature_table)
    gt = np.asarray([parse_float(row.get("gt_ttc_s")) for row in rows], dtype=np.float64)
    base_pred = np.clip(
        np.asarray([parse_float(row.get("lightgbm_l1_l2_geomean_pred_ttc_s")) for row in rows], dtype=np.float64),
        args.pred_min_s,
        args.pred_max_s,
    )
    sequence_ids = np.asarray([row.get("sequence_id") for row in rows], dtype=object)
    candidate_iqr = finite_with_median(
        np.asarray([parse_float(row.get("candidate_iqr_over_median")) for row in rows], dtype=np.float64)
    )
    model_matrix = np.column_stack(
        [
            np.asarray([parse_float(row.get(column)) for row in rows], dtype=np.float64)
            for column in BASE_MODEL_COLUMNS
        ]
    )
    model_log_std = finite_with_median(np.std(np.log(np.maximum(finite_with_median(model_matrix), 1e-9)), axis=1))
    keep_fracs = np.linspace(args.keep_frac_min, args.keep_frac_max, int(args.keep_frac_steps))

    predictions: dict[str, np.ndarray] = {"parent": base_pred}
    for mode in args.modes:
        for window in args.windows:
            predictions[f"{mode}_rolling_median_w{window}"] = rolling_prediction(
                rows,
                base_pred,
                window=window,
                mode=mode,
                pred_min_s=args.pred_min_s,
                pred_max_s=args.pred_max_s,
            )

    frontier_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    for pred_name, pred in predictions.items():
        all_summary = metric_summary(gt, pred, np.ones(len(rows), dtype=bool))
        frontier_rows.append(
            {
                "prediction": pred_name,
                "target_e_ttc_pct": "",
                "policy": "all_v13_valid",
                "est_valid": all_summary["count"],
                "mae_s": all_summary["mae_s"],
                "e_ttc_pct": all_summary["e_ttc_pct"],
                "median_e_ttc_pct": all_summary["median_e_ttc_pct"],
                "p90_e_ttc_pct": all_summary["p90_e_ttc_pct"],
                "coverage_of_v13_valid": float(all_summary["count"]) / max(len(rows), 1),
                "coverage_of_gt_valid": float(all_summary["count"]) / 3738.0,
                "nonempty_sequence_count": len(set(sequence_ids.tolist())),
                "over50": all_summary["over50"],
                "over100": all_summary["over100"],
                "success_lt_7pct": bool(float(all_summary["e_ttc_pct"]) < 7.0),
            }
        )
        for target in args.targets:
            mask, fold_summary_rows = run_gate(
                rows=rows,
                gt=gt,
                pred=pred,
                sequence_ids=sequence_ids,
                candidate_iqr=candidate_iqr,
                model_log_std=model_log_std,
                target=float(target),
                keep_fracs=keep_fracs,
                min_train_kept=int(args.min_train_kept),
            )
            summary = metric_summary(gt, pred, mask)
            nonempty = int(
                sum(np.any(mask & (sequence_ids == sequence_id)) for sequence_id in sorted(set(sequence_ids.tolist())))
            )
            frontier_rows.append(
                {
                    "prediction": pred_name,
                    "target_e_ttc_pct": float(target),
                    "policy": "iqr_modelstd_gate",
                    "est_valid": summary["count"],
                    "mae_s": summary["mae_s"],
                    "e_ttc_pct": summary["e_ttc_pct"],
                    "median_e_ttc_pct": summary["median_e_ttc_pct"],
                    "p90_e_ttc_pct": summary["p90_e_ttc_pct"],
                    "coverage_of_v13_valid": float(summary["count"]) / max(len(rows), 1),
                    "coverage_of_gt_valid": float(summary["count"]) / 3738.0,
                    "nonempty_sequence_count": nonempty,
                    "over50": summary["over50"],
                    "over100": summary["over100"],
                    "success_lt_7pct": bool(summary["e_ttc_pct"] is not None and float(summary["e_ttc_pct"]) < 7.0),
                }
            )
            for fold_row in fold_summary_rows:
                fold_rows.append({"prediction": pred_name, "target_e_ttc_pct": float(target), **fold_row})
            for idx, row in enumerate(rows):
                interval_rows.append(
                    {
                        "prediction": pred_name,
                        "target_e_ttc_pct": float(target),
                        "sample_id": row.get("sample_id"),
                        "sequence_id": row.get("sequence_id"),
                        "interval_idx": row.get("interval_idx"),
                        "gt_ttc_s": gt[idx],
                        "pred_ttc_s": pred[idx] if mask[idx] else "",
                        "kept": int(mask[idx]),
                    }
                )

    best_lt7 = [
        row
        for row in frontier_rows
        if row["policy"] == "iqr_modelstd_gate"
        and row["nonempty_sequence_count"] == len(set(sequence_ids.tolist()))
        and row["success_lt_7pct"]
    ]
    best_lt7.sort(key=lambda row: float(row["coverage_of_v13_valid"]), reverse=True)
    summary_payload = {
        "run_name": args.run_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "objective": "Search for higher-coverage post-V13 <7% operating points by combining sequence consistency and confidence gating.",
        "inputs": {
            "feature_table": str(args.feature_table),
            "windows": args.windows,
            "modes": args.modes,
            "targets": args.targets,
            "gate_features": ["candidate_iqr_over_median", "model_log_std"],
            "split": "leave-one-sequence-out gate threshold selection",
        },
        "row_count": len(rows),
        "sequence_count": len(set(sequence_ids.tolist())),
        "best_no_empty_lt7": best_lt7[0] if best_lt7 else None,
        "frontier_rows": frontier_rows,
    }
    write_json(output_dir / "Summary.json", summary_payload)
    write_csv(
        output_dir / "CoverageFrontier.csv",
        frontier_rows,
        [
            "prediction",
            "target_e_ttc_pct",
            "policy",
            "est_valid",
            "mae_s",
            "e_ttc_pct",
            "median_e_ttc_pct",
            "p90_e_ttc_pct",
            "coverage_of_v13_valid",
            "coverage_of_gt_valid",
            "nonempty_sequence_count",
            "over50",
            "over100",
            "success_lt_7pct",
        ],
    )
    write_csv(
        output_dir / "FoldGateSummary.csv",
        fold_rows,
        [
            "prediction",
            "target_e_ttc_pct",
            "sequence_id",
            "test_count",
            "test_kept",
            "test_e_ttc_pct",
            "test_mae_s",
            "keep_frac_a",
            "keep_frac_b",
            "threshold_a",
            "threshold_b",
            "train_count",
            "train_e_ttc_pct",
        ],
    )
    write_csv(
        output_dir / "GateIntervalMetrics.csv",
        interval_rows,
        ["prediction", "target_e_ttc_pct", "sample_id", "sequence_id", "interval_idx", "gt_ttc_s", "pred_ttc_s", "kept"],
    )
    chart_path = plot_frontier(output_dir / "CoverageFrontier.png", frontier_rows)

    lines = [
        f"# {args.run_name}",
        "",
        f"- generated_at: `{summary_payload['generated_at']}`",
        f"- row_count: `{len(rows)}`",
        f"- sequence_count: `{len(set(sequence_ids.tolist()))}`",
        "- gate_features: `candidate_iqr_over_median + model_log_std`",
        f"- chart: `{chart_path.name}`",
        "",
        "| prediction | target | policy | est_valid | coverage_v13 | coverage_gt | e_ttc_pct | median | p90 | nonempty | lt7 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(
        frontier_rows,
        key=lambda item: (
            str(item["prediction"]),
            0.0 if item["target_e_ttc_pct"] == "" else float(item["target_e_ttc_pct"]),
        ),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['prediction']}`",
                    "" if row["target_e_ttc_pct"] == "" else f"{float(row['target_e_ttc_pct']):.2f}",
                    f"`{row['policy']}`",
                    str(row["est_valid"]),
                    f"{float(row['coverage_of_v13_valid']):.3f}",
                    f"{float(row['coverage_of_gt_valid']):.3f}",
                    f"{float(row['e_ttc_pct']):.3f}",
                    f"{float(row['median_e_ttc_pct']):.3f}",
                    f"{float(row['p90_e_ttc_pct']):.3f}",
                    str(row["nonempty_sequence_count"]),
                    str(row["success_lt_7pct"]),
                ]
            )
            + " |"
        )
    if best_lt7:
        best = best_lt7[0]
        lines.extend(
            [
                "",
                "Best no-empty `<7%` operating point:",
                "",
                f"- `{best['prediction']} / iqr_modelstd_gate`",
                f"- target: `{float(best['target_e_ttc_pct']):.2f}`",
                f"- `est_valid={best['est_valid']}`",
                f"- `e_ttc_pct={float(best['e_ttc_pct']):.3f}`",
                f"- `coverage_of_v13_valid={float(best['coverage_of_v13_valid']):.3f}`",
                f"- `coverage_of_gt_valid={float(best['coverage_of_gt_valid']):.3f}`",
            ]
        )
    (output_dir / "Summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[Done] output={output_dir}")
    if best_lt7:
        best = best_lt7[0]
        print(
            f"[Best] {best['prediction']} target={float(best['target_e_ttc_pct']):.2f} "
            f"count={best['est_valid']} e_ttc_pct={float(best['e_ttc_pct']):.3f} "
            f"coverage_v13={float(best['coverage_of_v13_valid']):.3f}"
        )


if __name__ == "__main__":
    main()
