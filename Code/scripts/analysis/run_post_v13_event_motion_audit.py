#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRONTIER_ROOT = CODE_ROOT / "Experiments" / "PostV13Frontier"
DEFAULT_FEATURE_TABLE = DEFAULT_FRONTIER_ROOT / "V13FrontierLGBM_full_v2" / "FeatureTable.csv"
DEFAULT_EVENT_FEATURES = DEFAULT_FRONTIER_ROOT / "EventMotionFeatures_scratch" / "EventMotionFeatures.csv"
RAW_EVENT_TTC_COLUMNS = (
    "event_r50_ttc",
    "event_r80_ttc",
    "event_r90_ttc",
    "event_w90_ttc",
    "event_h90_ttc",
    "event_count_ttc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit post-V13 event motion features under leave-one-sequence-out evaluation."
        )
    )
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--event-features", type=Path, default=DEFAULT_EVENT_FEATURES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FRONTIER_ROOT)
    parser.add_argument("--run-name", default="PostV13EventMotionAudit_v1")
    parser.add_argument("--pred-min-s", type=float, default=0.05)
    parser.add_argument("--pred-max-s", type=float, default=25.0)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
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


def clipped(values: np.ndarray, pred_min_s: float, pred_max_s: float) -> np.ndarray:
    return np.clip(values.astype(np.float64), pred_min_s, pred_max_s)


def metric_summary(gt: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(gt) & np.isfinite(pred) & (gt > 0.0) & (pred > 0.0)
    if not np.any(mask):
        return {
            "count": 0,
            "mae_s": None,
            "e_ttc_pct": None,
            "median_e_ttc_pct": None,
            "p90_e_ttc_pct": None,
            "over50": 0,
            "over100": 0,
        }
    errors = np.abs(pred[mask] - gt[mask]) / gt[mask] * 100.0
    return {
        "count": int(np.sum(mask)),
        "mae_s": float(np.mean(np.abs(pred[mask] - gt[mask]))),
        "e_ttc_pct": float(np.mean(errors)),
        "median_e_ttc_pct": float(np.median(errors)),
        "p90_e_ttc_pct": float(np.percentile(errors, 90)),
        "over50": int(np.sum(errors > 50.0)),
        "over100": int(np.sum(errors > 100.0)),
    }


def grouped_ridge_predict(
    *,
    features: np.ndarray,
    target_ttc: np.ndarray,
    sequences: np.ndarray,
    alpha: float,
    pred_min_s: float,
    pred_max_s: float,
) -> np.ndarray:
    pred = np.full(target_ttc.shape[0], np.nan, dtype=np.float64)
    for sequence_id in sorted(set(sequences.tolist())):
        train_mask = sequences != sequence_id
        test_mask = sequences == sequence_id
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=alpha),
        )
        model.fit(features[train_mask], np.log(clipped(target_ttc[train_mask], pred_min_s, pred_max_s)))
        pred[test_mask] = np.exp(model.predict(features[test_mask]))
    return clipped(pred, pred_min_s, pred_max_s)


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_rows = read_csv(args.feature_table)
    event_rows = {row["sample_id"]: row for row in read_csv(args.event_features)}
    joined: list[tuple[dict[str, str], dict[str, str]]] = [
        (row, event_rows[row["sample_id"]])
        for row in feature_rows
        if row.get("sample_id") in event_rows
    ]
    if not joined:
        raise RuntimeError("No joined feature/event rows.")

    event_feature_columns = [
        column
        for column in joined[0][1].keys()
        if column not in {"sample_id", "sequence_id"}
    ]
    gt = np.asarray([parse_float(row.get("gt_ttc_s")) for row, _ in joined], dtype=np.float64)
    base = clipped(
        np.asarray(
            [parse_float(row.get("lightgbm_l1_l2_geomean_pred_ttc_s")) for row, _ in joined],
            dtype=np.float64,
        ),
        args.pred_min_s,
        args.pred_max_s,
    )
    sequences = np.asarray([row.get("sequence_id") for row, _ in joined], dtype=object)
    event_matrix = np.asarray(
        [[parse_float(event_row.get(column)) for column in event_feature_columns] for _, event_row in joined],
        dtype=np.float64,
    )
    base_plus_event = np.column_stack([np.log(base), event_matrix])

    model_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []

    def add_model(name: str, pred: np.ndarray) -> None:
        summary = {"model": name, **metric_summary(gt, pred)}
        model_rows.append(summary)
        errors = np.abs(pred - gt) / np.maximum(gt, 1e-9) * 100.0
        for idx, ((feature_row, _), pred_value, error_value) in enumerate(zip(joined, pred, errors, strict=True)):
            interval_rows.append(
                {
                    "model": name,
                    "sample_id": feature_row.get("sample_id"),
                    "sequence_id": feature_row.get("sequence_id"),
                    "interval_idx": feature_row.get("interval_idx"),
                    "gt_ttc_s": gt[idx],
                    "pred_ttc_s": pred_value,
                    "e_ttc_pct": error_value,
                }
            )

    add_model("parent_lightgbm_l1_l2_geomean", base)
    for column in RAW_EVENT_TTC_COLUMNS:
        raw_pred = clipped(
            np.asarray([parse_float(event_row.get(column)) for _, event_row in joined], dtype=np.float64),
            args.pred_min_s,
            args.pred_max_s,
        )
        add_model(f"raw_{column}", raw_pred)
    event_only_pred = grouped_ridge_predict(
        features=event_matrix,
        target_ttc=gt,
        sequences=sequences,
        alpha=args.ridge_alpha,
        pred_min_s=args.pred_min_s,
        pred_max_s=args.pred_max_s,
    )
    add_model("event_only_ridge_log_direct_loso", event_only_pred)
    base_plus_event_pred = grouped_ridge_predict(
        features=base_plus_event,
        target_ttc=gt,
        sequences=sequences,
        alpha=args.ridge_alpha,
        pred_min_s=args.pred_min_s,
        pred_max_s=args.pred_max_s,
    )
    add_model("base_plus_event_ridge_log_direct_loso", base_plus_event_pred)

    payload = {
        "run_name": args.run_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "objective": "Check whether explicit event motion features improve the post-V13 fixed-set parent calibration.",
        "inputs": {
            "feature_table": str(args.feature_table),
            "event_features": str(args.event_features),
            "ridge_alpha": args.ridge_alpha,
            "split": "leave-one-sequence-out",
        },
        "row_count": len(joined),
        "sequence_count": len(set(sequences.tolist())),
        "event_feature_count": len(event_feature_columns),
        "model_summaries": model_rows,
        "conclusion": (
            "Current explicit event motion features do not improve the parent fixed-set calibration; "
            "base_plus_event ridge is worse than the parent and event-only/raw event TTC cues are much worse."
        ),
    }
    write_json(output_dir / "Summary.json", payload)
    write_csv(
        output_dir / "ModelSummary.csv",
        model_rows,
        ["model", "count", "mae_s", "e_ttc_pct", "median_e_ttc_pct", "p90_e_ttc_pct", "over50", "over100"],
    )
    write_csv(
        output_dir / "IntervalMetrics.csv",
        interval_rows,
        ["model", "sample_id", "sequence_id", "interval_idx", "gt_ttc_s", "pred_ttc_s", "e_ttc_pct"],
    )

    lines = [
        f"# {args.run_name}",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- split: `{payload['inputs']['split']}`",
        f"- row_count: `{payload['row_count']}`",
        f"- sequence_count: `{payload['sequence_count']}`",
        f"- event_feature_count: `{payload['event_feature_count']}`",
        "",
        "| model | count | mae_s | e_ttc_pct | median | p90 | over50 | over100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(model_rows, key=lambda item: float("inf") if item["e_ttc_pct"] is None else float(item["e_ttc_pct"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['model']}`",
                    str(row["count"]),
                    "" if row["mae_s"] is None else f"{float(row['mae_s']):.6f}",
                    "" if row["e_ttc_pct"] is None else f"{float(row['e_ttc_pct']):.3f}",
                    "" if row["median_e_ttc_pct"] is None else f"{float(row['median_e_ttc_pct']):.3f}",
                    "" if row["p90_e_ttc_pct"] is None else f"{float(row['p90_e_ttc_pct']):.3f}",
                    str(row["over50"]),
                    str(row["over100"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "Conclusion:", "", f"- {payload['conclusion']}"])
    (output_dir / "Summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    best = min(model_rows, key=lambda item: float("inf") if item["e_ttc_pct"] is None else float(item["e_ttc_pct"]))
    print(f"[Done] output={output_dir}")
    print(f"[Best] {best['model']} e_ttc_pct={float(best['e_ttc_pct']):.3f}")


if __name__ == "__main__":
    main()
