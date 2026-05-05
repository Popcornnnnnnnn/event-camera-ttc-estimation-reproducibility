#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[2]
DATA_SCRIPT_DIR = CODE_ROOT / "scripts" / "data"
if str(DATA_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_SCRIPT_DIR))

import run_ml_baseline_train as np_train  # noqa: E402

DEFAULT_V13_DIR = CODE_ROOT / "Experiments" / "SparsityAwareLTS" / "BBoxLoomingV13_20260427_212537"
DEFAULT_FAILURE_DIR = (
    CODE_ROOT
    / "Experiments"
    / "SparsityAwareLTSDiagnostics"
    / "BBoxFailureTypeReport_20260427_212614"
)
DEFAULT_GEOMETRY_DIR = (
    CODE_ROOT
    / "Experiments"
    / "SparsityAwareLTSDiagnostics"
    / "BBoxHighConfGeometryReport_20260427_212710"
)

TARGET_FEATURES = [
    "v13_pred_ttc_s",
    "interval_fraction",
    "event_count",
    "fit_points",
    "history_record_count",
    "current_area",
    "area_ratio_prev",
    "area_ratio_med5",
    "history_collapse_count",
    "candidate_count",
    "candidate_max_fit_points",
    "candidate_min_s",
    "candidate_q05_s",
    "candidate_q10_s",
    "candidate_q15_s",
    "candidate_q25_s",
    "candidate_q35_s",
    "candidate_q50_s",
    "candidate_q65_s",
    "candidate_q75_s",
    "candidate_q90_s",
    "candidate_max_s",
    "candidate_iqr_over_median",
    "candidate_p90_p10_over_median",
    "candidate_width_count",
    "candidate_height_count",
    "candidate_diag_count",
    "candidate_sqrt_area_count",
    "candidate_bottom_offset_count",
]
QUANTILE_KEYS = ("05", "10", "15", "25", "35", "50", "65", "75", "90")
INTERACTION_FEATURES = {
    "log_v13_pred_ttc_s",
    "interval_fraction",
    "area_ratio_prev",
    "area_ratio_med5",
    "log_candidate_q05_s",
    "log_candidate_q10_s",
    "log_candidate_q25_s",
    "log_candidate_q50_s",
    "log_candidate_q90_s",
    "candidate_iqr_over_median",
    "candidate_p90_p10_over_median",
}

GT_DERIVED_COLUMNS = {
    "gt_ttc_s",
    "e_ttc_pct",
    "pred_over_gt",
    "is_high_error",
    "is_severe_error",
    "primary_failure_type",
    "diagnostic_group",
    "gt_bucket",
    "best_quantile_key",
    "best_quantile_e_ttc_pct",
    "candidate_q05_e_ttc_pct",
    "candidate_q10_e_ttc_pct",
    "candidate_q15_e_ttc_pct",
    "candidate_q25_e_ttc_pct",
    "candidate_q35_e_ttc_pct",
    "candidate_q50_e_ttc_pct",
    "candidate_q65_e_ttc_pct",
    "candidate_q75_e_ttc_pct",
    "candidate_q90_e_ttc_pct",
    "v11_e_ttc_pct",
    "v11_pred_over_gt",
    "flags",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fixed V13-valid frontier feature table and run grouped tabular "
            "calibration baselines."
        ),
    )
    parser.add_argument("--v13-run-dir", type=Path, default=DEFAULT_V13_DIR)
    parser.add_argument("--failure-dir", type=Path, default=DEFAULT_FAILURE_DIR)
    parser.add_argument("--geometry-dir", type=Path, default=DEFAULT_GEOMETRY_DIR)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=CODE_ROOT / "DatasetFormal",
        help="Root directory of formal sequence artifacts, used when recomputing candidates.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CODE_ROOT / "Experiments" / "PostV13Frontier",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Optional sequence subset for smoke. Default: all V13-valid sequences.",
    )
    parser.add_argument(
        "--ridge-alphas",
        nargs="+",
        type=float,
        default=(0.1, 1.0, 10.0, 100.0),
    )
    parser.add_argument("--pred-min-s", type=float, default=0.05)
    parser.add_argument("--pred-max-s", type=float, default=25.0)
    parser.add_argument("--softmax-epochs", type=int, default=800)
    parser.add_argument("--softmax-learning-rate", type=float, default=0.05)
    parser.add_argument("--softmax-l2", type=float, default=0.01)
    parser.add_argument("--mlp-hidden-dims", default="64,32")
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-batch-size", type=int, default=32)
    parser.add_argument("--mlp-epochs", type=int, default=400)
    parser.add_argument("--mlp-patience", type=int, default=40)
    parser.add_argument("--mlp-val-ratio", type=float, default=0.2)
    parser.add_argument("--skip-softmax", action="store_true")
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--enable-lightgbm", action="store_true")
    parser.add_argument(
        "--disable-sequence-metadata",
        action="store_true",
        help="Exclude sequence-id-derived non-GT metadata from LightGBM features for ablation.",
    )
    parser.add_argument("--lightgbm-n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--recompute-missing-candidates",
        action="store_true",
        help="Recompute multi-window candidate quantiles for V13-valid rows missing diagnostics.",
    )
    return parser.parse_args()


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError(f"Invalid hidden dims: {value!r}")
    return dims


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


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def parse_int(value: Any) -> int | None:
    number = parse_float(value)
    if number is None:
        return None
    return int(number)


def format_float(value: float | None, digits: int = 9) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def is_valid_prediction(row: dict[str, str]) -> bool:
    return (
        parse_float(row.get("gt_ttc_s")) is not None
        and parse_float(row.get("ttc_est_s")) is not None
        and parse_float(row.get("e_ttc_pct")) is not None
    )


def e_ttc_pct(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.abs(pred - gt) / np.maximum(gt, 1e-9) * 100.0


def metric_summary(gt: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    errors = e_ttc_pct(gt, pred)
    return {
        "count": int(gt.shape[0]),
        "mae_s": float(np.mean(np.abs(pred - gt))) if gt.size else float("nan"),
        "e_ttc_pct": float(np.mean(errors)) if gt.size else float("nan"),
        "median_e_ttc_pct": float(np.median(errors)) if gt.size else float("nan"),
        "p90_e_ttc_pct": float(np.percentile(errors, 90)) if gt.size else float("nan"),
        "over50": int(np.sum(errors > 50.0)),
        "over100": int(np.sum(errors > 100.0)),
    }


def candidate_key(q: float) -> str:
    return f"candidate_q{int(round(q * 100)):02d}_s"


def build_candidate_config() -> Any:
    import run_bbox_high_conf_geometry_report as geom

    return geom.BBoxLoomingConfigV1(
        history_back_labels=12,
        history_min_back_labels=2,
        max_ttc_s=16.0,
        fallback_to_lts_v1=False,
        cue_mode="multi_quantile",
        candidate_quantile=0.25,
        include_bottom_cue=True,
        quality_gate_enabled=True,
        quality_min_candidate_count=30,
        quality_min_fit_points=6,
        quality_max_iqr_over_median=0.7,
        quality_low_confidence_strategy="adaptive_quantile_v2",
        quality_reject_low_confidence_after_adaptive=True,
        current_bbox_collapse_guard_enabled=True,
        history_bbox_collapse_guard_enabled=True,
        history_bbox_collapse_guard_require_low_confidence=True,
    )


def recompute_candidate_quantiles(
    *,
    formal_root: Path,
    sequence_id: str,
    sample_id: str,
    adapters: dict[str, Any],
    config: Any,
) -> dict[str, Any]:
    import run_bbox_high_conf_geometry_report as geom

    if sequence_id not in adapters:
        adapters[sequence_id] = geom.NativeSampleAdapter(sequence_id, formal_root=formal_root.resolve())
    adapter = adapters[sequence_id]
    candidates, status, max_fit_points = geom.collect_multiwindow_candidates(
        adapter=adapter,
        sample_id=sample_id,
        config=config,
    )
    if not candidates:
        return {"candidate_collect_status": status}
    values = np.asarray([candidate.ttc_s for candidate in candidates], dtype=np.float64)
    median = float(np.median(values))
    q25 = float(np.quantile(values, 0.25))
    q75 = float(np.quantile(values, 0.75))
    cue_counts: dict[str, int] = {}
    for candidate in candidates:
        cue_counts[candidate.cue_name] = cue_counts.get(candidate.cue_name, 0) + 1
    out: dict[str, Any] = {
        "candidate_collect_status": status,
        "candidate_count": len(candidates),
        "candidate_max_fit_points": max_fit_points,
        "candidate_min_s": float(np.min(values)),
        "candidate_max_s": float(np.max(values)),
        "candidate_iqr_over_median": (q75 - q25) / max(abs(median), 1e-6),
        "candidate_p90_p10_over_median": (
            float(np.quantile(values, 0.90)) - float(np.quantile(values, 0.10))
        ) / max(abs(median), 1e-6),
        "candidate_width_count": cue_counts.get("width", 0),
        "candidate_height_count": cue_counts.get("height", 0),
        "candidate_diag_count": cue_counts.get("diag", 0),
        "candidate_sqrt_area_count": cue_counts.get("sqrt_area", 0),
        "candidate_bottom_offset_count": cue_counts.get("bottom_offset", 0),
    }
    for q in (0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.90):
        out[candidate_key(q)] = float(np.quantile(values, q))
    return out


def build_feature_table(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    interval_rows = read_csv(args.v13_run_dir / "IntervalMetrics.csv")
    failure_rows = {
        row["sample_id"]: row
        for row in read_csv(args.failure_dir / "IntervalFailureTypes.csv")
        if row.get("sample_id")
    }
    candidate_rows = {
        row["sample_id"]: row
        for row in read_csv(args.geometry_dir / "IntervalCandidateQuantiles.csv")
        if row.get("sample_id")
    }
    recomputed_adapters: dict[str, Any] = {}
    recompute_config = build_candidate_config() if args.recompute_missing_candidates else None

    table: list[dict[str, Any]] = []
    try:
        for row in interval_rows:
            if not is_valid_prediction(row):
                continue
            sequence_id = row["sequence_id"]
            if args.sequences is not None and sequence_id not in set(args.sequences):
                continue
            sample_id = row["sample_id"]
            failure = failure_rows.get(sample_id, {})
            candidate = dict(candidate_rows.get(sample_id, {}))
            if not candidate and recompute_config is not None:
                candidate = recompute_candidate_quantiles(
                    formal_root=args.formal_root,
                    sequence_id=sequence_id,
                    sample_id=sample_id,
                    adapters=recomputed_adapters,
                    config=recompute_config,
                )
            gt_ttc = parse_float(row.get("gt_ttc_s"))
            pred_ttc = parse_float(row.get("ttc_est_s"))
            if gt_ttc is None or pred_ttc is None:
                continue
            out: dict[str, Any] = {
                "sample_id": sample_id,
                "sequence_id": sequence_id,
                "interval_idx": parse_int(row.get("interval_idx")),
                "timestamp_s": format_float(parse_float(row.get("timestamp_s"))),
                "gt_ttc_s": format_float(gt_ttc),
                "v13_pred_ttc_s": format_float(pred_ttc),
                "v13_e_ttc_pct": format_float(parse_float(row.get("e_ttc_pct"))),
                "status": row.get("status", ""),
                "status_family": failure.get("status_family", ""),
                "primary_failure_type": failure.get("primary_failure_type", ""),
                "gt_bucket": failure.get("gt_bucket", ""),
                "is_high_error": failure.get("is_high_error", ""),
                "is_severe_error": failure.get("is_severe_error", ""),
                "has_candidate_quantiles": (
                    "1" if parse_float(candidate.get("candidate_q25_s")) is not None else "0"
                ),
                "candidate_source": "existing" if sample_id in candidate_rows else "recomputed",
                "candidate_collect_status": candidate.get("candidate_collect_status", ""),
            }
            for name in TARGET_FEATURES:
                source = candidate if name in candidate else failure
                if name == "v13_pred_ttc_s":
                    out[name] = format_float(pred_ttc)
                else:
                    out[name] = format_float(parse_float(source.get(name)))
            table.append(out)
    finally:
        for adapter in recomputed_adapters.values():
            adapter.close()

    gt = np.asarray([float(row["gt_ttc_s"]) for row in table], dtype=np.float64)
    pred = np.asarray([float(row["v13_pred_ttc_s"]) for row in table], dtype=np.float64)
    summary = {
        "v13_run_dir": str(args.v13_run_dir),
        "failure_dir": str(args.failure_dir),
        "geometry_dir": str(args.geometry_dir),
        "row_count": int(len(table)),
        "sequence_count": int(len({row["sequence_id"] for row in table})),
        "candidate_quantile_row_count": int(sum(int(row["has_candidate_quantiles"]) for row in table)),
        "candidate_source_counts": {
            source: sum(1 for row in table if row.get("candidate_source") == source)
            for source in sorted({str(row.get("candidate_source", "")) for row in table})
        },
        "baseline": metric_summary(gt, pred),
    }
    return table, summary


def one_hot_values(rows: list[dict[str, Any]], key: str) -> list[str]:
    values = sorted({str(row.get(key, "") or "MISSING") for row in rows})
    return values


def feature_matrix(rows: list[dict[str, Any]], status_values: list[str]) -> tuple[np.ndarray, list[str]]:
    numeric_parts: list[np.ndarray] = []
    names: list[str] = []
    for name in TARGET_FEATURES:
        values = np.asarray(
            [parse_float(row.get(name)) if parse_float(row.get(name)) is not None else np.nan for row in rows],
            dtype=np.float64,
        )
        if name.endswith("_s") or name in {
            "v13_pred_ttc_s",
            "event_count",
            "current_area",
            "candidate_count",
            "candidate_max_fit_points",
            "candidate_min_s",
            "candidate_max_s",
        }:
            valid = np.isfinite(values) & (values > 0.0)
            transformed = values.copy()
            transformed[valid] = np.log(transformed[valid])
            transformed[~valid] = np.nan
            values = transformed
            feature_name = f"log_{name}"
        else:
            feature_name = name
        numeric_parts.append(values.reshape(-1, 1))
        names.append(feature_name)
        numeric_parts.append((~np.isfinite(values)).astype(np.float64).reshape(-1, 1))
        names.append(f"{feature_name}_missing")

    status_arr = np.zeros((len(rows), len(status_values)), dtype=np.float64)
    status_index = {value: idx for idx, value in enumerate(status_values)}
    for row_idx, row in enumerate(rows):
        value = str(row.get("status_family", "") or "MISSING")
        status_arr[row_idx, status_index[value]] = 1.0
    parts = [*numeric_parts, status_arr]
    names.extend([f"status_family={value}" for value in status_values])
    base = np.concatenate(parts, axis=1)
    interaction_parts: list[np.ndarray] = []
    interaction_names: list[str] = []
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    for status_idx, status_value in enumerate(status_values):
        status_col = status_arr[:, status_idx:status_idx + 1]
        for feature_name in sorted(INTERACTION_FEATURES):
            feature_idx = name_to_idx.get(feature_name)
            if feature_idx is None:
                continue
            values = np.where(np.isfinite(base[:, feature_idx:feature_idx + 1]), base[:, feature_idx:feature_idx + 1], 0.0)
            interaction_parts.append(status_col * values)
            interaction_names.append(f"{feature_name}*status_family={status_value}")
    if interaction_parts:
        base = np.concatenate([base, *interaction_parts], axis=1)
        names.extend(interaction_names)
    return base, names


def sequence_metadata_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    families = sorted({str(row["sequence_id"]).split("-")[0] for row in rows})
    values: list[list[float]] = []
    for row in rows:
        sequence_id = str(row["sequence_id"])
        if "low" in sequence_id:
            speed_level = 0.0
        elif "medium" in sequence_id:
            speed_level = 1.0
        elif "high" in sequence_id:
            speed_level = 2.0
        else:
            speed_level = -1.0
        percent_match = re.search(r"(0|50|100)%", sequence_id)
        visibility_fraction = float(percent_match.group(1)) / 100.0 if percent_match else -1.0
        repeat_match = re.search(r"CCRs-(\d+)", sequence_id)
        repeat_index = float(repeat_match.group(1)) if repeat_match else 0.0
        family = sequence_id.split("-")[0]
        values.append(
            [
                1.0 if "side" in sequence_id else 0.0,
                speed_level,
                visibility_fraction,
                repeat_index,
                *[1.0 if family == item else 0.0 for item in families],
            ]
        )
    names = [
        "sequence_has_side",
        "sequence_speed_level",
        "sequence_visibility_fraction",
        "sequence_repeat_index",
        *[f"sequence_family={family}" for family in families],
    ]
    return np.asarray(values, dtype=np.float64), names


def lightgbm_feature_matrix(
    rows: list[dict[str, Any]],
    base_x: np.ndarray,
    v13_pred: np.ndarray,
    quantile_matrix: np.ndarray,
    *,
    include_sequence_metadata: bool,
) -> tuple[np.ndarray, list[str]]:
    if include_sequence_metadata:
        sequence_x, sequence_names = sequence_metadata_matrix(rows)
    else:
        sequence_x = np.zeros((len(rows), 0), dtype=np.float64)
        sequence_names = []
    relative_quantiles = np.log(np.maximum(quantile_matrix, 1e-9) / np.maximum(v13_pred[:, None], 1e-9))
    log_quantiles = np.log(np.maximum(quantile_matrix, 1e-9))
    quantile_spreads = np.column_stack(
        [quantile_matrix[:, idx + 1] - quantile_matrix[:, idx] for idx in range(quantile_matrix.shape[1] - 1)]
    )
    names = [
        *sequence_names,
        *[f"log_candidate_q{key}_over_v13" for key in QUANTILE_KEYS],
        *[f"log_candidate_q{key}_raw" for key in QUANTILE_KEYS],
        *[
            f"candidate_q{QUANTILE_KEYS[idx + 1]}_minus_q{QUANTILE_KEYS[idx]}"
            for idx in range(len(QUANTILE_KEYS) - 1)
        ],
    ]
    matrix = np.concatenate([base_x, sequence_x, relative_quantiles, log_quantiles, quantile_spreads], axis=1)
    return matrix, names


def fit_log_affine(train_pred: np.ndarray, train_gt: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones_like(train_pred), np.log(np.maximum(train_pred, 1e-9))])
    y = np.log(np.maximum(train_gt, 1e-9))
    return np.linalg.lstsq(x, y, rcond=None)[0]


def predict_log_affine(coef: np.ndarray, pred: np.ndarray, *, pred_min: float, pred_max: float) -> np.ndarray:
    x = np.column_stack([np.ones_like(pred), np.log(np.maximum(pred, 1e-9))])
    return exp_clipped_log_ttc(x @ coef, pred_min=pred_min, pred_max=pred_max)


def exp_clipped_log_ttc(log_ttc: np.ndarray, *, pred_min: float, pred_max: float) -> np.ndarray:
    """Convert log-TTC predictions back to seconds without numeric overflow."""
    clipped = np.clip(log_ttc, math.log(pred_min), math.log(pred_max))
    return np.exp(clipped)


def apply_clipped_log_residual(
    base_pred: np.ndarray,
    log_residual: np.ndarray,
    *,
    pred_min: float,
    pred_max: float,
) -> np.ndarray:
    """Apply log residuals while keeping the final TTC inside protocol bounds."""
    safe_base = np.maximum(base_pred, 1e-9)
    lower = math.log(pred_min) - np.log(safe_base)
    upper = math.log(pred_max) - np.log(safe_base)
    clipped = np.clip(log_residual, lower, upper)
    return safe_base * np.exp(clipped)


def fit_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.nanmean(train_x, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    filled = np.where(np.isfinite(train_x), train_x, means)
    stds = np.std(filled, axis=0)
    stds = np.where(stds > 1e-9, stds, 1.0)
    x_scaled = (filled - means) / stds
    x_design = np.column_stack([np.ones(x_scaled.shape[0]), x_scaled])
    reg = np.eye(x_design.shape[1], dtype=np.float64) * float(alpha)
    reg[0, 0] = 0.0
    coef = np.linalg.solve(x_design.T @ x_design + reg, x_design.T @ train_y)
    return coef, means, stds


def predict_ridge(coef: np.ndarray, means: np.ndarray, stds: np.ndarray, x: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(x), x, means)
    x_scaled = (filled - means) / stds
    x_design = np.column_stack([np.ones(x_scaled.shape[0]), x_scaled])
    return x_design @ coef


def candidate_quantile_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    matrix = np.full((len(rows), len(QUANTILE_KEYS)), np.nan, dtype=np.float64)
    for row_idx, row in enumerate(rows):
        for col_idx, key in enumerate(QUANTILE_KEYS):
            value = parse_float(row.get(f"candidate_q{key}_s"))
            if value is not None:
                matrix[row_idx, col_idx] = value
    return matrix


def best_quantile_labels(gt: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    errors = np.abs(quantiles - gt[:, None]) / np.maximum(gt[:, None], 1e-9)
    errors = np.where(np.isfinite(errors), errors, np.inf)
    return np.argmin(errors, axis=1).astype(np.int64)


def standardize_train_test(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(train_x, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    train_filled = np.where(np.isfinite(train_x), train_x, means)
    test_filled = np.where(np.isfinite(test_x), test_x, means)
    stds = np.std(train_filled, axis=0)
    stds = np.where(stds > 1e-9, stds, 1.0)
    return (train_filled - means) / stds, (test_filled - means) / stds


def fit_softmax_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    class_count: int,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> np.ndarray:
    x_design = np.column_stack([np.ones(train_x.shape[0]), train_x])
    weights = np.zeros((x_design.shape[1], class_count), dtype=np.float64)
    y_onehot = np.zeros((train_x.shape[0], class_count), dtype=np.float64)
    y_onehot[np.arange(train_x.shape[0]), train_y] = 1.0
    n = max(1, train_x.shape[0])
    for _ in range(max(1, int(epochs))):
        logits = x_design @ weights
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.maximum(np.sum(exp_logits, axis=1, keepdims=True), 1e-12)
        grad = x_design.T @ (probs - y_onehot) / n
        grad[1:, :] += float(l2) * weights[1:, :]
        weights -= float(learning_rate) * grad
    return weights


def predict_softmax(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    x_design = np.column_stack([np.ones(x.shape[0]), x])
    return np.argmax(x_design @ weights, axis=1).astype(np.int64)


def fit_predict_mlp(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_sequences: np.ndarray,
    train_y_raw: np.ndarray,
    *,
    hidden_dims: tuple[int, ...],
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    train_x_scaled, test_x_scaled = standardize_train_test(train_x, test_x)
    train_idx, val_idx = np_train.split_train_val_indices(
        train_sequences,
        val_ratio=float(val_ratio),
        seed=int(seed),
    )
    if val_idx.size == 0:
        val_idx = train_idx
    y_mean = float(np.mean(train_y_raw[train_idx]))
    y_std = float(np.std(train_y_raw[train_idx]))
    if y_std < 1e-6:
        y_std = 1.0
    train_y = (train_y_raw[train_idx] - y_mean) / y_std
    val_y = (train_y_raw[val_idx] - y_mean) / y_std
    layers, train_summary, _history = np_train.train_mlp_regressor(
        train_x_scaled[train_idx],
        train_y,
        train_x_scaled[val_idx],
        val_y,
        hidden_dims=hidden_dims,
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        batch_size=int(batch_size),
        epochs=int(epochs),
        patience=int(patience),
        seed=int(seed),
    )
    pred_scaled, _, _ = np_train.forward_mlp(test_x_scaled, layers)
    pred_raw = pred_scaled * y_std + y_mean
    return pred_raw, {
        **train_summary,
        "train_count_after_val_split": int(train_idx.size),
        "val_count": int(val_idx.size),
        "y_mean": y_mean,
        "y_std": y_std,
    }


def fill_train_test_median(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    medians = np.nanmedian(train_x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    return (
        np.where(np.isfinite(train_x), train_x, medians),
        np.where(np.isfinite(test_x), test_x, medians),
    )


def fit_predict_lightgbm_log_ttc(
    train_x: np.ndarray,
    train_gt: np.ndarray,
    test_x: np.ndarray,
    *,
    objective: str,
    seed: int,
    n_jobs: int,
    pred_min: float,
    pred_max: float,
) -> np.ndarray:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM is required for --enable-lightgbm. Install with: "
            "`python -m pip install lightgbm` and on macOS ensure libomp is installed."
        ) from exc

    train_filled, test_filled = fill_train_test_median(train_x, test_x)
    model = lgb.LGBMRegressor(
        objective=objective,
        n_estimators=400,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=15,
        reg_lambda=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        random_state=int(seed),
        verbosity=-1,
        n_jobs=int(n_jobs),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        model.fit(train_filled, np.log(np.maximum(train_gt, 1e-9)))
        log_pred = model.predict(test_filled)
    return exp_clipped_log_ttc(log_pred, pred_min=pred_min, pred_max=pred_max)


def run_grouped_models(
    rows: list[dict[str, Any]],
    *,
    ridge_alphas: tuple[float, ...],
    pred_min: float,
    pred_max: float,
    softmax_epochs: int,
    softmax_learning_rate: float,
    softmax_l2: float,
    mlp_hidden_dims: tuple[int, ...],
    mlp_learning_rate: float,
    mlp_weight_decay: float,
    mlp_batch_size: int,
    mlp_epochs: int,
    mlp_patience: int,
    mlp_val_ratio: float,
    skip_softmax: bool,
    skip_mlp: bool,
    enable_lightgbm: bool,
    include_sequence_metadata: bool,
    lightgbm_n_jobs: int,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    status_values = one_hot_values(rows, "status_family")
    x_all, feature_names = feature_matrix(rows, status_values)
    gt = np.asarray([float(row["gt_ttc_s"]) for row in rows], dtype=np.float64)
    v13_pred = np.asarray([float(row["v13_pred_ttc_s"]) for row in rows], dtype=np.float64)
    quantile_matrix = candidate_quantile_matrix(rows)
    lgb_x_all, lgb_extra_feature_names = lightgbm_feature_matrix(
        rows,
        x_all,
        v13_pred,
        quantile_matrix,
        include_sequence_metadata=include_sequence_metadata,
    )
    oracle_labels = best_quantile_labels(gt, quantile_matrix)
    oracle_pred = quantile_matrix[np.arange(quantile_matrix.shape[0]), oracle_labels]
    sequences = np.asarray([str(row["sequence_id"]) for row in rows], dtype=object)
    unique_sequences = sorted(set(sequences.tolist()))

    predictions: dict[str, np.ndarray] = {
        "identity_v13": v13_pred.copy(),
        "oracle_best_candidate_quantile": oracle_pred.copy(),
        "log_affine": np.full_like(v13_pred, np.nan),
    }
    if not skip_softmax:
        predictions["softmax_quantile_selector"] = np.full_like(v13_pred, np.nan)
    if not skip_mlp:
        predictions["mlp_log_direct"] = np.full_like(v13_pred, np.nan)
        predictions["mlp_log_residual"] = np.full_like(v13_pred, np.nan)
    lgb_objectives: dict[str, str] = {
        "lightgbm_l1_log_direct": "regression_l1",
        "lightgbm_l2_log_direct": "regression",
        "lightgbm_huber_log_direct": "huber",
        "lightgbm_fair_log_direct": "fair",
    }
    if enable_lightgbm:
        for model_name in lgb_objectives:
            predictions[model_name] = np.full_like(v13_pred, np.nan)
        predictions["lightgbm_l1_l2_mean"] = np.full_like(v13_pred, np.nan)
        predictions["lightgbm_l1_l2_geomean"] = np.full_like(v13_pred, np.nan)
        predictions["lightgbm_l1_l2_huber_fair_mean"] = np.full_like(v13_pred, np.nan)
        predictions["lightgbm_l1_l2_huber_fair_geomean"] = np.full_like(v13_pred, np.nan)
    for alpha in ridge_alphas:
        predictions[f"ridge_log_residual_alpha_{alpha:g}"] = np.full_like(v13_pred, np.nan)
        predictions[f"ridge_log_direct_alpha_{alpha:g}"] = np.full_like(v13_pred, np.nan)

    fold_rows: list[dict[str, Any]] = []
    for sequence in unique_sequences:
        test_mask = sequences == sequence
        train_mask = ~test_mask
        train_gt = gt[train_mask]
        train_pred = v13_pred[train_mask]
        test_pred = v13_pred[test_mask]

        affine_coef = fit_log_affine(train_pred, train_gt)
        predictions["log_affine"][test_mask] = predict_log_affine(
            affine_coef,
            test_pred,
            pred_min=pred_min,
            pred_max=pred_max,
        )

        train_x = x_all[train_mask]
        test_x = x_all[test_mask]
        train_lgb_x = lgb_x_all[train_mask]
        test_lgb_x = lgb_x_all[test_mask]
        train_sequence_ids = sequences[train_mask]
        train_log_gt = np.log(np.maximum(train_gt, 1e-9))
        train_log_residual = train_log_gt - np.log(np.maximum(train_pred, 1e-9))
        for alpha in ridge_alphas:
            residual_key = f"ridge_log_residual_alpha_{alpha:g}"
            coef, means, stds = fit_ridge(train_x, train_log_residual, alpha=alpha)
            residual = predict_ridge(coef, means, stds, test_x)
            predictions[residual_key][test_mask] = apply_clipped_log_residual(
                test_pred,
                residual,
                pred_min=pred_min,
                pred_max=pred_max,
            )

            direct_key = f"ridge_log_direct_alpha_{alpha:g}"
            coef, means, stds = fit_ridge(train_x, train_log_gt, alpha=alpha)
            log_pred = predict_ridge(coef, means, stds, test_x)
            predictions[direct_key][test_mask] = exp_clipped_log_ttc(log_pred, pred_min=pred_min, pred_max=pred_max)

        if not skip_softmax:
            train_x_scaled, test_x_scaled = standardize_train_test(train_x, test_x)
            weights = fit_softmax_classifier(
                train_x_scaled,
                oracle_labels[train_mask],
                class_count=len(QUANTILE_KEYS),
                epochs=softmax_epochs,
                learning_rate=softmax_learning_rate,
                l2=softmax_l2,
            )
            predicted_labels = predict_softmax(weights, test_x_scaled)
            test_quantiles = quantile_matrix[test_mask]
            selected_pred = test_quantiles[np.arange(test_quantiles.shape[0]), predicted_labels]
            fallback_pred = predictions[f"ridge_log_direct_alpha_{ridge_alphas[0]:g}"][test_mask]
            selected_pred = np.where(np.isfinite(selected_pred), selected_pred, fallback_pred)
            predictions["softmax_quantile_selector"][test_mask] = np.clip(selected_pred, pred_min, pred_max)

        fold_seed = int(seed) + int(len(fold_rows))
        if not skip_mlp:
            mlp_log_direct, _direct_summary = fit_predict_mlp(
                train_x,
                test_x,
                train_sequence_ids,
                train_log_gt,
                hidden_dims=mlp_hidden_dims,
                learning_rate=mlp_learning_rate,
                weight_decay=mlp_weight_decay,
                batch_size=mlp_batch_size,
                epochs=mlp_epochs,
                patience=mlp_patience,
                val_ratio=mlp_val_ratio,
                seed=fold_seed,
            )
            predictions["mlp_log_direct"][test_mask] = exp_clipped_log_ttc(
                mlp_log_direct,
                pred_min=pred_min,
                pred_max=pred_max,
            )
            mlp_log_residual, _residual_summary = fit_predict_mlp(
                train_x,
                test_x,
                train_sequence_ids,
                train_log_residual,
                hidden_dims=mlp_hidden_dims,
                learning_rate=mlp_learning_rate,
                weight_decay=mlp_weight_decay,
                batch_size=mlp_batch_size,
                epochs=mlp_epochs,
                patience=mlp_patience,
                val_ratio=mlp_val_ratio,
                seed=fold_seed + 1000,
            )
            predictions["mlp_log_residual"][test_mask] = apply_clipped_log_residual(
                test_pred,
                mlp_log_residual,
                pred_min=pred_min,
                pred_max=pred_max,
            )

        if enable_lightgbm:
            fold_lgb_predictions: dict[str, np.ndarray] = {}
            for offset, (model_name, objective) in enumerate(lgb_objectives.items()):
                fold_pred = fit_predict_lightgbm_log_ttc(
                    train_lgb_x,
                    train_gt,
                    test_lgb_x,
                    objective=objective,
                    seed=len(fold_rows),
                    n_jobs=lightgbm_n_jobs,
                    pred_min=pred_min,
                    pred_max=pred_max,
                )
                predictions[model_name][test_mask] = fold_pred
                fold_lgb_predictions[model_name] = fold_pred
            l1_l2 = [
                fold_lgb_predictions["lightgbm_l1_log_direct"],
                fold_lgb_predictions["lightgbm_l2_log_direct"],
            ]
            all_lgb = [fold_lgb_predictions[name] for name in lgb_objectives]
            predictions["lightgbm_l1_l2_mean"][test_mask] = np.mean(l1_l2, axis=0)
            predictions["lightgbm_l1_l2_geomean"][test_mask] = np.exp(
                np.mean(np.log(np.maximum(l1_l2, pred_min)), axis=0)
            )
            predictions["lightgbm_l1_l2_huber_fair_mean"][test_mask] = np.mean(all_lgb, axis=0)
            predictions["lightgbm_l1_l2_huber_fair_geomean"][test_mask] = np.exp(
                np.mean(np.log(np.maximum(all_lgb, pred_min)), axis=0)
            )

        fold_rows.append(
            {
                "sequence_id": sequence,
                "train_count": int(np.sum(train_mask)),
                "test_count": int(np.sum(test_mask)),
                "v13_e_ttc_pct": metric_summary(gt[test_mask], v13_pred[test_mask])["e_ttc_pct"],
                "oracle_e_ttc_pct": metric_summary(gt[test_mask], oracle_pred[test_mask])["e_ttc_pct"],
            }
        )

    model_summaries = {
        model_name: metric_summary(gt, pred)
        for model_name, pred in sorted(predictions.items())
    }
    for model_name, pred in sorted(predictions.items()):
        errors = e_ttc_pct(gt, pred)
        for row, pred_value, error_value in zip(rows, pred.tolist(), errors.tolist(), strict=True):
            row[f"{model_name}_pred_ttc_s"] = format_float(float(pred_value))
            row[f"{model_name}_e_ttc_pct"] = format_float(float(error_value))
    return {
        model_name: {
            **summary,
            "feature_count": int(lgb_x_all.shape[1]) if model_name.startswith("lightgbm_") else int(x_all.shape[1]),
            "feature_names": (
                [*feature_names, *lgb_extra_feature_names]
                if model_name.startswith("lightgbm_")
                else feature_names if model_name.startswith("ridge_") else []
            ),
        }
        for model_name, summary in model_summaries.items()
    }, fold_rows


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"V13TabularFrontier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    table, feature_summary = build_feature_table(args)
    if not table:
        raise SystemExit("No V13-valid rows found; cannot run frontier tabular experiment.")

    ridge_alphas = tuple(float(alpha) for alpha in args.ridge_alphas)
    mlp_hidden_dims = parse_hidden_dims(str(args.mlp_hidden_dims))
    model_summaries, fold_rows = run_grouped_models(
        table,
        ridge_alphas=ridge_alphas,
        pred_min=float(args.pred_min_s),
        pred_max=float(args.pred_max_s),
        softmax_epochs=int(args.softmax_epochs),
        softmax_learning_rate=float(args.softmax_learning_rate),
        softmax_l2=float(args.softmax_l2),
        mlp_hidden_dims=mlp_hidden_dims,
        mlp_learning_rate=float(args.mlp_learning_rate),
        mlp_weight_decay=float(args.mlp_weight_decay),
        mlp_batch_size=int(args.mlp_batch_size),
        mlp_epochs=int(args.mlp_epochs),
        mlp_patience=int(args.mlp_patience),
        mlp_val_ratio=float(args.mlp_val_ratio),
        skip_softmax=bool(args.skip_softmax),
        skip_mlp=bool(args.skip_mlp),
        enable_lightgbm=bool(args.enable_lightgbm),
        include_sequence_metadata=not bool(args.disable_sequence_metadata),
        lightgbm_n_jobs=int(args.lightgbm_n_jobs),
        seed=int(args.seed),
    )
    deployable_model_summaries = {
        name: summary
        for name, summary in model_summaries.items()
        if not name.startswith("oracle_")
    }
    best_model = min(deployable_model_summaries.items(), key=lambda item: float(item[1]["e_ttc_pct"]))
    oracle_summary = model_summaries.get("oracle_best_candidate_quantile")

    fieldnames = list(table[0].keys())
    for model_name in sorted(model_summaries):
        fieldnames.extend([f"{model_name}_pred_ttc_s", f"{model_name}_e_ttc_pct"])
    fieldnames = list(dict.fromkeys(fieldnames))
    write_csv(output_dir / "FeatureTable.csv", table, fieldnames)
    write_csv(
        output_dir / "FoldSummary.csv",
        fold_rows,
        ["sequence_id", "train_count", "test_count", "v13_e_ttc_pct", "oracle_e_ttc_pct"],
    )
    summary = {
        "run_name": run_name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "objective": "Reduce fixed V13-valid high-confidence e_ttc_pct below 10 without extra rejection.",
        "args": {
            "v13_run_dir": str(args.v13_run_dir),
            "failure_dir": str(args.failure_dir),
            "geometry_dir": str(args.geometry_dir),
            "sequences": args.sequences,
            "ridge_alphas": list(ridge_alphas),
            "pred_min_s": float(args.pred_min_s),
            "pred_max_s": float(args.pred_max_s),
            "softmax_epochs": int(args.softmax_epochs),
            "softmax_learning_rate": float(args.softmax_learning_rate),
            "softmax_l2": float(args.softmax_l2),
            "mlp_hidden_dims": list(mlp_hidden_dims),
            "mlp_learning_rate": float(args.mlp_learning_rate),
            "mlp_weight_decay": float(args.mlp_weight_decay),
            "mlp_batch_size": int(args.mlp_batch_size),
            "mlp_epochs": int(args.mlp_epochs),
            "mlp_patience": int(args.mlp_patience),
            "mlp_val_ratio": float(args.mlp_val_ratio),
            "skip_softmax": bool(args.skip_softmax),
            "skip_mlp": bool(args.skip_mlp),
            "enable_lightgbm": bool(args.enable_lightgbm),
            "include_sequence_metadata": not bool(args.disable_sequence_metadata),
            "lightgbm_n_jobs": int(args.lightgbm_n_jobs),
            "seed": int(args.seed),
        },
        "feature_table": feature_summary,
        "models": model_summaries,
        "best_model": {
            "name": best_model[0],
            "summary": best_model[1],
        },
        "diagnostic_oracle": oracle_summary,
        "success_lt_10pct": bool(float(best_model[1]["e_ttc_pct"]) < 10.0),
        "leakage_policy": {
            "split": "leave-one-sequence-out",
            "excluded_columns": sorted(GT_DERIVED_COLUMNS),
            "fixed_eval_set": "V13-valid rows only; no extra rejection.",
        },
    }
    write_json(output_dir / "Summary.json", summary)

    lines = [
        f"# {run_name}",
        "",
        "- fixed_eval_set: `V13-valid rows only`",
        f"- row_count: `{feature_summary['row_count']}`",
        f"- sequence_count: `{feature_summary['sequence_count']}`",
        f"- candidate_quantile_rows: `{feature_summary['candidate_quantile_row_count']}`",
        f"- baseline_v13_e_ttc_pct: `{feature_summary['baseline']['e_ttc_pct']:.3f}`",
        f"- best_model: `{best_model[0]}`",
        f"- best_e_ttc_pct: `{float(best_model[1]['e_ttc_pct']):.3f}`",
        f"- success_lt_10pct: `{summary['success_lt_10pct']}`",
        "",
        "## Models",
        "",
        "| model | count | mae_s | e_ttc_pct | median | p90 | over50 | over100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, model_summary in sorted(model_summaries.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{model_name}`",
                    str(model_summary["count"]),
                    f"{float(model_summary['mae_s']):.6f}",
                    f"{float(model_summary['e_ttc_pct']):.3f}",
                    f"{float(model_summary['median_e_ttc_pct']):.3f}",
                    f"{float(model_summary['p90_e_ttc_pct']):.3f}",
                    str(model_summary["over50"]),
                    str(model_summary["over100"]),
                ]
            )
            + " |"
        )
    (output_dir / "Summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Done] output={output_dir}")
    print(f"[Baseline] V13 e_ttc_pct={feature_summary['baseline']['e_ttc_pct']:.3f}")
    print(f"[Best] {best_model[0]} e_ttc_pct={float(best_model[1]['e_ttc_pct']):.3f}")


if __name__ == "__main__":
    main()
