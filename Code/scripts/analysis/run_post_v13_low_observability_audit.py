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


CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRONTIER_ROOT = CODE_ROOT / "Experiments" / "PostV13Frontier"
DEFAULT_FEATURE_TABLE = DEFAULT_FRONTIER_ROOT / "V13FrontierLGBM_full_v2" / "FeatureTable.csv"
LOW_OBSERVABILITY_FAMILIES = ("CPNA", "CPNAO")
BASE_MODEL_COLUMNS = (
    "lightgbm_l1_log_direct_pred_ttc_s",
    "lightgbm_l2_log_direct_pred_ttc_s",
    "lightgbm_huber_log_direct_pred_ttc_s",
    "lightgbm_fair_log_direct_pred_ttc_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit CPNA/CPNAO low-observability failure modes and targeted gate policies."
    )
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FRONTIER_ROOT)
    parser.add_argument("--run-name", default="PostV13LowObservabilityAudit_v1")
    parser.add_argument("--base-target", type=float, default=6.875)
    parser.add_argument("--robust-target", type=float, default=7.0)
    parser.add_argument("--low-targets", nargs="+", type=float, default=(5.5, 6.0, 6.5, 6.875, 7.0))
    parser.add_argument("--keep-frac-min", type=float, default=0.45)
    parser.add_argument("--keep-frac-max", type=float, default=0.90)
    parser.add_argument("--keep-frac-steps", type=int, default=31)
    parser.add_argument("--third-keep-frac-steps", type=int, default=9)
    parser.add_argument("--min-train-kept", type=int, default=20)
    parser.add_argument("--pred-min-s", type=float, default=0.05)
    parser.add_argument("--pred-max-s", type=float, default=25.0)
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
        }
    errors = np.abs(pred[valid] - gt[valid]) / gt[valid] * 100.0
    return {
        "count": int(np.sum(valid)),
        "mae_s": float(np.mean(np.abs(pred[valid] - gt[valid]))),
        "e_ttc_pct": float(np.mean(errors)),
        "median_e_ttc_pct": float(np.median(errors)),
        "p90_e_ttc_pct": float(np.percentile(errors, 90)),
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


def abs_log_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return finite_with_median(np.abs(np.log(np.maximum(a, 1e-9)) - np.log(np.maximum(b, 1e-9))))


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
        raise RuntimeError("No valid two-feature gate choice.")
    feasible_grid = valid_grid & (e_grid <= float(target_e_ttc_pct))
    if np.any(feasible_grid):
        index = np.unravel_index(int(np.argmax(np.where(feasible_grid, count_grid, -1.0))), count_grid.shape)
    else:
        index = np.unravel_index(int(np.argmin(np.where(valid_grid, e_grid, np.inf))), e_grid.shape)
    return {
        "keep_frac_a": float(keep_fracs[int(index[0])]),
        "keep_frac_b": float(keep_fracs[int(index[1])]),
        "threshold_a": float(thresholds_a[int(index[0])]),
        "threshold_b": float(thresholds_b[int(index[1])]),
        "train_count": int(count_grid[index]),
        "train_e_ttc_pct": float(e_grid[index]),
    }


def select_three_feature_gate(
    *,
    train_mask: np.ndarray,
    errors: np.ndarray,
    feature_a: np.ndarray,
    feature_b: np.ndarray,
    feature_c: np.ndarray,
    target_e_ttc_pct: float,
    keep_fracs: np.ndarray,
    min_train_kept: int,
) -> dict[str, Any]:
    train_feature_a = feature_a[train_mask]
    train_feature_b = feature_b[train_mask]
    train_feature_c = feature_c[train_mask]
    train_errors = errors[train_mask]
    thresholds_a = np.quantile(train_feature_a, keep_fracs)
    thresholds_b = np.quantile(train_feature_b, keep_fracs)
    thresholds_c = np.quantile(train_feature_c, keep_fracs)
    keep_a = train_feature_a[None, :] <= thresholds_a[:, None]
    keep_b = train_feature_b[None, :] <= thresholds_b[:, None]
    keep_a_f = keep_a.astype(np.float64)
    weighted_keep_a = keep_a_f * train_errors[None, :]
    best_feasible: dict[str, Any] | None = None
    best_any: dict[str, Any] | None = None
    for idx_c, threshold_c in enumerate(thresholds_c):
        keep_c = train_feature_c <= float(threshold_c)
        keep_bc = (keep_b & keep_c[None, :]).astype(np.float64)
        count_grid = keep_a_f @ keep_bc.T
        error_grid = weighted_keep_a @ keep_bc.T
        with np.errstate(divide="ignore", invalid="ignore"):
            e_grid = error_grid / count_grid
        valid_grid = count_grid >= float(min_train_kept)
        if not np.any(valid_grid):
            continue
        any_index = np.unravel_index(int(np.argmin(np.where(valid_grid, e_grid, np.inf))), e_grid.shape)
        any_choice = {
            "keep_frac_a": float(keep_fracs[int(any_index[0])]),
            "keep_frac_b": float(keep_fracs[int(any_index[1])]),
            "keep_frac_c": float(keep_fracs[idx_c]),
            "threshold_a": float(thresholds_a[int(any_index[0])]),
            "threshold_b": float(thresholds_b[int(any_index[1])]),
            "threshold_c": float(threshold_c),
            "train_count": int(count_grid[any_index]),
            "train_e_ttc_pct": float(e_grid[any_index]),
        }
        if best_any is None or float(any_choice["train_e_ttc_pct"]) < float(best_any["train_e_ttc_pct"]):
            best_any = any_choice
        feasible_grid = valid_grid & (e_grid <= float(target_e_ttc_pct))
        if np.any(feasible_grid):
            feasible_index = np.unravel_index(
                int(np.argmax(np.where(feasible_grid, count_grid, -1.0))), count_grid.shape
            )
            feasible_choice = {
                "keep_frac_a": float(keep_fracs[int(feasible_index[0])]),
                "keep_frac_b": float(keep_fracs[int(feasible_index[1])]),
                "keep_frac_c": float(keep_fracs[idx_c]),
                "threshold_a": float(thresholds_a[int(feasible_index[0])]),
                "threshold_b": float(thresholds_b[int(feasible_index[1])]),
                "threshold_c": float(threshold_c),
                "train_count": int(count_grid[feasible_index]),
                "train_e_ttc_pct": float(e_grid[feasible_index]),
            }
            if best_feasible is None or int(feasible_choice["train_count"]) > int(best_feasible["train_count"]):
                best_feasible = feasible_choice
    if best_feasible is not None:
        return best_feasible
    if best_any is not None:
        return best_any
    raise RuntimeError("No valid three-feature gate choice.")


def run_gate(
    *,
    rows: list[dict[str, str]],
    gt: np.ndarray,
    pred: np.ndarray,
    sequence_ids: np.ndarray,
    candidate_iqr: np.ndarray,
    model_log_std: np.ndarray,
    third_feature: np.ndarray | None,
    target: float,
    keep_fracs: np.ndarray,
    min_train_kept: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    errors = np.abs(pred - gt) / np.maximum(gt, 1e-9) * 100.0
    mask = np.zeros(len(rows), dtype=bool)
    fold_rows: list[dict[str, Any]] = []
    for sequence_id in sorted_sequence_ids(sequence_ids):
        test_mask = sequence_ids == sequence_id
        train_mask = ~test_mask
        if third_feature is None:
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
        else:
            choice = select_three_feature_gate(
                train_mask=train_mask,
                errors=errors,
                feature_a=candidate_iqr,
                feature_b=model_log_std,
                feature_c=third_feature,
                target_e_ttc_pct=target,
                keep_fracs=keep_fracs,
                min_train_kept=min_train_kept,
            )
            fold_keep = (
                (candidate_iqr[test_mask] <= float(choice["threshold_a"]))
                & (model_log_std[test_mask] <= float(choice["threshold_b"]))
                & (third_feature[test_mask] <= float(choice["threshold_c"]))
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


def sequence_family(sequence_id: str) -> str:
    if sequence_id.startswith("CPNAO"):
        return "CPNAO"
    if sequence_id.startswith("CPNA"):
        return "CPNA"
    if sequence_id.startswith("CPLA"):
        return "CPLA"
    if sequence_id.startswith("CCRm"):
        return "CCRm"
    if sequence_id.startswith("CCRs"):
        return "CCRs"
    return "other"


def is_low_observability_family(sequence_id: str) -> bool:
    return sequence_family(sequence_id) in LOW_OBSERVABILITY_FAMILIES


def sorted_sequence_ids(sequence_ids: np.ndarray) -> list[str]:
    return sorted({str(seq) for seq in sequence_ids.tolist()})


def vector(rows: list[dict[str, str]], column: str) -> np.ndarray:
    return np.asarray([parse_float(row.get(column)) for row in rows], dtype=np.float64)


def build_model_log_std(rows: list[dict[str, str]]) -> np.ndarray:
    model_matrix = np.column_stack([vector(rows, column) for column in BASE_MODEL_COLUMNS])
    return finite_with_median(np.std(np.log(np.maximum(finite_with_median(model_matrix), 1e-9)), axis=1))


def target_key(target: float) -> str:
    return f"{float(target):.12g}"


def run_family_specific_gate(
    *,
    rows: list[dict[str, str]],
    gt: np.ndarray,
    pred: np.ndarray,
    sequence_ids: np.ndarray,
    family_mask: np.ndarray,
    candidate_iqr: np.ndarray,
    model_log_std: np.ndarray,
    third_feature: np.ndarray | None,
    target: float,
    keep_fracs: np.ndarray,
    min_train_kept: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    errors = np.abs(pred - gt) / np.maximum(gt, 1e-9) * 100.0
    mask = np.zeros(len(rows), dtype=bool)
    fold_rows: list[dict[str, Any]] = []
    for sequence_id in sorted_sequence_ids(sequence_ids):
        test_mask = (sequence_ids == sequence_id) & family_mask
        if not np.any(test_mask):
            continue
        train_mask = family_mask & (sequence_ids != sequence_id)
        if int(np.sum(train_mask)) < min_train_kept:
            train_mask = sequence_ids != sequence_id
        if third_feature is None:
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
        else:
            choice = select_three_feature_gate(
                train_mask=train_mask,
                errors=errors,
                feature_a=candidate_iqr,
                feature_b=model_log_std,
                feature_c=third_feature,
                target_e_ttc_pct=target,
                keep_fracs=keep_fracs,
                min_train_kept=min_train_kept,
            )
            fold_keep = (
                (candidate_iqr[test_mask] <= float(choice["threshold_a"]))
                & (model_log_std[test_mask] <= float(choice["threshold_b"]))
                & (third_feature[test_mask] <= float(choice["threshold_c"]))
            )
        mask[test_mask] = fold_keep
        fold_summary = metric_summary(gt, pred, mask & test_mask)
        fold_rows.append(
            {
                "sequence_id": sequence_id,
                "target_e_ttc_pct": float(target),
                "test_count": int(np.sum(test_mask)),
                "test_kept": int(np.sum(fold_keep)),
                "test_e_ttc_pct": fold_summary["e_ttc_pct"],
                "test_mae_s": fold_summary["mae_s"],
                **choice,
            }
        )
    return mask, fold_rows


def family_summary_rows(
    *,
    rows: list[dict[str, str]],
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    sequence_ids: np.ndarray,
    candidate_iqr: np.ndarray,
    model_log_std: np.ndarray,
    smooth_ratio: np.ndarray,
    policy: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    families = np.asarray([sequence_family(str(seq)) for seq in sequence_ids], dtype=object)
    for family in sorted(set(families.tolist())):
        family_mask = families == family
        kept_mask = family_mask & mask
        rejected_mask = family_mask & ~mask
        summary = metric_summary(gt, pred, kept_mask)
        output.append(
            {
                "policy": policy,
                "family": family,
                "sequence_count": len({sequence_ids[idx] for idx in np.where(family_mask)[0]}),
                "valid_count": int(np.sum(family_mask)),
                "kept": int(np.sum(kept_mask)),
                "rejected": int(np.sum(rejected_mask)),
                "coverage": float(np.sum(kept_mask)) / max(int(np.sum(family_mask)), 1),
                "kept_e_ttc_pct": summary["e_ttc_pct"],
                "kept_p90_e_ttc_pct": summary["p90_e_ttc_pct"],
                "mean_candidate_iqr": float(np.mean(candidate_iqr[family_mask])),
                "mean_model_log_std": float(np.mean(model_log_std[family_mask])),
                "mean_smooth_ratio": float(np.mean(smooth_ratio[family_mask])),
            }
        )
    return output


def sequence_summary_rows(
    *,
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    sequence_ids: np.ndarray,
    policy: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for sequence_id in sorted_sequence_ids(sequence_ids):
        seq_mask = sequence_ids == sequence_id
        kept_mask = seq_mask & mask
        summary = metric_summary(gt, pred, kept_mask)
        output.append(
            {
                "policy": policy,
                "sequence_id": sequence_id,
                "family": sequence_family(sequence_id),
                "valid_count": int(np.sum(seq_mask)),
                "kept": int(np.sum(kept_mask)),
                "coverage": float(np.sum(kept_mask)) / max(int(np.sum(seq_mask)), 1),
                "kept_e_ttc_pct": summary["e_ttc_pct"],
                "kept_p90_e_ttc_pct": summary["p90_e_ttc_pct"],
            }
        )
    return output


def classify_interval(
    *,
    kept: bool,
    sequence_id: str,
    error_pct: float,
    candidate_iqr: float,
    model_log_std: float,
    smooth_ratio: float,
    event_count: float,
    fit_points: float,
    history_collapse_count: float,
    thresholds: dict[str, float],
) -> str:
    low_family = is_low_observability_family(sequence_id)
    if not kept:
        if low_family and event_count <= thresholds["event_count_q20"]:
            return "rejected_low_event_observability"
        if history_collapse_count > 0:
            return "rejected_bbox_collapse_history"
        if candidate_iqr >= thresholds["candidate_iqr_q80"]:
            return "rejected_candidate_disagreement"
        if model_log_std >= thresholds["model_log_std_q80"]:
            return "rejected_model_disagreement"
        if smooth_ratio >= thresholds["smooth_ratio_q80"]:
            return "rejected_smoothing_disagreement"
        if low_family:
            return "rejected_low_observability_family"
        return "rejected_general_confidence"
    if error_pct >= 15.0:
        if low_family and fit_points <= thresholds["fit_points_q20"]:
            return "kept_high_error_sparse_fit"
        if smooth_ratio >= thresholds["smooth_ratio_q80"]:
            return "kept_high_error_smoothing_bias"
        if low_family:
            return "kept_high_error_low_observability_family"
        return "kept_high_error_model_residual"
    if error_pct >= 7.0:
        if low_family:
            return "kept_moderate_error_low_observability_family"
        return "kept_moderate_error"
    return "kept_good"


def taxonomy_rows(
    *,
    rows: list[dict[str, str]],
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    sequence_ids: np.ndarray,
    candidate_iqr: np.ndarray,
    model_log_std: np.ndarray,
    smooth_ratio: np.ndarray,
    thresholds: dict[str, float],
    policy: str,
) -> list[dict[str, Any]]:
    event_count = finite_with_median(vector(rows, "event_count"))
    fit_points = finite_with_median(vector(rows, "fit_points"))
    history_collapse_count = finite_with_median(vector(rows, "history_collapse_count"))
    output: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        error_pct = abs(float(pred[idx]) - float(gt[idx])) / max(float(gt[idx]), 1e-9) * 100.0
        label = classify_interval(
            kept=bool(mask[idx]),
            sequence_id=str(sequence_ids[idx]),
            error_pct=error_pct,
            candidate_iqr=float(candidate_iqr[idx]),
            model_log_std=float(model_log_std[idx]),
            smooth_ratio=float(smooth_ratio[idx]),
            event_count=float(event_count[idx]),
            fit_points=float(fit_points[idx]),
            history_collapse_count=float(history_collapse_count[idx]),
            thresholds=thresholds,
        )
        output.append(
            {
                "policy": policy,
                "sample_id": row.get("sample_id"),
                "sequence_id": sequence_ids[idx],
                "family": sequence_family(str(sequence_ids[idx])),
                "interval_idx": row.get("interval_idx"),
                "kept": int(mask[idx]),
                "gt_ttc_s": float(gt[idx]),
                "pred_ttc_s": float(pred[idx]),
                "e_ttc_pct": error_pct,
                "taxonomy": label,
                "candidate_iqr_over_median": float(candidate_iqr[idx]),
                "model_log_std": float(model_log_std[idx]),
                "smooth_parent_log_ratio": float(smooth_ratio[idx]),
                "event_count": float(event_count[idx]),
                "fit_points": float(fit_points[idx]),
            }
        )
    return output


def aggregate_taxonomy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["policy"], row["family"], row["taxonomy"])].append(row)
    output: list[dict[str, Any]] = []
    for (policy, family, taxonomy), items in sorted(grouped.items()):
        errors = [float(item["e_ttc_pct"]) for item in items if int(item["kept"])]
        output.append(
            {
                "policy": policy,
                "family": family,
                "taxonomy": taxonomy,
                "count": len(items),
                "kept_count": sum(int(item["kept"]) for item in items),
                "mean_kept_e_ttc_pct": float(np.mean(errors)) if errors else None,
            }
        )
    return output


def write_taxonomy_svg(output_path: Path, rows: list[dict[str, Any]], *, policy: str) -> None:
    filtered = [row for row in rows if row["policy"] == policy and row["family"] in LOW_OBSERVABILITY_FAMILIES]
    filtered.sort(key=lambda row: (row["family"], -int(row["count"]), row["taxonomy"]))
    width = 980
    height = 70 + max(len(filtered), 1) * 28
    left = 260
    bar_width = 520
    max_count = max([int(row["count"]) for row in filtered] + [1])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="19" font-weight="700">Low-observability taxonomy: {policy}</text>',
    ]
    for idx, row in enumerate(filtered):
        y = 58 + idx * 28
        count = int(row["count"])
        w = count / max_count * bar_width
        color = "#4575b4" if row["taxonomy"].startswith("kept") else "#d73027"
        parts.extend(
            [
                f'<text x="20" y="{y + 5}" font-family="Arial, sans-serif" font-size="12">{row["family"]} / {row["taxonomy"]}</text>',
                f'<rect x="{left}" y="{y - 10}" width="{bar_width}" height="17" fill="#f1f1f1" stroke="#dddddd"/>',
                f'<rect x="{left}" y="{y - 10}" width="{w:.1f}" height="17" fill="{color}"/>',
                f'<text x="{left + bar_width + 12}" y="{y + 4}" font-family="Arial, sans-serif" font-size="12">{count}</text>',
            ]
        )
    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.feature_table)
    gt = vector(rows, "gt_ttc_s")
    base_pred = np.clip(vector(rows, "lightgbm_l1_l2_geomean_pred_ttc_s"), args.pred_min_s, args.pred_max_s)
    sequence_ids = np.asarray([row.get("sequence_id") for row in rows], dtype=object)
    candidate_iqr = finite_with_median(vector(rows, "candidate_iqr_over_median"))
    model_log_std = build_model_log_std(rows)
    center_w11 = rolling_prediction(
        rows,
        base_pred,
        window=11,
        mode="center",
        pred_min_s=args.pred_min_s,
        pred_max_s=args.pred_max_s,
    )
    smooth_ratio = abs_log_ratio(center_w11, base_pred)
    keep_fracs = np.linspace(args.keep_frac_min, args.keep_frac_max, args.keep_frac_steps)
    third_keep_fracs = np.linspace(args.keep_frac_min, args.keep_frac_max, args.third_keep_frac_steps)
    low_family_mask = np.asarray([is_low_observability_family(str(seq)) for seq in sequence_ids], dtype=bool)

    base_mask, base_folds = run_gate(
        rows=rows,
        gt=gt,
        pred=center_w11,
        sequence_ids=sequence_ids,
        candidate_iqr=candidate_iqr,
        model_log_std=model_log_std,
        third_feature=None,
        target=args.base_target,
        keep_fracs=keep_fracs,
        min_train_kept=args.min_train_kept,
    )
    robust_mask, robust_folds = run_gate(
        rows=rows,
        gt=gt,
        pred=center_w11,
        sequence_ids=sequence_ids,
        candidate_iqr=candidate_iqr,
        model_log_std=model_log_std,
        third_feature=smooth_ratio,
        target=args.robust_target,
        keep_fracs=third_keep_fracs,
        min_train_kept=args.min_train_kept,
    )

    policies: list[dict[str, Any]] = []
    masks: dict[str, np.ndarray] = {}
    folds: list[dict[str, Any]] = []
    for name, mask, policy_folds in [
        (f"global_center_w11_target{args.base_target:g}", base_mask, base_folds),
        (f"global_smoothratio_target{args.robust_target:g}", robust_mask, robust_folds),
    ]:
        summary = metric_summary(gt, center_w11, mask)
        low_summary = metric_summary(gt, center_w11, mask & low_family_mask)
        masks[name] = mask
        policies.append(
            {
                "policy": name,
                "est_valid": summary["count"],
                "e_ttc_pct": summary["e_ttc_pct"],
                "p90_e_ttc_pct": summary["p90_e_ttc_pct"],
                "coverage_of_v13_valid": summary["count"] / max(len(rows), 1),
                "coverage_of_gt_valid": summary["count"] / 3738.0,
                "low_family_est_valid": low_summary["count"],
                "low_family_e_ttc_pct": low_summary["e_ttc_pct"],
                "low_family_coverage": low_summary["count"] / max(int(np.sum(low_family_mask)), 1),
            }
        )
        for fold in policy_folds:
            folds.append({"policy": name, **fold})

    non_low_mask = ~low_family_mask
    fit_points = finite_with_median(vector(rows, "fit_points"))
    history_collapse_count = finite_with_median(vector(rows, "history_collapse_count"))
    event_count = finite_with_median(vector(rows, "event_count"))
    low_smooth_q75 = float(np.quantile(smooth_ratio[low_family_mask], 0.75))
    low_fit_q25 = float(np.quantile(fit_points[low_family_mask], 0.25))
    low_event_q20 = float(np.quantile(event_count[low_family_mask], 0.20))
    guard_specs = [
        ("low_family_smoothratio_q75_guard", low_family_mask & (smooth_ratio >= low_smooth_q75)),
        ("low_family_sparsefit_q25_guard", low_family_mask & (fit_points <= low_fit_q25)),
        ("low_family_event_q20_guard", low_family_mask & (event_count <= low_event_q20)),
        ("low_family_bboxcollapse_guard", low_family_mask & (history_collapse_count > 0)),
        (
            "low_family_combined_observability_guard",
            low_family_mask
            & (
                (smooth_ratio >= low_smooth_q75)
                | (fit_points <= low_fit_q25)
                | (event_count <= low_event_q20)
                | (history_collapse_count > 0)
            ),
        ),
    ]
    for guard_name, reject_mask in guard_specs:
        guarded_mask = base_mask & ~reject_mask
        summary = metric_summary(gt, center_w11, guarded_mask)
        low_summary = metric_summary(gt, center_w11, guarded_mask & low_family_mask)
        policies.append(
            {
                "policy": f"{guard_name}_plus_global_rest",
                "est_valid": summary["count"],
                "e_ttc_pct": summary["e_ttc_pct"],
                "p90_e_ttc_pct": summary["p90_e_ttc_pct"],
                "coverage_of_v13_valid": summary["count"] / max(len(rows), 1),
                "coverage_of_gt_valid": summary["count"] / 3738.0,
                "low_family_est_valid": low_summary["count"],
                "low_family_e_ttc_pct": low_summary["e_ttc_pct"],
                "low_family_coverage": low_summary["count"] / max(int(np.sum(low_family_mask)), 1),
            }
        )
        masks[f"{guard_name}_plus_global_rest"] = guarded_mask

    for target in args.low_targets:
        low_mask, low_folds = run_family_specific_gate(
            rows=rows,
            gt=gt,
            pred=center_w11,
            sequence_ids=sequence_ids,
            family_mask=low_family_mask,
            candidate_iqr=candidate_iqr,
            model_log_std=model_log_std,
            third_feature=None,
            target=float(target),
            keep_fracs=keep_fracs,
            min_train_kept=args.min_train_kept,
        )
        hybrid_mask = (base_mask & non_low_mask) | low_mask
        policy_name = f"low_family_specific_target{float(target):g}_plus_global_rest"
        summary = metric_summary(gt, center_w11, hybrid_mask)
        low_summary = metric_summary(gt, center_w11, hybrid_mask & low_family_mask)
        masks[policy_name] = hybrid_mask
        policies.append(
            {
                "policy": policy_name,
                "est_valid": summary["count"],
                "e_ttc_pct": summary["e_ttc_pct"],
                "p90_e_ttc_pct": summary["p90_e_ttc_pct"],
                "coverage_of_v13_valid": summary["count"] / max(len(rows), 1),
                "coverage_of_gt_valid": summary["count"] / 3738.0,
                "low_family_est_valid": low_summary["count"],
                "low_family_e_ttc_pct": low_summary["e_ttc_pct"],
                "low_family_coverage": low_summary["count"] / max(int(np.sum(low_family_mask)), 1),
            }
        )
        for fold in low_folds:
            folds.append({"policy": policy_name, **fold})

    policies.sort(key=lambda row: (float(row["e_ttc_pct"] or 1e9) >= 7.0, -float(row["coverage_of_v13_valid"])))
    best_policy = next((row for row in policies if float(row["e_ttc_pct"]) < 7.0), policies[0])

    thresholds = {
        "candidate_iqr_q80": float(np.quantile(candidate_iqr, 0.80)),
        "model_log_std_q80": float(np.quantile(model_log_std, 0.80)),
        "smooth_ratio_q80": float(np.quantile(smooth_ratio, 0.80)),
        "event_count_q20": float(np.quantile(finite_with_median(vector(rows, "event_count")), 0.20)),
        "fit_points_q20": float(np.quantile(finite_with_median(vector(rows, "fit_points")), 0.20)),
    }

    all_family_rows: list[dict[str, Any]] = []
    all_sequence_rows: list[dict[str, Any]] = []
    all_interval_rows: list[dict[str, Any]] = []
    for policy, mask in masks.items():
        all_family_rows.extend(
            family_summary_rows(
                rows=rows,
                gt=gt,
                pred=center_w11,
                mask=mask,
                sequence_ids=sequence_ids,
                candidate_iqr=candidate_iqr,
                model_log_std=model_log_std,
                smooth_ratio=smooth_ratio,
                policy=policy,
            )
        )
        all_sequence_rows.extend(
            sequence_summary_rows(gt=gt, pred=center_w11, mask=mask, sequence_ids=sequence_ids, policy=policy)
        )
    taxonomy_policies = list(
        dict.fromkeys(
            [
                f"global_center_w11_target{args.base_target:g}",
                f"global_smoothratio_target{args.robust_target:g}",
                str(best_policy["policy"]),
            ]
        )
    )
    for policy in taxonomy_policies:
        mask = masks[policy]
        all_interval_rows.extend(
            taxonomy_rows(
                rows=rows,
                gt=gt,
                pred=center_w11,
                mask=mask,
                sequence_ids=sequence_ids,
                candidate_iqr=candidate_iqr,
                model_log_std=model_log_std,
                smooth_ratio=smooth_ratio,
                thresholds=thresholds,
                policy=policy,
            )
        )

    taxonomy_summary = aggregate_taxonomy(all_interval_rows)
    write_csv(
        output_dir / "PolicySweep.csv",
        policies,
        [
            "policy",
            "est_valid",
            "e_ttc_pct",
            "p90_e_ttc_pct",
            "coverage_of_v13_valid",
            "coverage_of_gt_valid",
            "low_family_est_valid",
            "low_family_e_ttc_pct",
            "low_family_coverage",
        ],
    )
    write_csv(output_dir / "FoldSummary.csv", folds, sorted({key for row in folds for key in row.keys()}))
    write_csv(
        output_dir / "FamilySummary.csv",
        all_family_rows,
        [
            "policy",
            "family",
            "sequence_count",
            "valid_count",
            "kept",
            "rejected",
            "coverage",
            "kept_e_ttc_pct",
            "kept_p90_e_ttc_pct",
            "mean_candidate_iqr",
            "mean_model_log_std",
            "mean_smooth_ratio",
        ],
    )
    write_csv(
        output_dir / "SequenceSummary.csv",
        all_sequence_rows,
        ["policy", "sequence_id", "family", "valid_count", "kept", "coverage", "kept_e_ttc_pct", "kept_p90_e_ttc_pct"],
    )
    write_csv(
        output_dir / "IntervalTaxonomy.csv",
        all_interval_rows,
        [
            "policy",
            "sample_id",
            "sequence_id",
            "family",
            "interval_idx",
            "kept",
            "gt_ttc_s",
            "pred_ttc_s",
            "e_ttc_pct",
            "taxonomy",
            "candidate_iqr_over_median",
            "model_log_std",
            "smooth_parent_log_ratio",
            "event_count",
            "fit_points",
        ],
    )
    write_csv(
        output_dir / "TaxonomySummary.csv",
        taxonomy_summary,
        ["policy", "family", "taxonomy", "count", "kept_count", "mean_kept_e_ttc_pct"],
    )
    write_taxonomy_svg(output_dir / "LowObservabilityTaxonomy.svg", taxonomy_summary, policy=str(best_policy["policy"]))

    payload = {
        "run_name": args.run_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "feature_table": str(args.feature_table),
        "best_policy": best_policy,
        "thresholds": thresholds,
        "policies": policies,
        "guard_thresholds": {
            "low_smooth_q75": low_smooth_q75,
            "low_fit_q25": low_fit_q25,
            "low_event_q20": low_event_q20,
        },
    }
    write_json(output_dir / "Summary.json", payload)

    lines = [
        f"# {args.run_name}",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- row_count: `{len(rows)}`",
        f"- low_observability_families: `{', '.join(LOW_OBSERVABILITY_FAMILIES)}`",
        f"- chart: `LowObservabilityTaxonomy.svg`",
        "",
        "| policy | est_valid | coverage_v13 | coverage_gt | e_ttc_pct | p90 | low_family_valid | low_family_cov | low_family_e |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in policies:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['policy']}`",
                    str(row["est_valid"]),
                    f"{float(row['coverage_of_v13_valid']):.3f}",
                    f"{float(row['coverage_of_gt_valid']):.3f}",
                    f"{float(row['e_ttc_pct']):.3f}",
                    f"{float(row['p90_e_ttc_pct']):.3f}",
                    str(row["low_family_est_valid"]),
                    f"{float(row['low_family_coverage']):.3f}",
                    f"{float(row['low_family_e_ttc_pct']):.3f}" if row["low_family_e_ttc_pct"] is not None else "",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Best `<7%` policy by coverage:",
            "",
            f"- `{best_policy['policy']}`",
            f"- `est_valid={best_policy['est_valid']}`",
            f"- `e_ttc_pct={float(best_policy['e_ttc_pct']):.3f}`",
            f"- `coverage_of_v13_valid={float(best_policy['coverage_of_v13_valid']):.3f}`",
            f"- `low_family_coverage={float(best_policy['low_family_coverage']):.3f}`",
            f"- `low_family_e_ttc_pct={float(best_policy['low_family_e_ttc_pct']):.3f}`",
        ]
    )
    (output_dir / "Summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[Done] output={output_dir}")
    print(
        f"[Best] {best_policy['policy']} count={best_policy['est_valid']} "
        f"e_ttc_pct={float(best_policy['e_ttc_pct']):.3f} "
        f"coverage_v13={float(best_policy['coverage_of_v13_valid']):.3f} "
        f"low_family_e={float(best_policy['low_family_e_ttc_pct']):.3f}"
    )


if __name__ == "__main__":
    main()
