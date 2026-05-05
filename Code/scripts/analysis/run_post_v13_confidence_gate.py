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


CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_TABLE = (
    CODE_ROOT
    / "Experiments"
    / "PostV13Frontier"
    / "V13FrontierLGBM_full_v2"
    / "FeatureTable.csv"
)
DEFAULT_V13_SUMMARY = (
    CODE_ROOT
    / "Experiments"
    / "SparsityAwareLTS"
    / "BBoxLoomingV13_20260427_212537"
    / "Summary.json"
)

BASE_MODEL_COLUMNS = [
    "lightgbm_l1_log_direct_pred_ttc_s",
    "lightgbm_l2_log_direct_pred_ttc_s",
    "lightgbm_huber_log_direct_pred_ttc_s",
    "lightgbm_fair_log_direct_pred_ttc_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run post-V13 high-confidence gates on fixed V13-valid learned-calibration "
            "outputs using leave-one-sequence-out threshold selection."
        ),
    )
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--v13-summary", type=Path, default=DEFAULT_V13_SUMMARY)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CODE_ROOT / "Experiments" / "PostV13Frontier",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--base-pred-column",
        default="lightgbm_l1_l2_geomean_pred_ttc_s",
        help="Prediction column to gate. Default is the previous best full-metadata calibration.",
    )
    parser.add_argument("--target-e-ttc-pct", type=float, default=6.9)
    parser.add_argument("--keep-frac-min", type=float, default=0.45)
    parser.add_argument("--keep-frac-max", type=float, default=0.85)
    parser.add_argument("--keep-frac-steps", type=int, default=41)
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


def metric_summary(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    valid = mask & np.isfinite(gt) & np.isfinite(pred) & (gt > 0.0)
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


def quantile_grid(start: float, stop: float, steps: int) -> np.ndarray:
    if steps <= 1:
        return np.asarray([float(stop)], dtype=np.float64)
    return np.linspace(float(start), float(stop), int(steps), dtype=np.float64)


def finite_with_median(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float64, copy=True)
    median = float(np.nanmedian(out)) if np.any(np.isfinite(out)) else 0.0
    out[~np.isfinite(out)] = median
    return out


def build_gate_features(rows: list[dict[str, str]], base_pred: np.ndarray) -> dict[str, np.ndarray]:
    candidate_iqr = finite_with_median(
        np.asarray([parse_float(row.get("candidate_iqr_over_median")) for row in rows], dtype=np.float64)
    )
    model_matrix = np.column_stack(
        [
            np.asarray([parse_float(row.get(column)) for row in rows], dtype=np.float64)
            for column in BASE_MODEL_COLUMNS
        ]
    )
    model_log_std = np.std(np.log(np.maximum(finite_with_median(model_matrix), 1e-9)), axis=1)
    return {
        "candidate_iqr_over_median": candidate_iqr,
        "model_log_std": finite_with_median(model_log_std),
        "base_pred_ttc_s": finite_with_median(base_pred),
    }


def select_single_feature_gate(
    *,
    train_mask: np.ndarray,
    errors: np.ndarray,
    feature: np.ndarray,
    target_e_ttc_pct: float,
    keep_fracs: np.ndarray,
    min_train_kept: int,
) -> dict[str, float | int]:
    choices: list[dict[str, float | int]] = []
    for keep_frac in keep_fracs:
        threshold = float(np.quantile(feature[train_mask], keep_frac))
        kept = train_mask & (feature <= threshold)
        count = int(np.sum(kept))
        if count < min_train_kept:
            continue
        choices.append(
            {
                "keep_frac": float(keep_frac),
                "threshold": threshold,
                "train_count": count,
                "train_e_ttc_pct": float(np.mean(errors[kept])),
            }
        )
    if not choices:
        raise RuntimeError("No valid single-feature gate choices.")
    feasible = [item for item in choices if float(item["train_e_ttc_pct"]) <= target_e_ttc_pct]
    if feasible:
        return max(feasible, key=lambda item: int(item["train_count"]))
    return min(choices, key=lambda item: float(item["train_e_ttc_pct"]))


def select_two_feature_gate(
    *,
    train_mask: np.ndarray,
    errors: np.ndarray,
    feature_a: np.ndarray,
    feature_b: np.ndarray,
    target_e_ttc_pct: float,
    keep_fracs: np.ndarray,
    min_train_kept: int,
) -> dict[str, float | int]:
    choices: list[dict[str, float | int]] = []
    for keep_frac_a in keep_fracs:
        threshold_a = float(np.quantile(feature_a[train_mask], keep_frac_a))
        for keep_frac_b in keep_fracs:
            threshold_b = float(np.quantile(feature_b[train_mask], keep_frac_b))
            kept = train_mask & (feature_a <= threshold_a) & (feature_b <= threshold_b)
            count = int(np.sum(kept))
            if count < min_train_kept:
                continue
            choices.append(
                {
                    "keep_frac_a": float(keep_frac_a),
                    "keep_frac_b": float(keep_frac_b),
                    "threshold_a": threshold_a,
                    "threshold_b": threshold_b,
                    "train_count": count,
                    "train_e_ttc_pct": float(np.mean(errors[kept])),
                }
            )
    if not choices:
        raise RuntimeError("No valid two-feature gate choices.")
    feasible = [item for item in choices if float(item["train_e_ttc_pct"]) <= target_e_ttc_pct]
    if feasible:
        return max(feasible, key=lambda item: int(item["train_count"]))
    return min(choices, key=lambda item: float(item["train_e_ttc_pct"]))


def run_gates(
    *,
    rows: list[dict[str, str]],
    gt: np.ndarray,
    pred: np.ndarray,
    features: dict[str, np.ndarray],
    target_e_ttc_pct: float,
    keep_fracs: np.ndarray,
    min_train_kept: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    sequences = np.asarray([row["sequence_id"] for row in rows], dtype=object)
    errors = np.abs(pred - gt) / np.maximum(gt, 1e-9) * 100.0
    policies = {
        "iqr_gate": np.zeros(len(rows), dtype=bool),
        "iqr_modelstd_gate": np.zeros(len(rows), dtype=bool),
    }
    fold_rows: list[dict[str, Any]] = []
    for sequence_id in sorted(set(sequences.tolist())):
        test_mask = sequences == sequence_id
        train_mask = ~test_mask
        iqr_choice = select_single_feature_gate(
            train_mask=train_mask,
            errors=errors,
            feature=features["candidate_iqr_over_median"],
            target_e_ttc_pct=target_e_ttc_pct,
            keep_fracs=keep_fracs,
            min_train_kept=min_train_kept,
        )
        policies["iqr_gate"][test_mask] = (
            features["candidate_iqr_over_median"][test_mask] <= float(iqr_choice["threshold"])
        )
        two_choice = select_two_feature_gate(
            train_mask=train_mask,
            errors=errors,
            feature_a=features["candidate_iqr_over_median"],
            feature_b=features["model_log_std"],
            target_e_ttc_pct=target_e_ttc_pct,
            keep_fracs=keep_fracs,
            min_train_kept=min_train_kept,
        )
        policies["iqr_modelstd_gate"][test_mask] = (
            (features["candidate_iqr_over_median"][test_mask] <= float(two_choice["threshold_a"]))
            & (features["model_log_std"][test_mask] <= float(two_choice["threshold_b"]))
        )
        for policy_name, choice, test_kept in [
            ("iqr_gate", iqr_choice, policies["iqr_gate"][test_mask]),
            ("iqr_modelstd_gate", two_choice, policies["iqr_modelstd_gate"][test_mask]),
        ]:
            test_indices = np.where(test_mask)[0]
            selected = np.zeros(len(rows), dtype=bool)
            selected[test_indices] = test_kept
            fold_summary = metric_summary(gt, pred, selected)
            fold_rows.append(
                {
                    "policy": policy_name,
                    "sequence_id": sequence_id,
                    "test_count": int(np.sum(test_mask)),
                    "test_kept": int(np.sum(test_kept)),
                    "test_e_ttc_pct": fold_summary["e_ttc_pct"],
                    "test_mae_s": fold_summary["mae_s"],
                    **choice,
                }
            )
    return policies, fold_rows


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"V13ConfidenceGate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.feature_table)
    gt = np.asarray([parse_float(row.get("gt_ttc_s")) for row in rows], dtype=np.float64)
    pred = np.asarray([parse_float(row.get(args.base_pred_column)) for row in rows], dtype=np.float64)
    features = build_gate_features(rows, pred)
    keep_fracs = quantile_grid(args.keep_frac_min, args.keep_frac_max, args.keep_frac_steps)
    policies, fold_rows = run_gates(
        rows=rows,
        gt=gt,
        pred=pred,
        features=features,
        target_e_ttc_pct=float(args.target_e_ttc_pct),
        keep_fracs=keep_fracs,
        min_train_kept=int(args.min_train_kept),
    )

    v13_summary = json.loads(args.v13_summary.read_text(encoding="utf-8"))
    base_mask = np.ones(len(rows), dtype=bool)
    policy_summaries: dict[str, Any] = {
        "base_all_v13_valid": metric_summary(gt, pred, base_mask),
    }
    for policy_name, mask in policies.items():
        summary = metric_summary(gt, pred, mask)
        empty_sequences = [
            sequence_id
            for sequence_id in sorted({row["sequence_id"] for row in rows})
            if not np.any(mask & (np.asarray([row["sequence_id"] for row in rows], dtype=object) == sequence_id))
        ]
        summary["rejected_from_v13_valid"] = int(len(rows) - int(summary["count"]))
        summary["coverage_of_v13_valid"] = float(int(summary["count"]) / max(len(rows), 1))
        summary["coverage_of_gt_valid"] = float(
            int(summary["count"]) / max(int(v13_summary["gt_valid_count"]), 1)
        )
        summary["total_failure_if_counting_rejection"] = int(
            v13_summary["failure_count"] + len(rows) - int(summary["count"])
        )
        summary["nonempty_sequence_count"] = int(len({row["sequence_id"] for row in rows}) - len(empty_sequences))
        summary["empty_sequences"] = empty_sequences
        policy_summaries[policy_name] = summary

    interval_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        out = {
            "sample_id": row["sample_id"],
            "sequence_id": row["sequence_id"],
            "interval_idx": row.get("interval_idx", ""),
            "gt_ttc_s": row.get("gt_ttc_s", ""),
            "base_pred_ttc_s": f"{pred[idx]:.9f}",
            "base_e_ttc_pct": f"{abs(pred[idx] - gt[idx]) / max(gt[idx], 1e-9) * 100.0:.9f}",
            "candidate_iqr_over_median": f"{features['candidate_iqr_over_median'][idx]:.9f}",
            "model_log_std": f"{features['model_log_std'][idx]:.9f}",
        }
        for policy_name, mask in policies.items():
            out[f"{policy_name}_kept"] = "1" if bool(mask[idx]) else "0"
            out[f"{policy_name}_pred_ttc_s"] = f"{pred[idx]:.9f}" if bool(mask[idx]) else ""
        interval_rows.append(out)

    write_csv(
        output_dir / "GateIntervalMetrics.csv",
        interval_rows,
        list(interval_rows[0].keys()),
    )
    write_csv(
        output_dir / "FoldGateSummary.csv",
        fold_rows,
        [
            "policy",
            "sequence_id",
            "test_count",
            "test_kept",
            "test_e_ttc_pct",
            "test_mae_s",
            "keep_frac",
            "threshold",
            "keep_frac_a",
            "keep_frac_b",
            "threshold_a",
            "threshold_b",
            "train_count",
            "train_e_ttc_pct",
        ],
    )

    best_policy = min(
        policy_summaries.items(),
        key=lambda item: float("inf") if item[1]["e_ttc_pct"] is None else float(item[1]["e_ttc_pct"]),
    )
    no_empty_policy_items = [
        item
        for item in policy_summaries.items()
        if not item[0].endswith("_all_v13_valid") and not item[1].get("empty_sequences")
    ]
    best_no_empty_policy = (
        min(no_empty_policy_items, key=lambda item: float(item[1]["e_ttc_pct"]))
        if no_empty_policy_items
        else None
    )
    summary = {
        "run_name": run_name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "objective": "Explore post-V13 high-confidence gating below 7% e_ttc_pct.",
        "args": {
            "feature_table": str(args.feature_table),
            "v13_summary": str(args.v13_summary),
            "base_pred_column": args.base_pred_column,
            "target_e_ttc_pct": float(args.target_e_ttc_pct),
            "keep_frac_min": float(args.keep_frac_min),
            "keep_frac_max": float(args.keep_frac_max),
            "keep_frac_steps": int(args.keep_frac_steps),
            "min_train_kept": int(args.min_train_kept),
        },
        "fixed_parent_eval": {
            "v13_valid_count": len(rows),
            "gt_valid": int(v13_summary["gt_valid_count"]),
            "v13_failure": int(v13_summary["failure_count"]),
            "parent_calibration": "V13FrontierLGBM_full_v2 / lightgbm_l1_l2_geomean",
        },
        "policy_summaries": policy_summaries,
        "best_policy": {
            "name": best_policy[0],
            "summary": best_policy[1],
        },
        "best_no_empty_policy": None
        if best_no_empty_policy is None
        else {
            "name": best_no_empty_policy[0],
            "summary": best_no_empty_policy[1],
        },
        "success_lt_7pct": bool(
            best_policy[1]["e_ttc_pct"] is not None and float(best_policy[1]["e_ttc_pct"]) < 7.0
        ),
        "leakage_policy": {
            "base_prediction_split": "parent predictions are grouped OOF from V13FrontierLGBM_full_v2",
            "gate_threshold_split": "for each held-out sequence, thresholds are selected using only the other sequences",
            "test_time_inputs": [
                "candidate_iqr_over_median",
                "model_log_std",
                "parent calibration prediction",
            ],
            "gt_usage": "GT is used only on training folds to select confidence thresholds, never on the held-out sequence.",
        },
    }
    write_json(output_dir / "Summary.json", summary)

    lines = [
        f"# {run_name}",
        "",
        "- parent: `V13FrontierLGBM_full_v2 / lightgbm_l1_l2_geomean`",
        "- split: `leave-one-sequence-out threshold selection`",
        f"- target_e_ttc_pct: `{args.target_e_ttc_pct:.3f}`",
        f"- success_lt_7pct: `{summary['success_lt_7pct']}`",
        "",
        "| policy | count | seq_nonempty | coverage_of_v13 | coverage_of_gt | mae_s | e_ttc_pct | median | p90 | over50 | over100 | total_failure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy_name, policy_summary in policy_summaries.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{policy_name}`",
                    str(policy_summary["count"]),
                    str(policy_summary.get("nonempty_sequence_count", len({row["sequence_id"] for row in rows}))),
                    f"{float(policy_summary.get('coverage_of_v13_valid', 1.0)):.3f}",
                    f"{float(policy_summary.get('coverage_of_gt_valid', len(rows) / int(v13_summary['gt_valid_count']))):.3f}",
                    "" if policy_summary["mae_s"] is None else f"{float(policy_summary['mae_s']):.6f}",
                    "" if policy_summary["e_ttc_pct"] is None else f"{float(policy_summary['e_ttc_pct']):.3f}",
                    ""
                    if policy_summary["median_e_ttc_pct"] is None
                    else f"{float(policy_summary['median_e_ttc_pct']):.3f}",
                    ""
                    if policy_summary["p90_e_ttc_pct"] is None
                    else f"{float(policy_summary['p90_e_ttc_pct']):.3f}",
                    str(policy_summary["over50"]),
                    str(policy_summary["over100"]),
                    str(policy_summary.get("total_failure_if_counting_rejection", v13_summary["failure_count"])),
                ]
            )
            + " |"
        )
    (output_dir / "Summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[Done] output={output_dir}")
    print(f"[Best] {best_policy[0]} e_ttc_pct={float(best_policy[1]['e_ttc_pct']):.3f}")


if __name__ == "__main__":
    main()
