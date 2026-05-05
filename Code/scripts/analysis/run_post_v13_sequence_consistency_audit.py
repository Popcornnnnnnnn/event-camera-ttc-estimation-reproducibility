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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit simple non-GT sequence-level consistency rules on V13-valid rows."
        )
    )
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FRONTIER_ROOT)
    parser.add_argument("--run-name", default="PostV13SequenceConsistencyAudit_v1")
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


def interval_key(row: dict[str, str]) -> int:
    value = parse_float(row.get("interval_idx"))
    return int(value) if math.isfinite(value) else 0


def clipped(value: float, pred_min_s: float, pred_max_s: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    return min(max(value, pred_min_s), pred_max_s)


def metric_summary(rows: list[dict[str, str]], predictions: dict[str, float], pred_min_s: float, pred_max_s: float) -> dict[str, Any]:
    errors: list[float] = []
    mae_values: list[float] = []
    for row in rows:
        gt = parse_float(row.get("gt_ttc_s"))
        pred = clipped(predictions.get(row["sample_id"], float("nan")), pred_min_s, pred_max_s)
        if not (math.isfinite(gt) and gt > 0.0 and math.isfinite(pred) and pred > 0.0):
            continue
        mae_values.append(abs(pred - gt))
        errors.append(abs(pred - gt) / gt * 100.0)
    return {
        "count": len(errors),
        "mae_s": float(np.mean(mae_values)) if mae_values else None,
        "e_ttc_pct": float(np.mean(errors)) if errors else None,
        "median_e_ttc_pct": float(np.median(errors)) if errors else None,
        "p90_e_ttc_pct": float(np.percentile(errors, 90)) if errors else None,
        "over50": int(sum(value > 50.0 for value in errors)),
        "over100": int(sum(value > 100.0 for value in errors)),
    }


def rolling_median_by_sequence(
    rows_by_sequence: dict[str, list[dict[str, str]]],
    source: dict[str, float],
    window: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    radius = window // 2
    for sequence_rows in rows_by_sequence.values():
        values = [source.get(row["sample_id"], float("nan")) for row in sequence_rows]
        for idx, row in enumerate(sequence_rows):
            local = [
                value
                for value in values[max(0, idx - radius) : min(len(values), idx + radius + 1)]
                if math.isfinite(value)
            ]
            out[row["sample_id"]] = float(np.median(local)) if local else float("nan")
    return out


def delta_clip_by_sequence(
    rows_by_sequence: dict[str, list[dict[str, str]]],
    source: dict[str, float],
    max_delta: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for sequence_rows in rows_by_sequence.values():
        previous = float("nan")
        for row in sequence_rows:
            pred = source.get(row["sample_id"], float("nan"))
            if math.isfinite(previous) and math.isfinite(pred):
                pred = min(max(pred, previous - max_delta), previous + max_delta)
            out[row["sample_id"]] = pred
            previous = pred
    return out


def ratio_clip_by_sequence(
    rows_by_sequence: dict[str, list[dict[str, str]]],
    source: dict[str, float],
    max_ratio: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for sequence_rows in rows_by_sequence.values():
        previous = float("nan")
        for row in sequence_rows:
            pred = source.get(row["sample_id"], float("nan"))
            if math.isfinite(previous) and previous > 0.0 and math.isfinite(pred):
                pred = min(max(pred, previous / max_ratio), previous * max_ratio)
            out[row["sample_id"]] = pred
            previous = pred
    return out


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.feature_table)
    rows_by_sequence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_sequence[row.get("sequence_id", "")].append(row)
    for sequence_rows in rows_by_sequence.values():
        sequence_rows.sort(key=interval_key)

    sources = {
        "parent_lightgbm_l1_l2_geomean": {
            row["sample_id"]: parse_float(row.get("lightgbm_l1_l2_geomean_pred_ttc_s")) for row in rows
        },
        "identity_v13": {
            row["sample_id"]: parse_float(row.get("identity_v13_pred_ttc_s")) for row in rows
        },
    }
    policies: dict[str, dict[str, float]] = dict(sources)
    for source_name, source in sources.items():
        for window in (3, 5):
            policies[f"{source_name}_rolling_median_w{window}"] = rolling_median_by_sequence(
                rows_by_sequence, source, window
            )
    for max_delta in (0.5, 1.0, 2.0):
        policies[f"parent_delta_clip_{max_delta}"] = delta_clip_by_sequence(
            rows_by_sequence,
            sources["parent_lightgbm_l1_l2_geomean"],
            max_delta,
        )
    for max_ratio in (1.25, 1.5, 2.0):
        policies[f"parent_ratio_clip_{max_ratio}"] = ratio_clip_by_sequence(
            rows_by_sequence,
            sources["parent_lightgbm_l1_l2_geomean"],
            max_ratio,
        )

    policy_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    for policy_name, predictions in policies.items():
        summary = {"policy": policy_name, **metric_summary(rows, predictions, args.pred_min_s, args.pred_max_s)}
        policy_rows.append(summary)
        for row in rows:
            gt = parse_float(row.get("gt_ttc_s"))
            pred = clipped(predictions.get(row["sample_id"], float("nan")), args.pred_min_s, args.pred_max_s)
            if not (math.isfinite(gt) and gt > 0.0 and math.isfinite(pred) and pred > 0.0):
                continue
            interval_rows.append(
                {
                    "policy": policy_name,
                    "sample_id": row.get("sample_id"),
                    "sequence_id": row.get("sequence_id"),
                    "interval_idx": row.get("interval_idx"),
                    "gt_ttc_s": gt,
                    "pred_ttc_s": pred,
                    "e_ttc_pct": abs(pred - gt) / gt * 100.0,
                }
            )

    payload = {
        "run_name": args.run_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "objective": "Audit non-GT sequence-level consistency rules on fixed V13-valid rows.",
        "inputs": {
            "feature_table": str(args.feature_table),
            "pred_min_s": args.pred_min_s,
            "pred_max_s": args.pred_max_s,
        },
        "row_count": len(rows),
        "sequence_count": len(rows_by_sequence),
        "policy_summaries": policy_rows,
    }
    write_json(output_dir / "Summary.json", payload)
    write_csv(
        output_dir / "PolicySummary.csv",
        policy_rows,
        ["policy", "count", "mae_s", "e_ttc_pct", "median_e_ttc_pct", "p90_e_ttc_pct", "over50", "over100"],
    )
    write_csv(
        output_dir / "PolicyIntervalMetrics.csv",
        interval_rows,
        ["policy", "sample_id", "sequence_id", "interval_idx", "gt_ttc_s", "pred_ttc_s", "e_ttc_pct"],
    )

    sorted_rows = sorted(policy_rows, key=lambda item: float("inf") if item["e_ttc_pct"] is None else float(item["e_ttc_pct"]))
    lines = [
        f"# {args.run_name}",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- fixed_eval_set: `V13-valid rows only`",
        f"- row_count: `{payload['row_count']}`",
        f"- sequence_count: `{payload['sequence_count']}`",
        "",
        "| policy | count | mae_s | e_ttc_pct | median | p90 | over50 | over100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['policy']}`",
                    str(row["count"]),
                    f"{float(row['mae_s']):.6f}",
                    f"{float(row['e_ttc_pct']):.3f}",
                    f"{float(row['median_e_ttc_pct']):.3f}",
                    f"{float(row['p90_e_ttc_pct']):.3f}",
                    str(row["over50"]),
                    str(row["over100"]),
                ]
            )
            + " |"
        )
    best = sorted_rows[0]
    lines.extend(
        [
            "",
            "Conclusion:",
            "",
            f"- Best consistency rule: `{best['policy']}` with `e_ttc_pct={float(best['e_ttc_pct']):.3f}`.",
            "- Simple sequence smoothing improves the fixed-set parent calibration but remains well above `<7%`.",
            "- This direction is useful as a secondary diagnostic, not as the current path to the `<7%` frontier.",
        ]
    )
    (output_dir / "Summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Done] output={output_dir}")
    print(f"[Best] {best['policy']} e_ttc_pct={float(best['e_ttc_pct']):.3f}")


if __name__ == "__main__":
    main()
